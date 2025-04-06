import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from optimizer.basic_optimizer import Basic_Optimizer
from scipy.stats import qmc
from torch.distributions import Normal


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32, hidden_dim=256, num_layers=1, 
                 lr=0.0003, beta=1.0, device=None, initial_max_std=0.2, initial_min_std=0.1):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.beta = beta
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        log_max_std = np.log(initial_max_std).astype(np.float32)
        log_min_std = np.log(initial_min_std).astype(np.float32)
        self.max_logstd = nn.Parameter(torch.full((1, 1), log_max_std))
        self.min_logstd = nn.Parameter(torch.full((1, 1), log_min_std))
        
        # encoder layer
        self.encoder_layers = []
        for i in range(num_layers):
            if i == 0:
                self.encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.encoder_layers.append(nn.LeakyReLU())
        self.encoder_layers.append(nn.Linear(hidden_dim, latent_dim * 2))
        self.encoder = nn.Sequential(*self.encoder_layers)
        
        # decoder layer
        self.decoder_layers = []
        for i in range(num_layers):
            if i == 0:
                self.decoder_layers.append(nn.Linear(latent_dim, hidden_dim))
            else:
                self.decoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.decoder_layers.append(nn.LeakyReLU())
        self.decoder_layers.append(nn.Linear(hidden_dim, input_dim * 2))
        self.decoder = nn.Sequential(*self.decoder_layers)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.to(self.device)
        
    def encode(self, x):
        h = self.encoder(x)
        mean, logstd = torch.chunk(h, 2, dim=-1)
        
        logstd = self.max_logstd - F.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + F.softplus(logstd - self.min_logstd)
        
        return mean, logstd
    
    def decode(self, z):
        h = self.decoder(z)
        mean, logstd = torch.chunk(h, 2, dim=-1)
        
        logstd = self.max_logstd - F.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + F.softplus(logstd - self.min_logstd)
        
        return mean, logstd
    
    def reparameterize(self, mean, logstd):
        std = torch.exp(logstd)  
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x):
        mean, logstd = self.encode(x)
        z = self.reparameterize(mean, logstd)
        x_mean, x_logstd = self.decode(z)
        return x_mean, x_logstd, mean, logstd, z
    
    def sample_from_z(self, z):
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            x_mean, x_logstd = self.decode(z)
            std = torch.exp(x_logstd)
            eps = torch.randn_like(std)
            samples = x_mean + eps * std
        
        self.train(was_training)
        return samples
    
    def log_prob(self, x, z=None):
        was_training = self.training
        self.eval()
        
        if x.device != self.device:
            x = x.to(self.device)
        with torch.no_grad():
            if z is None:
                mean, logstd = self.encode(x)
                z = self.reparameterize(mean, logstd)
            elif z.device != self.device:
                z = z.to(self.device)
            
            x_mean, x_logstd = self.decode(z)

            dist = Normal(loc=x_mean, scale=torch.exp(x_logstd))
            result = dist.log_prob(x).sum(dim=-1)
        
        self.train(was_training)
        return result
    
    def train_model(self, x_data, weights=None, batch_size=32, epochs=10):
        x_tensor = torch.tensor(x_data, dtype=torch.float32, device=self.device)
        
        if weights is None:
            weights = np.ones((len(x_data), 1))
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)

        dataset = TensorDataset(x_tensor, weights_tensor)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            self.train()
            for batch_x, batch_w in train_loader:
                self.optimizer.zero_grad()
                x_mean, x_logstd, z_mean, z_logstd, z = self.forward(batch_x)
                x_var = torch.exp(2 * x_logstd)
                recon_loss = 0.5 * torch.sum(
                    (x_logstd + (batch_x - x_mean) ** 2 / x_var + np.log(2 * np.pi)), 
                    dim=1, keepdim=True
                )
                kl_loss = -0.5 * torch.sum(
                    1 + 2 * z_logstd - z_mean.pow(2) - torch.exp(2 * z_logstd),
                    dim=1, keepdim=True
                )
                loss = torch.mean(batch_w * (recon_loss + self.beta * kl_loss))
                loss.backward()
                self.optimizer.step()
        
        return True
    
    def copy_weights_from(self, other_vae):
        self.load_state_dict(other_vae.state_dict())

class CbAS_Optimizer(Basic_Optimizer):
    def __init__(self, config):
        super().__init__(config)

        config.cbas_latent_dim = 32
        # VAE 
        config.cbas_hidden_dim = 256
        config.cbas_num_layers = 1
        config.cbas_percentile = 25
        # ensemble
        config.cbas_num_models = 5
        config.cbas_vae_epochs = 10
        config.cbas_ensemble_epochs = 100
        # CbAS
        config.cbas_batch_size = 100
        if config.dim <= 10:
            config.cbas_samples_per_iter = 100 * config.dim
        else:
            config.cbas_samples_per_iter = 200 * config.dim
            
        self.__config = config
        
        self.log_interval = config.log_interval
        
    def run_episode(self, problem):
        torch.set_grad_enabled(True)
        fes = 0
        log_index = 1
        best_y = float('inf')
        cost = []

        sampler = qmc.LatinHypercube(d=self.__config.dim)
        x_normalized = sampler.random(n=200000)
        x_dataset = problem.lb + x_normalized * (problem.ub - problem.lb)

        if problem.optimum is None:
            y_dataset = problem.eval(x_dataset)
        else:
            y_dataset = problem.eval(x_dataset) - problem.optimum


        sort_indices = np.argsort( y_dataset.flatten())
        # to get the 25% samples which have the lower y 
        top_25_percent = int(0.25 * len(sort_indices))
        top_indices = sort_indices[:top_25_percent]
        high_quality_x = x_dataset[top_indices]

        prior_vae = VAE(
            input_dim=self.__config.dim,
            latent_dim=self.__config.cbas_latent_dim,
            hidden_dim=self.__config.cbas_hidden_dim,
            num_layers=self.__config.cbas_num_layers,
            device=self.__config.device
        )
        
        working_vae = VAE(
            input_dim=self.__config.dim,
            latent_dim=self.__config.cbas_latent_dim,
            hidden_dim=self.__config.cbas_hidden_dim,
            num_layers=self.__config.cbas_num_layers,
            device=self.__config.device
        )


        prior_vae.train_model(high_quality_x, batch_size=self.__config.cbas_batch_size,
                            epochs=self.__config.cbas_vae_epochs * 2) 
        working_vae.copy_weights_from(prior_vae)
        y_star = np.percentile(y_dataset, self.__config.cbas_percentile)


        iteration = 0
        done = False
        while not done:
            iteration += 1
            z = torch.randn(self.__config.cbas_samples_per_iter, self.__config.cbas_latent_dim, device=self.__config.device)

            vae_working_samples = working_vae.sample_from_z(z).cpu().numpy()
            vae_prior_samples = prior_vae.sample_from_z(z).cpu().numpy()

            
            if problem.optimum is None:
                y_samples = problem.eval(vae_working_samples)
            else:
                y_samples = problem.eval(vae_working_samples) - problem.optimum
            fes += self.__config.cbas_samples_per_iter

            min_y = np.min(y_samples)
            if min_y < best_y:
                best_y = min_y
            
            vae_prior_samples_tensor = torch.tensor(vae_prior_samples, dtype=torch.float32, device=self.__config.device)
            prior_log_prob = prior_vae.log_prob(vae_prior_samples_tensor,z) 
            vae_working_samples_tensor = torch.tensor(vae_working_samples, dtype=torch.float32, device=self.__config.device)
            current_log_prob = working_vae.log_prob(vae_working_samples_tensor,z)
            log_ratio = prior_log_prob - current_log_prob
            prob_ratio = torch.exp(log_ratio).cpu().numpy().reshape(-1, 1)

            y_star_new = np.percentile(y_samples, self.__config.cbas_percentile)
            y_star = min(y_star,y_star_new)
            
            # prob_below_threshold = ((y_samples < y_star).astype(float).sum()/len(y_samples))
            prob_below_threshold = (y_samples < y_star).astype(float).reshape(-1, 1)
            weights = prob_ratio * prob_below_threshold
            weights = np.clip(weights, -20.0, 20.0)
            weights = weights / np.mean(weights)
            working_vae.train_model(vae_working_samples, weights=weights, batch_size=self.__config.cbas_batch_size, epochs=self.__config.cbas_vae_epochs)

            if fes >= log_index * self.log_interval:
                log_index += 1
                cost.append(best_y)

            if problem.optimum is None:
                done = fes >= self.__config.maxFEs
            else:
                done = fes >= self.__config.maxFEs or best_y <= 1e-8
                if best_y <= 1e-8:
                    print("达到收敛阈值，提前终止优化")

        if len(cost) < self.__config.n_logpoint + 1:
            cost.append(best_y)
        else:
            cost[-1] = best_y 
        return {'cost': cost, 'fes': fes}   