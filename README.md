# CbAS Optimizer for MetaBox
## 项目介绍 | Introduction
我为MetaBox框架添加了CbAS（Conditioning by Adaptive Sampling）优化算法的实现。CbAS是一种基于变分自编码器和重要性采样的优化算法，特别适用于复杂设计空间的探索。
I've added a CbAS (Conditioning by Adaptive Sampling) optimizer implementation to the MetaBox framework. CbAS is an optimization algorithm based on variational autoencoders and importance sampling, particularly suitable for exploring complex design spaces.

相关文献(related paper): [ICML 2019 paper 'Conditioning by adaptive sampling for robust design'](https://arxiv.org/abs/1901.10060)

## 实现位置 | Implementation Location
代码位于：`MetaBox/src/optimizer/cbas_optimizer.py`

## 注意 | Attention
为了将CbAS算法应用于metabox框架，本模型做了如下改动：
- 移除预测模型，使用metabox 自带的 promblem.eval()
- 先验概率模型(prior_vae)用抽样得到的 200000个数据中的前得分前25%的输入x训练
- 每次将阙值与新的25分位数比较，替换为小的那个,以此来表示越来越严格的条件
- 因为没有预测模型，所以直接将采样得到的y与 阙值对比，用0/1变量表示，以此代表论文中提到的$P(S|x)$
由于对深度学习模型vae的依赖，模型的运行时间会显著高于其他模型
## 效果展示 | Demonstration
在2维度的sphere问题上，ln cost 在负7到负9之间，**强于Random_search** 
略微弱于DEAP_CMAES. 明显弱于 GL_PSO
![image](https://github.com/user-attachments/assets/5a1d1c81-9603-473d-b0ca-aad6c2ca9f23)


在10维度的sphere问题上的分数ln cost大约在0-1之间
弱于DEAP_CMAES 弱于 GL_PSO
![image](https://github.com/user-attachments/assets/30e0efbc-cc8b-4e7c-9759-0310fae84aa2)
![image](https://github.com/user-attachments/assets/72d2821c-698b-469a-ab42-70e0c6d6efeb)
![image](https://github.com/user-attachments/assets/b11b9630-d7a0-47ca-a9e8-0c3a02a00351)
由于运行时间问题，对其他问题上的跑分还未进行测试
## 代码结构 | Code Structure
```
cbas_optimizer.py
│
├── VAE(nn.Module)               
│   ├── __init__                   
│   ├── encode                       
│   ├── decode                    
│   ├── reparameterize              
│   ├── forward                   
│   ├── sample_from_z                      
│   ├── log_prob                     
│   ├── train_model               
│   └── copy_weights_from          
│
└── CbAS_Optimizer(Basic_Optimizer) 
    ├── __init__                    
    └── run_episode                  
```


## 使用示例 | Usage examples
```shell
python main.py --test --problem bbob --difficulty easy --optimizer CbAS_Optimizer --t_optimizer_for_cp DEAP_CMAES Random_search

python main.py --test --problem bbob --difficulty easy --optimizer CbAS_Optimizer --cbas_latent_dim 64 --cbas_percentile 90.0 --cbas_hidden_dim 512
```

CbAS优化器的可选参数：
- `--cbas_latent_dim`: VAE潜变量维度，默认值32
- `--cbas_hidden_dim`: 神经网络隐藏层维度，默认值256
- `--cbas_num_layers`: 神经网络层数，默认值1
- `--cbas_percentile`: 用于计算阈值的百分位数，默认值25
- `--cbas_num_models`: 集成模型数量，默认值5
- `--cbas_vae_epochs`: VAE训练轮数，默认值10
- `--cbas_ensemble_epochs`: 集成模型训练轮数，默认值100
- `--cbas_batch_size`: 批处理大小，默认值100
- `--cbas_samples_per_iter`: 每次迭代的采样数，维度<=10时默认为$100\times dim$，否则为$200\times dim$



---

# MetaBox: A Benchmark Platform for Meta-Black-Box Optimization with Reinforcement Learning

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2023-b31b1b.svg)]([https://proceedings.neurips.cc/paper_files/paper/2023/hash/232eee8ef411a0a316efa298d7be3c2b-Abstract-Datasets_and_Benchmarks.html]) **MetaBox has been published at NeurIPS 2023！**

MetaBox is the first benchmark platform expressly tailored for developing and evaluating MetaBBO-RL methods. MetaBox offers a flexible algorithmic template that allows users to effortlessly implement their unique designs within the platform. Moreover, it provides a broad spectrum of over 300 problem instances, collected from synthetic to realistic scenarios, and an extensive library of 19 baseline methods, including both traditional black-box optimizers and recent MetaBBO-RL methods. Besides, MetaBox introduces three standardized performance metrics, enabling a more thorough assessment of the methods.

## 😁Contact Us
👨‍💻👩‍💻We are a research team mainly focus on Meta-Black-Box-Optimization (MetaBBO), which assists automated algorithm design for Evolutionary Computation. 

Here is our [homepage](https://gmc-drl.github.io/) and [github](https://github.com/GMC-DRL). **🥰🥰🥰Please feel free to contact us—any suggestions are welcome!**

If you have any question or want to contact us：
- 🌱Fork, Add, and Merge
- ❓️Report an [issue](https://github.com/GMC-DRL/MetaBox/issues)
- 📧Contact WenJie Qiu ([wukongqwj@gmail.com](mailto:wukongqwj@gmail.com))
- 🚨**We warmly invite you to join our QQ group for further communication (Group Number: 952185139).**

## Installations

You can access all MetaBox files with the command:

```shell
git clone git@github.com:GMC-DRL/MetaBox.git
cd MetaBox
```

## Citing MetaBox

The PDF version of the paper is available [here](https://arxiv.org/abs/2310.08252). If you find our MetaBox useful, please cite it in your publications or projects.

```latex
@inproceedings{metabox,
author={Ma, Zeyuan and Guo, Hongshu and Chen, Jiacheng and Li, Zhenrui and Peng, Guojun and Gong, Yue-Jiao and Ma, Yining and Cao, Zhiguang},
title={MetaBox: A Benchmark Platform for Meta-Black-Box Optimization with Reinforcement Learning},
booktitle = {Advances in Neural Information Processing Systems},
year={2023},
volume = {36}
}
```

## Requirements

`Python` >=3.7.1 with the following packages installed:  

* `numpy`==1.21.2  
* `torch`==1.9.0  
* `matplotlib`==3.4.3  
* `pandas`==1.3.3  
* `scipy`==1.7.1
* `scikit_optimize`==0.9.0  
* `deap`==1.3.3  
* `tqdm`==4.62.3  
* `openpyxl`==3.1.2

## Quick Start

* To obtain the figures in our paper, run the following commands:

  ```shell
  cd for_review
  python paper_experiment.py
  ```

  then corresponding figures will be output to `for_revivew/pics`.

  ---

  The quick usage of the four main running interfaces is listed as follows, in the following command, we specifically take `RLEPSO` as an example.

  Firstly, get into the main code folder, src:

  ```shell
  cd ../src
  ```

* To trigger the entire workflow, including **train, rollout and test**, run the following command:

  ```shell
  python main.py --run_experiment --problem bbob --difficulty easy --train_agent RLEPSO_Agent --train_optimizer RLEPSO_Optimizer
  ```

* To trigger the standalone process of **training**:

  ```shell
  python main.py --train --problem bbob --difficulty easy --train_agent RLEPSO_Agent --train_optimizer RLEPSO_Optimizer 
  ```

* To trigger the standalone process of **testing**:

  ```shell
  python main.py --test --problem bbob --difficulty easy --agent_load_dir agent_model/test/bbob_easy/ --agent_for_cp RLEPSO_Agent --l_optimizer_for_cp RLEPSO_Optimizer --t_optimizer_for_cp DEAP_CMAES Random_search
  ```


## Documentation

For more details about the usage of `MetaBox`, please refer to [MetaBox User's Guide](https://gmc-drl.github.io/MetaBox/).

## Datasets


At present, three benchmark suites are integrated in `MetaBox`:  

* `Synthetic` contains 24 noiseless functions, borrowed from [coco](https://github.com/numbbo/coco):bbob with [original paper](https://www.tandfonline.com/eprint/DQPF7YXFJVMTQBH8NKR8/pdf?target=10.1080/10556788.2020.1808977).
* `Noisy-Synthetic` contains 30 noisy functions, borrowed from [coco](https://github.com/numbbo/coco):bbob-noisy with [original paper](https://www.tandfonline.com/eprint/DQPF7YXFJVMTQBH8NKR8/pdf?target=10.1080/10556788.2020.1808977).
* `Protein-Docking` contains 280 problem instances, which simulate the application of protein docking as a 12-dimensional optimization problem, borrowed from [LOIS](https://github.com/Shen-Lab/LOIS) with [original paper](http://papers.nips.cc/paper/9641-learning-to-optimize-in-swarms).

## Baseline Library

**7 MetaBBO-RL optimizers, 1 MetaBBO-SL optimizer and 11 classic optimizers have been integrated into `MetaBox`.** They are listed below.
<!-- Choose one or more of them to be the baseline(s) to test the performance of your own optimizer. -->

**Supported MetaBBO-RL optimizers**:

|   Name   | Year |                        Related paper                         |
| :------: | :--: | :----------------------------------------------------------: |
| DE-DDQN  | 2019 | [Deep reinforcement learning based parameter control in differential evolution](https://dl.acm.org/doi/10.1145/3321707.3321813) |
|  QLPSO   | 2019 | [A reinforcement learning-based communication topology in particle swarm optimization](https://link.springer.com/article/10.1007/s00521-019-04527-9) |
|  DEDQN   | 2021 | [Differential evolution with mixed mutation strategy based on deep reinforcement learning](https://www.sciencedirect.com/science/article/pii/S1568494621005998) |
|   LDE    | 2021 | [Learning Adaptive Differential Evolution Algorithm From Optimization Experiences by Policy Gradient](https://ieeexplore.ieee.org/document/9359652) |
|  RL-PSO  | 2021 | [Employing reinforcement learning to enhance particle swarm optimization methods](https://www.tandfonline.com/doi/full/10.1080/0305215X.2020.1867120) |
|  RLEPSO  | 2022 | [RLEPSO:Reinforcement learning based Ensemble particle swarm optimizer✱](https://dl.acm.org/doi/abs/10.1145/3508546.3508599) |
| RL-HPSDE | 2022 | [Differential evolution with hybrid parameters and mutation strategies based on reinforcement learning](https://www.sciencedirect.com/science/article/pii/S2210650222001602) |
| GLEET    | 2024 | [Auto-configuring Exploration-Exploitation Tradeoff in Evolutionary Computation via Deep Reinforcement Learning](https://arxiv.org/abs/2404.08239) |
| SYMBOL   | 2024 | [Symbol: Generating Flexible Black-Box Optimizers through Symbolic Equation Learning](https://iclr.cc/virtual/2024/poster/17539) |
| RL-DAS   | 2024 | [Deep Reinforcement Learning for Dynamic Algorithm Selection: A Proof-of-Principle Study on Differential Evolution](https://ieeexplore.ieee.org/abstract/document/10496708/) |

**Supported MetaBBO-SL optimizer**:

|  Name  | Year |                        Related paper                         |
| :----: | :--: | :----------------------------------------------------------: |
| RNN-OI | 2017 | [Learning to learn without gradient descent by gradient descent](https://dl.acm.org/doi/10.5555/3305381.3305459) |

**Supported MetaBBO-NE optimizer**:

|  Name  | Year |                        Related paper                         |
| :----: | :--: | :----------------------------------------------------------: |
| LES      | 2023 | [Discovering evolution strategies via meta-black-box optimization](https://iclr.cc/virtual/2023/poster/11005) |

**Supported classic optimizers**:

|         Name          | Year |                        Related paper                         |
| :-------------------: | :--: | :----------------------------------------------------------: |
|          PSO          | 1995 | [Particle swarm optimization](https://ieeexplore.ieee.org/abstract/document/488968) |
|          DE           | 1997 | [Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces](https://dl.acm.org/doi/abs/10.1023/A%3A1008202821328) |
|        CMA-ES         | 2001 | [Completely Derandomized Self-Adaptation in Evolution Strategies](https://ieeexplore.ieee.org/document/6790628) |
| Bayesian Optimization | 2014 | [Bayesian Optimization: Open source constrained global optimization tool for Python](https://github.com/bayesian-optimization/BayesianOptimization) |
|        GL-PSO         | 2015 | [Genetic Learning Particle Swarm Optimization](https://ieeexplore.ieee.org/abstract/document/7271066/) |
|       sDMS_PSO        | 2015 | [A Self-adaptive Dynamic Particle Swarm Optimizer](https://ieeexplore.ieee.org/document/7257290) |
|          j21          | 2021 | [Self-adaptive Differential Evolution Algorithm with Population Size Reduction for Single Objective Bound-Constrained Optimization: Algorithm j21](https://ieeexplore.ieee.org/document/9504782) |
|         MadDE         | 2021 | [Improving Differential Evolution through Bayesian Hyperparameter Optimization](https://ieeexplore.ieee.org/document/9504792) |
|        SAHLPSO        | 2021 | [Self-Adaptive two roles hybrid learning strategies-based particle swarm optimization](https://www.sciencedirect.com/science/article/pii/S0020025521006988) |
|     NL_SHADE_LBC      | 2022 | [NL-SHADE-LBC algorithm with linear parameter adaptation bias change for CEC 2022 Numerical Optimization](https://ieeexplore.ieee.org/abstract/document/9870295) |
|     Random Search     |  -   |                              -                               |

Note that `Random Search` is to randomly sample candidate solutions from the searching space. 

## Post-processing
In a bid to illustrate the utility of MetaBox for facilitating rigorous evaluation and in-depth analysis, as mentioned in our paper, we carry out a wide-ranging benchmarking study on existing MetaBBO-RL methods. The post-processed data is available in [content.md](post_processed_data/content.md).

<!-- To facilitate the observation of our baselines and related metrics, we tested our baselines on two levels of difficulty on three datasets. Post-processed data are provided in [content.md](post_processed_data/content.md). -->



## Acknowledgements

The code and the framework are based on the repos [DEAP](https://github.com/DEAP/deap), [coco](https://github.com/numbbo/coco) and [Protein-protein docking V4.0](https://zlab.umassmed.edu/benchmark/).

