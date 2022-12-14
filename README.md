# CausalEGM

Estimating Causal Effect by Deep Encoding Generative Modeling. CausalEGM utilizes deep generative neural newtworks for estimating the causal effect by decoupling the high-dimensional confounder into a set of different latent variables with specific dependency on treatment or potential outcome.

![model](https://github.com/kimmo1019/CausalEGM/blob/main/model.jpg)

## Requirements

- TensorFlow>=2.4.1
- Python>=3.6.1

## Install

CausalEGM can be installed by
```shell
pip install causalEGM
```
Software has been tested on a Linux (Centos 7) with Python3.9. A GPU card is recommended for accelerating the training process.


## Reproduction

This section provides instructions on how to reproduce results in the our paper.

### Simulation data

We tested CausalEGM with simulation datasets first. 

