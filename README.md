[![PyPI](https://img.shields.io/pypi/v/CausalEGM)](https://pypi.org/project/CausalEGM/)
[![Travis (.org)](https://img.shields.io/travis/kimmo1019/CausalEGM)](https://app.travis-ci.com/github/kimmo1019/CausalEGM)
[![Documentation Status](https://readthedocs.org/projects/causalegm/badge/?version=latest)](https://causalegm.readthedocs.io/en/latest/?badge=latest)


# CausalEGM

Estimating Causal Effect by Deep Encoding Generative Modeling. CausalEGM utilizes deep generative neural newtworks for estimating the causal effect by decoupling the high-dimensional confounder into a set of different latent variables with specific dependency on treatment or potential outcome.

![model](https://github.com/SUwonglab/CausalEGM/blob/main/model.jpg)

## Requirements

- TensorFlow>=2.4.1
- Python>=3.6.1

## Install

CausalEGM can be installed by
```shell
pip install CausalEGM
```
Software has been tested on a Linux (Centos 7) with Python3.9. A GPU card is recommended for accelerating the training process.


## Reproduction

This section provides instructions on how to reproduce results in the our paper.

### Simulation data

We tested CausalEGM with simulation datasets first. 

## Contact

Please feel free to open an issue in Github or contact `liuqiao@stanford.edu` if you have any problem in CausalEGM.


## Citation

If you find CausalEGM useful for your work, please consider citing our [paper](https://arxiv.org/abs/2212.05925):

Qiao Liu, Zhongren Chen, Wing Hung Wong. CausalEGM: a general causal inference framework by encoding generative modeling[J]. arXiv preprint arXiv:2212.05925, 2022.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

