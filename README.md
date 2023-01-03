[![PyPI](https://img.shields.io/pypi/v/CausalEGM)](https://pypi.org/project/CausalEGM/)
[![Travis (.org)](https://img.shields.io/travis/kimmo1019/CausalEGM)](https://app.travis-ci.com/github/kimmo1019/CausalEGM)
[![Documentation Status](https://readthedocs.org/projects/causalegm/badge/?version=latest)](https://causalegm.readthedocs.io)

CausalEGM - A general causal inference framework by encoding generative modeling
================================================================================

<img align="left" src="https://github.com/SUwonglab/CausalEGM/blob/main/model.jpg" width="350">
   
**CausalEGM** is a general causal inference framework `Liu et al. (arXiv, 2022) <https://arxiv.org/pdf/2212.05925.pdf>`_ for estimating causal effects by encoding generative modeling, which can be applied in both discrete and continuous treatment settings. 

CausalEGM simultaneously decouples the dependencies of confounders on both treatment and outcome and maps the confounders to the low-dimensional latent space. By conditioning on the low-dimensional latent features, CausalEGM can estimate the causal effect for each individual or the average causal effect within a population.



## Requirements

- TensorFlow>=2.4.1
- Python>=3.6.1

## Install

CausalEGM can be installed by
```shell
pip install CausalEGM
```

Note that a GPU is recommended for accelerating the model training. However, GPU is not a must, CausalEGM can be installed on any personal computer (e.g, Macbook) or computational cluster.


## Reproduction

This section provides instructions on how to reproduce results in the our paper.

We provide the config files for all the datasets used in our study. These config files can be found in `configs` folder.

```shell
cd src
python3 main.py -c CONFIG_PATH
```

## Contact

Please feel free to open an issue in Github or contact `liuqiao@stanford.edu` if you have any problem in CausalEGM.


## Citation

If you find CausalEGM useful for your work, please consider citing our [paper](https://arxiv.org/abs/2212.05925):

Qiao Liu, Zhongren Chen, Wing Hung Wong. CausalEGM: a general causal inference framework by encoding generative modeling[J]. arXiv preprint arXiv:2212.05925, 2022.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

