# Installation

## Prerequisites

CausalEGM can be installed via [Pip], [Conda], and GitHub for Python users. CausalEGM can also be installed via CRAN and GitHub for R users. 

### pip prerequisites

1. Install [Python]. we recommend Python>=3.9 and the [venv](https://docs.python.org/3/library/venv.html) or [pyenv](https://github.com/pyenv/pyenv/) for creating a virtual environment and version management system.

2. Take venv for instance. Create a virtual environment:

    ```shell
    python3 -m venv <venv_path>
    ```

3. Activate the virtual environment:

    ```shell
    source <venv_path>/bin/activate
    ```

### conda prerequisites

1. Install conda through [miniconda](http://conda.pydata.org/miniconda.html) or [anaconda](https://www.anaconda.com/). 

2. Create a new conda environment:

    ```shell
    conda create -n causalegm-env python=3.9
    ```

3. Activate your environment:

    ```shell
    conda activate causalegm-env
    ```


### GPU prerequisites (optional)

Training CausalEGM model will be faster when accelerated with a GPU (not a must). Before installing CausalEGM, the CUDA and cuDNN environment should be setup.


## Install with pip

Install CausalEGM from PyPI using:

    ```shell
    pip install CausalEGM
    ```

If you get a `Permission denied` error, use `pip install CausalEGM --user` instead. Pip will automatically install all the dependent packages, such as TensorFlow.

Alteratively, CausalEGM can also be installed through GitHub using::

    ```shell
    pip install git+https://github.com/SUwonglab/CausalEGM.git
    ```
    
or:

    ``` shell
    git clone https://github.com/SUwonglab/CausalEGM && cd CausalEGM/src
    pip install -e .
    ```

``-e`` is short for ``--editable`` and links the package to the original cloned
location such that pulled changes are also reflected in the environment.

## Install with conda

1. CausalEGM can also be downloaded through conda-forge. Add `conda-forge` as the highest priority channel:

    ```shell
    conda config --add channels conda-forge
    ```

2. Activate strict channel priority:

    ```shell
    conda config --set channel_priority strict
    ```

3. Install CausalEGM from conda-forge channel:

    ```shell
    conda install -c conda-forge causalegm
    ```

## Install R package (RcausalEGM)


We provide a standard alone R package of CausalEGM via Reticulate.

The easiest way to install CausalEGM for R is via CRAN:

    ```R
    install.packages("RcausalEGM")
    ```

Alternatively, users can also install RcausalEGM from GitHub source using devtools: 

    ```R
    devtools::install_github("SUwonglab/CausalEGM", subdir = "r-package/RcausalEGM")
    ```

[Python]: https://www.python.org/downloads/
[Pip]: https://pypi.org/project/CausalEGM/
[Conda]: https://anaconda.org/conda-forge/causalegm
[Tensorflow]: https://www.tensorflow.org/
[jax]: https://jax.readthedocs.io/en/latest/
[reticulate]: https://rstudio.github.io/reticulate/