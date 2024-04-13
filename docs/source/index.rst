|PyPI| |CRAN| |Anaconda| |travis| |Platforms| |Docs|

CausalEGM - An Encoding Generative Modeling Approach to Dimension Reduction and Covariate Adjustment in Causal Inference
========================================================================================================================

.. image:: https://raw.githubusercontent.com/SUwonglab/CausalEGM/main/model.jpg
   :width: 300px
   :align: left

.. include:: _key_contributors.rst

Causal inference has become extremely essential in modern observational studies with rich covariate information. However, it is often challenging to estimate the causal effect with high-dimensional covariates
due to the “curse of dimensionality”.

We develop **CausalEGM**, a deep learning framework for nonlinear dimension reduction and generative modeling of the dependency among covariate features affecting treatment and response. 
The key idea is to identify a latent covariate feature set (e.g., latent confounders) that affects both treatment and outcome. By conditioning on these features, 
one can mitigate the confounding effect of the high dimensional covariate on the estimation of the causal relation between treatment and response.

CausalEGM provides a flexible and powerful framework to develop deep learning-based estimates for treatment effect. Both empirical and theoretical results are provided to demonstrate the effectiveness of CausalEGM.

CausalEGM Wide Applicability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Estimate counterfactual outcome.
- Estimate average treatment effect (ATE).
- Estimate individual treatment effect (ITE).
- Estiamte average dose response function (ADRF).
- Estimate conditional average treatment effect (CATE).
- Built-in simulation and semi-simulation datasets.

CausalEGM Highlighted Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Capable of handling both continuous and binary treatment settings.
- Support big dataset with large sample size (e.g, >10M) and number of covariates (e.g., >10k) in a personal PC.
- Provide both `Python PyPi package <https://pypi.org/project/CausalEGM/>`__ and `R CRAN package <https://cran.r-project.org/web/packages/RcausalEGM/index.html>`__, incluidng a user-friendly command-line interface (CLI).


Main References
^^^^^^^^^^^^^^^
Liu *et al.* (2022), CausalEGM: a general causal inference framework by encoding generative modeling,
`arXiv <https://arxiv.org/abs/2212.05925>`__.

Liu *et al.* (2021), Density estimation using deep generative neural networks, `PNAS <https://www.pnas.org/doi/abs/10.1073/pnas.2101344118>`_.


Support
^^^^^^^
Found a bug or would like to see a feature implemented? Feel free to submit an
`issue <https://github.com/SUwonglab/CausalEGM/issues/new/choose>`_.
Have a question or would like to start a new discussion? You can also always send us an `email <liuqiao@stanford.edu>`_.
Your help to improve CausalEGM is highly appreciated!


.. toctree::
   :caption: Main
   :maxdepth: 1
   :hidden:

   about
   installation
   api
   release_notes

.. toctree::
   :caption: Tutorials
   :maxdepth: 1
   :hidden:

   tutorial_py
   tutorial_r
   


.. |PyPI| image:: https://img.shields.io/pypi/v/CausalEGM
   :target: https://pypi.org/project/CausalEGM/

.. |CRAN| image:: https://www.r-pkg.org/badges/version/RcausalEGM
   :target: https://cran.r-project.org/web/packages/RcausalEGM/index.html

.. |Anaconda| image:: https://anaconda.org/conda-forge/causalegm/badges/version.svg
   :target: https://anaconda.org/conda-forge/causalegm

.. |Platforms| image:: https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/causalegm-feedstock?branchName=main
   :target: https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=18625&branchName=main

.. |Docs| image:: https://readthedocs.org/projects/causalegm/badge/?version=latest
   :target: https://causalegm.readthedocs.io

.. |travis| image:: https://app.travis-ci.com/kimmo1019/CausalEGM.svg?branch=main
   :target: https://app.travis-ci.com/github/kimmo1019/CausalEG

.. _Scanpy: https://scanpy.readthedocs.io

.. _calendly: https://calendly.com/scvelo

.. |br| raw:: html

  <br/>

.. |meet| raw:: html

  <!-- Calendly link widget begin -->
  <link href="https://assets.calendly.com/assets/external/widget.css" rel="stylesheet">
  <script src="https://assets.calendly.com/assets/external/widget.js" type="text/javascript"></script>
  <a href="" onclick="Calendly.initPopupWidget({url: 'https://calendly.com/scvelo'});return false;">here</a>
  <!-- Calendly link widget end -->

.. |dim| raw:: html

   <span class="__dimensions_badge_embed__" data-id="pub.1129830274" data-style="small_rectangle"></span>
   <script async src="https://badge.dimensions.ai/badge.js" charset="utf-8"></script>
