|PyPI| |travis| |Docs|



CausalEGM - A general causal inference framework by encoding generative modeling
================================================================================

.. image:: https://raw.githubusercontent.com/SUwonglab/CausalEGM/main/model.jpg
   :width: 300px
   :align: left

.. include:: _key_contributors.rst

**CausalEGM** is a general causal inference framework `Liu et al. (arXiv, 2022) <https://arxiv.org/pdf/2212.05925.pdf>`_ for estimating causal effects by encoding generative modeling, which can be applied in both binary and continuous treatment settings.

CausalEGM simultaneously decouples the dependencies of confounders on both treatment and outcome and maps the confounders to the low-dimensional latent space. 
By conditioning on the low-dimensional latent features, CausalEGM can estimate the causal effect for each individual or the average causal effect within a population.

CausalEGM's key applications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- estimate average treatment effect (ATE).
- estimate individual treatment effect (ITE).
- estiamte average dose response function (ADRF).
- estimate heterogenous treatment effect (HTE).
- estimate reaction rates of transcription, splicing and degradation.
- built-in simulation and semi-simulation datasets.


Latest news
^^^^^^^^^^^
- Dec/2021: Preprint paper of CausalEGM is out on `arXiv <https://arxiv.org/abs/2212.05925/>`__ 
- Aug/2022: Version 0.1.0 of CausalEGM is released on `PyPI <https://pypi.org/project/epiaster/>`_ 

Main References
^^^^^^^^^^^^^^^
Liu *et al.* (2021), Density estimation using deep generative neural networks, `PNAS <https://www.pnas.org/doi/abs/10.1073/pnas.2101344118>`_.

Liu *et al.* (2022), CausalEGM: a general causal inference framework by encoding generative modeling,
`arXiv <https://arxiv.org/abs/2212.05925>`__.

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
   release_notes

.. toctree::
   :caption: Tutorials
   :maxdepth: 1
   :hidden:

   get_started



.. |PyPI| image:: https://img.shields.io/pypi/v/CausalEGM
   :target: https://pypi.org/project/CausalEGM/

.. |Docs| image:: https://readthedocs.org/projects/causalegm/badge/?version=latest
   :target: https://causalegm.readthedocs.io

.. |travis| image:: https://img.shields.io/travis/kimmo1019/CausalEGM
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
