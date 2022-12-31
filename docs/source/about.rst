About CausalEGM
---------------

Given the observational data, drawing inferences about the causal effect of a treatment is crucial to many
scientific and engineering problems and attracts immense interest in a wide variety of areas. The most
effective way to learn the causality is to conduct a randomized controlled trial (RCT). However, RCT is
time-consuming, expensive, and problematic with generalisability. In contrast, observational studies
can provide valuable evidence and examine effects in “real world” settings. In real-world applications,
treatments are typically not assigned at random due to the selection bias introduced by confounders.
Accurate estimation of the causal effect involves dealing with confounders, which may affect both
treatment and outcome. Failing to adjust for confounding effect may lead to biased estimates and wrong conclusions.
CausalEGM is a causal inference framework for obervational study.


Highlighted features
~~~~~~~~~~~~~~~~~~~~
In observational study, many frameworks have been proposed to adjust for confounding effect under the potential outcome model (`Rubin et al. 1974 <http://www.fsb.muohio.edu/lij14/420_paper_Rubin74.pdf>`_). CausalEGM explores the advances in machine learning, especially deep learning, for improving the performance in causal effect estimation. Inspired by our previous work Roundtrip (`Liu et al. 2021 <https://www.pnas.org/doi/abs/10.1073/pnas.2101344118>`_), we propose CausalEGM based on
bidirectional deep generative neural networks to learn the latent representation of high-dimensional confounders. The independence of treatment and confounding variables conditioning on the latent representation provides a new aspect to handle the high-dimensional confounders. In brief, given the observation data with treatment X, outcome Y, covariates V, if V can be effectively encoded by Z, with unconfoundedness assumption, the population average estimate can be represented as

.. math::
   \begin{align}
   \mu(x)=\int \mathbb{E}(Y|X=x,Z_0=z_0)p_{Z_0}(z_0)dz_0,
   \end{align}
   
Several unique features of CausalEGM are summarized as follows:

- CausalEGM simultaneously decouples the dependencies of confounders on treatment and outcome and apply dimension reduction to confounders

- CausalEGM supports inferring the causal relationship given extremely large sample size (e.g, >10M)

- CausalEGM is able to handle high-dimensional confounders (e.g, >10K)


See `Liu et al. (2020) <https://arxiv.org/abs/2212.05925>`_ for a detailed exposition of the methods.
