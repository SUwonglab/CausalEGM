About CausalEGM
---------------

We introduce a new approach by encoding generative modeling (EGM) for handling high-dimensional covariates by a dependency-aware dimension reduction strategy where the key idea is to identify a latent covariate feature set (e.g., latent confounders) that affects both treatment and outcome. 
EGM provides a flexible and powerful framework for us to develop deep learning-based estimates for the structural equation modeling that describes the causal relations among variables. Comprehensive numerical experiments suggest that the proposed method is effective and scalable in estimating the causal effect of one variable on another under various settings.

Background and Challengings
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given data in an observational study, a central problem in causal inference is to estimate the effect of one variable (e.g., treatment) on another variable (e.g., outcome) in the presence of a covariate vector that represents all other variables observed in the study.
Under the well-known “unconfoundedness” condition, valid estimates of the desired effect of treatment on outcome can be obtained by alternative approaches, including matching, weighting, stratification, and regression-based methods.
Covariate adjustment plays an important role in these methods. When the covariate is of high dimension, as is often the case in modern applications, covariate adjustment becomes difficult because of the ``curse of dimensionality".

Core Idea in CausalEGM
~~~~~~~~~~~~~~~~~~~~~~~
In observational study, many methods have been proposed for covariate adjustment under the potential outcome model (`Rubin et al. 1974 <http://www.fsb.muohio.edu/lij14/420_paper_Rubin74.pdf>`_), which often involves the estimation of the expectation of the outcome conditional on the treatment and the covariate.
CausalEGM simultaneously learns to (1) embed the high-dimensional covariates into a low-dimensional latent space where the distribution of the embeddings (latent covariate features) is pre-specified. (2) build generative models for treatment given latent features and for outcome given treatment and latent features. 
The key idea of this method is to partition the latent feature vector into different independent components that play different roles in the above two generative models. This partitioning then allows us to identify a minimal latent covariate feature subvector that affects both treatment and outcome.
Once the latent confounding variable $Z_0$ can be learned, the average dose-response function $\mu(x)$ can be estimated by the following formula:

.. math::
   \begin{align}
   \mu(x)=\int \mathbb{E}(Y|X=x,Z_0=z_0)p_{Z_0}(z_0)dz_0,
   \end{align}
where $X$ and $Y$ are the treatment and outcome variables, respectively. We show that the original high-dimensional covariate $V$ can be replaced by a low-dimensional covariate feature.


See `Liu et al. (2020) <https://arxiv.org/abs/2212.05925>`_ for a detailed exposition of the methods.
