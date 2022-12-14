import setuptools

setuptools.setup(
    name="CausalEGM", 
    version="0.2.2",
    author="Qiao Liu",
    author_email="liuqiao@stanford.edu",
    description="CausalEGM: a general causal inference framework by encoding generative modeling",
    long_description="Understanding and characterizing causal effect has become essential in observational studies while it is still challenging if the confounders are high-dimensional. In this article, we develop a general framework CausalEGM, for estimating causal effect by encoding generative modeling, which can be applied in both binary and continuous treatment settings. In the potential outcome framework with unconfoundedness, we build a bidirectional transformation between the high-dimensional confounders space and a low-dimensional latent space where the density is known (e.g., Gaussian). Through this, CausalEGM enables simultaneously decoupling the dependencies of confounders on both treatment and outcome, and mapping the confounders to the low-dimensional latent space. By conditioning on the low-dimensional latent features, CausalEGM is able to estimate the causal effect for each individual or estimate the average causal effect within a population. Our theoretical analysis shows that the excess risk for CausalEGM can be bounded through empirical process theory. Under an assumption on encoder-decoder networks, the consistency of the estimate can also be guaranteed. In a series of experiments, CausalEGM demonstrates superior performance against existing methods in both binary and continuous settings. Specifically, we find CausalEGM to be substantially more powerful than competing methods in the presence of large sample size and high dimensional confounders. CausalEGM is freely available at https://github.com/SUwonglab/CausalEGM.",
    long_description_content_type="text/markdown",
    url="https://github.com/SUwonglab/CausalEGM",
    packages=setuptools.find_packages(),
    install_requires=["tensorflow-gpu==2.8.0", 
                "tensorflow-determinism==0.3.0", 
                "scikit-learn", "pandas",
                "python-dateutil"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
    'console_scripts': [
        'causalEGM = CausalEGM.cli:main',
    ]},
)