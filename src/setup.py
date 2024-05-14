import setuptools

setuptools.setup(
    name="CausalEGM", 
    version="0.4.0",
    author="Qiao Liu",
    author_email="liuqiao@stanford.edu",
    description="CausalEGM: an encoding generative modeling approach to dimension reduction and covariate adjustment in causal inference with observational studies",
    long_description="In this article, we develop CausalEGM, a deep learning framework for nonlinear dimension reduction and generative modeling of the dependency among covariate features affecting treatment and response. CausalEGM can be used for estimating causal effects in both binary and continuous treatment settings. By learning a bidirectional transformation between the high-dimensional covariate space and a low-dimensional latent space and then modeling the dependencies of different subsets of the latent variables on the treatment and response, CausalEGM can extract the latent covariate features that affect both treatment and response. By conditioning on these features, one can mitigate the confounding effect of the high dimensional covariate on the estimation of the causal relation between treatment and response. In a series of experiments, the proposed method is shown to achieve superior performance over existing methods in both binary and continuous treatment settings. The improvement is substantial when the sample size is large and the covariate is of high dimension. Finally, we established excess risk bounds and consistency results for our method, and discuss how our approach is related to and improves upon other dimension reduction approaches in causal inference. CausalEGM is freely available at https://github.com/SUwonglab/CausalEGM.",
    long_description_content_type="text/markdown",
    url="https://github.com/SUwonglab/CausalEGM",
    packages=setuptools.find_packages(),
    install_requires=[
   'tensorflow>=2.8.0',
   'scikit-learn',
   'pandas',
   'python-dateutil'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    entry_points={
    'console_scripts': [
        'causalEGM = CausalEGM.cli:main',
    ]},
)
