Installation
------------

CausalEGM requires Python 3 and TensorFlow 2. We recommend to use Miniconda_.

Install from PyPI
^^^^^^^^^^^^^^^^^

Make sure pip is available, then install CausalEGM from PyPI_ using::

    pip install CausalEGM

If you get a ``Permission denied`` error, use ``pip install CausalEGM --user`` instead.

Install from GitHub
^^^^^^^^^^^^^^^^^^^

Alteratively, CausalEGM can also be installed through GitHub_ using::

    pip install git+https://github.com/SUwonglab/CausalEGM.git
    
or::
    
    git clone https://github.com/SUwonglab/CausalEGM && cd CausalEGM/src
    pip install -e .

``-e`` is short for ``--editable`` and links the package to the original cloned
location such that pulled changes are also reflected in the environment.


Dependencies
^^^^^^^^^^^^

- `TensorFlow <https://www.tensorflow.org/>`_ - deep learning framework.
- `numpy <https://docs.scipy.org/>`_, `pandas <https://pandas.pydata.org/>`_, `scikit-learn <https://scikit-learn.org/>`_.

Note that a GPU is recommended for accelerating the model training. However, **GPU is not a must**, CausalEGM can be installed on any personal computer (e.g, Macbook) or computational cluster.


If you run into issues, do not hesitate to raise a `GitHub issue`_.

.. _Miniconda: http://conda.pydata.org/miniconda.html
.. _PyPI: https://pypi.org/project/CausalEGM
.. _Github: https://github.com/SUwonglab/CausalEGM
.. _`Github issue`: https://github.com/SUwonglab/CausalEGM/issues/new/choose
