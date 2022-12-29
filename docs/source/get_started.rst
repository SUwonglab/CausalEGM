Getting Started
---------------

Here, you will be briefly guided through the basics of how to use CausalEGM.

First of all, you need to make sure CausalEGM is installed.
    
Users can use CausalEGM by a **single command line** or **use the API**.
    
Use CausalEGM command line
^^^^^^^^^^^^^^^^^^^^^^^^^^

When installing the CausalEGM, setuptools will add the console script to PATH and make it available for general use. This has advantage of being generalizeable to non-python scripts!

one can get the instuctions for running CausalEGM directly in a Linux/Unix terminal using::

    $ CausalEGM -h

Then the instructions for all arguments will be printed. Most of the parameters have default values.


Use CausalEGM API
^^^^^^^^^^^^^^^^^

Model initialization
''''''''''''''''''''

Import CausalEGM as::

    import CausalEGM as cegm
    
Model initialization using::

    model = cegm.CausalEGM(params, random_seed=123)
    
Model training
''''''''''''''

Model training using::

    model.train(data, batch_size=32)
