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


Use CausalEGM Python API
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

Use CausalEGM R API
^^^^^^^^^^^^^^^^^

Import CausalEGM into a R enviornment
''''''''''''''''''''

Import CausalEGM as::

    library(reticulate)
    cegm <- import("CausalEGM")
    np <- import("numpy")
    
Run CausalEGM using exmaple data as::

    n <- 10000
    p <- 10
    v <- matrix(rnorm(n * p), n, p)
    x <- rbinom(n, 1, 0.4 + 0.2 * (x[, 1] > 0))
    y <- pmax(v[, 1], 0) * x + v[, 2] + pmin(v[, 3], 0) + rnorm(n)
    x <- np$array(x)
    y <- np$array(y)
    v <- np$array(v)
    model <- cegm$CausalEGM(params,random_seed=123)
    
Model training
''''''''''''''

Model training using::

    model$train([x ,y, v])