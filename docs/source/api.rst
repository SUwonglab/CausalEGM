.. automodule:: CausalEGM

API
===

Import CausalEGM as::

   import CausalEGM as cegm


After reading the data (``scv.read``) or loading an in-built dataset (``scv.datasets.*``),
the typical workflow consists of subsequent calls of
preprocessing (``scv.pp.*``), analysis tools (``scv.tl.*``) and plotting (``scv.pl.*``).
Further, several utilities (``scv.utils.*``) are provided to facilitate data analysis.


Models
-----------

.. autosummary::
   :toctree: .

   causalEGM.CausalEGM
   causalEGM.VariationalCausalEGM


Datasets
-------------


.. autosummary::
   :toctree: .

   util.Base_sampler
   util.Sim_Hirano_Imbens_sampler
   util.Sim_Sun_sampler
   util.Sim_Colangelo_sampler
   util.Semi_Twins_sampler
   util.Semi_acic_sampler