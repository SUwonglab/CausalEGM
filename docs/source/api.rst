
API
===

Import CausalEGM as::

   import CausalEGM as cegm


After reading the data from an text file (``cegm.load_from_text``) or loading an in-built dataset (``cegm.datasets.*``),
the typical workflow consists of subsequent calls of
preprocessing (``scv.pp.*``), analysis tools (``scv.tl.*``) and plotting (``scv.pl.*``).
Further, several utilities (``scv.utils.*``) are provided to facilitate data analysis.


Model initialization
--------------------

