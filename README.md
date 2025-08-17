# A lightweight, partial, rewrite of 21CMMC

This is a lightweight rewrite of [21CMMC](https://github.com/21cmfast/21CMMC), with the aim of:

- 1. Support [21cmFAST](https://github.com/21cmfast/21cmFAST) v4.
- 2. Support [nautilus](https://nautilus-sampler.readthedocs.io/).
- 3. Deprecate the cosmohammer backend, focus on constructing the model and the likelihood, and leave the sampler completely to the user
- 4. Each likelihood object can be used as a standalone instance, with the context saved entirely based on the cache of `21cmFAST` and attributes of the class. This should make writing and testing new likelihood functions much easier.


