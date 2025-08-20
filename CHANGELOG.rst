Changelog
=========

dev
---

Added
~~~~~

- Add MPI support for emcee and nautilus
- A simple lightcone simulator that can be used to simulate the lightcone and calculate the global neutral fraction
- Allow likelihood class to invoke the lightcone simulator

Fixed
~~~~~

- Fixed a bug when no blob is returned in the sampler

v0.0.1 [18 Aug 2025]
----------------------

- Initial release

Added
~~~~~

- A basic structure of the package
- A simulator class that interact with 21cmfast
- A likelihood class that can be used to invoke 21cmfast simulations
- A simple likelihood that constraints the neutral fraction at a given redshift
- A sampler class that can be used to run Bayesian inference for given likelihoods
- A top level sampler class that runs MCMC using emcee
- A top level sampler class that runs INC using nautilus
