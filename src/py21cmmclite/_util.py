import logging
import numpy as np
import py21cmfast as p21
from py21cmfast.io.caching import OutputCache

logger = logging.getLogger("21cmFAST")


class BaseSimulator:
    """
    Base class for constructing simulators and likelihoods.
    """

    def __init__(
        self,
        inputs_21cmfast,
        varied_params,
        cache_dir,
        regenerate=False,
        global_params=None,
    ):
        self.inputs_21cmfast = inputs_21cmfast
        self.varied_params = varied_params
        self.inputs_simulator = self.inputs_21cmfast.evolve_input_structs(
            **self.varied_params
        )
        self.cache_dir = cache_dir
        self.regenerate = regenerate
        self.global_params = global_params or {}
