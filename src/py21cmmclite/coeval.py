import logging
import numpy as np
import py21cmfast as p21
from py21cmfast.io.caching import OutputCache
from os import path
from powerbox.tools import get_power


logger = logging.getLogger("21cmFAST")


class CoevalSimulator:
    """
    Basic simulator for coeval cubes.
    """

    def __init__(
        self,
        redshifts,
        inputs_21cmfast,
        varied_params,
        cache_dir,
        regenerate=False,
        global_params=None,
    ):
        self.redshifts = redshifts
        self.inputs_21cmfast = inputs_21cmfast
        self.varied_params = varied_params
        self.inputs_simulator = self.inputs_21cmfast.evolve_input_structs(
            **self.varied_params
        )
        self.cache_dir = cache_dir
        self.regenerate = regenerate
        self.global_params = global_params or {}

    def simulate(self):
        """
        Simulate the coeval cubes.
        """
        cache = OutputCache(self.cache_dir)
        coeval = p21.run_coeval(
            out_redshifts=self.redshifts,
            inputs=self.inputs_simulator,
            write=True,
            regenerate=self.regenerate,
            cache=cache,
            **self.global_params,
        )
        # coeval is reshuffled in the order of redshift, so we need to get the redshift of each coeval
        return 1.0

    def build_model_data(self):
        pass


class CoevalNeutralFraction(CoevalSimulator):
    """
    Simulate the coeval cubes and calculate the 1DPS.
    """

    def __init__(
        self,
        redshifts,
        inputs_21cmfast,
        varied_params,
        cache_dir,
        n_psbins=None,
        min_k=0.1,
        max_k=1.0,
        logk=True,
        ignore_kperp_zero=True,
        ignore_kpar_zero=False,
        ignore_k_zero=False,
    ):
        super().__init__(redshifts, inputs_21cmfast, varied_params, cache_dir)
        self.n_psbins = n_psbins
        self.min_k = min_k
        self.max_k = max_k
        self.logk = logk
        self.ignore_k_zero = ignore_k_zero
        self.ignore_kperp_zero = ignore_kperp_zero
        self.ignore_kpar_zero = ignore_kpar_zero

    def build_model_data(self):
        super().build_model_data()
        xhibox = [
            p21.IonizedBox.new(redshift=z, inputs=self.inputs_simulator)
            for z in self.redshifts
        ]
        cache = OutputCache(self.cache_dir)
        xhibox = [
            p21.io.h5.read_output_struct(cache.find_existing(path)) for path in xhibox
        ]
        global_xhi = [xhi.get("neutral_fraction").mean() for xhi in xhibox]
        return global_xhi
