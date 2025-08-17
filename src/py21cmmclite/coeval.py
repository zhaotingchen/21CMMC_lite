import logging
import numpy as np
import py21cmfast as p21
from py21cmfast.io.caching import OutputCache
from ._util import BaseSimulator

logger = logging.getLogger("21cmFAST")


class CoevalSimulator(BaseSimulator):
    """
    Basic simulator for coeval cubes.
    """

    def __init__(
        self,
        redshifts: list[float] | np.ndarray,
        inputs_21cmfast: p21.InputParameters,
        varied_params: dict,
        cache_dir: str,
        regenerate: bool = False,
        global_params: dict | None = None,
    ):
        super().__init__(inputs_21cmfast, varied_params, cache_dir, regenerate, global_params)
        self.redshifts = redshifts

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
