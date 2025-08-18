import logging
import numpy as np
import py21cmfast as p21
from py21cmfast.io.caching import OutputCache

logger = logging.getLogger("21cmFAST")


class BaseSimulator:
    """
    Base class for all simulators.
    """

    def __init__(
        self,
        inputs,
    ):
        self.inputs = inputs

    def build_model_data(self, update_params: dict = {}):
        pass

    def simulate(self, update_params: dict = {}):
        pass

    def build_blob_data(self, update_params: dict = {}):
        pass


class EoRSimulator(BaseSimulator):
    """
    Base class for constructing simulators of EoR using 21cmFAST.
    """

    def __init__(
        self,
        inputs: p21.InputParameters,
        cache_dir: str,
        regenerate: bool = False,
        global_params: dict | None = None,
    ):
        super().__init__(inputs)
        self.cache_dir = cache_dir
        self.regenerate = regenerate
        self.global_params = global_params or {}

    def get_update_input(self, update_dict: dict):
        return self.inputs.evolve_input_structs(**update_dict)

    def build_model_data(self, update_params: dict = {}):
        pass

    def simulate(self, update_params: dict = {}):
        pass

    def build_blob_data(self, update_params: dict = {}):
        pass


class CoevalSimulator(EoRSimulator):
    """
    Basic simulator for coeval cubes.
    """

    def __init__(
        self,
        redshifts: list[float] | np.ndarray,
        inputs: p21.InputParameters,
        cache_dir: str,
        regenerate: bool = False,
        global_params: dict | None = None,
    ):
        super().__init__(inputs, cache_dir, regenerate, global_params)
        self.redshifts = redshifts

    def simulate(self, update_params: dict = {}):
        """
        Simulate the coeval cubes.
        """
        inputs = self.get_update_input(update_params)
        cache = OutputCache(self.cache_dir)
        p21.run_coeval(
            out_redshifts=self.redshifts,
            inputs=inputs,
            write=True,
            regenerate=self.regenerate,
            cache=cache,
            **self.global_params,
        )
        return 1.0


class CoevalNeutralFraction(CoevalSimulator):
    """
    Simulate the coeval cubes and calculate the 1DPS.
    """

    def __init__(
        self,
        redshifts: list[float] | np.ndarray,
        inputs: p21.InputParameters,
        cache_dir: str,
        regenerate: bool = False,
        global_params: dict | None = None,
        save_global_xhi: bool = False,
        save_xhi_box: bool = False,
    ):
        super().__init__(redshifts, inputs, cache_dir, regenerate, global_params)
        self.save_global_xhi = save_global_xhi
        self.save_xhi_box = save_xhi_box

    def build_model_data(self, update_params: dict = {}):
        inputs = self.get_update_input(update_params)
        xhibox = [p21.IonizedBox.new(redshift=z, inputs=inputs) for z in self.redshifts]
        cache = OutputCache(self.cache_dir)
        xhibox = [
            p21.io.h5.read_output_struct(cache.find_existing(path)) for path in xhibox
        ]
        global_xhi = [xhi.get("neutral_fraction").mean() for xhi in xhibox]
        blob = {}
        if self.save_global_xhi:
            blob["global_xhi_coeval"] = global_xhi
        if self.save_xhi_box:
            blob["xhi_box_coeval"] = [xhi.get("neutral_fraction") for xhi in xhibox]
        return global_xhi, blob
