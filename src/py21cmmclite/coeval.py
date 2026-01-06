import logging
import numpy as np
import py21cmfast as p21
from py21cmfast.io.caching import OutputCache
from py21cmfast.wrapper.cfuncs import compute_luminosity_function

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


class LuminosityFunctionSimulator(BaseSimulator):
    """
    Base class for simulating luminosity function.
    Note that, the simulation of UVLF is purely analytical (see Equation 11 of https://arxiv.org/pdf/1809.08995), and therefore does not require invoking
    the actual run of 21cmFAST.

    Parameters
    ----------
    inputs: p21.InputParameters
        Input parameters for 21cmFAST.
    redshifts: list[float] | np.ndarray
        Redshifts to calculate the UV luminosity function.
    n_uv_bins: int
        Number of UV bins.
    save_uvlf: bool
        Whether to save the UV luminosity function to the blob.


    """

    def __init__(
        self,
        inputs: p21.InputParameters,
        redshifts: list[float] | np.ndarray,
        n_uv_bins: int = 100,
        save_uvlf: bool = False,
    ):
        super().__init__(inputs)
        self.redshifts = redshifts
        self.save_uvlf = save_uvlf
        self.n_uv_bins = n_uv_bins

    def get_update_input(self, update_dict: dict):
        return self.inputs.evolve_input_structs(**update_dict)

    def build_model_data(self, update_params: dict = {}):
        inputs = self.get_update_input(update_params)
        Muvfunc, Mhfunc, lfunc = compute_luminosity_function(
            redshifts=self.redshifts,
            inputs=inputs,
            nbins=self.n_uv_bins,
            mturnovers=inputs.astro_params.M_TURN * np.ones(len(self.redshifts)),
        )
        blob = {}
        if self.save_uvlf:
            blob["uvlf"] = np.array([Muvfunc, lfunc])
        return [Muvfunc, lfunc], blob


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
    Simulate the coeval cubes and calculate the global neutral fraction.
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
            blob["global_xhi_coeval"] = np.array(global_xhi)
        if self.save_xhi_box:
            blob["xhi_box_coeval"] = np.array(
                [xhi.get("neutral_fraction") for xhi in xhibox]
            )
        return global_xhi, blob


class CoevalPhotonConsFlag(CoevalNeutralFraction):
    """
    Simulate the coeval cube at a specific redshift without photon conservation,
    and calculate the global neutral fraction at that redshift. Returns a
    flag indicating whether the global neutral fraction is smaller/larger than the threshold.
    This is useful in an inference run to be the first simulator to check,
    and if flagged, the parameter set should be rejected before running other simulators,
    to avoid an error being raised.

    If the PHOTON_CONS_TYPE is "no-photoncons", this simulator will be skipped.
    """

    def __init__(
        self,
        inputs: p21.InputParameters,
        cache_dir: str,
        regenerate: bool = False,
        global_params: dict | None = None,
        xhi_threshold: float = 0.01,
        redshifts: list[float] | np.ndarray = None,
        flag_type: str = "larger",
    ):
        # only run at the end z of photoncon calibration
        if redshifts is None:
            redshifts = [
                inputs.astro_params.PHOTONCONS_CALIBRATION_END,
            ]
        if inputs.astro_options.PHOTON_CONS_TYPE == "no-photoncons":
            self.run_photoncons = False
        else:
            self.run_photoncons = True
            # evolve input struct to turn off photon conservation
            # also turn off TS fluct, inhomo reco, and mini halos
            # zstep_factor=0.9 to avoid having node redshifts
            inputs = inputs.evolve_input_structs(
                PHOTON_CONS_TYPE="no-photoncons",
                USE_TS_FLUCT=False,
                INHOMO_RECO=False,
                USE_MINI_HALOS=False,
            ).with_logspaced_redshifts(zmin=redshifts[0], zmax=35.0, zstep_factor=0.9)
        super().__init__(redshifts, inputs, cache_dir, regenerate, global_params)
        self.xhi_threshold = xhi_threshold
        self.flag_type = flag_type

    def simulate(self, update_params: dict = {}):
        """
        To avoid caching error for input and output consistency,
        we do not save the coeval cube at this step.
        Simulate therefore is empty.
        """
        return 1.0

    def build_model_data(self, update_params: dict = {}):
        """
        To avoid caching error for input and output consistency,
        we do not save the coeval cube at this step.
        Simulate therefore is empty and build_model_data runs the
        coeval cube without photon conservation with caching off.
        """
        if not self.run_photoncons:
            return False, {}
        cacheconfig = p21.CacheConfig.off()
        cache = OutputCache(self.cache_dir)
        inputs = self.get_update_input(update_params)
        coeval = p21.run_coeval(
            out_redshifts=self.redshifts,
            inputs=inputs,
            write=cacheconfig,
            regenerate=self.regenerate,
            cache=cache,
            **self.global_params,
        )[0]
        global_xhi = coeval.neutral_fraction.mean()
        #xhibox = [p21.IonizedBox.new(redshift=z, inputs=inputs) for z in self.redshifts]
        #cache = OutputCache(self.cache_dir)
        #xhibox = [
        #    p21.io.h5.read_output_struct(cache.find_existing(path)) for path in xhibox
        #]
        #global_xhi = [xhi.get("neutral_fraction").mean() for xhi in xhibox][0]
        flag = global_xhi < self.xhi_threshold if self.flag_type == "smaller" else global_xhi > self.xhi_threshold
        return flag, {"global_xhi": global_xhi}
