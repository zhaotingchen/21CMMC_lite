import logging
import numpy as np
import py21cmfast as p21
from py21cmfast.io.caching import OutputCache
from .coeval import EoRSimulator
import hashlib
import re
import os
import h5py

logger = logging.getLogger("21cmFAST")


class LightconeSimulator(EoRSimulator):
    """
    Base class for constructing simulators of lightcone using 21cmFAST.
    """

    def __init__(
        self,
        inputs: p21.InputParameters,
        cache_dir: str,
        regenerate: bool = False,
        global_params: dict | None = None,
        lc_min_redshift: float | None = None,
        lc_max_redshift: float | None = None,
        lc_quantities: list[str] = ["brightness_temp", "neutral_fraction"],
    ):
        super().__init__(inputs, cache_dir, regenerate, global_params)
        self.lc_min_redshift = lc_min_redshift
        self.lc_max_redshift = lc_max_redshift
        if self.lc_min_redshift is None:
            self.lc_min_redshift = np.min(inputs.node_redshifts)
        if self.lc_max_redshift is None:
            self.lc_max_redshift = np.max(inputs.node_redshifts)
        self.lc_quantities = lc_quantities

    def get_lc_file_path(self, inputs: p21.InputParameters):
        cache = OutputCache(self.cache_dir)
        datasets = cache.list_datasets(
            inputs=inputs,
            all_seeds=False,
            redshift=inputs.node_redshifts[0],
        )
        file_path = [x for x in datasets if "BrightnessTemp" in str(x)]
        if len(file_path) == 0:
            return None
        assert (
            len(file_path) == 1
        ), f"Multiple files found for {inputs.node_redshifts[0]}"
        file_path = str(file_path[0])
        h = hashlib.new("sha256")
        h.update(str(inputs.node_redshifts).encode())
        lc_id = h.hexdigest()
        file_path = file_path.replace("BrightnessTemp", f"Lightcone_{lc_id}")
        return file_path

    def simulate(self, update_params: dict = {}):
        inputs = self.get_update_input(update_params)
        # try to see if lightcone file exists
        lc_file_path = self.get_lc_file_path(inputs)
        if lc_file_path is not None:
            if os.path.exists(lc_file_path):
                return 1.0
        cache = OutputCache(self.cache_dir)
        # maybe should allow select angular lightcone as well
        lc_cfg = p21.RectilinearLightconer.between_redshifts(
            min_redshift=self.lc_min_redshift,
            max_redshift=self.lc_max_redshift,
            resolution=inputs.simulation_options.cell_size,
            quantities=self.lc_quantities,
        )
        # if cache did not exist, after running it will
        lc = p21.run_lightcone(
            lightconer=lc_cfg,
            inputs=inputs,
            cache=cache,
        )
        # re-obtain the file path
        lc_file_path = self.get_lc_file_path(inputs)
        lc.save(lc_file_path, clobber=True)
        with h5py.File(lc_file_path, "a") as f:
            f.create_dataset("lightcone_redshifts", data=lc.lightcone_redshifts)
        return 1.0


class LightconeNeutralFraction(LightconeSimulator):
    """
    Simulate the lightcone and calculate the global neutral fraction.
    """

    def __init__(
        self,
        xhi_z_edges_low: list[float],
        xhi_z_edges_high: list[float],
        inputs: p21.InputParameters,
        cache_dir: str,
        regenerate: bool = False,
        global_params: dict | None = None,
        lc_min_redshift: float | None = None,
        lc_max_redshift: float | None = None,
        lc_quantities: list[str] = ["brightness_temp", "neutral_fraction"],
        save_xhi_points: bool = False,
        save_xhi_lc: bool = False,
    ):
        super().__init__(
            inputs,
            cache_dir,
            regenerate,
            global_params,
            lc_min_redshift,
            lc_max_redshift,
            lc_quantities,
        )
        self.xhi_z_edges_low = xhi_z_edges_low
        self.xhi_z_edges_high = xhi_z_edges_high
        self.save_xhi_points = save_xhi_points
        self.save_xhi_lc = save_xhi_lc
        assert len(xhi_z_edges_low) == len(
            xhi_z_edges_high
        ), "xhi_z_edges_low and xhi_z_edges_high must have the same length"
        assert (
            "neutral_fraction" in self.lc_quantities
        ), "neutral_fraction must be in lc_quantities"

    def build_model_data(self, update_params: dict = {}):
        inputs = self.get_update_input(update_params)
        lc_file_path = self.get_lc_file_path(inputs)
        if not os.path.exists(lc_file_path):
            raise FileNotFoundError(f"Lightcone file not found: {lc_file_path}")
        with h5py.File(lc_file_path, "r") as f:
            xhi_lc = np.array(f["lightcones"]["neutral_fraction"])
            z_lc = np.array(f["lightcone_redshifts"])
        xhi_points = []
        for i in range(len(self.xhi_z_edges_low)):
            z_sel = np.where(
                (z_lc >= self.xhi_z_edges_low[i]) & (z_lc <= self.xhi_z_edges_high[i])
            )[0]
            xhi_points.append(np.mean(xhi_lc[:, :, z_sel]))
        blob = {}
        if self.save_xhi_points:
            blob["xhi_points_lc"] = np.array(xhi_points)
        if self.save_xhi_lc:
            blob["xhi_lc"] = np.array(xhi_lc)
        return xhi_points, blob
