import logging
import numpy as np
import py21cmfast as p21
from py21cmfast.io.caching import OutputCache
from .coeval import EoRSimulator
import hashlib
import os
import h5py
from scipy.interpolate import InterpolatedUnivariateSpline
from py21cmfast import wrapper as lib

logger = logging.getLogger("21cmFAST")


def get_lc_file_path(cache_dir: str, inputs: p21.InputParameters):
    cache = OutputCache(cache_dir)
    datasets = cache.list_datasets(
        inputs=inputs,
        all_seeds=False,
        redshift=inputs.node_redshifts[0],
    )
    file_path = [x for x in datasets if "BrightnessTemp" in str(x)]
    if len(file_path) == 0:
        return None
    assert len(file_path) == 1, f"Multiple files found for {inputs.node_redshifts[0]}"
    file_path = str(file_path[0])
    h = hashlib.new("sha256")
    h.update(str(inputs.node_redshifts).encode())
    lc_id = h.hexdigest()
    file_path = file_path.replace("BrightnessTemp", f"Lightcone_{lc_id}")
    return file_path


def tau_GP(cosmo_params, gamma_bg, delta, temp, redshifts):
    r"""Calculating the lyman-alpha optical depth in each pixel using the fluctuating GP approximation.
    Parameters
    ----------
    gamma_bg : float or array_like
        The background photonionization rate in units of 1e-12 s**-1
    delta : float or array_like
        The underlying overdensity
    temp : float or array_like
        The kinectic temperature of the gas in 1e4 K
    redshifts : float or array_like
        Correspoding redshifts along the los
    """
    gamma_local = np.zeros_like(gamma_bg)
    residual_xHI = np.zeros_like(gamma_bg, dtype=np.float64)
    flag_neutral = gamma_bg == 0
    flag_zerodelta = delta == 0
    if gamma_bg.shape != redshifts.shape:
        redshifts = np.tile(redshifts, (*gamma_bg.shape[:-1], 1))
    delta_ss = 2.67e4 * temp**0.17 * (1.0 + redshifts) ** -3 * gamma_bg ** (2.0 / 3.0)
    gamma_local[~flag_neutral] = gamma_bg[~flag_neutral] * (
        0.98
        * ((1.0 + (delta[~flag_neutral] / delta_ss[~flag_neutral]) ** 1.64) ** -2.28)
        + 0.02 * (1.0 + (delta[~flag_neutral] / delta_ss[~flag_neutral])) ** -0.84
    )
    Y_He = 0.245
    # TODO: use global_params
    residual_xHI[~flag_zerodelta] = 1 + gamma_local[~flag_zerodelta] * 1.0155e7 / (
        1.0 + 1.0 / (4.0 / Y_He - 3)
    ) * temp[~flag_zerodelta] ** 0.75 / (
        delta[~flag_zerodelta] * (1.0 + redshifts[~flag_zerodelta]) ** 3
    )
    residual_xHI[~flag_zerodelta] = residual_xHI[~flag_zerodelta] - np.sqrt(
        residual_xHI[~flag_zerodelta] ** 2 - 1.0
    )
    return (
        7875.053145028655
        / (
            cosmo_params.hlittle
            * np.sqrt(cosmo_params.OMm * (1.0 + redshifts) ** 3 + cosmo_params.OMl)
        )
        * delta
        * (1.0 + redshifts) ** 3
        * residual_xHI
    )


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
        only_save_lc: bool = False,
        subdir_for_only_save_lc: bool = False,
    ):
        super().__init__(inputs, cache_dir, regenerate, global_params)
        self.lc_min_redshift = lc_min_redshift
        self.lc_max_redshift = lc_max_redshift
        if self.lc_min_redshift is None:
            self.lc_min_redshift = np.min(inputs.node_redshifts)
        if self.lc_max_redshift is None:
            self.lc_max_redshift = np.max(inputs.node_redshifts)
        self.lc_quantities = lc_quantities
        self.only_save_lc = only_save_lc
        self.subdir_for_only_save_lc = subdir_for_only_save_lc

    def generate_lc_file_path(self, update_params: dict = {}):
        if self.only_save_lc:
            h = hashlib.new("sha256")
            h.update(str(list(update_params.values())).encode())
            lc_id = h.hexdigest()
            if self.subdir_for_only_save_lc:
                file_name = os.path.join(self.cache_dir, f"{lc_id}/lightcone.h5")
            else:
                file_name = os.path.join(self.cache_dir, f"lightcone_{lc_id}.h5")
            return file_name
        else:
            inputs = self.get_update_input(update_params)
            return get_lc_file_path(self.cache_dir, inputs)

    def simulate(self, update_params: dict = {}):
        inputs = self.get_update_input(update_params)
        # try to see if lightcone file exists
        lc_file_path = self.generate_lc_file_path(update_params)
        if lc_file_path is not None:
            if os.path.exists(lc_file_path):
                return 1.0
        cache = OutputCache(self.cache_dir)
        if self.subdir_for_only_save_lc:
            cache_dir = os.path.join(self.cache_dir, lc_file_path.split("/")[-2])
            cache = OutputCache(cache_dir)
        if self.only_save_lc:
            cacheconfig = p21.CacheConfig.off()
        else:
            cacheconfig = p21.CacheConfig.on()
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
            write=cacheconfig,
        )
        # re-obtain the file path
        lc_file_path = self.generate_lc_file_path(update_params)
        lc.save(lc_file_path, clobber=True)
        with h5py.File(lc_file_path, "a") as f:
            f.create_dataset("lightcone_redshifts", data=lc.lightcone_redshifts)
        return 1.0


class LightconeCMBTau(LightconeSimulator):
    """
    Simulate the lightcone and calcualte the CMB optical depth.
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
        save_global_xhi: bool = False,
        save_tau_value: bool = False,
        use_node_boxes: bool = True,
        use_lightcone: bool = False,
        z_extrap_min: float = 5,
        z_extrap_max: float = 25,
        n_z_interp: int = 41,
        only_save_lc: bool = False,
        subdir_for_only_save_lc: bool = False,
    ):
        super().__init__(
            inputs,
            cache_dir,
            regenerate,
            global_params,
            lc_min_redshift,
            lc_max_redshift,
            lc_quantities,
            only_save_lc,
            subdir_for_only_save_lc,
        )
        self.save_global_xhi = save_global_xhi
        self.save_tau_value = save_tau_value
        self.use_node_boxes = use_node_boxes
        self.use_lightcone = use_lightcone
        self.z_extrap_min = z_extrap_min
        self.z_extrap_max = z_extrap_max
        self.n_z_interp = n_z_interp
        if self.use_node_boxes == self.use_lightcone:
            raise ValueError("use_node_boxes and use_lightcone cannot be the same")
        if self.only_save_lc and self.use_node_boxes:
            raise ValueError(
                "only_save_lc and use_node_boxes cannot be true at the same time"
            )

    @property
    def z_interp_arr(self):
        return np.linspace(self.z_extrap_min, self.z_extrap_max, self.n_z_interp)

    def build_model_data(self, update_params: dict = {}):
        inputs = self.get_update_input(update_params)
        if self.use_node_boxes:
            redshifts = inputs.node_redshifts
            cache = OutputCache(self.cache_dir)
            xhibox = [p21.IonizedBox.new(redshift=z, inputs=inputs) for z in redshifts]
            xhibox = [
                p21.io.h5.read_output_struct(cache.find_existing(path))
                for path in xhibox
            ]
            global_xhi = [xhi.get("neutral_fraction").mean() for xhi in xhibox]
        elif self.use_lightcone:
            lc_file_path = self.generate_lc_file_path(update_params)
            if not os.path.exists(lc_file_path):
                raise FileNotFoundError(f"Lightcone file not found: {lc_file_path}")
            with h5py.File(lc_file_path, "r") as f:
                global_xhi = np.array(f["lightcones"]["neutral_fraction"]).mean(
                    axis=(0, 1)
                )
                redshifts = np.array(f["lightcone_redshifts"])
        # Order the redshifts in increasing order
        redshifts, global_xhi = np.sort(np.array([redshifts, global_xhi]))
        neutral_frac_func = InterpolatedUnivariateSpline(redshifts, global_xhi, k=1)
        global_xhi = neutral_frac_func(self.z_interp_arr)
        # Ensure that the neutral fraction does not exceed unity, or go negative
        np.clip(global_xhi, 0, 1, global_xhi)
        tau_value = lib.cfuncs.compute_tau(
            inputs=inputs,
            redshifts=self.z_interp_arr,
            global_xHI=global_xhi,
        )
        blob = {}
        if self.save_global_xhi:
            blob["global_xhi_interp"] = np.array(global_xhi)
        if self.save_tau_value:
            blob["cmb_tau"] = np.array(tau_value)
        return tau_value, blob


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
        only_save_lc: bool = False,
        subdir_for_only_save_lc: bool = False,
    ):
        super().__init__(
            inputs,
            cache_dir,
            regenerate,
            global_params,
            lc_min_redshift,
            lc_max_redshift,
            lc_quantities,
            only_save_lc,
            subdir_for_only_save_lc,
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
        lc_file_path = self.generate_lc_file_path(update_params)
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


class LightconeLyaOpticalDepth(LightconeSimulator):
    """
    Simulate the lightcone and calculate the lya optical depth.
    """

    def __init__(
        self,
        redshift_bin_edges: list[float],
        inputs: p21.InputParameters,
        cache_dir: str,
        regenerate: bool = False,
        global_params: dict | None = None,
        lc_min_redshift: float | None = None,
        lc_max_redshift: float | None = None,
        lc_quantities: list[str] = [
            "brightness_temp",
            "neutral_fraction",
            "ionisation_rate_G12",
            "density",
            "kinetic_temperature",
        ],
        correct_gp_to_hydro: bool = False,
        max_correct_filling_factor: float = 0.7,
        kde_repeat_num: int = 30,
        inverse_tau_bin_edges: list[float] = np.linspace(0 - 0.0025, 1 + 0.0025, 202),
        save_tau_gp: bool = False,
        save_inv_tau_pdf: bool = False,
        model_err_fraction: float = 0.0,
        only_save_lc: bool = False,
        subdir_for_only_save_lc: bool = False,
    ):
        super().__init__(
            inputs,
            cache_dir,
            regenerate,
            global_params,
            lc_min_redshift,
            lc_max_redshift,
            lc_quantities,
            only_save_lc,
            subdir_for_only_save_lc,
        )
        self.redshift_bin_edges = redshift_bin_edges
        self.save_tau_gp = save_tau_gp
        self.save_inv_tau_pdf = save_inv_tau_pdf
        self.correct_gp_to_hydro = correct_gp_to_hydro
        if self.correct_gp_to_hydro:
            self.correct_gp_to_hydro_mapping = os.path.join(
                os.path.dirname(__file__), "data/Forests/tau_mapping.npy"
            )
            self.cpdf = np.load(
                self.correct_gp_to_hydro_mapping, allow_pickle=True
            ).item()
        self.max_correct_filling_factor = max_correct_filling_factor
        self.kde_repeat_num = kde_repeat_num
        self.inverse_tau_bin_edges = inverse_tau_bin_edges
        self.model_err_fraction = model_err_fraction

    @property
    def redshift_bin_centers(self):
        return (self.redshift_bin_edges[:-1] + self.redshift_bin_edges[1:]) / 2

    def build_model_data(self, update_params: dict = {}):
        lc_file_path = self.generate_lc_file_path(update_params)
        if not os.path.exists(lc_file_path):
            raise FileNotFoundError(f"Lightcone file not found: {lc_file_path}")
        with h5py.File(lc_file_path, "r") as f:
            z_lc = np.array(f["lightcone_redshifts"])
            z_sel_list = [
                np.logical_and(
                    z_lc >= self.redshift_bin_edges[i],
                    z_lc < self.redshift_bin_edges[i + 1],
                )
                for i in range(len(self.redshift_bin_edges) - 1)
            ]
            tau_gp = [
                tau_GP(
                    self.inputs.cosmo_params,
                    np.array(f["lightcones"]["ionisation_rate_G12"])[:, :, z_sel],
                    1 + np.array(f["lightcones"]["density"])[:, :, z_sel],
                    np.array(f["lightcones"]["kinetic_temperature"])[:, :, z_sel] / 1e4,
                    z_lc[z_sel],
                )
                for z_sel in z_sel_list
            ]
            tau_gp = [
                -np.log(np.mean(np.exp(-tau_i), axis=-1)).ravel() for tau_i in tau_gp
            ]
            tau_gp = np.array(tau_gp)
            tau_gp[tau_gp < 0] = 1e-4
            if self.correct_gp_to_hydro:
                tau_hydros = []
                filling_factor = [
                    np.array(f["lightcones"]["neutral_fraction"])[:, :, z_sel].mean()
                    for z_sel in z_sel_list
                ]
                for i in range(len(self.redshift_bin_edges) - 1):
                    if filling_factor[i] < self.max_correct_filling_factor:
                        tau_hydros_i = (
                            self.cpdf.sample(
                                inherent_conditionals={
                                    "z": self.redshift_bin_centers[i],
                                    "xHI": filling_factor[i],
                                },
                                conditionals={
                                    "inversetau_GP": np.array(
                                        [
                                            tau_gp[i] ** -1,
                                        ]
                                        * self.kde_repeat_num
                                    ).ravel()
                                },
                                n_samples=1,
                                random_state=None,  # different realizations
                                keep_dims=False,
                            ).flatten()
                            ** -1
                        )
                        tau_hydros.append(tau_hydros_i)
                    else:
                        tau_hydros.append(
                            np.array(
                                [
                                    tau_gp[i] ** -1,
                                ]
                                * self.kde_repeat_num
                            ).ravel()
                        )
                tau_hydros = np.array(tau_hydros)
                tau_gp = tau_hydros
        tau_pdf = []
        for i in range(len(self.redshift_bin_edges) - 1):
            tau_gp_i = tau_gp[i]
            if self.model_err_fraction > 0:
                tau_gp_i = tau_gp_i * (
                    1 + np.random.normal(0, self.model_err_fraction, tau_gp[i].shape)
                )
                tau_gp_i[tau_gp_i <= 0] = 1e-4
            tau_pdf_i = np.histogram(
                tau_gp_i**-1, bins=self.inverse_tau_bin_edges, density=True
            )[0]
            tau_pdf_i[np.isnan(tau_pdf_i)] = 0.0
            tau_pdf.append(tau_pdf_i)
        tau_pdf = np.array(tau_pdf)
        blob = {}
        if self.save_tau_gp:
            blob["tau_forest_gp"] = np.nan_to_num(tau_gp, posinf=1e10, neginf=-1e10)
        if self.save_inv_tau_pdf:
            blob["inv_tau_forest_pdf"] = np.nan_to_num(tau_pdf, posinf=1e10, neginf=-1e10)
        return tau_pdf, blob
