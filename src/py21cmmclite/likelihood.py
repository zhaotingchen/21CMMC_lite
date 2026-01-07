from .coeval import (
    CoevalNeutralFraction,
    EoRSimulator,
    LuminosityFunctionSimulator,
    CoevalPhotonConsFlag,
)
import numpy as np
import py21cmfast as p21
from .lightcone import (
    LightconeNeutralFraction,
    LightconeCMBTau,
    LightconeLyaOpticalDepth,
)
import os
from scipy.interpolate import InterpolatedUnivariateSpline
import logging

logging.basicConfig(level=logging.INFO)


def compute_chi2(model_vector, data_vector, inv_covariance):
    diff = np.array(model_vector - data_vector).ravel()
    chi2 = np.dot(diff, np.dot(inv_covariance, diff))
    chi2 = np.ravel(chi2)
    return float(chi2[0])


class LikelihoodBase:
    """
    Base class for constructing likelihoods.
    """

    def __init__(
        self,
        varied_params: list[str],
    ):
        self.varied_params = varied_params
        self.simulators = []

    def get_update_dict(self, varied_params_values):
        if not len(varied_params_values) == len(self.varied_params):
            raise ValueError("input values must have the same length as varied_params")
        update_params = dict(zip(self.varied_params, varied_params_values))
        return update_params

    def gather_data(self):
        pass

    def invoke_simulators(self, params_values=None):
        if params_values is None:
            update_dict = {}
        else:
            update_dict = self.get_update_dict(params_values)
        for simulator in self.simulators:
            simulator.simulate(update_dict)

    def gather_model(self, params_values=None):
        self.invoke_simulators(params_values)
        if params_values is None:
            update_dict = {}
        else:
            update_dict = self.get_update_dict(params_values)
        model_vector = []
        blob_tot = {}
        for simulator in self.simulators:
            model, blob = simulator.build_model_data(update_dict)
            additional_blob = simulator.build_blob_data(update_dict)
            model_vector.append(model)
            if blob is not None:
                blob_tot.update(blob)
            if additional_blob is not None:
                blob_tot.update(additional_blob)
        return model_vector, blob_tot

    def likelihood_function(self, model, data):
        pass

    def compute_likelihood(self, varied_params_values):
        model, blob = self.gather_model(varied_params_values)
        ll = self.likelihood_function(model, self.gather_data())
        if np.isnan(ll):
            logging.warning(
                f"Likelihood is nan for params {varied_params_values}"
                f"for simulator {self.__class__.__name__}. "
                "setting to -inf"
            )
            ll = -np.inf
        return ll, blob


class LikelihoodPhotonConsFlag(EoRSimulator, LikelihoodBase):
    """
    Likelihood for photon conservation flag.

    If the flag is true, rest of the likelihoods will be skipped
    in a sampler. Therefore, to ensure the blob shape is correct,
    the blob_shape should be provided unless no blob is saved.
    """

    def __init__(
        self,
        inputs_21cmfast: p21.InputParameters,
        varied_params: list[str],
        cache_dir: str,
        regenerate: bool = False,
        global_params: dict | None = None,
        xhi_threshold: float = 0.001,
        redshifts: list[float] | np.ndarray = None,
        flag_type: str = "larger",
        blob_shape: tuple[int, ...] | None = None,
    ):
        EoRSimulator.__init__(
            self, inputs_21cmfast, cache_dir, regenerate, global_params
        )
        LikelihoodBase.__init__(self, varied_params)
        self.simulators = [
            CoevalPhotonConsFlag(
                inputs=inputs_21cmfast,
                cache_dir=cache_dir,
                regenerate=regenerate,
                global_params=global_params,
                xhi_threshold=xhi_threshold,
                redshifts=redshifts,
                flag_type=flag_type,
            )
        ]
        self.blob_shape = blob_shape

    def likelihood_function(self, model, data):
        if model[0]:
            return -np.inf
        return 0.0

    def compute_likelihood(self, varied_params_values):
        model, blob = self.gather_model(varied_params_values)
        ll = self.likelihood_function(model, self.gather_data())
        if ll == 0.0 or self.blob_shape is None:
            return ll, {}
        else:
            return ll, {"empty_blob": np.zeros(self.blob_shape)}


class LikelihoodForest(EoRSimulator, LikelihoodBase):
    """
    Likelihood for forest.
    """

    def __init__(
        self,
        inputs_21cmfast: p21.InputParameters,
        varied_params: list[str],
        cache_dir: str,
        regenerate: bool = False,
        global_params: dict | None = None,
        lc_min_redshift: float | None = None,
        lc_max_redshift: float | None = None,
        observation: str = "xqr30",
        use_limit_pdf: bool = False,
        correct_gp_to_hydro: bool = False,
        save_tau_gp: bool = False,
        save_inv_tau_pdf: bool = False,
        model_err_fraction: float = 0.0,
        only_save_lc: bool = False,
        subdir_for_only_save_lc: bool = False,
        z_min: float = -np.inf,
        z_max: float = np.inf,
        cut_threshold: float = 0.001,
    ):
        EoRSimulator.__init__(
            self, inputs_21cmfast, cache_dir, regenerate, global_params
        )
        LikelihoodBase.__init__(self, varied_params)
        self.observation = observation
        if observation != "xqr30":
            raise ValueError(f"Observation {observation} not supported")
        self.use_limit_pdf = use_limit_pdf
        self.only_save_lc = only_save_lc
        self.subdir_for_only_save_lc = subdir_for_only_save_lc
        self.cut_threshold = cut_threshold
        data_dir = os.path.join(
            os.path.dirname(__file__), f"data/Forests/{observation}"
        )
        # hard coded for now
        inverse_tau_bin_edges = np.linspace(0 - 0.0025, 1 + 0.0025, 202)
        redshift_bin_centers = np.linspace(5, 6.2, 7)
        #redshift_bin_edges = np.linspace(4.9, 6.3, 8)
        sel = np.logical_and(redshift_bin_centers >= z_min, redshift_bin_centers <= z_max)
        redshift_bin_centers = redshift_bin_centers[sel]
        redshift_bin_edges = np.append(redshift_bin_centers-0.1, redshift_bin_centers[-1] + 0.1)
        data = []
        for z in redshift_bin_centers:
            data_i = np.load(
                os.path.join(
                    data_dir,
                    f"dz0pt2/inv_tau_dect_z{str(np.round(z,1)).replace('.', 'pt')}.npy",
                )
            )
            if use_limit_pdf:
                data_i_limit = np.load(
                    os.path.join(
                        data_dir,
                        f"dz0pt2/inv_tau_limit_z{str(np.round(z,1)).replace('.', 'pt')}.npy",
                    )
                )
                if data_i_limit.size > 0:
                    data_i = np.vstack((data_i, data_i_limit))
            data.append(data_i)
        # data is sum normalised, whereas we need the PDF
        self.inv_tau_pdf_data = [data_i / np.diff(inverse_tau_bin_edges)[0] for data_i in data]
        # some data produce negative values of tau, which then adds to zeroth bin. Set to zero.
        # This is better than throwing them away, because then the PDF will not be normalised to 1,
        # instead, a small part is missing due to these negative values.
        for i, data_i in enumerate(self.inv_tau_pdf_data):
            data_i[:,0] = 0.0
            self.inv_tau_pdf_data[i] = data_i
        lc_quantities = [
            "brightness_temp",
            "neutral_fraction",
            "ionisation_rate_G12",
            "density",
            "kinetic_temperature",
        ]
        max_correct_filling_factor = 0.7
        kde_repeat_num = 30
        self.simulators = [
            LightconeLyaOpticalDepth(
                redshift_bin_edges=redshift_bin_edges,
                inputs=inputs_21cmfast,
                cache_dir=cache_dir,
                regenerate=regenerate,
                global_params=global_params,
                lc_min_redshift=lc_min_redshift,
                lc_max_redshift=lc_max_redshift,
                lc_quantities=lc_quantities,
                correct_gp_to_hydro=correct_gp_to_hydro,
                max_correct_filling_factor=max_correct_filling_factor,
                kde_repeat_num=kde_repeat_num,
                inverse_tau_bin_edges=inverse_tau_bin_edges,
                save_tau_gp=save_tau_gp,
                save_inv_tau_pdf=save_inv_tau_pdf,
                model_err_fraction=model_err_fraction,
                only_save_lc=only_save_lc,
                subdir_for_only_save_lc=subdir_for_only_save_lc,
            )
        ]

    def gather_data(self):
        return [self.inv_tau_pdf_data]

    #def likelihood_function(self, model, data):
    #    log_p = 0.0
    #    model = model[0]
    #    data = data[0]
    #    for i in range(len(self.simulators[0].redshift_bin_edges) - 1):
    #        log_prob_i = np.log(
    #            (self.inv_tau_pdf_data[i] * model[i] / model[i].max()).sum(1)
    #        ) * (self.inv_tau_pdf_data[i]).sum(1)
    #        log_p += log_prob_i.sum()
    #    if np.isnan(log_p) or np.isinf(log_p):
    #        log_p = -np.inf
    #    return log_p
    def likelihood_function(self, model, data):
        """
        Cross-entropy likelihood
        """
        log_l = 0.0
        model = model[0]
        data = data[0]
        delta_bin = np.diff(self.simulators[0].inverse_tau_bin_edges)[0]
        cut_threshold = self.cut_threshold
        for i in range(len(self.simulators[0].redshift_bin_edges) - 1):
            model_i = model[i]
            sel_i = model_i>0
            data_i = data[i]
            for j in range(len(data_i)):
                data_q_j = data_i[j]
                if any(model_i[(data_q_j / data_q_j.max())>cut_threshold]==0):
                    return -np.inf
            log_l += (np.log(model_i[sel_i]) * data_q_j[sel_i]).sum() * delta_bin
        return log_l


class LikelihoodGaussian(LikelihoodBase):
    """
    Class for Gaussian likelihoods.
    """

    def __init__(
        self,
        varied_params: list[str],
        data_dict: dict | None = None,
        simulate_data: bool = False,
        simulate_error_fraction: float = 0.1,
    ):
        super().__init__(varied_params)
        if data_dict is None:
            data_vector = None
            data_inv_covariance = None
        else:
            data_vector = data_dict["data_vector"]
            data_inv_covariance = data_dict["data_inv_covariance"]
        if data_inv_covariance is not None:
            if data_inv_covariance.shape == data_vector.shape:
                self.data_inv_covariance = np.diag(data_inv_covariance.ravel())
            elif data_inv_covariance.shape == (data_vector.size, data_vector.size):
                self.data_inv_covariance = data_inv_covariance
            else:
                raise ValueError(
                    "data_inv_covariance must be a square matrix with the same size as data_vector,"
                    "or a vector with the same size as data_vector assuming diagonal covariance"
                )
        self.data_dict = {
            "data_vector": data_vector,
            "data_inv_covariance": data_inv_covariance,
        }
        self.simulate_data = simulate_data
        self.simulate_error_fraction = simulate_error_fraction

    def gather_data(self):
        if self.simulate_data and self.data_dict["data_vector"] is None:
            # use default params to simulate data
            data_vec = np.array(self.gather_model()[0]).ravel()
            inv_cov = np.diag(1 / data_vec / self.simulate_error_fraction) ** 2
            self.data_dict = {
                "data_vector": data_vec,
                "data_inv_covariance": inv_cov,
            }
        return self.data_dict

    def likelihood_function(self, model, data):
        model_vector = np.array(model).ravel()
        data_vector = data["data_vector"]
        data_inv_covariance = data["data_inv_covariance"]
        chi2 = compute_chi2(model_vector, data_vector, data_inv_covariance)
        return -0.5 * chi2


class LikelihoodCoevalNeutralFraction(EoRSimulator, LikelihoodGaussian):
    """
    Likelihood for neutral fraction for coeval cubes.
    """

    def __init__(
        self,
        redshifts: list[float] | np.ndarray,
        inputs_21cmfast: p21.InputParameters,
        varied_params: list[str],
        cache_dir: str,
        regenerate: bool = False,
        global_params: dict | None = None,
        data_dict: dict | None = None,
        simulate_data: bool = False,
        simulate_error_fraction: float = 0.1,
        save_global_xhi: bool = False,
        save_xhi_box: bool = False,
    ):
        EoRSimulator.__init__(
            self, inputs_21cmfast, cache_dir, regenerate, global_params
        )
        LikelihoodGaussian.__init__(
            self, varied_params, data_dict, simulate_data, simulate_error_fraction
        )
        self.redshifts = redshifts
        self.save_global_xhi = save_global_xhi
        self.save_xhi_box = save_xhi_box
        self.simulators = [
            CoevalNeutralFraction(
                redshifts=redshifts,
                inputs=inputs_21cmfast,
                cache_dir=cache_dir,
                regenerate=regenerate,
                global_params=global_params,
                save_global_xhi=save_global_xhi,
                save_xhi_box=save_xhi_box,
            )
        ]


class LikelihoodLightconeNeutralFraction(EoRSimulator, LikelihoodGaussian):
    """
    Likelihood for neutral fraction for lightcone at different redshift bins.
    """

    def __init__(
        self,
        xhi_z_edges_low: list[float],
        xhi_z_edges_high: list[float],
        inputs_21cmfast: p21.InputParameters,
        varied_params: list[str],
        cache_dir: str,
        regenerate: bool = False,
        global_params: dict | None = None,
        lc_min_redshift: float | None = None,
        lc_max_redshift: float | None = None,
        lc_quantities: list[str] = ["brightness_temp", "neutral_fraction"],
        data_dict: dict | None = None,
        simulate_data: bool = False,
        simulate_error_fraction: float = 0.1,
        save_xhi_points: bool = False,
        save_xhi_per_slice: bool = False,
        save_xhi_lc: bool = False,
        only_save_lc: bool = False,
        subdir_for_only_save_lc: bool = False,
    ):
        EoRSimulator.__init__(
            self, inputs_21cmfast, cache_dir, regenerate, global_params
        )
        LikelihoodGaussian.__init__(
            self, varied_params, data_dict, simulate_data, simulate_error_fraction
        )
        self.xhi_z_edges_low = xhi_z_edges_low
        self.xhi_z_edges_high = xhi_z_edges_high
        self.lc_min_redshift = lc_min_redshift
        self.lc_max_redshift = lc_max_redshift
        self.lc_quantities = lc_quantities
        self.save_xhi_points = save_xhi_points
        self.save_xhi_lc = save_xhi_lc
        self.save_xhi_per_slice = save_xhi_per_slice
        self.only_save_lc = only_save_lc
        self.subdir_for_only_save_lc = subdir_for_only_save_lc
        self.simulators = [
            LightconeNeutralFraction(
                xhi_z_edges_low=xhi_z_edges_low,
                xhi_z_edges_high=xhi_z_edges_high,
                inputs=inputs_21cmfast,
                cache_dir=cache_dir,
                regenerate=regenerate,
                global_params=global_params,
                lc_min_redshift=lc_min_redshift,
                lc_max_redshift=lc_max_redshift,
                lc_quantities=lc_quantities,
                save_xhi_points=save_xhi_points,
                save_xhi_per_slice=save_xhi_per_slice,
                save_xhi_lc=save_xhi_lc,
                only_save_lc=only_save_lc,
                subdir_for_only_save_lc=subdir_for_only_save_lc,
            )
        ]


class LikelihoodLightconeCMBTau(EoRSimulator, LikelihoodGaussian):
    """
    Likelihood for CMB optical depth.
    """

    def __init__(
        self,
        inputs_21cmfast: p21.InputParameters,
        cache_dir: str,
        varied_params: list[str],
        data_dict: dict | None = None,
        simulate_data: bool = False,
        simulate_error_fraction: float = 0.1,
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
        EoRSimulator.__init__(
            self, inputs_21cmfast, cache_dir, regenerate, global_params
        )
        LikelihoodGaussian.__init__(
            self, varied_params, data_dict, simulate_data, simulate_error_fraction
        )
        # for CMB tau, if no input data default should be Planck prior
        if data_dict is None:
            # 1908.09856
            self.data_dict["data_vector"] = np.array([0.059])
            self.data_dict["data_inv_covariance"] = 1 / (np.array([0.006]) ** 2)
        self.lc_min_redshift = lc_min_redshift
        self.lc_max_redshift = lc_max_redshift
        self.lc_quantities = lc_quantities
        self.save_global_xhi = save_global_xhi
        self.save_tau_value = save_tau_value
        self.use_node_boxes = use_node_boxes
        self.use_lightcone = use_lightcone
        self.z_extrap_min = z_extrap_min
        self.z_extrap_max = z_extrap_max
        self.n_z_interp = n_z_interp
        self.only_save_lc = only_save_lc
        self.subdir_for_only_save_lc = subdir_for_only_save_lc
        self.simulators = [
            LightconeCMBTau(
                inputs=inputs_21cmfast,
                cache_dir=cache_dir,
                regenerate=regenerate,
                global_params=global_params,
                lc_min_redshift=lc_min_redshift,
                lc_max_redshift=lc_max_redshift,
                lc_quantities=lc_quantities,
                save_global_xhi=save_global_xhi,
                save_tau_value=save_tau_value,
                use_node_boxes=use_node_boxes,
                use_lightcone=use_lightcone,
                z_extrap_min=z_extrap_min,
                z_extrap_max=z_extrap_max,
                n_z_interp=n_z_interp,
                only_save_lc=only_save_lc,
                subdir_for_only_save_lc=subdir_for_only_save_lc,
            )
        ]


class LikelihoodLuminosityFunction(LikelihoodGaussian):
    """
    Likelihood for luminosity function.
    """

    def __init__(
        self,
        inputs_21cmfast: p21.InputParameters,
        cache_dir: str,
        varied_params: list[str],
        data_dict: dict | None = None,
        redshifts: list[float] | np.ndarray | None = None,
        n_uv_bins: int = 100,
        save_uvlf: bool = False,
        simulate_data: bool = False,
        simulate_error_fraction: float = 0.1,
        Muv_range: tuple[float, float] | None = None,
    ):
        super().__init__(
            varied_params, data_dict, simulate_data, simulate_error_fraction
        )
        self.n_uv_bins = n_uv_bins
        self.save_uvlf = save_uvlf
        self.data_dict = data_dict
        if data_dict is None and redshifts is None:
            redshifts = np.array([6, 7, 8, 10])
            data_dir = [
                os.path.join(
                    os.path.dirname(__file__),
                    f"data/UVLF/LF_lfuncs_z{str(np.round(z,0))}.npz",
                )
                for z in redshifts
            ]
            noise_dir = [
                os.path.join(
                    os.path.dirname(__file__),
                    f"data/UVLF/LF_sigmas_z{str(np.round(z,0))}.npz",
                )
                for z in redshifts
            ]
            data = [np.load(data_dir) for data_dir in data_dir]
            noise = [np.load(noise_dir) for noise_dir in noise_dir]
            self.data_dict = {
                "x_vector": [d["Muv"] for d in data],
                # "data_vector": np.concatenate([d["lfunc"] for d in data]),
                "data_vector": [d["lfunc"] for d in data],
                # "data_inv_covariance": np.diag(
                #    np.concatenate([1 / d["sigma"] ** 2 for d in noise])
                # ),
                "data_inv_covariance": [1 / d["sigma"] ** 2 for d in noise],
            }
        self.Muv_range = Muv_range
        if Muv_range is not None:
            x_vec_new = []
            data_vec_new = []
            data_inv_cov_new = []
            for i, x_vec in enumerate(self.data_dict["x_vector"]):
                x_sel = np.logical_and(x_vec >= Muv_range[0], x_vec <= Muv_range[1])
                x_vec_new.append(x_vec[x_sel])
                data_vec_new.append(self.data_dict["data_vector"][i][x_sel])
                data_inv_cov_new.append(self.data_dict["data_inv_covariance"][i][x_sel])
            self.data_dict["x_vector"] = x_vec_new
            self.data_dict["data_vector"] = data_vec_new
            self.data_dict["data_inv_covariance"] = data_inv_cov_new
        # contatenate data_vector and data_inv_covariance
        self.data_dict["data_vector"] = np.concatenate(self.data_dict["data_vector"])
        self.data_dict["data_inv_covariance"] = np.diag(
            np.concatenate(self.data_dict["data_inv_covariance"])
        )
        self.redshifts = redshifts
        self.simulators = [
            LuminosityFunctionSimulator(
                inputs=inputs_21cmfast,
                redshifts=redshifts,
                n_uv_bins=n_uv_bins,
                save_uvlf=save_uvlf,
            )
        ]

    def likelihood_function(self, model, data):
        Muvfunc, lfunc = model[0]
        lf_interp = []
        for i in range(len(self.redshifts)):
            x = Muvfunc[i]
            y = lfunc[i]
            sel = ~np.isnan(y)
            loglf_func = InterpolatedUnivariateSpline(x[sel][::-1], y[sel][::-1])
            lf_i = 10 ** loglf_func(data["x_vector"][i])
            lf_interp.append(lf_i)
        lf_interp = np.concatenate(lf_interp)
        return super().likelihood_function(lf_interp, data)
