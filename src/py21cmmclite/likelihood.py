from .coeval import CoevalNeutralFraction, EoRSimulator
import numpy as np
import py21cmfast as p21
from .lightcone import LightconeNeutralFraction


def compute_chi2(model_vector, data_vector, inv_covariance):
    diff = np.array(model_vector - data_vector).ravel()
    chi2 = np.dot(diff, np.dot(inv_covariance, diff))
    return chi2


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
        update_params = {
            self.varied_params[i]: varied_params_values[i]
            for i in range(len(self.varied_params))
        }
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
        return ll, blob


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
        save_xhi_lc: bool = False,
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
                save_xhi_lc=save_xhi_lc,
            )
        ]
