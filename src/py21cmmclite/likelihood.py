from .coeval import CoevalNeutralFraction, EoRSimulator
import numpy as np
import py21cmfast as p21


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
        for simulator in self.simulators:
            model_vector.append(simulator.build_model_data(update_dict))
        return model_vector

    def likelihood_function(self, model, data):
        pass

    def compute_likelihood(self, varied_params_values):
        return self.likelihood_function(
            self.gather_model(varied_params_values), self.gather_data()
        )


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
            data_vec = np.array(self.gather_model()).ravel()
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


class LikelihoodNeutralFraction(EoRSimulator, LikelihoodGaussian):
    """
    Likelihood for neutral fraction.
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
    ):
        EoRSimulator.__init__(
            self, inputs_21cmfast, cache_dir, regenerate, global_params
        )
        LikelihoodGaussian.__init__(
            self, varied_params, data_dict, simulate_data, simulate_error_fraction
        )
        self.redshifts = redshifts
        self.simulators = [
            CoevalNeutralFraction(
                redshifts=redshifts,
                inputs=inputs_21cmfast,
                cache_dir=cache_dir,
                regenerate=regenerate,
                global_params=global_params,
            )
        ]
