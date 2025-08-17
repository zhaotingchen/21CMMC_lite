from .coeval import CoevalNeutralFraction
from ._util import BaseSimulator
import numpy as np
import py21cmfast as p21


def compute_chi2(model_vector, data_vector, inv_covariance):
    diff = np.array(model_vector - data_vector).ravel()
    chi2 = np.dot(diff, np.dot(inv_covariance, diff))
    return chi2


class LikelihoodBase(BaseSimulator):
    """
    Base class for constructing likelihoods.
    """

    def __init__(
        self,
        inputs_21cmfast: p21.InputParameters,
        varied_params: dict,
        cache_dir: str,
        regenerate: bool = False,
        global_params: dict | None = None,
    ):
        super().__init__(
            inputs_21cmfast, varied_params, cache_dir, regenerate, global_params
        )
        self.simulators = []
        self._model_vector = []
        self._simulated = False

    def invoke_simulators(self):
        for simulator in self.simulators:
            simulator.simulate()
        self._simulated = True

    def gather_model_vector(self):
        if not self._simulated:
            raise ValueError(
                "Simulators have not been invoked yet. "
                "Please run invoke_simulators() first."
            )
        model_vector = []
        for simulator in self.simulators:
            model_vector.append(simulator.build_model_data())
        self._model_vector = model_vector
        return model_vector

    def gather_data_vector(self):
        pass

    def compute_likelihood(self):
        pass


class LikelihoodNeutralFraction(LikelihoodBase):
    """
    Likelihood for neutral fraction.
    """

    def __init__(
        self,
        redshifts: list[float] | np.ndarray,
        inputs_21cmfast: p21.InputParameters,
        varied_params: dict,
        cache_dir: str,
        data_vector: np.ndarray,
        data_inv_covariance: np.ndarray,
        regenerate: bool = False,
        global_params: dict | None = None,
    ):
        super().__init__(
            inputs_21cmfast, varied_params, cache_dir, regenerate, global_params
        )
        self.redshifts = redshifts
        self.simulators = [
            CoevalNeutralFraction(
                redshifts=redshifts,
                inputs_21cmfast=inputs_21cmfast,
                varied_params=varied_params,
                cache_dir=cache_dir,
                regenerate=regenerate,
                global_params=global_params,
            )
        ]
        self.data_vector = data_vector
        if np.allclose(data_inv_covariance.shape, (data_vector.size, data_vector.size)):
            self.data_inv_covariance = data_inv_covariance
        else:
            raise ValueError(
                "data_inv_covariance must be a square matrix with the same size as data_vector"
            )

    def compute_likelihood(self):
        model_vector = self._model_vector
        chi2 = compute_chi2(model_vector, self.data_vector, self.data_inv_covariance)
        return -0.5 * chi2
