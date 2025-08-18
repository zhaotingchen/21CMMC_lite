import emcee
from .likelihood import LikelihoodBase
import py21cmfast as p21
import numpy as np
from multiprocessing import Pool
from .coeval import EoRSimulator
from py21cmfast.io.caching import OutputCache

inputs_21cmfast = p21.InputParameters.from_template(
    "simple-small",
    random_seed=1234,
)
astro_pars_21cmfast = list(inputs_21cmfast.astro_params.asdict().keys())
cosmo_pars_21cmfast = list(inputs_21cmfast.cosmo_params.asdict().keys())


class SamplerBase:
    """
    Base class for all samplers.
    """

    def __init__(
        self,
        params_name: list[str],
        init_pos: list[float] | np.ndarray,
        params_prior: list[tuple[str, float, float]],
        likelihood: list[LikelihoodBase],
    ):
        self.params_name = params_name
        self.init_pos = init_pos
        self.params_prior = params_prior
        if not len(params_name) == len(init_pos) == len(params_prior):
            raise ValueError(
                "params_names, init_pos and params_prior must have the same length"
            )
        self.likelihood = likelihood

    @property
    def is_varying_astro(self):
        return any(param in astro_pars_21cmfast for param in self.params_name)

    @property
    def is_varying_cosmo(self):
        return any(param in cosmo_pars_21cmfast for param in self.params_name)

    def validate_input(self):
        cache_dir = None
        for likelihood in self.likelihood:
            flag = [param in self.params_name for param in likelihood.varied_params]
            if not all(flag):
                raise ValueError(
                    f"varied_params of likelihood {likelihood} must be a subset of params_name"
                )
            if hasattr(likelihood, "cache_dir"):
                if cache_dir is None:
                    cache_dir = likelihood.cache_dir
                else:
                    if cache_dir != likelihood.cache_dir:
                        raise ValueError("cache_dir of likelihoods must be the same")
        self.cache_dir = cache_dir

    @property
    def ndim(self):
        return len(self.params_name)

    def find_subset_for_likelihood(self, likelihood):
        flag = [
            # this is to make sure the parameters are passed in with the correct order
            np.where(np.array(self.params_name) == param)[0][0]
            for param in likelihood.varied_params
        ]
        return flag

    def find_21cmfast_cache_files(self, params_values):
        params_values = np.array(params_values)
        datasets = []
        for likelihood in self.likelihood:
            if isinstance(likelihood, EoRSimulator):
                flag = self.find_subset_for_likelihood(likelihood)
                inputs = likelihood.get_update_input(
                    likelihood.get_update_dict(params_values[flag])
                )
                cache = OutputCache(direc=self.cache_dir)
                datasets.extend(cache.list_datasets(inputs=inputs, all_seeds=False))
        return datasets


class SamplerEmcee(SamplerBase):
    """
    Sampler using emcee.
    """

    def __init__(
        self,
        params_name: list[str],
        init_pos: list[float] | np.ndarray,
        params_prior: list[tuple[str, float, float]],
        likelihood: list[LikelihoodBase],
        nwalkers: int,
        nsteps: int,
        nthreads: int,
        mp_backend: str = "multiprocessing",
        save: bool = False,
        save_filename: str | None = None,
    ):
        super().__init__(params_name, init_pos, params_prior, likelihood)
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.nthreads = nthreads
        self.mp_backend = mp_backend
        self.validate_input()
        self._blob_dtype = None
        self.save = save
        self.save_filename = save_filename
        if self.save and self.save_filename is None:
            raise ValueError("save_filename must be provided if save is True")

    def log_prior_gaussian(self, value, mean, sigma):
        return -0.5 * (value - mean) ** 2 / sigma**2

    def log_prior_uniform(self, value, low, high):
        if value < low or value > high:
            return -np.inf
        else:
            return 0.0

    def log_prior(self, params_values):
        log_prior = 0.0
        for i, param_name in enumerate(self.params_name):
            prior_func = getattr(self, f"log_prior_{self.params_prior[i][0]}")
            log_prior_i = prior_func(
                params_values[i],
                self.params_prior[i][1],
                self.params_prior[i][2],
            )
            log_prior += log_prior_i
        return log_prior

    def log_likelihood(self, params_values):
        log_likelihood = 0.0
        blob = {}
        for likelihood in self.likelihood:
            pars_value_i = params_values[self.find_subset_for_likelihood(likelihood)]
            log_likelihood_i, blob_i = likelihood.compute_likelihood(pars_value_i)
            log_likelihood += log_likelihood_i
            blob.update(blob_i)
        return log_likelihood, blob

    def convert_blob_dict_to_array(self, blob_dict):
        blob_type = np.dtype(
            [
                (key, blob_dict[key].dtype, blob_dict[key].shape)
                for key in blob_dict.keys()
            ]
        )
        blob_array = np.zeros(1, dtype=blob_type)
        for key in blob_dict.keys():
            blob_array[key] = blob_dict[key]
        return blob_array

    def compute_at_init_pos(self):
        ll, blob_dict = self.log_likelihood(self.init_pos)
        blob_array = self.convert_blob_dict_to_array(blob_dict)
        self._blob_dtype = blob_array.dtype
        return ll, blob_array

    @property
    def blob_size(self):
        return np.sum(
            [np.prod(self._blob_dtype[i].shape) for i in range(len(self._blob_dtype))]
        )

    def compute_log_likelihood(self, params_values):
        if self._blob_dtype is None:
            raise ValueError(
                "blob_dtype is not set, either set it manually "
                "or run the sampler with the initial position by "
                "self.compute_at_init_pos()"
            )
        lp = self.log_prior(params_values)
        # no need to compute if already out of prior range
        if not np.isfinite(lp):
            return lp, np.zeros(self.blob_size) + np.nan
        ll, blob = self.log_likelihood(params_values)
        blob_array = [arr.ravel() for arr in blob.values()]
        blob_array = np.concatenate(blob_array)
        return ll + lp, blob_array

    def run(self, continue_from_last: bool = True, progress: bool = True):
        if self._blob_dtype is None:
            raise ValueError(
                "blob_dtype is not set, either set it manually "
                "or run the sampler with the initial position by "
                "self.compute_at_init_pos()"
            )
        if continue_from_last and not self.save:
            raise UserWarning(
                "continue_from_last is True but save is False, "
                "so the sampler will not continue from the last position"
            )
        if continue_from_last and self.save:
            start_coord = None
        else:
            start_coord = (
                1 + np.random.uniform(-1e-2, 1e-2, size=(self.nwalkers, self.ndim))
            ) * self.init_pos[None, :]
        nsteps = self.nsteps
        if self.save:
            backend = emcee.backends.HDFBackend(self.save_filename)
            if not continue_from_last:
                backend.reset(self.nwalkers, self.ndim)
            nsteps = nsteps - backend.iteration
        else:
            backend = None
        if self.mp_backend == "multiprocessing":
            pool = Pool(self.nthreads)
        sampler = emcee.EnsembleSampler(
            self.nwalkers,
            self.ndim,
            self.compute_log_likelihood,
            backend=backend,
            pool=pool,
        )
        sampler.run_mcmc(start_coord, nsteps, progress=progress)
        return sampler

    def get_chain(self):
        if not self.save:
            raise UserWarning("save is False, so the chain cannot be retrieved")
        backend = emcee.backends.HDFBackend(self.save_filename, read_only=True)
        return backend.get_chain()

    def get_blobs(self, return_structured_dict: bool = True):
        if not self.save:
            raise UserWarning("save is False, so the blobs cannot be retrieved")
        backend = emcee.backends.HDFBackend(self.save_filename, read_only=True)
        blobs = backend.get_blobs()
        if return_structured_dict:
            blob_dict = {}
            index = 0
            for i in range(len(self._blob_dtype)):
                dtype = self._blob_dtype[i]
                blob_dict[self._blob_dtype.names[i]] = blobs[
                    :, :, index : index + np.prod(dtype.shape)
                ].reshape((backend.iteration, self.nwalkers) + dtype.shape)
            return blob_dict
        else:
            return blobs
