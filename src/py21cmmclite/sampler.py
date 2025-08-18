import emcee
from .likelihood import LikelihoodBase
import py21cmfast as p21
import numpy as np
from multiprocessing import Pool
from .coeval import EoRSimulator
from py21cmfast.io.caching import OutputCache
import os
import nautilus
from scipy.stats import norm

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
        clear_cache: bool = True,
        save: bool = False,
        save_filename: str | None = None,
    ):
        self.params_name = params_name
        self.init_pos = init_pos
        self.params_prior = params_prior
        if not len(params_name) == len(init_pos) == len(params_prior):
            raise ValueError(
                "params_names, init_pos and params_prior must have the same length"
            )
        self.likelihood = likelihood
        self.clear_cache = clear_cache
        self.save = save
        self.save_filename = save_filename
        self.validate_input()

    @property
    def is_varying_astro(self):
        return any(param in astro_pars_21cmfast for param in self.params_name)

    @property
    def is_varying_cosmo(self):
        return any(param in cosmo_pars_21cmfast for param in self.params_name)

    def validate_input(self):
        if self.save and self.save_filename is None:
            raise ValueError("save_filename must be provided if save is True")
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
        for prior in self.params_prior:
            if prior[0] not in ["uniform", "gaussian"]:
                raise ValueError(f"Unsupported prior type: {prior[0]}")

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

    def find_21cmfast_cache_files(self, params_values, only_astro: bool = False):
        params_values = np.array(params_values)
        datasets = []
        for likelihood in self.likelihood:
            if isinstance(likelihood, EoRSimulator):
                flag = self.find_subset_for_likelihood(likelihood)
                inputs = likelihood.get_update_input(
                    likelihood.get_update_dict(params_values[flag])
                )
                cache = OutputCache(direc=self.cache_dir)
                dataset_i = cache.list_datasets(inputs=inputs, all_seeds=False)
                if only_astro:
                    # change random params to change hash code
                    # may be wrong
                    inputs_alt = inputs.evolve_input_structs(
                        F_ESC10=inputs.astro_params.F_ESC10 + 1,
                    )
                    dataset_i_alt = cache.list_datasets(
                        inputs=inputs_alt, all_seeds=False
                    )
                    for dat in dataset_i_alt:
                        if dat in dataset_i:
                            dataset_i.remove(dat)
                datasets.extend(dataset_i)
        return datasets

    def clear_cache_for_one_params_set(self, params_values, only_astro: bool = False):
        # if cosmology is fixed, only clear astro cache
        datasets = self.find_21cmfast_cache_files(params_values, only_astro=only_astro)
        for file in datasets:
            os.remove(file)

    def clear_empty_cache_subdir(self):
        deleted = set()
        for current_dir, subdirs, files in os.walk(self.cache_dir, topdown=False):
            still_has_subdirs = False
            for subdir in subdirs:
                if os.path.join(current_dir, subdir) not in deleted:
                    still_has_subdirs = True
                    break
            if not any(files) and not still_has_subdirs:
                os.rmdir(current_dir)
                deleted.add(current_dir)
        return deleted


class SamplerNautilus(SamplerBase):
    """
    Sampler using Nautilus.
    """

    def __init__(
        self,
        params_name: list[str],
        init_pos: list[float] | np.ndarray,
        params_prior: list[tuple[str, float, float]],
        likelihood: list[LikelihoodBase],
        clear_cache: bool = True,
        n_live_points: int = 2000,
        f_live: float = 0.01,
        n_shell: int = 1,
        n_eff: int = 10000,
        save: bool = False,
        save_filename: str | None = None,
        mp_backend: str = "multiprocessing",
        nthreads: int = 1,
    ):
        self.n_live_points = n_live_points
        self.f_live = f_live
        self.n_shell = n_shell
        self.n_eff = n_eff
        self.mp_backend = mp_backend
        self.nthreads = nthreads
        super().__init__(
            params_name,
            init_pos,
            params_prior,
            likelihood,
            clear_cache,
            save,
            save_filename,
        )

    def get_nautilus_prior(self):
        prior = nautilus.Prior()
        for i, param_name in enumerate(self.params_name):
            if self.params_prior[i][0] == "uniform":
                dist = (self.params_prior[i][1], self.params_prior[i][2])
            elif self.params_prior[i][0] == "gaussian":
                dist = norm(loc=self.params_prior[i][1], scale=self.params_prior[i][2])
            prior.add_parameter(param_name, dist=dist)
        return prior

    def log_likelihood(self, params_values):
        log_likelihood = 0.0
        blob = {}
        for likelihood in self.likelihood:
            pars_value_i = params_values[self.find_subset_for_likelihood(likelihood)]
            log_likelihood_i, blob_i = likelihood.compute_likelihood(pars_value_i)
            log_likelihood += log_likelihood_i
            blob.update(blob_i)
        # clear cache after computing likelihood
        if self.clear_cache:
            # if cosmology is fixed, only clear astro cache
            self.clear_cache_for_one_params_set(
                params_values,
                only_astro=not self.is_varying_cosmo,
            )
        return log_likelihood, blob

    def compute_log_likelihood(self, params_values):
        ll, blob = self.log_likelihood(params_values)
        blob_array = [arr.ravel() for arr in blob.values()]
        blob_array = np.concatenate(blob_array)
        return ll, blob_array

    def run(self, continue_from_last: bool = True, verbose: bool = True):
        pool = None
        filepath = None
        resume = continue_from_last
        if self.mp_backend == "multiprocessing":
            pool = self.nthreads
        if self.save:
            filepath = self.save_filename
            file_exist = os.path.isfile(filepath)
        if not self.save and continue_from_last:
            raise UserWarning(
                "continue_from_last is True but save is False, "
                "so the sampler will not continue from the last position"
            )
            resume = False
        elif self.save and not file_exist and continue_from_last:
            raise UserWarning(
                "continue_from_last is True but the file does not exist, "
                "so the sampler will not continue from the last position"
            )
            resume = False
        sampler = nautilus.Sampler(
            self.get_nautilus_prior(),
            self.compute_log_likelihood,
            pass_dict=False,
            pool=pool,
            n_live=self.n_live_points,
            resume=resume,
            filepath=filepath,
        )
        sampler.run(
            f_live=self.f_live,
            n_shell=self.n_shell,
            n_eff=self.n_eff,
            discard_exploration=True,
            verbose=verbose,
        )
        return sampler

    def get_posterior(self):
        if not self.save:
            raise UserWarning("save is False, so the chain cannot be retrieved")
        sampler = nautilus.Sampler(
            self.get_nautilus_prior(),
            self.compute_log_likelihood,
            pass_dict=False,
            filepath=self.save_filename,
        )
        points, log_w, log_l, blob = sampler.posterior(return_blobs=True)
        return points, log_w, log_l, blob


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
        clear_cache: bool = True,
    ):
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.nthreads = nthreads
        self.mp_backend = mp_backend
        self._blob_dtype = None
        super().__init__(
            params_name,
            init_pos,
            params_prior,
            likelihood,
            clear_cache,
            save,
            save_filename,
        )

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
        # clear cache after computing likelihood
        if self.clear_cache:
            # if cosmology is fixed, only clear astro cache
            self.clear_cache_for_one_params_set(
                params_values,
                only_astro=not self.is_varying_cosmo,
            )
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
