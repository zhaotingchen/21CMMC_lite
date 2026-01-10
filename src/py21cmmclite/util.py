from cobaya.model import get_model
from cobaya.input import get_default_info
import numpy as np
from nautilus import Prior

def get_cobaya_sampled_params(model):
    """
    Get the list of sampled parameters from the model.
    """
    input_params = model.parameterization.sampled_params().keys()
    return input_params

def get_cobaya_nuisance_reference_values(model):
    """
    Retrieve the reference values of the nuisance parameters from the model.
    """
    likelihood_names = list(model.likelihood.keys())
    likelihood_params = {}
    for like_name in likelihood_names[:]:
        # Get default info
        default_info = get_default_info(like_name, kind='likelihood')
        params = default_info.get('params', {})
        
        # Separate sampled (nuisance) vs derived/fixed
        sampled = {}
        fixed = {}
        
        for pname, pinfo in params.items():
            if isinstance(pinfo, dict):
                if 'value' in pinfo:
                    fixed[pname] = pinfo
                elif 'prior' in pinfo:
                    sampled[pname] = pinfo
        
        likelihood_params[like_name] = {
            'sampled': sampled,
            'fixed': fixed
        }
    init_params_dict = {}
    for like in likelihood_params:
        data = likelihood_params[like]
        for key in data['fixed']:
            init_params_dict[key] = data['fixed'][key]['value']
        for key in data['sampled']:
            init_params_dict[key] = data['sampled'][key]['ref']['loc']
    return init_params_dict


def get_nautilus_prior_for_cobaya(model):
    """
    Get the nautilus prior for the model.
    """
    prior_dict = dict(zip(model.prior.params, model.prior.pdf))
    prior = Prior()
    for key, item in prior_dict.items():
        prior.add_parameter(key, dist = item)
    return prior