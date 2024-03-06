import torch
import numpy as np

const_bnn_prior_parameters = {
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": -3.0,
            "type": "Reparameterization",  # Flipout or Reparameterization
            "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
            "moped_delta": 0.5,
        }

def exact_match(truth, pred):
    len_truth = len(truth)
    len_pred = len(pred)
    max_len = max(len_truth, len_pred)
    a = truth + [""] * (max_len - len_truth)
    b = pred + [""] * (max_len - len_pred)
    em = np.mean(np.array(a) == np.array(b))
    return torch.tensor(em)
