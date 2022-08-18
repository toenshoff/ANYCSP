import subprocess
from src.csp import csp_data
import numpy as np
import torch


def get_random_RB(k, n):
    min_d = int(np.floor(n ** (1 / k))) + 1
    d = np.random.randint(min_d, 2 * min_d + 1)

    min_m = int(np.ceil(n * (np.log(d) / np.log(k))))
    m = np.random.randint(min_m, 2 * min_m + 1)

    a = np.log(d) / np.log(n)
    r = m/n/np.log(n)
    p = 1 - np.exp(-a/r)
    p *= 0.9

    negate = p < 0.5
    if not negate:
        p = 1.0 - p
    
    csp = csp_data.CSP_Data(n, torch.LongTensor([d] * n))
    cst_idx, var_idx, val_idx = sample_distinct_fast(n, k, d, m, p)
    csp.add_constraint_data_fixed_arity(negate, torch.LongTensor(cst_idx), torch.LongTensor(var_idx), torch.LongTensor(val_idx))
    return csp


def sample_distinct_fast(n, k, d, m, p):
    var_idx = np.vstack([np.random.choice(n, k, replace=False).reshape(1, -1) for _ in range(m)])
    cst_idx = np.arange(m)

    num_all_tup = int(np.power(d, k))
    mod = d ** np.arange(k, 0, -1).reshape(1, -1)
    div = d ** np.arange(k-1, -1, -1).reshape(1, -1)

    num_sampled_tup = np.random.binomial(num_all_tup, p, (m,))
    num_sampled_tup = np.maximum(num_sampled_tup, 1)
    var_idx = np.repeat(var_idx, num_sampled_tup, axis=0)
    cst_idx = np.repeat(cst_idx, num_sampled_tup, axis=0)

    tup = np.vstack([np.random.choice(num_all_tup, num_sampled_tup[i], replace=False).reshape(-1, 1) for i in range(m)])
    tup = tup % mod
    tup //= div
    val_idx = tup.reshape(-1, k)
    return cst_idx, var_idx, val_idx
