import numpy
import numpy as np
import torch


def linear_fct(coeff):
    return lambda x, y: x.reshape(-1, 1) + coeff * y.reshape(1, -1)


def comp_fct(b, comp):
    if comp == 'eq':
        return lambda x: np.equal(x, b)
    elif comp == 'ne':
        return lambda x: np.not_equal(x, b)
    elif comp == 'ge':
        return lambda x: np.greater_equal(x, b)
    elif comp == 'le':
        return lambda x: np.less_equal(x, b)
    elif comp == 'gt':
        return lambda x: np.greater(x, b)
    elif comp == 'lt':
        return lambda x: np.less(x, b)


def tuples_from_linear(coeffs, domain, b, comp, max_num_tuples=1000000):
    fct_list = [linear_fct(c) for c in coeffs]
    domains = [domain] * len(fct_list)
    fct_filter = comp_fct(b, comp)

    tuples, negate = generate_tuples(fct_list, domains, fct_filter, max_num_tuples)
    return tuples, negate


def generate_tuples(fct_list, domains, fct_filter, max_num_tuples):
    val = np.array([0])

    out_list = []
    val_list = [val]
    for f, dom in zip(fct_list, domains):
        out = f(val, dom)
        val = np.unique(out)

        out_list.append(out)
        val_list.append(val)

    mask = fct_filter(val)

    if mask.sum() <= (~mask).sum():
        negate = False
    else:
        negate = True
        mask = ~mask

    cur_targets = val[mask]

    tuples = None
    for i in range(len(domains)-1, -1, -1):
        val = val_list[i]
        out = out_list[i]
        dom = domains[i]

        edge_map = {t: np.where(out == t) for t in np.unique(cur_targets)}
        target_list, tuples_list = [], []
        for i, t in enumerate(cur_targets):
            val_idx, dom_idx = edge_map[t]
            expansion = val_idx.shape[0]
            target_list.append(val[val_idx])
            new_tuples = dom_idx.reshape(-1, 1)
            if tuples is not None:
                new_tuples = np.hstack([new_tuples, tuples[i].reshape(1, -1).repeat(expansion, axis=0)])
            tuples_list.append(new_tuples)

        cur_targets = np.concatenate(target_list)
        tuples = np.vstack(tuples_list)

        if tuples.shape[0] > max_num_tuples:
            raise ValueError('To many tuples!')

    return tuples, negate


def comp_edges_from_linear(coeffs, domain, b, comp):
    fct_list = [linear_fct(c) for c in coeffs]
    domains = [domain] * len(fct_list)
    fct_filter = comp_fct(b, comp)

    comp_edge_list, negate = get_comp_edge_list(fct_list, domains, fct_filter)
    return comp_edge_list, negate


def get_comp_edge_list(fct_list, domains, fct_filter):
    res = np.array([0])

    out_list = []
    res_list = [res]
    for f, dom in zip(fct_list, domains):
        out = f(res, dom)
        res = np.unique(out)

        out_idx_dict = {x: i for i, x in enumerate(res)}
        out_idx_map = np.vectorize(lambda x: out_idx_dict[x])
        out = out_idx_map(out)

        out_list.append(out)
        res_list.append(res)

    mask = fct_filter(res)
    if mask.sum() <= (~mask).sum():
        negate = False
    else:
        negate = True
        mask = ~mask

    pre_res_mask = mask
    comp_edge_list = []
    for i in range(len(domains)-1, -1, -1):
        num_cur_res = np.count_nonzero(pre_res_mask)
        cur_res_idx = np.where(pre_res_mask)[0]

        pre_res = res_list[i]
        out = out_list[i]

        edge_maps = [np.where(out == j) for j in cur_res_idx]
        max_num_edges = max([e[0].shape[0] for e in edge_maps])

        pre_res_mask = pre_res_mask[out].max(axis=1)
        res_reindex_dict = {j: k for k, j in enumerate(np.where(pre_res_mask)[0])}
        res_reindex = np.vectorize(lambda x: res_reindex_dict[x])

        res_idx = np.zeros((num_cur_res, max_num_edges), dtype=np.int64)
        dom_idx = np.zeros((num_cur_res, max_num_edges), dtype=np.int64)
        pad_mask = np.zeros((num_cur_res, max_num_edges), dtype=np.bool_)
        for i, (cur_res_edges, cur_dom_edges) in enumerate(edge_maps):
            cur_num_edges = cur_res_idx.shape[0]
            res_idx[i, :cur_num_edges] = res_reindex(cur_res_edges)
            dom_idx[i, :cur_num_edges] = cur_dom_edges
            pad_mask[i, :cur_num_edges] = True

        comp_edge_list = [(res_idx, dom_idx, pad_mask)] + comp_edge_list

    return comp_edge_list, negate


def sample_torch(val_prob, var_idx, var_off, comp_idx, comp_edge_list, num_samples):
    num_cst, arity = var_idx.shape
    offset = var_off[var_idx]

    sel_idx = torch.arange(num_cst, device=val_prob.device).view(-1, 1, 1)
    cur_res_prob = torch.ones((num_cst, 1), dtype=torch.float32, device=val_prob.device)
    edge_prob_list = []
    res_idx_list = []
    val_idx_list = []
    for i, (res_idx, dom_idx, mask) in enumerate(comp_edge_list):
        res_idx = res_idx[comp_idx]
        res_prob = cur_res_prob[sel_idx, res_idx]

        val_idx = offset[:, i].view(-1, 1, 1) + dom_idx[comp_idx]
        dom_prob = val_prob[val_idx]
        edge_prob = res_prob * dom_prob

        mask = (~mask)[comp_idx]
        edge_prob[mask] = 0.0
        cur_res_prob = edge_prob.sum(dim=2)
        cur_res_prob /= cur_res_prob.sum(dim=1).view(-1, 1) + 1.0e-6

        edge_prob_list.append(edge_prob)
        res_idx_list.append(res_idx)
        val_idx_list.append(val_idx)

    cur_res_node = torch.multinomial(cur_res_prob, num_samples, replacement=True).view(-1)

    sel_idx = torch.repeat_interleave(sel_idx, num_samples)
    sampled_val_idx = []
    for i in range(arity - 1, -1, -1):
        edge_prob = edge_prob_list[i]
        res_idx = res_idx_list[i]
        val_idx = val_idx_list[i]

        edge_prob = edge_prob[sel_idx, cur_res_node]
        sampled_edge = torch.multinomial(edge_prob, 1, replacement=True).view(-1)

        sampled_val_idx = [val_idx[sel_idx, cur_res_node, sampled_edge]] + sampled_val_idx
        cur_res_node = res_idx[sel_idx, cur_res_node, sampled_edge].view(-1)

    val_idx = torch.stack(sampled_val_idx, dim=1)
    return val_idx


if __name__ == '__main__':
    coeffs = np.array([-1, 1, 1, 1, 1])
    domain = np.array([0, 1])
    comp_edges, negate = comp_edges_from_linear(coeffs, domain, 0, 'ge')

    val_prob = 0.5 * torch.ones((10,), dtype=torch.float)
    var_idx = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    var_off = 2 * torch.arange(5)
    comp_idx = torch.zeros((2,), dtype=torch.int64)
    comp_edges = [
        (
            torch.tensor(r).unsqueeze(0),
            torch.tensor(d).unsqueeze(0),
            torch.tensor(m).unsqueeze(0)
        ) for r, d, m in comp_edges
    ]
    tuples = sample_torch(val_prob, var_idx, var_off, comp_idx, comp_edges, num_samples=4)
