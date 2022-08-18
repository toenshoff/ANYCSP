import torch
from torch.nn import Module, Linear, Sequential, LayerNorm, ReLU
from torch_scatter import scatter_max, scatter_mean, scatter_sum

try:
    from spmm_coo import spmm_coo_sum, spmm_coo_mean, spmm_coo_max
    SPMM_COO_AVAIL = True
    print('Using Spmm COO')
except ImportError:
    SPMM_COO_AVAIL = False


# general aggregation method with/without sparse_coo installation
def aggregate(msg, out_idx, in_idx, dim_size, aggr):
    if aggr == 'sum':
        if SPMM_COO_AVAIL:
            rec = spmm_coo_sum(msg, out_idx, in_idx, dim_size)
        else:
            rec = scatter_sum(msg[out_idx], in_idx, dim=0, dim_size=dim_size)
    elif aggr == 'mean':
        if SPMM_COO_AVAIL:
            rec = spmm_coo_mean(msg, out_idx, in_idx, dim_size)
        else:
            rec = scatter_mean(msg[out_idx], in_idx, dim=0, dim_size=dim_size)
    elif aggr == 'max':
        if SPMM_COO_AVAIL:
            rec = spmm_coo_max(msg, out_idx, in_idx, dim_size)[0]
        else:
            rec = scatter_max(msg[out_idx], in_idx, dim=0, dim_size=dim_size)[0]
    return rec


class Val2Cst_Layer(Module):

    def __init__(self, config):
        super(Val2Cst_Layer, self).__init__()
        self.hidden_dim = config['hidden_dim']
        self.aggr = config['aggr_val2cst']
        assert self.aggr in ['sum', 'mean', 'max']

        # E
        self.val_enc = Sequential(
            Linear(self.hidden_dim + 1, self.hidden_dim, bias=True),
            ReLU(),
            Linear(self.hidden_dim, self.hidden_dim, bias=False),
            LayerNorm(self.hidden_dim),
        )

        # M_V
        self.val_send = Sequential(
            Linear(self.hidden_dim, 2 * self.hidden_dim, bias=False),
            LayerNorm(2 * self.hidden_dim),
        )

    def forward(self, data, h_val, assign):
        # Encode value state
        x_val = torch.cat([h_val, assign.view(-1, 1).half()], dim=1)
        x_val = self.val_enc(x_val)

        # Values generate msg.
        m_val = self.val_send(x_val)
        m_val = m_val.view(2 * data.num_val, self.hidden_dim)

        # Edge index from edge labels
        out_idx = 2 * data.cst_edges[1] + data.LE
        in_idx = data.cst_edges[0]

        # Aggregate
        r_cst = aggregate(m_val, out_idx, in_idx, data.num_cst, self.aggr)
        return r_cst, x_val


class Cst2Val_Layer(Module):

    def __init__(self, config):
        super(Cst2Val_Layer, self).__init__()
        self.hidden_dim = config['hidden_dim']
        self.aggr = config['aggr_cst2val']
        assert self.aggr in ['sum', 'mean', 'max']

        # M_C
        self.cst_send = Sequential(
            Linear(self.hidden_dim, self.hidden_dim, bias=True),
            ReLU(),
            Linear(self.hidden_dim, 2 * self.hidden_dim, bias=False),
            LayerNorm(2 * self.hidden_dim),
        )

        # U_V
        self.val_rec = Sequential(
            Linear(self.hidden_dim, self.hidden_dim, bias=True),
            ReLU(),
            Linear(self.hidden_dim, self.hidden_dim, bias=False),
            LayerNorm(self.hidden_dim)
        )

    def forward(self, data, x_val, r_cst):
        # Constraints generate msg
        m_cst = self.cst_send(r_cst)
        m_cst = m_cst.view(2 * data.num_cst, self.hidden_dim)

        # Edge index from edge labels
        out_idx = 2 * data.cst_edges[0] + data.LE
        in_idx = data.cst_edges[1]

        # Aggregate and update
        r_val = aggregate(m_cst, out_idx, in_idx, data.num_val, self.aggr)
        x_val = self.val_rec(x_val + r_val) + x_val
        return x_val


class Val2Val_Layer(Module):

    def __init__(self, config):
        super(Val2Val_Layer, self).__init__()
        self.hidden_dim = config['hidden_dim']
        self.aggr = config['aggr_val2var']

        assert self.aggr in ['sum', 'mean', 'max']

        # U_X
        self.var_enc = Sequential(
            Linear(self.hidden_dim, self.hidden_dim, bias=True),
            ReLU(),
            Linear(self.hidden_dim, self.hidden_dim, bias=False),
            LayerNorm(self.hidden_dim),
        )

    def forward(self, data, y_val):
        # Pool value states of the same variable
        if self.aggr == 'max':
            z_var = scatter_max(y_val, data.var_idx, dim=0, dim_size=data.num_var)[0]
        elif self.aggr == 'mean':
            z_var = scatter_mean(y_val, data.var_idx, dim=0, dim_size=data.num_var)
        else:
            z_var = scatter_sum(y_val, data.var_idx, dim=0, dim_size=data.num_var)

        # Apply U_X and send result back
        z_var = self.var_enc(z_var)
        y_val += z_var[data.var_idx]
        return y_val


class Policy(Module):

    def __init__(self, config):
        super(Policy, self).__init__()
        self.hidden_dim = config['hidden_dim']

        # O
        self.mlp = Sequential(
            LayerNorm(self.hidden_dim),
            Linear(self.hidden_dim, self.hidden_dim, bias=True),
            ReLU(),
            Linear(self.hidden_dim, 1, bias=False),
        )

    def forward(self, h_val):
        logits = self.mlp(h_val)
        return logits
