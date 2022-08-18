import numpy as np
import torch
from torch_geometric.utils import degree
from torch_scatter import scatter_softmax, scatter_sum, scatter_max

from src.csp.constraints import Constraint_Data, Constraint_Data_All_Diff, Constraint_Data_Linear


# Class for representing CSP instances for torch
class CSP_Data:

    def __init__(self, num_var, domain_size, domain=None, batch=None, path=None):
        """
        @param num_var: Number of variables
        @param domain_size: Python Integer or Long-Tensor specifying the domain size of each variable
        @param domain: Long-Tensor specifying the values in each domain. Only needed for numerical domains with intension constraints.
        @param batch: Long-Tensor containing the index of the batch instance each variable belongs to.
        @param path: Path to the file the instance was parsed from (optional)
        """
        self.path = path
        self.num_var = num_var
        self.num_cst = 0
        self.num_tup_sampled = 0

        if isinstance(domain_size, torch.Tensor):
            self.domain_size = domain_size
        else:
            self.domain_size = domain_size * torch.ones((num_var,), dtype=torch.int64)

        self.num_val = int(self.domain_size.sum().numpy())
        self.max_dom = int(self.domain_size.max().numpy())

        self.var_idx = torch.repeat_interleave(torch.arange(0, num_var, dtype=torch.int64), self.domain_size)
        self.var_off = torch.cat([torch.tensor([0]), torch.cumsum(self.domain_size, dim=0)[:-1]], dim=0)
        self.dom_idx = torch.arange(self.num_val) - self.var_off[self.var_idx]

        if domain is not None:
            self.domain = domain
        else:
            self.domain = torch.arange(self.num_val) - self.var_off[self.var_idx]

        self.var_reg = 1.0 / (self.domain_size + 1.0e-8).view(-1, 1)
        self.cst_reg = None
        self.val_reg = None

        self.batch = torch.zeros((num_var,), dtype=torch.int64) if batch is None else batch
        self.batch_size = int(self.batch.max().numpy()) + 1
        self.batch_num_cst = torch.zeros((self.batch_size,), dtype=torch.float32)

        self.batch_num_val = scatter_sum(self.domain_size, self.batch, dim=0)
        self.batch_val_off = torch.zeros((self.batch_size,), dtype=torch.long)
        self.batch_val_off[1:] = torch.cumsum(self.batch_num_val[:-1], dim=0)
        self.batch_val_idx = torch.arange(self.num_val)
        self.batch_val_idx -= self.batch_val_off[self.batch[self.var_idx]]
        self.max_num_val = self.batch_num_val.max().numpy()

        self.num_edges = 0
        self.constraints = {}
        self.cst_batch = None
        self.cst_edges = None
        self.LE = None

        self.initialized = False
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.var_idx = self.var_idx.to(device)
        self.var_reg = self.var_reg.to(device)
        self.var_off = self.var_off.to(device)
        self.dom_idx = self.dom_idx.to(device)
        self.domain = self.domain.to(device)
        self.domain_size = self.domain_size.to(device)
        self.batch = self.batch.to(device)
        self.batch_val_off = self.batch_val_off.to(device)
        self.batch_val_idx = self.batch_val_idx.to(device)
        self.batch_num_val = self.batch_num_val.to(device)
        for cst_data in self.constraints.values():
            cst_data.to(device)

    @staticmethod
    def collate(batch):
        num_var = [d.num_var for d in batch]
        var_off = np.concatenate([[0], np.cumsum(num_var)[:-1]], axis=0)
        num_val = [d.num_val for d in batch]
        val_off = np.concatenate([[0], np.cumsum(num_val)[:-1]], axis=0)

        num_var = sum(num_var)

        domain_size = torch.cat([d.domain_size for d in batch])
        domain = torch.cat([d.domain for d in batch])
        batch_idx = torch.cat([d.batch + i for i, d in enumerate(batch)])
        batch_data = CSP_Data(num_var, domain_size, domain=domain, batch=batch_idx)

        cst_batch_dict = {}
        for i, data in enumerate(batch):
            for key, cst_data in data.constraints.items():
                batch_item = (cst_data, var_off[i], val_off[i], i)
                if key in cst_batch_dict:
                    cst_batch_dict[key].append(batch_item)
                else:
                    cst_batch_dict[key] = [batch_item]

        for key, batch_list in cst_batch_dict.items():
            const_data = batch_list[0][0].collate(batch_list, batch_data)
            batch_data.add_constraint_data_(const_data, key)

        return batch_data

    def init_adj(self):
        cst_edges, cst_batch = [], []
        cst_off = 0
        for cst_data in self.constraints.values():
            cur_edges = cst_data.cst_edges.clone()
            cur_edges[0] += cst_off
            cst_edges.append(cur_edges)

            cst_batch.append(cst_data.batch)
            self.num_edges += cst_data.num_edges
            cst_off += cst_data.num_cst

        self.cst_edges = torch.cat(cst_edges, dim=1)
        self.cst_batch = torch.cat(cst_batch, dim=0)
        self.batch_num_cst = degree(self.cst_batch, num_nodes=self.batch_size, dtype=torch.int64)

    def update_LE(self):
        self.LE = torch.cat([cst_data.LE for cst_data in self.constraints.values()], dim=0).flatten().long()

    def add_constraint_data_(self, cst_data, name):
        self.num_cst += cst_data.num_cst
        if name not in self.constraints:
            self.constraints[name] = cst_data
        else:
            cst_old = self.constraints[name]
            cst_data = cst_data.collate([(cst_old, 0, 0, 0), (cst_data, 0, 0, 0)], self)
            self.constraints[name] = cst_data

    def add_constraint_data(self, negate, cst_idx, tup_idx, var_idx, val_idx, cst_type=None):
        num_cst = cst_idx.max().numpy() + 1
        if cst_type is None:
            cst_type = torch.ones((num_cst,), dtype=torch.int64) if negate else torch.zeros((num_cst,), dtype=torch.int64)

        var_idx = var_idx.view(-1)
        val_idx = val_idx.view(-1)
        val_node_idx = self.var_off[var_idx] + val_idx

        cst_data = Constraint_Data(
            csp_data=self,
            cst_idx=cst_idx,
            tup_idx=tup_idx,
            val_idx=val_node_idx,
            cst_type=cst_type
        )

        self.add_constraint_data_(cst_data, 'ext')

    def add_uniform_constraint_data(self, negate, var_idx, val_idx):
        num_cst, num_tup, arity = val_idx.shape
        cst_idx = torch.repeat_interleave(torch.arange(num_cst), num_tup, dim=0)
        self.add_constraint_data_fixed_arity(
            negate,
            cst_idx,
            var_idx.view(-1, arity),
            val_idx.view(-1, arity),
        )

    def add_constraint_data_fixed_arity(self, negate, cst_idx, var_idx, val_idx):
        num_tup, arity = val_idx.shape
        num_cst = cst_idx.max() + 1
        tup_idx = torch.repeat_interleave(torch.arange(num_tup), arity)
        val_idx = self.var_off[var_idx] + val_idx
        val_idx = val_idx.flatten()
        cst_type = torch.ones((num_cst,), dtype=torch.int64) if negate else torch.zeros((num_cst,), dtype=torch.int64)

        cst_data = Constraint_Data(
            csp_data=self,
            cst_idx=cst_idx,
            tup_idx=tup_idx,
            val_idx=val_idx,
            cst_type=cst_type,
        )
        self.add_constraint_data_(cst_data, f'{arity}_sampled')

    def add_all_different_constraint_data(self, var_idx):
        num_cst, num_var = var_idx.shape
        cst_idx = torch.repeat_interleave(torch.arange(num_cst), num_var, dim=0)
        var_idx = var_idx.flatten()

        cst_data = Constraint_Data_All_Diff(
            csp_data=self,
            cst_idx=cst_idx,
            var_idx=var_idx
        )
        self.add_constraint_data_(cst_data, f'all_diff')

    def add_linear_constraint_data(self, var_idx, coeffs, b, comp):
        cst_idx = torch.arange(var_idx.shape[0])
        cst_idx = torch.repeat_interleave(cst_idx, var_idx.shape[1])

        #comp_op = torch.zeros(var_idx.shape[0], dtype=torch.int64)
        #comp_op += Constraint_Data_Linear.comp_idx(comp)

        comp_op = torch.tensor([Constraint_Data_Linear.comp_idx(c) for c in comp])

        cst_data = Constraint_Data_Linear(self, cst_idx, var_idx.flatten(), coeffs.flatten(), b.flatten(), comp_op)
        self.add_constraint_data_(cst_data, f'linear')

    def value_softmax(self, value_logits):
        with torch.cuda.amp.autocast(enabled=False):
            value_logits = value_logits.view(self.num_val, -1)
            value_prob = scatter_softmax(value_logits.float(), self.var_idx, dim=0)
        return value_prob

    def value_softmax_local(self, value_logits, cur_assignment):
        with torch.cuda.amp.autocast(enabled=False):
            value_logits = value_logits.float().view(self.num_val, -1)
            value_logits -= 10000.0 * cur_assignment.view(self.num_val, -1)
            value_prob = scatter_softmax(value_logits.float(), self.batch[self.var_idx], dim=0)
        return value_prob

    def round_to_one_hot(self, value_prob):
        value_prob = value_prob.view(self.num_val, -1)
        max_idx = scatter_max(value_prob, self.var_idx, dim=0)[1]
        step_idx = torch.arange(value_prob.shape[1], device=value_prob.device).view(1, -1)
        one_hot = torch.zeros_like(value_prob)
        one_hot[max_idx, step_idx] = 1.0
        return one_hot

    def hard_assign_max(self, value_prob):
        value_prob = value_prob.view(self.num_val, -1)
        value_idx = scatter_max(value_prob, self.var_idx, dim=0)[1]
        value_idx -= self.var_off.view(-1, 1)
        return value_idx

    def hard_assign_sample(self, logits):
        value_prob = self.value_softmax(logits).view(self.num_val, 1)
        with torch.no_grad():
            dense_probs = torch.zeros((self.num_var, self.max_dom), dtype=torch.float32, device=self.device)
            dense_probs[self.var_idx, self.dom_idx] = value_prob.view(-1)

            idx = torch.multinomial(dense_probs, 1)
            idx += self.var_off.view(-1, 1)

            assignment = torch.zeros((self.num_val, 1), dtype=torch.float32, device=self.device)
            assignment[idx.view(-1)] = 1.0

        sampled_prob = value_prob[idx]
        log_prob = scatter_sum(torch.log(sampled_prob + 1.0e-5), self.batch, dim=0).view(-1, 1)
        return assignment, log_prob

    def hard_assign_sample_local(self, logits, assignment):
        value_prob = self.value_softmax_local(logits, assignment).view(self.num_val, 1)
        with torch.no_grad():
            value_assignment = self.dom_idx[assignment.bool().flatten()]

            dense_probs = torch.zeros((self.batch_size, self.max_num_val), dtype=torch.float32, device=self.device)
            dense_probs[self.batch[self.var_idx], self.batch_val_idx] = value_prob.view(-1)

            idx = torch.multinomial(dense_probs, 1).flatten()
            idx += self.batch_val_off

            value_assignment[self.var_idx[idx]] = self.dom_idx[idx]
            assignment = torch.zeros((self.num_val, 1), dtype=torch.float32, device=self.device)
            assignment[self.var_off + value_assignment] = 1.0

        log_prob = torch.log(value_prob[idx] + 1.0e-5).view(-1, 1)
        return assignment, log_prob

    def constraint_is_sat(self, assignment_one_hot, update_LE=False):
        assignment_one_hot = assignment_one_hot.view(self.num_val, -1)
        sat = torch.cat([c.is_sat(assignment_one_hot, update_LE) for k, c in self.constraints.items()], dim=0)
        if update_LE:
            self.update_LE()
        return sat

    def count_unsat(self, assignment_one_hot):
        unsat = 1.0 - self.constraint_is_sat(assignment_one_hot).float()
        num_unsat = scatter_sum(unsat, self.cst_batch, dim=0, dim_size=self.batch_size)
        return num_unsat

    def count_sat(self, assignment_one_hot):
        sat = self.constraint_is_sat(assignment_one_hot).float()
        sat = scatter_sum(sat, self.cst_batch, dim=0, dim_size=self.batch_size)
        return sat
