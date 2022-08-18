import torch
from torch_geometric.utils import degree
from torch_scatter import scatter_sum, scatter_min, scatter_max, segment_max_csr


class Constraint_Data_Base:

    def __init__(self, csp_data, cst_edges, cst_var_edges, batch=None):
        self.cst_edges = cst_edges
        self.cst_var_edges = cst_var_edges

        self.LE = None
        self.num_cst = cst_edges[0].max().numpy() + 1
        self.num_edges = cst_edges.shape[1]

        self.cst_deg = degree(cst_edges[0], dtype=torch.int64)
        self.cst_arity = degree(cst_var_edges[0], dtype=torch.int64)
        self.val_deg = degree(cst_edges[1], dtype=torch.int64, num_nodes=csp_data.num_val)

        self.batch = batch if batch is not None else torch.zeros((self.num_cst,), dtype=torch.int64)
        self.batch_size = int(self.batch.max().numpy())

        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.cst_edges = self.cst_edges.to(device)
        self.cst_var_edges = self.cst_var_edges.to(device)
        self.batch = self.batch.to(device)
        self.cst_deg = self.cst_deg.to(device)
        self.cst_arity = self.cst_arity.to(device)
        self.val_deg = self.val_deg.to(device)

    def update_LE_(self, **kwargs):
        raise NotImplementedError

    def is_sat(self, assignment):
        raise NotImplementedError


class Constraint_Data(Constraint_Data_Base):

    def __init__(self, csp_data, cst_idx, tup_idx, val_idx, cst_type, cst_edges=None, cst_var_edges=None, lit_edge_map=None, batch=None):
        self.cst_idx = cst_idx
        self.tup_idx = tup_idx
        self.val_idx = val_idx
        self.cst_type = cst_type

        if cst_var_edges is None:
            cst_var_edges = torch.unique(torch.stack([cst_idx[tup_idx], csp_data.var_idx[self.val_idx]], dim=0), dim=1)
        if cst_edges is None or lit_edge_map is None:
            cst_var_dom_size = csp_data.domain_size[cst_var_edges[1]]
            cst_edges = torch.repeat_interleave(cst_var_edges, cst_var_dom_size, dim=1)
            cst_edges[1] = csp_data.var_off[cst_edges[1]]

            cst_val_off = torch.zeros_like(cst_var_dom_size)
            cst_val_off[1:] += torch.cumsum(cst_var_dom_size[:-1], dim=0)
            cst_val_shift = torch.arange(cst_edges.shape[1])
            cst_val_shift -= torch.repeat_interleave(cst_val_off, cst_var_dom_size, dim=0)
            cst_edges[1] += cst_val_shift

            cst_edges = torch.cat([
                cst_edges,
                torch.stack([cst_idx[tup_idx], self.val_idx], dim=0)
            ], dim=1)
            cst_edges, lit_edge_map = torch.unique(cst_edges, dim=1, return_inverse=True)
            lit_edge_map = lit_edge_map[cst_edges.shape[1]:].contiguous()

        super(Constraint_Data, self).__init__(
                csp_data=csp_data,
                cst_edges=cst_edges,
                cst_var_edges=cst_var_edges,
                batch=batch,
        )

        self.num_tup = tup_idx.max().numpy() + 1
        self.tup_deg = degree(tup_idx, num_nodes=self.num_tup, dtype=torch.uint8).view(-1, 1)
        self.cst_num_tup = degree(cst_idx, num_nodes=self.num_cst, dtype=torch.float32).view(-1, 1)

        self.lit_edge_map = lit_edge_map
        self.cst_neg_mask = self.cst_type.bool()
        self.neg_edge_mask = self.cst_neg_mask[self.cst_edges[0]]
        self.edge_comp_thresh = (self.cst_arity.int()-1)[self.cst_edges[0]].view(-1, 1)

    def to(self, device):
        super(Constraint_Data, self).to(device)
        self.cst_idx = self.cst_idx.to(device)
        self.val_idx = self.val_idx.to(device)
        self.tup_idx = self.tup_idx.to(device)
        self.tup_deg = self.tup_deg.to(device)
        self.cst_num_tup = self.cst_num_tup.to(device)
        self.lit_edge_map = self.lit_edge_map.to(device)
        self.cst_type = self.cst_type.to(device)
        self.cst_neg_mask = self.cst_neg_mask.to(device)
        self.neg_edge_mask = self.neg_edge_mask.to(device)
        self.edge_comp_thresh = self.edge_comp_thresh.to(device)

    @staticmethod
    def collate(batch_list, merged_csp_data):
        cst_idx, tup_idx, val_idx, cst_type, batch_idx = [], [], [], [], []
        cst_off, tup_off, edge_off = 0, 0, 0
        cst_val_edges, cst_var_edges, lit_edge_map = [], [], []
        for cst_data, var_off, val_off, i in batch_list:
            cst_idx.append(cst_data.cst_idx + cst_off)
            tup_idx.append(cst_data.tup_idx + tup_off)
            val_idx.append(cst_data.val_idx + val_off)

            cur_cst_val_edges = cst_data.cst_edges.clone()
            cur_cst_val_edges[0] += cst_off
            cur_cst_val_edges[1] += val_off
            cst_val_edges.append(cur_cst_val_edges)

            cur_cst_var_edges = cst_data.cst_var_edges.clone()
            cur_cst_var_edges[0] += cst_off
            cur_cst_var_edges[1] += var_off
            cst_var_edges.append(cur_cst_var_edges)

            lit_edge_map.append(cst_data.lit_edge_map + edge_off)

            cst_type.append(cst_data.cst_type)
            batch_idx.append(cst_data.batch + i)

            cst_off += cst_data.num_cst
            tup_off += cst_data.num_tup
            edge_off += cst_data.num_edges

        cst_idx = torch.cat(cst_idx, dim=0)
        tup_idx = torch.cat(tup_idx, dim=0)
        val_idx = torch.cat(val_idx, dim=0)
        cst_type = torch.cat(cst_type, dim=0)
        batch_idx = torch.cat(batch_idx, dim=0)
        cst_val_edges = torch.cat(cst_val_edges, dim=1)
        cst_var_edges = torch.cat(cst_var_edges, dim=1)
        lit_edge_map = torch.cat(lit_edge_map, dim=0)

        batch_cst_data = Constraint_Data(
            csp_data=merged_csp_data,
            cst_idx=cst_idx,
            tup_idx=tup_idx,
            val_idx=val_idx,
            cst_type=cst_type,
            batch=batch_idx,
            cst_edges=cst_val_edges,
            cst_var_edges=cst_var_edges,
            lit_edge_map=lit_edge_map,
        )
        return batch_cst_data

    def update_LE_(self, assignment, tup_sum, **kwargs):
        num_sat_neigh = scatter_max(tup_sum[self.tup_idx], self.lit_edge_map, dim=0, dim_size=self.num_edges)[0]
        num_sat_neigh = num_sat_neigh.int()
        num_sat_neigh -= assignment[self.cst_edges[1]]

        comp_mask = num_sat_neigh != self.edge_comp_thresh
        comp_mask[self.neg_edge_mask] = ~comp_mask[self.neg_edge_mask]
        self.LE = comp_mask.long()

    def is_sat(self, assignment, update_val_comp=False):
        assignment = assignment.byte()
        tup_sum = scatter_sum(assignment[self.val_idx], self.tup_idx, dim=0, dim_size=self.num_tup)
        tup_sat = self.tup_deg - tup_sum
        cst_sat = scatter_min(tup_sat, self.cst_idx, dim=0, dim_size=self.num_cst)[0]
        cst_sat = cst_sat == 0
        cst_sat[self.cst_neg_mask] = ~cst_sat[self.cst_neg_mask]

        if update_val_comp:
            self.update_LE_(assignment, tup_sum)

        return cst_sat.float()


class Constraint_Data_All_Diff(Constraint_Data_Base):

    def __init__(self, csp_data, cst_idx, var_idx, cst_edges=None, cst_var_edges=None, count_edge_map=None, batch=None):
        self.cst_idx = cst_idx
        self.var_idx = var_idx

        cst_var_dom_size = csp_data.domain_size[var_idx]
        self.cst_dom_size = scatter_max(cst_var_dom_size, cst_idx, dim=0)[0]
        self.num_val = self.cst_dom_size.sum().cpu().numpy()
        self.cst_ptr = torch.zeros((self.cst_dom_size.shape[0]+1,), dtype=torch.int64)
        self.cst_ptr[1:] = torch.cumsum(self.cst_dom_size, dim=0)

        if cst_var_edges is None:
            cst_var_edges = torch.stack([cst_idx, var_idx], dim=0)
        if cst_edges is None:
            cst_edges = torch.repeat_interleave(cst_var_edges, cst_var_dom_size, dim=1)
            cst_edges[1] = csp_data.var_off[cst_edges[1]]

            cst_val_off = torch.zeros_like(cst_var_dom_size)
            cst_val_off[1:] += torch.cumsum(cst_var_dom_size[:-1], dim=0)
            cst_val_shift = torch.arange(cst_edges.shape[1])
            cst_val_shift -= torch.repeat_interleave(cst_val_off, cst_var_dom_size, dim=0)
            cst_edges[1] += cst_val_shift

        super(Constraint_Data_All_Diff, self).__init__(
            csp_data=csp_data,
            cst_edges=cst_edges,
            cst_var_edges=cst_var_edges,
            batch=batch,
        )

        self.count_edge_map = self.cst_ptr[cst_edges[0]] + csp_data.dom_idx[cst_edges[1]]
        self.val_var_idx = csp_data.var_idx
        self.var_off = csp_data.var_off

    def to(self, device):
        super(Constraint_Data_All_Diff, self).to(device)
        self.cst_idx = self.cst_idx.to(device)
        self.var_idx = self.var_idx.to(device)
        self.val_var_idx = self.val_var_idx.to(device)
        self.var_off = self.var_off.to(device)
        self.cst_ptr = self.cst_ptr.to(device)
        self.cst_dom_size = self.cst_dom_size.to(device)
        self.count_edge_map = self.count_edge_map.to(device)

    @staticmethod
    def collate(batch_list, merged_csp_data):
        cst_idx, var_idx, batch_idx = [], [], []
        cst_off, edge_off = 0, 0
        cst_val_edges, cst_var_edges, count_edge_map = [], [], []
        for cst_data, var_off, val_off, i in batch_list:
            cst_idx.append(cst_data.cst_idx + cst_off)
            var_idx.append(cst_data.var_idx + var_off)

            cur_cst_val_edges = cst_data.cst_edges.clone()
            cur_cst_val_edges[0] += cst_off
            cur_cst_val_edges[1] += val_off
            cst_val_edges.append(cur_cst_val_edges)

            cur_cst_var_edges = cst_data.cst_var_edges.clone()
            cur_cst_var_edges[0] += cst_off
            cur_cst_var_edges[1] += var_off
            cst_var_edges.append(cur_cst_var_edges)

            count_edge_map.append(cst_data.count_edge_map + edge_off)

            batch_idx.append(cst_data.batch + i)

            cst_off += cst_data.num_cst
            edge_off += cst_data.num_edges

        cst_idx = torch.cat(cst_idx, dim=0)
        var_idx = torch.cat(var_idx, dim=0)
        batch_idx = torch.cat(batch_idx, dim=0)
        cst_val_edges = torch.cat(cst_val_edges, dim=1)
        cst_var_edges = torch.cat(cst_var_edges, dim=1)
        count_edge_map = torch.cat(count_edge_map, dim=0)

        batch_cst_data = Constraint_Data_All_Diff(
            csp_data=merged_csp_data,
            cst_idx=cst_idx,
            var_idx=var_idx,
            batch=batch_idx,
            cst_edges=cst_val_edges,
            cst_var_edges=cst_var_edges,
            count_edge_map=count_edge_map,
        )
        return batch_cst_data

    def update_LE_(self, assignment, val_count, **kwargs):
        val_count = val_count[self.count_edge_map]
        val_count -= assignment[self.cst_edges[1]].int().flatten()
        comp_mask = val_count == 0
        self.LE = 1.0 - comp_mask.float().view(-1, 1)

    def is_sat(self, assignment, update_val_comp=False):
        idx_assignment = scatter_max(assignment, self.val_var_idx, dim=0)[1] - self.var_off.view(-1, 1)
        idx_assignment = idx_assignment[self.var_idx] + self.cst_ptr[self.cst_idx].view(-1, 1)
        value_count = degree(idx_assignment.flatten(), num_nodes=self.num_val, dtype=torch.int32)
        cst_sat = segment_max_csr(value_count, self.cst_ptr)[0]
        cst_sat = cst_sat <= 1

        if update_val_comp:
            self.update_LE_(assignment, value_count)

        return cst_sat.view(-1, assignment.shape[1]).float()


class Constraint_Data_Linear(Constraint_Data_Base):

    def __init__(self, csp_data, cst_idx, var_idx, coeffs, b, comp_op, cst_edges=None, cst_var_edges=None, cst_var_dom_size=None, batch=None):
        self.cst_idx = cst_idx
        self.var_idx = var_idx
        self.coeffs = coeffs
        self.comp_op = comp_op
        self.b = b

        if cst_var_edges is None:
            cst_var_edges = torch.stack([cst_idx, var_idx], dim=0)
        if cst_edges is None:
            cst_var_dom_size = csp_data.domain_size[cst_var_edges[1]]
            cst_edges = torch.repeat_interleave(cst_var_edges, cst_var_dom_size, dim=1)
            cst_edges[1] = csp_data.var_off[cst_edges[1]]

            cst_val_off = torch.zeros_like(cst_var_dom_size)
            cst_val_off[1:] += torch.cumsum(cst_var_dom_size[:-1], dim=0)
            cst_val_shift = torch.arange(cst_edges.shape[1])
            cst_val_shift -= torch.repeat_interleave(cst_val_off, cst_var_dom_size, dim=0)
            cst_edges[1] += cst_val_shift

        super(Constraint_Data_Linear, self).__init__(
            csp_data=csp_data,
            cst_edges=cst_edges,
            cst_var_edges=cst_var_edges,
            batch=batch
        )

        self.cst_var_dom_size = cst_var_dom_size
        self.b_expanded = b[cst_edges[0]]
        self.comp_op_expanded = comp_op[cst_edges[0]]
        self.scaled_val = torch.repeat_interleave(coeffs.int(), self.cst_var_dom_size) * csp_data.domain[cst_edges[1]].int()

    def to(self, device):
        super(Constraint_Data_Linear, self).to(device)
        self.cst_idx = self.cst_idx.to(device)
        self.var_idx = self.var_idx.to(device)
        self.b = self.b.to(device)
        self.coeffs = self.coeffs.to(device)
        self.comp_op = self.comp_op.to(device)
        self.cst_var_dom_size = self.cst_var_dom_size.to(device)
        self.b_expanded = self.b_expanded.to(device)
        self.comp_op_expanded = self.comp_op_expanded.to(device)
        self.scaled_val = self.scaled_val.to(device)

    @staticmethod
    def collate(batch_list, merged_csp_data):
        cst_idx, var_idx, coeffs, b, comp_op, batch_idx = [], [], [], [], [], []
        cst_off, edge_off = 0, 0
        cst_val_edges, cst_var_edges, cst_var_dom_size = [], [], []
        for cst_data, var_off, val_off, i in batch_list:
            cst_idx.append(cst_data.cst_idx + cst_off)
            var_idx.append(cst_data.var_idx + var_off)
            coeffs.append(cst_data.coeffs)
            b.append(cst_data.b)
            comp_op.append(cst_data.comp_op)

            cur_cst_val_edges = cst_data.cst_edges.clone()
            cur_cst_val_edges[0] += cst_off
            cur_cst_val_edges[1] += val_off
            cst_val_edges.append(cur_cst_val_edges)

            cur_cst_var_edges = cst_data.cst_var_edges.clone()
            cur_cst_var_edges[0] += cst_off
            cur_cst_var_edges[1] += var_off
            cst_var_edges.append(cur_cst_var_edges)

            cst_var_dom_size.append(cst_data.cst_var_dom_size)

            batch_idx.append(cst_data.batch + i)

            cst_off += cst_data.num_cst
            edge_off += cst_data.num_edges

        cst_idx = torch.cat(cst_idx, dim=0)
        var_idx = torch.cat(var_idx, dim=0)
        coeffs = torch.cat(coeffs, dim=0)
        b = torch.cat(b, dim=0)
        comp_op = torch.cat(comp_op, dim=0)
        batch_idx = torch.cat(batch_idx, dim=0)
        cst_val_edges = torch.cat(cst_val_edges, dim=1)
        cst_var_edges = torch.cat(cst_var_edges, dim=1)
        cst_var_dom_size = torch.cat(cst_var_dom_size, dim=0)

        batch_cst_data = Constraint_Data_Linear(
            csp_data=merged_csp_data,
            cst_idx=cst_idx,
            var_idx=var_idx,
            coeffs=coeffs,
            b=b,
            comp_op=comp_op,
            batch=batch_idx,
            cst_edges=cst_val_edges,
            cst_var_edges=cst_var_edges,
            cst_var_dom_size=cst_var_dom_size,
        )
        return batch_cst_data

    @staticmethod
    def comp_idx(comp):
        if comp == 'eq':
            return 0
        elif comp == 'ne':
            return 1
        elif comp == 'le':
            return 2
        elif comp == 'ge':
            return 3
        else:
            raise ValueError(f'Operator {comp} not supported for linear constraints!')

    def compare_(self, x, b, op):
        y = torch.stack([
            x == b,
            x != b,
            x <= b,
            x >= b,
        ], dim=1)
        y = y[torch.arange(y.shape[0], device=x.device), op]
        return y

    def update_LE_(self, assignment, lin_comb, val_sel, **kwargs):
        var_lin_comb = lin_comb[self.cst_var_edges[0]] - val_sel
        val_lin_comb = torch.repeat_interleave(var_lin_comb, self.cst_var_dom_size) + self.scaled_val
        #comp_mask = self.compare_(val_lin_comb, self.b_expanded, self.comp_op_expanded)

        diff = val_lin_comb - self.b_expanded
        cost = torch.stack([
            torch.abs(diff),
            (diff == 0).float(),
            torch.relu(diff),
            ], dim=1
        )
        cost = cost[torch.arange(diff.shape[0], device=diff.device), self.comp_op_expanded]
        self.LE = cost.float().view(-1, 1)

    def is_sat(self, assignment, update_val_comp=False):
        assignment = assignment.int()
        val_sel = self.scaled_val[assignment[self.cst_edges[1]].flatten().bool()]
        lin_comb = scatter_sum(val_sel, self.cst_var_edges[0], dim=0)
        cst_sat = self.compare_(lin_comb, self.b, self.comp_op)

        if update_val_comp:
            self.update_LE_(assignment, lin_comb, val_sel)

        return cst_sat.view(-1, assignment.shape[1]).float()
