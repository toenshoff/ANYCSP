import os
from timeit import default_timer as timer
import torch
from torch.nn import Module, GRUCell
from torch_scatter import scatter_sum

from src.model.layers import Val2Val_Layer, Cst2Val_Layer, Val2Cst_Layer, Policy
from src.utils.config_utils import read_config, write_config


class ANYCSP(Module):

    def __init__(self, model_dir, config):
        super(ANYCSP, self).__init__()
        self.model_dir = model_dir
        self.config = config
        self.hidden_dim = config['hidden_dim']
        self.sampling = config['sampling']

        # GRU cell and its initial state
        self.h_val_init = torch.nn.Parameter(torch.normal(0.0, 1.0, (1, self.hidden_dim), dtype=torch.float32))
        self.val_cell = GRUCell(self.hidden_dim, self.hidden_dim)

        # module for msg. pass from values to constraints
        self.val2cst = Val2Cst_Layer(config)

        # module for msg. pass from constraints to values
        self.cst2val = Cst2Val_Layer(config)

        # module for msg. from values tor variables and back
        self.val2val = Val2Val_Layer(config)

        # Output mlp (O)
        self.policy = Policy(config)

        self.global_step = 0

    def save_model(self, model_dir=None, name='model'):
        if model_dir is None:
            model_dir = self.model_dir
        os.makedirs(model_dir, exist_ok=True)
        write_config(self.config, model_dir)
        state_dict = self.state_dict()
        state_dict['global_step'] = self.global_step
        torch.save(state_dict, os.path.join(model_dir, f'{name}.pkl'))

    @staticmethod
    def load_model(model_dir, name='model'):
        config = read_config(os.path.join(model_dir, 'config.json'))
        model = ANYCSP(model_dir, config)
        state_dict = torch.load(os.path.join(model_dir, f'{name}.pkl'))
        model.load_state_dict(state_dict, strict=False)
        model.global_step = state_dict['global_step']
        return model

    def init_assignment(self, data):
        logits = torch.ones((data.num_val,), device=data.device, dtype=torch.float32)
        assignment, _ = data.hard_assign_sample(logits)
        cst_sat = data.constraint_is_sat(assignment, update_LE=True)
        num_unsat = scatter_sum(1.0 - cst_sat, data.cst_batch, dim=0, dim_size=data.batch_size)
        return assignment, num_unsat

    def update_assignment(self, data, logits, assignment):
        if self.sampling == 'local':
            assignment, log_prob = data.hard_assign_sample_local(logits, assignment)
        else:
            assignment, log_prob = data.hard_assign_sample(logits)
        cst_sat = data.constraint_is_sat(assignment, update_LE=True)
        num_unsat = scatter_sum(1.0 - cst_sat, data.cst_batch, dim=0, dim_size=data.batch_size)
        return assignment, num_unsat, log_prob

    def forward(
            self,
            data,
            steps,
            stop_early=False,
            return_log_probs=False,
            return_all_assignments=False,
            return_all_unsat=False,
            verbose=False,
            keep_time=False,
            timeout=None
    ):
        data.init_adj()

        # initialize first assignment and states
        assignment, num_unsat = self.init_assignment(data)
        h_val = self.h_val_init.tile(data.num_val, 1)

        data.best_num_unsat = num_unsat.min(dim=1)[0]
        data.num_steps = 0

        value_assignment = data.domain[assignment.flatten().bool()]
        assignment_list = [value_assignment.view(-1, 1)]
        num_unsat_list = [data.best_num_unsat.view(-1, 1)]
        log_prob_list = []

        opt = data.best_num_unsat.min()
        data.opt_step = 0
        if verbose:
            print(f'o {opt.int().cpu().numpy()}')

        keep_time |= timeout is not None
        if keep_time:
            start = timer()
            data.opt_time = 0.0

        for s in range(steps):
            # one round of mes passes
            r_cst, x_val = self.val2cst(data, h_val, assignment)
            y_val = self.cst2val(data, x_val, r_cst)
            z_val = self.val2val(data, y_val)

            # update states and predict logit scores
            h_val = self.val_cell(z_val, h_val)
            logits = self.policy(h_val)

            # sample next assignment
            assignment, num_unsat, log_prob = self.update_assignment(data, logits, assignment)
            data.num_steps = s + 1

            # update all kinds of metrics...
            num_unsat, best_assign = num_unsat.min(dim=1)
            data.best_num_unsat = torch.minimum(data.best_num_unsat, num_unsat)
            cur_opt = data.best_num_unsat.min()

            if return_log_probs:
                log_prob_list.append(log_prob.view(-1, 1))
            if return_all_unsat:
                num_unsat_list.append(num_unsat.view(-1, 1))
            if keep_time:
                cur = timer()
                time = float(cur - start)
            if cur_opt < opt:
                opt = cur_opt
                if verbose:
                    print(f'o {opt.int().cpu().numpy()}')
                if keep_time:
                    data.opt_time = float(time)
                    data.opt_step = s + 1
            if stop_early and num_unsat.min() == 0.0:
                break
            if return_all_assignments:
                value_assignment = data.domain[assignment.flatten().bool()]
                assignment_list.append(value_assignment.view(-1, 1))
            if timeout is not None and timeout < time:
                break

        if return_log_probs:
            data.all_log_probs = torch.cat(log_prob_list, dim=1)
        if return_all_assignments:
            data.all_assignments = torch.cat(assignment_list, dim=1)
        if return_all_unsat:
            data.all_num_unsat = torch.cat(num_unsat_list, dim=1)
        if keep_time:
            data.total_time = time
        return data
