import torch
import numpy as np
from src.csp.csp_data import CSP_Data
from src.model.model import ANYCSP

from argparse import ArgumentParser
from src.data.dataset import File_Dataset


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="Model directory")
    parser.add_argument("--data_path", type=str, help="Path to the training data")
    parser.add_argument("--checkpoint", type=str, default='best', help="Name of the checkpoint")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    parser.add_argument("--network_steps", type=int, default=100000, help="Number of network steps during evaluation")
    parser.add_argument("--num_boost", type=int, default=20, help="Number of parallel evaluate runs")
    parser.add_argument("--verbose", action='store_true', default=False, help="Output intermediate optima")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout in seconds")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dict_args = vars(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    name = 'model' if args.checkpoint is None else f'{args.checkpoint}'
    model = ANYCSP.load_model(args.model_dir, name)
    model.eval()
    model.to(device)

    dataset = File_Dataset(args.data_path)

    num_solved = 0
    num_total = len(dataset)

    for data in dataset:
        file = data.path
        max_val = data.constraints['ext'].cst_neg_mask.int().sum().cpu().numpy()
        if args.num_boost > 1:
            data = CSP_Data.collate([data for _ in range(args.num_boost)])
        data.to(device)

        if args.verbose:
            print(f'Solving {file}:')
        #with torch.cuda.amp.autocast():
        with torch.inference_mode():
            data = model(
                data,
                args.network_steps,
                return_all_assignments=False,
                return_log_probs=False,
                stop_early=True,
                verbose=args.verbose,
                keep_time=True,
                timeout=args.timeout,
            )

        best_per_run = data.best_num_unsat
        mean_best = best_per_run.mean()
        best = best_per_run.min().cpu().numpy()
        solved = best == 0
        num_solved += int(solved)
        best_cut_val = max_val - best

        print(
            f'{file}: {"Solved" if solved else "Unsolved"}, '
            f'Num Unsat: {int(best)}, '
            f'Cut Value: {best_cut_val}, '
            f'Steps: {data.num_steps}, '
            f'Opt Time: {data.opt_time:.2f}s, '
            f'Opt Step: {data.opt_step}'
        )

    print(f'Solved {100 * num_solved / num_total:.2f}%')
