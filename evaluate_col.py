import torch
import numpy as np
from glob import glob
from argparse import ArgumentParser
from tqdm import tqdm

from src.csp.csp_data import CSP_Data
from src.model.model import ANYCSP
from src.data.dataset import nx_to_col
from src.utils.data_utils import load_dimacs_graph


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="Model directory")
    parser.add_argument("--data_path", type=str, help="Path to the training data")
    parser.add_argument("--checkpoint", type=str, default='best', help="Name of the checkpoint")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    parser.add_argument("--network_steps", type=int, default=1000000, help="Number of network steps during evaluation")
    parser.add_argument("--num_boost", type=int, default=1, help="Number of parallel evaluate runs")
    parser.add_argument("--verbose", action='store_true', default=False, help="Output intermediate optima")
    parser.add_argument("--timeout", type=int, default=1200, help="Timeout in seconds")
    parser.add_argument("--num_colors", type=int, help="Number of colors")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    name = 'model' if args.checkpoint is None else f'{args.checkpoint}'
    model = ANYCSP.load_model(args.model_dir, name)
    model.eval()
    model.to(device)

    data_dict = {p: load_dimacs_graph(p) for p in tqdm(glob(args.data_path))}
    data_dict = {p: nx_to_col(g, args.num_colors) for p, g in data_dict.items()}

    num_solved = 0
    num_total = len(data_dict)
    for file, data in data_dict.items():
        if args.num_boost > 1:
            data = CSP_Data.collate([data for _ in range(args.num_boost)])
        data.to(device)

        if args.verbose:
            print(f'Solving {file}:')
        # with torch.cuda.amp.autocast():
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

        print(
            f'{file}: {"Solved" if solved else "Unsolved"}, '
            f'Num Unsat: {int(best)}, '
            f'Steps: {data.num_steps}, '
            f'Opt Time: {data.opt_time:.2f}s, '
            f'Opt Step: {data.opt_step}'
        )

    print(f'Solved {100 * num_solved / num_total:.2f}%')
