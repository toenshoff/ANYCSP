import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

from src.utils.config_utils import read_config, dataset_from_config
from src.model.model import ANYCSP
from src.model.loss import reinforce_loss
from src.csp.csp_data import CSP_Data

from argparse import ArgumentParser
from tqdm import tqdm
import os


torch.multiprocessing.set_sharing_strategy('file_system')


def get_linear_scheduler():
    training_steps = config['epochs'] * len(train_loader)
    decay = config['lr_decay']
    lr_fn = lambda step: max(1.0 - ((1.0 - decay) * (step / training_steps)), decay)
    scheduler = LambdaLR(opt, lr_lambda=lr_fn)
    return scheduler


def save_opt_states(model_dir):
    torch.save(
        {
            'opt_state_dict': opt.state_dict(),
            'sched_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        },
        os.path.join(model_dir, 'opt_state_dict.pt')
    )


def load_opt_states(model_dir):
    state_dicts = torch.load(os.path.join(model_dir, 'opt_state_dict.pt'))
    opt.load_state_dict(state_dicts['opt_state_dict'])
    scheduler.load_state_dict(state_dicts['sched_state_dict'])
    scaler.load_state_dict(state_dicts['scaler_state_dict'])
    return opt, scaler


def train_epoch():
    model.train()
    unsat_list = []
    unsat_ratio_list = []
    solved_list = []

    for data in tqdm(train_loader, total=len(train_loader), disable=args.no_bar, desc=f'Training Epoch {epoch+1}'):
        opt.zero_grad()
        data.to(device)

        with torch.cuda.amp.autocast():
            data = model(
                data,
                config['T_train'],
                return_log_probs=True,
                return_all_unsat=True,
                return_all_assignments=True
            )

            loss = reinforce_loss(data, config)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)

        scaler.step(opt)
        scaler.update()
        scheduler.step()

        best_unsat = data.best_num_unsat.view(-1)
        unsat_ratio = best_unsat / data.batch_num_cst.view(-1)
        solved = best_unsat == 0

        unsat_list.append(best_unsat.cpu())
        unsat_ratio_list.append(unsat_ratio.cpu())
        solved_list.append(solved.cpu())

        if (model.global_step + 1) % args.logging_steps == 0:
            unsat = torch.cat(unsat_list, dim=0)
            unsat_ratio = torch.cat(unsat_ratio_list, dim=0)
            solved = torch.cat(solved_list, dim=0)
            logger.add_scalar('Train/Loss', loss.mean(), model.global_step)
            logger.add_scalar('Train/Solved_Ratio', solved.float().mean(), model.global_step)
            logger.add_scalar('Train/Unsat_Count', unsat.float().mean(), model.global_step)
            logger.add_scalar('Train/Unsat_Ratio', unsat_ratio.float().mean(), model.global_step)
            unsat_list = []
            unsat_ratio_list = []
            solved_list = []

        if (model.global_step + 1) % args.checkpoint_steps == 0:
            model.save_model(name=f'checkpoint_{model.global_step}')

        model.global_step += 1


def validate():
    model.eval()

    total_unsat = 0
    total_solved = 0
    total_count = 0

    for data in tqdm(val_loader, disable=args.no_bar, desc=f'Validating'):
        data.to(device)
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                data = model(
                    data,
                    config['T_val'],
                    return_log_probs=False,
                    return_all_unsat=True,
                    return_all_assignments=False
                )

        best_unsat = data.best_num_unsat.view(-1)
        total_unsat += best_unsat.float().sum().cpu().numpy()
        total_solved += (best_unsat == 0).float().sum().cpu().numpy()
        total_count += data.batch_size

    unsat = total_unsat / total_count
    solved = total_solved / total_count
    logger.add_scalar('Val/Solved_Ratio', solved, model.global_step)
    logger.add_scalar('Val/Unsat_Count', unsat, model.global_step)
    return unsat, solved


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='models/comb/test', help="Model directory")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    parser.add_argument("--logging_steps", type=int, default=10, help="Training steps between logging")
    parser.add_argument("--checkpoint_steps", type=int, default=5000, help="Training steps between saving checkpoints")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of workers")
    parser.add_argument("--no_bar", action='store_true', default=False, help="Turn of tqdm bar")
    parser.add_argument("--from_last", action='store_true', default=False, help="Continue from existing last checkpoint")
    parser.add_argument("--pretrained_dir", type=str, default=None, help="Pretrained Model directory")
    parser.add_argument("--config", type=str, help="path the config file")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.from_last:
        args.pretrained_dir = args.model_dir
        args.config = os.path.join(args.model_dir, 'config.json')

    config = read_config(args.config)

    if args.pretrained_dir is None:
        model = ANYCSP(args.model_dir, config)
    else:
        model = ANYCSP.load(args.pretrained_dir, f'last')
        model.model_dir = args.model_dir

    model.to(device)
    model.train()

    train_data = dataset_from_config(config['train_data'], config['epoch_steps'] * config['batch_size'])
    train_loader = DataLoader(
        train_data,
        batch_size=config['batch_size'],
        num_workers=args.num_workers,
        collate_fn=CSP_Data.collate
    )

    if 'val_data' in config:
        val_data = dataset_from_config(config['val_data'])
        val_loader = DataLoader(
            val_data,
            batch_size=config['val_batch_size'],
            num_workers=args.num_workers,
            collate_fn=CSP_Data.collate
        )
    else:
        val_loader = None

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = get_linear_scheduler()
    scaler = torch.cuda.amp.GradScaler()

    if args.pretrained_dir is not None:
        opt, scaler = load_opt_states(args.pretrained_dir)

    logger = SummaryWriter(args.model_dir)
    best_unsat = np.float32('inf')
    best_solved = 0.0
    start_step = 0
    for epoch in range(config['epochs']):
        train_epoch()

        if val_loader is not None:
            unsat, solved = validate()

            print(f'Mean Unsat Count: {unsat:.2f}, Solved: {100 * solved:.2f}%')
            if unsat < best_unsat:
                model.save_model(name='best')
                best_unsat = unsat

        model.save_model(name='last')
        save_opt_states(model.model_dir)
