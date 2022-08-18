import torch


def reward_improve(data):
    # Our default reward scheme: Difference to previous best clipped at 0
    with torch.no_grad():
        reward = data.batch_num_cst.view(-1, 1) - data.all_num_unsat
        reward /= data.batch_num_cst.view(-1, 1) + 1.0e-8
        reward = reward - reward[:, 0].view(-1, 1)

        max_prior = torch.cummax(reward, dim=1)[0]
        reward[:, 1:] -= max_prior[:, :-1]
        reward[reward < 0.0] = 0.0
        reward[:, 0] = 0.0
        return reward


def reward_quality(data):
    # Naive reward scheme used in our ablation study
    with torch.no_grad():
        reward = data.batch_num_cst.view(-1, 1) - data.all_num_unsat
        reward /= data.batch_num_cst.view(-1, 1) + 1.0e-8
        reward = reward - reward[:, 0].view(-1, 1)
        return reward


def reinforce_loss(data, config):
    # get reward in each step t
    assert config['reward'] in {'improve', 'quality'}
    if config['reward'] == 'improve':
        reward = reward_improve(data)
    else:
        reward = reward_quality(data)

    # accumulate discounted future rewards
    with torch.no_grad():
        discount = config['discount']
        return_disc = torch.zeros((reward.shape[0], reward.shape[1]-1), device=data.device)
        weights = discount ** torch.arange(0, reward.shape[1], device=data.device)
        weights = weights.view(1, -1)
        for i in range(return_disc.shape[1]):
            r = reward[:, i+1:]
            w = weights[:, :r.shape[1]]
            return_disc[:, i] = (r * w).sum(dim=1)

    # loss term for the SGD optimizer
    loss = - (data.all_log_probs * return_disc).mean()
    return loss
