import optuna
import os
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname("C:/Users/Etienne/Desktop/TradeMaster-PAMI/TradeMaster")
sys.path.append(ROOT)

import torch
import argparse
import os.path as osp
from mmcv import Config
from trademaster.utils import replace_cfg_vals
from trademaster.nets.builder import build_net
from trademaster.environments.builder import build_environment
from trademaster.datasets.builder import build_dataset
from trademaster.agents.builder import build_agent
from trademaster.optimizers.builder import build_optimizer
from trademaster.losses.builder import build_loss
from trademaster.trainers.builder import build_trainer
from trademaster.utils import plot
from trademaster.utils import set_seed
import time
set_seed(2023)

def objective(trial):
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Download Alpaca Datasets')
    parser.add_argument("--config", default=osp.join(ROOT, "Trademaster", "configs", "algorithmic_trading", "algorithmic_trading_dj30_dqn_dqn_adam_mse.py"))
    parser.add_argument("--task_name", type=str, default="train")

    args, _ = parser.parse_known_args()
    cfg = Config.fromfile(args.config)
    task_name = args.task_name
    cfg = replace_cfg_vals(cfg)

    # Sample hyperparameters using Optuna
    dims = trial.suggest_categorical('dims', [(64, 64, 64)])
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)  # Learning rate
    gamma = trial.suggest_float('gamma', 0.8, 0.99)  # Discount factor
    soft_update_tau = trial.suggest_float('soft_update_tau', 0.001, 0.1, log=True)  # Soft update tau
    horizon_len = trial.suggest_int('horizon_len', 64, 512, step=64)  # Horizon length
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])  # Batch size
    a = str(dims) + "_" + str(lr)+ "_" + str(gamma)+ "_" + str(soft_update_tau)+ "_" + str(horizon_len)+ "_" + str(batch_size)
    # Update the configuration with sampled hyperparameters
    cfg.act.update(dict(dims=dims))
    cfg.optimizer.update(dict(lr=lr))
    cfg.trainer.update(dict(horizon_len=horizon_len, batch_size=batch_size))
    cfg.agent.update(dict(gamma=gamma, soft_update_tau=soft_update_tau))

    # Build dataset, environment, agent, and trainer
    dataset = build_dataset(cfg)
    train_environment = build_environment(cfg, default_args=dict(dataset=dataset, task="train"))
    valid_environment = build_environment(cfg, default_args=dict(dataset=dataset, task="valid"))
    test_environment = build_environment(cfg, default_args=dict(dataset=dataset, task="test"))

    action_dim = train_environment.action_dim
    state_dim = train_environment.state_dim

    cfg.act.update(dict(action_dim=action_dim, state_dim=state_dim))
    act = build_net(cfg.act)
    act_optimizer = build_optimizer(cfg, default_args=dict(params=act.parameters()))

    criterion = build_loss(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = build_agent(cfg, default_args=dict(
        action_dim=action_dim,
        state_dim=state_dim,
        act=act,
        cri=None,
        act_optimizer=act_optimizer,
        cri_optimizer=None,
        criterion=criterion,
        device=device
    ))

    trainer = build_trainer(cfg, default_args=dict(
        train_environment=train_environment,
        valid_environment=valid_environment,
        test_environment=test_environment,
        agent=agent,
        device=device
    ))

    # Train and validate
    trainer.train_and_valid()
    episode_reward_sum = trainer.test(a)
    print(time.time() - start_time)
    # Evaluate the performance

    return episode_reward_sum

# Run Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Print best hyperparameters
print("Best hyperparameters: ", study.best_params)




