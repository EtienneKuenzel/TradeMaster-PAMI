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
from trademaster.transition.builder import build_transition
from trademaster.trainers.builder import build_trainer
from trademaster.utils import plot
from trademaster.utils import set_seed
import time
set_seed(2019)

start_time = time.time()
parser = argparse.ArgumentParser(description='Download Alpaca Datasets')
parser.add_argument("--config", default=osp.join(ROOT, "Trademaster", "configs", "algorithmic_trading", "algorithmic_trading_dj30_dqn_dqn_adam_mse.py"))
parser.add_argument("--task_name", type=str, default="train")

args, _ = parser.parse_known_args()
cfg = Config.fromfile(args.config)
cfg = replace_cfg_vals(cfg)

# Update the configuration with sampled hyperparameters

# Build dataset, environment, agent, and trainer
dataset = build_dataset(cfg)
train_environment = build_environment(cfg, default_args=dict(dataset=dataset, task="train"))
valid_environment = build_environment(cfg, default_args=dict(dataset=dataset, task="valid"))
test_environment = build_environment(cfg, default_args=dict(dataset=dataset, task="test"))
train_environment.df.head()

action_dim = train_environment.action_dim
state_dim = train_environment.state_dim
cfg.act.update(dict(action_dim=action_dim, state_dim=state_dim))
act = build_net(cfg.act)
act_optimizer = build_optimizer(cfg, default_args=dict(params=act.parameters()))
if cfg.cri:
    cfg.cri.update(dict(action_dim=action_dim, state_dim=state_dim))
    cri = build_net(cfg.cri)
    cri_optimizer = build_optimizer(cfg, default_args=dict(params=cri.parameters()))
else:
    cri = None
    cri_optimizer = None

criterion = build_loss(cfg)
transition = build_transition(cfg)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent = build_agent(cfg, default_args=dict(action_dim = action_dim,state_dim = state_dim,act = act,cri = cri,act_optimizer = act_optimizer,cri_optimizer = cri_optimizer, criterion = criterion,transition = transition,device=device))

trainer = build_trainer(cfg, default_args=dict(train_environment=train_environment,valid_environment=valid_environment,test_environment=test_environment,agent=agent,device=device))

trainer.train_and_valid()
trainer.test()
print(time.time() - start_time)



