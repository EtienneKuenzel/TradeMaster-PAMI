import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = "C:\\Users\\Etienne\\Desktop\\TradeMaster-PAMI\\TradeMaster"  # Include the TradeMaster directory
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
from trademaster.transition.builder import build_transition
from trademaster.utils import plot
from trademaster.utils import set_seed
set_seed(2023)
parser = argparse.ArgumentParser(description='Download Alpaca Datasets')
parser.add_argument("--config", default=osp.join(ROOT, "configs", "high_frequency_trading", "high_frequency_trading_BTC_dqn_dqn_adam_mse.py"), help="download datasets config file path")
parser.add_argument("--task_name", type=str, default="train")


args, _ = parser.parse_known_args()
print(f"Looking for config file at: {args.config}")

try:
    cfg = Config.fromfile(args.config)
except FileNotFoundError as e:
    print(e)
    print(f"Ensure the file exists at: {args.config}")
    exit(1)

task_name = args.task_name
cfg = replace_cfg_vals(cfg)

dataset = build_dataset(cfg)
print("a")

valid_environment = build_environment(cfg, default_args=dict(dataset=dataset, task="valid"))
test_environment = build_environment(cfg, default_args=dict(dataset=dataset, task="test"))
cfg.environment = cfg.train_environment
train_environment = build_environment(cfg, default_args=dict(dataset=dataset, task="train"))

train_environment.df.head()

action_dim = train_environment.action_dim
state_dim = train_environment.state_dim
print("a")

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
print("a")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent = build_agent(cfg,default_args=dict(action_dim=action_dim,state_dim=state_dim,act=act,cri=cri,act_optimizer=act_optimizer,cri_optimizer=cri_optimizer,criterion=criterion,device=device))

trainer = build_trainer(cfg, default_args=dict(train_environment=train_environment,valid_environment=valid_environment,test_environment=test_environment,agent=agent,device=device))

cfg.dump(osp.join(ROOT, cfg.work_dir, osp.basename(args.config)))

trainer.train_and_valid()
trainer.test()
plot(trainer.test_environment.save_asset_memoey(),alg="DDQN")