task_name = "algorithmic_trading"
dataset_name = "BTC"
optimizer_name = "adam"
loss_name = "mse"
auxiliry_loss_name = "KLdiv"
net_name = "dqn"
agent_name = "ddqn"
work_dir = f"work_dir/{task_name}_{dataset_name}_{net_name}_{agent_name}_{optimizer_name}_{loss_name}"

_base_ = [
    f"../_base_/datasets/{task_name}/{dataset_name}.py",
    f"../_base_/environments/{task_name}/env.py",
    f"../_base_/agents/{task_name}/{agent_name}.py",
    f"../_base_/trainers/{task_name}/trainer.py",
    f"../_base_/losses/{loss_name}.py",
    f"../_base_/optimizers/{optimizer_name}.py",
    f"../_base_/nets/{net_name}.py",
]

batch_size = 128
data = dict(
    type='AlgorithmicTradingDataset',
    data_path='data/algorithmic_trading/BTC',
    train_path='data/algorithmic_trading/BTC/movavg.csv',
    valid_path='data/algorithmic_trading/BTC/test_4_doubled.csv',
    test_path='data/algorithmic_trading/BTC/test_1.csv',
    tech_indicator_list=[
        'high', 'low', 'open', 'close', 'adjcp', 'zopen', 'zhigh', 'zlow',
        'zadjcp', 'zclose', 'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
    ],
    backward_num_day=5,
    forward_num_day=5,
    test_dynamic='-1')

environment = dict(type='AlgorithmicTradingEnvironment')
transition = dict(type = "Transition")
agent = dict(
    type='AlgorithmicTradingDQN',
    max_step=12345,
    reward_scale=1,
    repeat_times=1,
    gamma=0.8803679826789824,
    batch_size=batch_size,
    clip_grad_norm=3.0,
    soft_update_tau=0.005,
    state_value_tau=0.005
)
trainer = dict(
    type='AlgorithmicTradingTrainer',
    epochs=1,
    work_dir=work_dir,
    seeds_list=(4, ),
    batch_size=batch_size,
    horizon_len= 120,
    buffer_size=1000000.0,
    num_threads=8,
    if_remove=False,
    if_discrete=True,
    if_off_policy=True,
    if_keep_save=True,
    if_over_write=False,
    if_save_buffer=False,)
loss = dict(type='MSELoss')
optimizer = dict(type="Adam", lr=0.01)
act = dict(type='QNet', state_dim=82, action_dim=9, dims=(64, 64), explore_rate=0.5)
cri = None
