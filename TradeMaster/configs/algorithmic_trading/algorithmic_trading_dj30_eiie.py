task_name = "algorithmic_trading"
dataset_name = "dj30"
net_name = "eiie"
agent_name = "eiie"
optimizer_name = "adam"
loss_name = "mse"
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

batch_size = 512
data = dict(
    type='AlgorithmicTradingDataset',
    data_path='data/algorithmic_trading/dj30',
    train_path='data/algorithmic_trading/dj30/train.csv',
    valid_path='data/algorithmic_trading/dj30/valid.csv',
    test_path='data/algorithmic_trading/dj30/test.csv',
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
    type='PortfolioManagementEIIE',
    max_step=12345,
    reward_scale=1,
    repeat_times=1,
    gamma=0.9,
    batch_size=batch_size,
    clip_grad_norm=3.0,
    soft_update_tau=0,
    state_value_tau=0.005
)
trainer = dict(
    type='AlgorithmicTradingTrainer',
    epochs=100,
    work_dir=work_dir,
    seeds_list=(42, ),
    batch_size=batch_size,
    horizon_len= 128,
    buffer_size=1000000.0,
    num_threads=8,
    if_remove=False,
    if_discrete=True,
    if_off_policy=True,
    if_keep_save=True,
    if_over_write=False,
    if_save_buffer=False,)
loss = dict(type='MSELoss')
act = dict(
    type = "EIIEConv",
    input_dim = None,
    output_dim=1,
    time_steps=10,
    kernel_size=3,
    dims = [32]
)

cri = dict(
    type = "EIIECritic",
    input_dim = None,
    action_dim = None,
    output_dim=1,
    time_steps=None,
    num_layers = 1,
    hidden_size=32
)

optimizer = dict(type='Adam', lr=0.001)