task_name = "portfolio_management"
dataset_name = "dj30"
net_name = "dqn"
agent_name = "ddqn"
optimizer_name = "adam"
loss_name = "mse"
work_dir = f"work_dir/{task_name}_{dataset_name}_{net_name}_{agent_name}_{optimizer_name}_{loss_name}"

_base_ = [
    f"../_base_/datasets/{task_name}/{dataset_name}.py",
    f"../_base_/environments/{task_name}/env.py",
    f"../_base_/agents/{task_name}/{agent_name}.py",#
    f"../_base_/trainers/{task_name}/trainer.py",#
    f"../_base_/losses/{loss_name}.py",
    f"../_base_/optimizers/{optimizer_name}.py",
    f"../_base_/nets/{net_name}.py",#
    f"../_base_/transition/transition.py"
]

data = dict(
    type='PortfolioManagementDataset',
    data_path='data/portfolio_management/dj30',
    train_path='data/portfolio_management/dj30/train.csv',
    valid_path='data/portfolio_management/dj30/valid.csv',
    test_path='data/portfolio_management/dj30/test.csv',
    test_dynamic_path='data/portfolio_management/dj30/test_with_label.csv',
    tech_indicator_list=[
        'zopen', 'zhigh', 'zlow', 'zadjcp', 'zclose',
        'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
    ],
    length_day=10,
    initial_amount=100000,
    transaction_cost_pct=0.001)

environment = dict(type='PortfolioManagementEIIEEnvironment')
transition = dict(type = "Transition")
agent = dict(
    type='HighFrequencyTradingDDQN',
    memory_capacity=1000,
    gamma=0.99,
    policy_update_frequency=500)

trainer = dict(
    type='PortfolioManagementTrainer',
    epochs=1,
    work_dir=work_dir,
    if_remove=False )

loss = dict(type='MSELoss')

optimizer = dict(type='Adam', lr=0.001)

act = dict(type='QNet', state_dim=82, action_dim=3, dims=(64, 32), explore_rate=0.25)
cri = None