
data = dict(
    type = "AlgorithmicTradingDataset",
    data_path = "data/algorithmic_trading/dj30",
    train_path = "data/algorithmic_trading/dj30/train.csv",
    valid_path = "data/algorithmic_trading/dj30/valid.csv",
    test_path = "data/algorithmic_trading/dj30/test.csv",
    test_dynamic_path="", #'data/algorithmic_trading/dj30/test_labeled_3_24_-0.15_0.15.csv',
    tech_indicator_list = [
        "high",
        "low",
        "open",
        "close",
        "adjcp",
        "zopen",
        "zhigh",
        "zlow",
        "zadjcp",
        "zclose",
        "zd_5",
        "zd_10",
        "zd_15",
        "zd_20",
        "zd_25",
        "zd_30"
    ],
    backward_num_day = 5,
    forward_num_day = 5,
)