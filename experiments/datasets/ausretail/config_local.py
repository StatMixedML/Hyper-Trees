import torch.nn as nn

config = {
    "general": {
        "model_scope": "Local",
        "loss_function": nn.MSELoss(),
        "round_decimals": 4,
    },

    "hyper_tree_params": {
        "num_boost_round": 100,
        "eta": 1e-01,
        "linear_tree": True,
    },

    "hyper_treenet_params": {
        "num_boost_round": 100,
        "eta": 1e-01,
        "linear_tree": True,
        "embedding_dimension": 1,
        "use_random_projection": True,
    },

    "hyper_tree_ets_params": {
        "num_boost_round": 100,
        "eta": 1e-01,
        "linear_tree": True,
        "season_length": 12,
        "ets_type": "triple",
        "manual_param": 0.3,
        "scaling": False,
        "train": True
    },

    "lgb_params": {
        "num_boost_round": 100,
        "eta": 1e-01,
        "linear_tree": True,
        "use_time_index": True
    },

    "lgb_stl_params": {
        "degree": 3,
    },

    "deep_learning": {
        "num_epochs": 100,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "num_layers": 3,
        "num_heads": 4,
        "hidden_size": 128,
        "dropout": 0.1,
        "variable_dim": 128,
        "num_samples": 1000,
        "num_samples_chronos": 50,
        "quantiles_tft": [0.5]
    }
}