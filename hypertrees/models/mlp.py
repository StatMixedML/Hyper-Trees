import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class RPLayer(nn.Module):
    """
    Random Projection Layer as implemented in

    Chin-Chia Michael Yeh, Yujie Fan, Xin Dai, Uday Singh Saini, Vivian Lai, Prince Osei Aboagye, Junpeng Wang,
    Huiyuan Chen, Yan Zheng, Zhongfang Zhuang, Liang Wang, and Wei Zhang. 2024.

    RPMixer: Shaking Up Time Series Forecasting with Random Projections for Large Spatial-Temporal Data.

    In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '24).
    Association for Computing Machinery, New York, NY, USA, 3919–3930. https://doi.org/10.1145/3637528.3671881

    Parameters
    ----------
    in_dim : int
        Dimension of the input (the tree embeddings).
    out_dim : int
        Output dimension of the random projection.
    seed : int
        Random seed used to draw the fixed (non-trainable) projection weights.
    """

    def __init__(self, in_dim: int, out_dim: int, seed: int):
        super().__init__()
        torch.manual_seed(seed=seed)
        weight = torch.randn(out_dim, in_dim, requires_grad=False)
        self.register_buffer('weight', weight, persistent=True)
        self.register_buffer('bias', None)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class MLP(nn.Module):
    """
    Multi-Layer Perceptron with optional Random Projection

    Parameters
    ----------
    tree_embed_dim : int
        Dimension of the input tree embeddings.
    output_dim : int
        Dimension of the output layer.
    hidden_dim : int
        Dimension of the hidden layer.
    use_random_projection : bool, optional
        Whether to use random projection, by default False.
    rp_embed_dim : Optional[int], optional
        Dimension of the random projection layer, by default None.
    dropout_rate : float, optional
        Dropout rate applied after the output layer, by default 0.1.
    seed : int, optional
        Random seed for reproducibility, by default 123.
    """
    def __init__(self,
                 tree_embed_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 use_random_projection: bool = False,
                 rp_embed_dim: Optional[int] = None,
                 dropout_rate: float = 0.1,
                 seed: int = 123):
        super().__init__()

        layers = []
        if use_random_projection:
            if rp_embed_dim is None:
                raise ValueError("rp_embed_dim must be provided when use_random_projection=True.")
            layers.append(RPLayer(tree_embed_dim, rp_embed_dim, seed=seed))
            input_dim = rp_embed_dim
        else:
            input_dim = tree_embed_dim

        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            # Dropout deliberately sits after the output layer, as specified
            # in the paper (Section 2.2, footnote on the MLP architecture):
            # it randomly zeros individual target-model coefficients during
            # training, acting as parameter regularization.
            nn.Dropout(dropout_rate)
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
