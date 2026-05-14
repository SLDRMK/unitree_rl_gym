import torch.nn as nn


class AMPDiscriminator(nn.Module):
    """Logits-only MLP discriminator (use BCEWithLogitsLoss for training)."""

    def __init__(self, input_dim: int, hidden_dims: list, activation: str = "elu"):
        super().__init__()
        activation = activation.lower().strip()
        act_cls = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "lrelu": nn.LeakyReLU}.get(activation, nn.ELU)
        layers = []
        d_in = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(d_in, h), act_cls()])
            d_in = h
        layers.append(nn.Linear(d_in, 1))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x).squeeze(-1)
