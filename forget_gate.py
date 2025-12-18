import torch.nn as nn
import torch




class M4ForgetGate(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim + 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, emb, age, bm25):
        x = torch.cat([emb, age, bm25], dim=1)
        logits = self.net(x).squeeze(1)
        f = torch.sigmoid(logits)
        return f, logits