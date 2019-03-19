from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP,self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.seq = nn.Sequential(
        nn.Linear(input_dim,256),
        nn.ReLU(),
        nn.Linear(256,128),
        nn.ReLU(),
        nn.Linear(128, output_dim)
        )
    def forward(self, feature):
        return self.seq(feature)

    def net_forward(self, feature):
        return self.forward(feature)

