# This is a wrapper for torchsummary to be used with models that includes
# gradient checkpointing. Since gradient checkpointing expects tensors with
# requires_grad True

import torch.nn as nn

class torchSummaryWrapper(nn.Module):
    def __init__(self, model):
        super(torchSummaryWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        x = x.detach()
        x.requires_grad = True
        return self.model(x)
    
def get_torchSummaryWrapper( model ):
    
    return torchSummaryWrapper( model )