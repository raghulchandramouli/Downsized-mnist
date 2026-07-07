import torch
from torch import nn

# LinearBase - Acts as logic for Linear and LinearWithBias classes
class LinearBase(nn.Module):

    """
    Base class for linear layers. 
    It provides the common functionality for linear layers, including weight initialization and forward pass.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

# MLPBase - Acts as a 2-hidden layer MLP with ReLU activation
class MLPBase(nn.Module):
    
    """
    Base class for a Multi-Layer Perceptron (MLP) with two hidden layers and ReLU activation.
    """

    def __init__(self, input_size, output_size, hidden_size = 100):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = self.linear1(x).relu()
        h = h + self.linear2(h).relu() # Residual connection
        return self.linear3(h)
    
# ConvBase - Acts as a 2D convolutional layer with ReLU activation
class ConvBase(nn.Module):
    
    """
    the 1D-CNN, needs modifications
    Explicit:
        all three conv layers are using, 'Stride=2' - that's already doing downsampling,
        This is exactly what the pooling benchmark experiment.
    """

    def __init__(self, output_size, channels=25, linear_in=125):
        super().__init__()
        self.conv1 = nn.Conv1d(1, channels, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.linear = nn.Linear(linear_in, output_size)

    def forward(self, x):
        x = x.view(-1, 1, x.shape[-1])
        h1 = self.conv1(x).relu()
        h2 = self.conv2(h1).relu()
        h3 = self.conv3(h2).relu()
        h3 = h3.view(h3.shape[0], -1)
        return self.linear(h3)
    
# GRUBase - Acts as a GRU layer with a linear output layer
class GRUBase(nn.Module):
    
    """
    Base class for a Gated Recurrent Unit (GRU) layer followed by a linear output layer.
    """

    def __init__(self, input_size, output_size, hidden_size = 6, time_steps = 40, bidirectional = False):
        super().__init__()
        self.gru  = nn.GRU(input_size = input_size, hidden_size = hidden_size, batch_first = True, bidirectional = bidirectional)
        flat_size = 2 * hidden_size * time_steps if bidirectional else hidden_size * time_steps
        self.linear = nn.Linear(flat_size, output_size)
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

    def forward(self, x, h0 = None):
        x = x.view(*x.shape[:2], 1) # [Batch, time] -> [Batch, time, 1 feature]
        k = 2 if self.bidirectional else 1
        h0 = torch.zeros(k, x.shape[0], self.hidden_size) if h0 is None else h0
        output, hn = self.gru(x, h0)
        output = output.reshape(output.shape[0], -1) # Concat all timesteps outputs
        return self.linear(output)

# SSLBase - Acts as a base class for self-supervised learning models
class SSLProjectionHead(nn.Module):
    """
    Base class for self-supervised learning projection heads.
    It provides a linear layer to project the input features to a lower-dimensional space.
    """

    def __init__(self, encoder, embed_dim, proj_dim = 64, hidden_dim=64):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        return self.head(h)