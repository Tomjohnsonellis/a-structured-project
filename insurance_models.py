from torch import nn

# This is a model specific for this task, the data has 10 dimensions
class InsurancePriceNN(nn.Module):
    # Define the layers
    def __init__(self, hidden_sizes):
        super().__init__()
        self.layer1 = nn.Linear(10, hidden_sizes[0])
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.activation2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_sizes[1], 1)
    
    # Define the forward pass
    def forward(self, sample):
        layer1_results = self.layer1(sample)
        layer1_output = self.activation1(layer1_results)
        layer2_results = self.layer2(layer1_output)
        layer2_output = self.activation2(layer2_results)
        network_output = self.layer3(layer2_output)
        return network_output

class AwfulModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)

    def forward(self, sample):
        network_output = self.layer(sample)
        return network_output
