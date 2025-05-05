import torch.nn as nn
 
class NeuralNetwork(nn.Module):
    
    """
    
    """
    def __init__(self) -> None: 
        super().__init__() 
        self.flatten = nn.Flatten() 
        self.network_stack = nn.Sequential( 
            nn.Linear(in_features = 28*28, out_features = 512), 
            nn.ReLU(), 
            nn.Linear(in_features = 512, out_features = 10) 
        )
            
    def forward(self, x): #defines how data flows through the network
        x = self.flatten(x) #First flattens the input image
        output = self.network_stack(x) #Then passes it through the network stack
        return output #Returns the final output (scores for each of the 10 classes)