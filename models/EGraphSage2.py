import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data


class EGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_layers=2, residual=True):
        """
        E-GraphSAGE with support for edge features.

        Parameters:
        - in_channels (int): Size of input node features.
        - hidden_channels (int): Size of hidden layer features.
        - out_channels (int): Number of classes for classification.
        - edge_dim (int): Size of edge features.
        - num_layers (int): Number of GNN layers.
        - residual (bool): Whether to include edge features in residual connections.
        """
        super(EGraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.residual = residual

        # Define SAGEConv layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        # Edge feature transformation layer
        if residual:
            self.edge_transform = nn.Linear(edge_dim, hidden_channels)

    def forward(self, data):
        """
        Forward pass for E-GraphSAGE.

        Parameters:
        - data (torch_geometric.data.Data): Input graph with x (node features), 
                                            edge_index (edge connectivity), and edge_attr (edge features).

        Returns:
        - logits (torch.Tensor): Logits for each node.
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Node embedding propagation
        for i, conv in enumerate(list(self.convs)[:-1]):
            x = conv(x, edge_index)
            if self.residual and i == 0:
                edge_embeds = self.edge_transform(edge_attr)
                x[edge_index[1]] += edge_embeds  # Add edge features to target nodes
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        # Final layer (no activation)
        x = self.convs[-1](x, edge_index)
        return x

    def loss(self, logits, labels):
        """
        Compute cross-entropy loss for node classification.

        Parameters:
        - logits (torch.Tensor): Model predictions for each node.
        - labels (torch.Tensor): Ground truth labels for each node.

        Returns:
        - loss (torch.Tensor): Cross-entropy loss value.
        """
        return F.cross_entropy(logits, labels)



