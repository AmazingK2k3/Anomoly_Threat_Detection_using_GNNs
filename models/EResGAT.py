import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GATConv


class ModifiedEResGAT(torch.nn.Module):
    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, edge_feat_dim, device, 
                 add_skip_connection=False, residual=False, dropout=0.2):
        super(ModifiedEResGAT, self).__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, \
            f'Invalid architecture parameters.'

        self.num_of_layers = num_of_layers
        self.device = device
        self.edge_feat_dim = edge_feat_dim
        self.dropout = nn.Dropout(p=dropout)

        gat_layers = []
        for i in range(num_of_layers):
            gat_layers.append(
                GATLayerWithEdgeFeatures(
                    num_in_features=num_features_per_layer[i] if i == 0 else num_features_per_layer[i] * num_heads_per_layer[i],
                    num_out_features=num_features_per_layer[i + 1],
                    edge_feat_dim=self.edge_feat_dim,
                    num_heads=num_heads_per_layer[i],
                    concat=True if i < num_of_layers - 1 else False,
                    activation=nn.ELU() if i < num_of_layers - 1 else None,
                    dropout_prob=dropout,
                    add_skip_connection=add_skip_connection,
                    residual=residual
                )
            )

        self.gat_net = nn.Sequential(*gat_layers)

    def forward(self, node_features, edge_features, edge_index):
        """
        Forward pass for the model.
        :param node_features: Tensor of shape [N, F], where N is the number of nodes and F is the number of features.
        :param edge_features: Tensor of shape [E, D], where E is the number of edges and D is the edge feature dimension.
        :param edge_index: Tensor of shape [2, E], representing the edge connectivity.
        """
        data = (node_features, edge_features, edge_index)
        return self.gat_net(data)


class GATLayerWithEdgeFeatures(torch.nn.Module):
    def __init__(self, num_in_features, num_out_features, edge_feat_dim, num_heads, concat, activation, 
                 dropout_prob, add_skip_connection, residual):
        super(GATLayerWithEdgeFeatures, self).__init__()

        self.num_heads = num_heads
        self.concat = concat
        self.residual = residual
        self.dropout = nn.Dropout(p=dropout_prob)

        self.gat_conv = GATConv(
            in_channels=num_in_features,
            out_channels=num_out_features,
            heads=num_heads,
            concat=concat,
            dropout=dropout_prob
        )
        self.edge_linear = nn.Linear(edge_feat_dim, num_out_features * num_heads, bias=False)
        self.skip_connection = nn.Linear(num_in_features, num_out_features * num_heads, bias=False) \
            if add_skip_connection else None

        self.activation = activation

    def forward(self, data):
        node_features, edge_features, edge_index = data

        # Apply GAT convolution
        edge_feat_transformed = self.edge_linear(edge_features)
        edge_feat_transformed = edge_feat_transformed.view(-1, self.num_heads, edge_feat_transformed.size(-1) // self.num_heads)
        out = self.gat_conv(node_features, edge_index, edge_attr=edge_feat_transformed)

        # Add skip connection if enabled
        if self.skip_connection is not None:
            out += self.skip_connection(node_features)

        if self.activation is not None:
            out = self.activation(out)

        return out
