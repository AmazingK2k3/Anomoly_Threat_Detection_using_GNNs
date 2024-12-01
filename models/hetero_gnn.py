import torch
import torch.nn.functional as F
from torch_geometric.nn import Linear, HeteroConv, SAGEConv

class HeteroGNN(torch.nn.Module):
    def __init__(self, dim_host, dim_flow, dim_h, dim_out, num_layers):
        """
        HeteroGNN Model for Host and Flow Nodes

        Parameters:
        - dim_host: Dimension of initial host node features
        - dim_flow: Dimension of initial flow node features
        - dim_h: Hidden layer dimension
        - dim_out: Output layer dimension (number of classes)
        - num_layers: Number of message-passing layers
        """
        super().__init__()

        # Input projections for node types
        self.host_proj = Linear(dim_host, dim_h)  # Projection for host node features
        self.flow_proj = Linear(dim_flow, dim_h)  # Projection for flow node features

        # HeteroConv layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('host', 'to', 'flow'): SAGEConv((-1, -1), dim_h, add_self_loops=False),
                ('flow', 'to', 'host'): SAGEConv((-1, -1), dim_h, add_self_loops=False),
            }, aggr='sum')
            self.convs.append(conv)

        # Readout (classification) layer for flow nodes
        self.lin = Linear(dim_h, dim_out)

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass of the HeteroGNN model.

        Parameters:
        - x_dict: Dictionary of node features for each node type
        - edge_index_dict: Dictionary of edge indices for each edge type

        Returns:
        - Output logits for 'flow' nodes
        """
        # Project initial features to hidden dimension
        x_dict['host'] = F.relu(self.host_proj(x_dict['host']))
        x_dict['flow'] = F.relu(self.flow_proj(x_dict['flow']))

        # Apply message-passing layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        # Final classification on 'flow' nodes
        return self.lin(x_dict['flow'])
