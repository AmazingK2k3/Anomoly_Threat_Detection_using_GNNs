import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_mean_pool

class CustomGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomGNN, self).__init__()
        # Edge-specific message functions
        self.msg_sf = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.msg_fd = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GRU-based update functions
        self.gru_h = nn.GRUCell(hidden_dim, hidden_dim)
        self.gru_f = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Readout function
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index, edge_type):
        # x: Node features [num_nodes, input_dim]
        # edge_index: [2, num_edges]
        # edge_type: Edge type (0 for sf, 1 for fd)
        
        hidden_states = x
        for _ in range(3):  # Number of message-passing steps
            messages = torch.zeros_like(hidden_states)
            
            # Compute messages for each edge type
            for edge_type_id, msg_func in enumerate([self.msg_sf, self.msg_fd]):
                edge_mask = (edge_type == edge_type_id)
                source_nodes = edge_index[0, edge_mask]
                target_nodes = edge_index[1, edge_mask]
                edge_features = torch.cat(
                    [hidden_states[source_nodes], hidden_states[target_nodes]], dim=-1
                )
                messages[target_nodes] += msg_func(edge_features)
            
            # Aggregate messages
            aggregated_messages = messages.mean(dim=1)
            
            # Update hidden states using GRUs
            hidden_states = self.gru_h(aggregated_messages, hidden_states)
        
        # Apply readout
        out = self.readout(hidden_states)
        return F.softmax(out, dim=1)
