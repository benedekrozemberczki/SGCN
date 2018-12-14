import math
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.utils import remove_self_loops, add_self_loops

def uniform(size, tensor):
    """
    """
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

class SignedSAGEConvolution(torch.nn.Module):
    """
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm=True,
                 norm_embed=True,
                 bias=True):
        super(SignedSAGEConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = norm
        self.norm_embed = norm_embed
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class SignedSAGEConvolutionBase(SignedSAGEConvolution):
    """
    """
    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index, None)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        row, col = edge_index

        if self.norm:
            out = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))
        else:
            out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))


        out = torch.cat((out,x),1)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            out = out + self.bias

        if self.norm_embed:
            out = F.normalize(out, p=2, dim=-1)

        return out

class SignedSAGEConvolutionDeep(SignedSAGEConvolution):
    """
    """
    def forward(self, x_1, x_2, edge_index_pos, edge_index_neg):

        edge_index_pos, _ = remove_self_loops(edge_index_pos, None)
        edge_index_pos = add_self_loops(edge_index_pos, num_nodes=x_1.size(0))
        edge_index_neg, _ = remove_self_loops(edge_index_neg, None)
        edge_index_neg = add_self_loops(edge_index_neg, num_nodes=x_2.size(0))

        row_pos, col_pos = edge_index_pos
        row_neg, col_neg = edge_index_neg
        
        if self.norm:
            out_1 = scatter_mean(x_1[col_pos], row_pos, dim=0, dim_size=x_1.size(0))
            out_2 = scatter_mean(x_2[col_neg], row_neg, dim=0, dim_size=x_2.size(0))
        else:
            out_1 = scatter_add(x_1[col_pos], row_pos, dim=0, dim_size=x_1.size(0))
            out_2 = scatter_add(x_2[col_neg], row_neg, dim=0, dim_size=x_2.size(0))

        out = torch.cat((out_1,out_2,x_1),1)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            out = out + self.bias

        if self.norm_embed:
            out = F.normalize(out, p=2, dim=-1)

        return out
