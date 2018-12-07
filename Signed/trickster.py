from torch_geometric.datasets import Planetoid
from tqdm import tqdm
dataset = Planetoid(root='/tmp/Cora', name='Cora')
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(128, 32)
        self.conv2 = SAGEConv(32, 10)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        y = np.array(data.edge_index)
        x = self.conv1(x, edge_index)
        print(y.shape)
        
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)

model.train()
for epoch in tqdm(range(200)):
    optimizer.zero_grad()
    out = model(data)
    print(np.array(data.y).shape)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))
