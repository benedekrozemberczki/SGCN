import random
import torch
import random
import numpy as np
from tqdm import tqdm
import networkx as nx
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.nn import Parameter
from tqdm import trange


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

g = nx.erdos_renyi_graph(2000,0.01)
edges = g.edges()
random.shuffle(edges)
e_count = len(g.edges())


positive_edges = edges[int(len(edges)/2):]
negative_edges = edges[:int(len(edges)/2)]

positive_edges =  torch.from_numpy(np.array(positive_edges, dtype=np.int64).T).type(torch.long).to(device)
negative_edges =  torch.from_numpy(np.array(negative_edges, dtype=np.int64).T).type(torch.long).to(device)

X_1 = torch.from_numpy(np.random.uniform(0,1,(2000,128))).float().to(device)
X_2 = torch.from_numpy(np.random.uniform(0,1,(2000,128))).float().to(device)

y = np.array([0 if i< int(e_count/2) else 1 for i in range(e_count)] +[2]*(e_count*2))
y = torch.from_numpy(y).type(torch.LongTensor).to(device)
e_count = len(g.edges())
v_count = len(g.nodes())

class Net(torch.nn.Module):
    def __init__(self, graph):
        super(Net, self).__init__()
        self.graph = graph
        self.neurons = [38, 17, 40, 23]
        self.layers = len(self.neurons)
        self.positive_base_aggregator = SAGEConv(128, self.neurons[0]).to(device)
        self.negative_base_aggregator = SAGEConv(128, self.neurons[0]).to(device)
        if self.layers > 1:
            self.setup_additional_layers()
        self.regression_weights = Parameter(torch.Tensor(4*self.neurons[-1], 3))
        init.xavier_normal(self.regression_weights)
        print(self.regression_weights.shape)
        self.lam = 0.1

    def setup_additional_layers(self):
        self.positive_aggregators = []
        self.negative_aggregators = []
        for i in range(1,self.layers):
            self.positive_aggregators.append(SAGEConv(2*self.neurons[i-1], self.neurons[i]).to(device))
            self.negative_aggregators.append(SAGEConv(2*self.neurons[i-1], self.neurons[i]).to(device))
 

    def calculate_regression_loss(self,z, target):
        pos = torch.cat((self.positive_z_i, self.positive_z_j),1)
        neg = torch.cat((self.negative_z_i, self.negative_z_j),1)
        surr_neg_i = torch.cat((self.negative_z_i, self.negative_z_k),1)
        surr_neg_j = torch.cat((self.negative_z_j, self.negative_z_k),1)
        surr_pos_i = torch.cat((self.positive_z_i, self.positive_z_k),1)
        surr_pos_j = torch.cat((self.positive_z_j, self.positive_z_k),1)
        features = torch.cat((pos,neg,surr_neg_i,surr_neg_j,surr_pos_i,surr_pos_j))
        predictions = torch.mm(features,self.regression_weights)
        predictions = F.log_softmax(predictions, dim=1)
        loss_term = F.nll_loss(predictions, target)
        return loss_term        

    def calculate_positive_embedding_loss(self, z, positive_edges):
        self.positive_surrogates = [random.choice(self.graph.nodes()) for node in range(positive_edges.shape[1])]
        self.positive_surrogates = torch.from_numpy(np.array(self.positive_surrogates, dtype=np.int64).T).type(torch.long).to(device)
        
        positive_edges = torch.t(positive_edges)
        self.positive_z_i, self.positive_z_j = z[positive_edges[:,0],:],z[positive_edges[:,1],:]
        self.positive_z_k = z[self.positive_surrogates,:]
        norm_i_j = torch.norm(self.positive_z_i-self.positive_z_j, 2, 1, True).pow(2)
        norm_i_k = torch.norm(self.positive_z_i-self.positive_z_k, 2, 1, True).pow(2)
        term = norm_i_j-norm_i_k
        term[term<0] = 0
        loss_term = term.mean()
        return loss_term

    def calculate_negative_embedding_loss(self, z, negative_edges):
        self.negative_surrogates = [random.choice(self.graph.nodes()) for node in range(negative_edges.shape[1])]
        self.negative_surrogates = torch.from_numpy(np.array(self.negative_surrogates, dtype=np.int64).T).type(torch.long).to(device)
        
        negative_edges = torch.t(negative_edges)
        self.negative_z_i, self.negative_z_j = z[negative_edges[:,0],:], z[negative_edges[:,1],:]
        self.negative_z_k = z[self.negative_surrogates,:]
        norm_i_j = torch.norm(self.negative_z_i-self.negative_z_j, 2, 1, True).pow(2)
        norm_i_k = torch.norm(self.negative_z_i-self.negative_z_k, 2, 1, True).pow(2)
        term = norm_i_k-norm_i_j
        term[term<0] = 0
        loss_term = term.mean()
        return loss_term

    def calculate_loss_function(self, z, positive_edges, negative_edges, target):
        loss_term_1 = self.calculate_positive_embedding_loss(z, positive_edges)
        loss_term_2 = self.calculate_negative_embedding_loss(z, negative_edges)
        regression_loss = self.calculate_regression_loss(z,target)
       
        loss_term = regression_loss + self.lam*(loss_term_1+ loss_term_2)
        return loss_term

    def forward(self, data_1, data_2, positive_edges, negative_edges,target):
        self.h_pos, self.h_neg = [],[]
        self.h_pos.append(torch.sigmoid(self.positive_base_aggregator(data_1, positive_edges)))
        self.h_neg.append(torch.sigmoid(self.negative_base_aggregator(data_2, negative_edges)))
        for i in range(1,self.layers):
            print(i)
            self.h_pos.append(torch.sigmoid(self.positive_aggregators[i-1](torch.cat((self.h_pos[i-1],self.h_neg[i-1]), 1), positive_edges)))
            self.h_neg.append(torch.sigmoid(self.negative_aggregators[i-1](torch.cat((self.h_neg[i-1],self.h_pos[i-1]), 1), negative_edges)))
        z = torch.cat((self.h_pos[-1],self.h_neg[-1]), 1)
        loss = self.calculate_loss_function(z, positive_edges, negative_edges,target)

            
        return loss


model = Net(g).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

model.train()

t = trange(20, desc='ML')

for epoch in t:
    optimizer.zero_grad()
    loss = model(X_1, X_2, positive_edges, negative_edges,y).cuda()
    loss.backward()
    t.set_description('ML (loss=%g)' % round(loss.item(),4))
    optimizer.step()

