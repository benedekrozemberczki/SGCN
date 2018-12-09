import json
import torch
import random
import pandas as pd
import numpy as np
from tqdm import trange
from scipy import sparse
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import SAGEConv
from sklearn.decomposition import TruncatedSVD
from utils import calculate_auc
from sklearn.model_selection import train_test_split

class SignedGraphConvolutionalNetwork(torch.nn.Module):
    """
    Signed Graph Convolutional Network Class.
    For details see:

    """
    def __init__(self, device, args, X):
        super(SignedGraphConvolutionalNetwork, self).__init__()
        """
        SGCN Initialization.
        :param device: Device for calculations.
        :param args: Arguments object.
        :param X: Node features.
        """
        self.args = args
        torch.manual_seed(self.args.seed)
        self.device = device
        self.X = X
        self.nodes = range(self.X.shape[0])
        self.neurons = self.args.layers
        self.layers = len(self.neurons)
        self.positive_base_aggregator = SAGEConv(self.X.shape[1], self.neurons[0]).to(self.device)
        self.negative_base_aggregator = SAGEConv(self.X.shape[1], self.neurons[0]).to(self.device)
        if self.layers > 1:
            self.setup_additional_layers()
        self.regression_weights = Parameter(torch.Tensor(4*self.neurons[-1], 3))
        init.xavier_normal_(self.regression_weights)

    def setup_additional_layers(self):
        """
        Adding Deep Signed GraphSAGE layers if the model is not a single layer model.
        """
        self.positive_aggregators = []
        self.negative_aggregators = []
        for i in range(1,self.layers):
            self.positive_aggregators.append(SAGEConv(2*self.neurons[i-1], self.neurons[i]).to(self.device))
            self.negative_aggregators.append(SAGEConv(2*self.neurons[i-1], self.neurons[i]).to(self.device))
 

    def calculate_regression_loss(self,z, target):
        """
        Calculating the regression loss for all pairs of nodes.
        :param z: Hidden vertex representations.
        :param target: Target vector.
        :return loss_term: Regression loss.
        :return predictions_soft: Predictions for each vertex pair. 
        """
        pos = torch.cat((self.positive_z_i, self.positive_z_j),1)
        neg = torch.cat((self.negative_z_i, self.negative_z_j),1)
        surr_neg_i = torch.cat((self.negative_z_i, self.negative_z_k),1)
        surr_neg_j = torch.cat((self.negative_z_j, self.negative_z_k),1)
        surr_pos_i = torch.cat((self.positive_z_i, self.positive_z_k),1)
        surr_pos_j = torch.cat((self.positive_z_j, self.positive_z_k),1)
        features = torch.cat((pos,neg,surr_neg_i,surr_neg_j,surr_pos_i,surr_pos_j))
        predictions = torch.mm(features,self.regression_weights)
        predictions_soft = F.log_softmax(predictions, dim=1)
        loss_term = F.nll_loss(predictions_soft, target)
        return loss_term, predictions_soft        

    def calculate_positive_embedding_loss(self, z, positive_edges):
        """
        Calculating the loss on the positive edge embedding distances
        :param z: Hidden vertex representation.
        :param positive_edges: Positive training edges.
        :return loss_term: Loss value on positive edge embedding.
        """
        self.positive_surrogates = [random.choice(self.nodes) for node in range(positive_edges.shape[1])]
        self.positive_surrogates = torch.from_numpy(np.array(self.positive_surrogates, dtype=np.int64).T).type(torch.long).to(self.device)
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
        """
        Calculating the loss on the negative edge embedding distances
        :param z: Hidden vertex representation.
        :param negative_edges: Negative training edges.
        :return loss_term: Loss value on negative edge embedding.
        """
        self.negative_surrogates = [random.choice(self.nodes) for node in range(negative_edges.shape[1])]
        self.negative_surrogates = torch.from_numpy(np.array(self.negative_surrogates, dtype=np.int64).T).type(torch.long).to(self.device)
        negative_edges = torch.t(negative_edges)
        self.negative_z_i, self.negative_z_j = z[negative_edges[:,0],:], z[negative_edges[:,1],:]
        self.negative_z_k = z[self.negative_surrogates,:]
        norm_i_j = torch.norm(self.negative_z_i-self.negative_z_j, 2, 1, True).pow(2)
        norm_i_k = torch.norm(self.negative_z_i-self.negative_z_k, 2, 1, True).pow(2)
        term = norm_i_k-norm_i_j
        term[term<0] = 0
        loss_term = term.mean()
        return loss_term

    def calculate_regularization_loss(self):
        """
        Calculate the regularization of model weights.
        1. Base positive and negative SAGE embedding weights.
        2. Regression weights.
        3. Deep SAGE weights if the number of layers > 1.
        :return regul_loss: regularization loss.
        """
        regul_base_pos = torch.norm(self.positive_base_aggregator.weight,2,1,True).mean()
        regul_base_neg = torch.norm(self.negative_base_aggregator.weight,2,1,True).mean()
        regul_reg = torch.norm(self.regression_weights,2,1,True).mean()
        regularization_loss = regul_base_pos + regul_base_neg + regul_reg
        for i in range(1,self.layers):
            regularization_loss = regularization_loss + torch.norm(self.positive_aggregators[i-1].weight,2,1,True).mean()
            regularization_loss = regularization_loss + torch.norm(self.negative_aggregators[i-1].weight,2,1,True).mean()
        return regularization_loss

    def calculate_loss_function(self, z, positive_edges, negative_edges, target):
        """
        Calculating the embedding losses, regression loss and weight regularization loss.
        :param z: Node embedding.
        :param positive_edges: Positive edge pairs.
        :param negative_edges: Negative edge pairs.
        :param target: Target vector.
        :return loss: Value of loss.
        """
        loss_term_1 = self.calculate_positive_embedding_loss(z, positive_edges)
        loss_term_2 = self.calculate_negative_embedding_loss(z, negative_edges)
        regression_loss, self.predictions = self.calculate_regression_loss(z,target)
        regularization_loss = self.calculate_regularization_loss()
        loss_term = regression_loss+self.args.lamb*(loss_term_1+loss_term_2)+self.args.gamma*regularization_loss
        return loss_term

    def forward(self, positive_edges, negative_edges, target):
        """
        Model forward propagation pass. Can fit deep and single layer SGCN models.
        :param positive_edges: Positive edges.
        :param negative_edges: Negative edges.
        :param target: Target vectors.
        :return loss: Loss value.
        :return self.z: Hidden vertex representations.
        """
        self.h_pos, self.h_neg = [],[]
        self.h_pos.append(torch.tanh(self.positive_base_aggregator(self.X, positive_edges)))
        self.h_neg.append(torch.tanh(self.negative_base_aggregator(self.X, negative_edges)))
        for i in range(1,self.layers):
            self.h_pos.append(torch.tanh(self.positive_aggregators[i-1](torch.cat((self.h_pos[i-1],self.h_neg[i-1]), 1), positive_edges)))
            self.h_neg.append(torch.tanh(self.negative_aggregators[i-1](torch.cat((self.h_neg[i-1],self.h_pos[i-1]), 1), negative_edges)))
        self.z = torch.cat((self.h_pos[-1], self.h_neg[-1]), 1)
        loss = self.calculate_loss_function(self.z, positive_edges, negative_edges, target)
        return loss, self.z

class SignedGCNTrainer(object):
    """
    Object to train and score the SGCN, log the model behaviour and save the output.
    """
    def __init__(self, args, edges):
        """
        Constructing the trainer instance and setting up logs.
        :param args: Arguments object.
        :param edges: Edge data structure with positive and negative edges separated.
        """
        self.args = args
        self.edges = edges 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logs()

    def setup_logs(self):
        """
        Creating a log dictionary.
        """
        self.logs = {}
        self.logs["parameters"] =  vars(self.args)
        self.logs["performance"] = [["Epoch","AUC","F1"]]
        self.logs["losses"] = []
        self.logs["training_time"] = []

    def setup_features(self):
        
        self.p_edges = self.positive_edges + [[edge[1],edge[0]] for edge in self.positive_edges]
        self.n_edges = self.negative_edges + [[edge[1],edge[0]] for edge in self.negative_edges]
        self.train_edges = self.p_edges + self.n_edges
        self.index_1 = [edge[0] for edge in self.train_edges]
        self.index_2 = [edge[1] for edge in self.train_edges]
        self.values = [1]*len(self.p_edges) + [-1]*len(self.n_edges)
        shaping = (self.node_count,self.node_count)
        self.signed_A = sparse.csr_matrix(sparse.coo_matrix((self.values,(self.index_1,self.index_2)),shape=shaping,dtype=np.float32))
        svd = TruncatedSVD(n_components=self.args.reduction_dimensions, n_iter=self.args.reduction_iterations, random_state=self.args.seed)
        svd.fit(self.signed_A)
        return svd.components_.T

    def setup_dataset(self):
        self.positive_edges, self.test_positive_edges = train_test_split(self.edges["positive_edges"], test_size = self.args.test_size)
        self.negative_edges, self.test_negative_edges = train_test_split(self.edges["negative_edges"], test_size = self.args.test_size)
        ecount = len(self.positive_edges + self.negative_edges)
        self.node_count = self.edges["ncount"]
        self.X = self.setup_features()
        self.positive_edges = torch.from_numpy(np.array(self.positive_edges, dtype=np.int64).T).type(torch.long).to(self.device)
        self.negative_edges = torch.from_numpy(np.array(self.negative_edges, dtype=np.int64).T).type(torch.long).to(self.device)

        self.y = np.array([0 if i< int(ecount/2) else 1 for i in range(ecount)] +[2]*(ecount*2))
        self.y = torch.from_numpy(self.y).type(torch.LongTensor).to(self.device)

        self.X = torch.from_numpy(self.X).float().to(self.device)

    def create_and_train_model(self):
        """
        """
        print("\nTraining started.\n")
        self.model = SignedGraphConvolutionalNetwork(self.device, self.args, self.X).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=5e-4)
        self.model.train()
        self.epochs = trange(self.args.epochs, desc='Loss')

        for epoch in self.epochs:
            self.optimizer.zero_grad()
            loss, _ = self.model(self.positive_edges, self.negative_edges, self.y)
            loss.backward()
            self.epochs.set_description('SGCN (Loss=%g)' % round(loss.item(),4))
            self.optimizer.step()
            self.score_model(epoch)

    def score_model(self, epoch):
        """
        """
        loss, self.train_z = self.model(self.positive_edges, self.negative_edges, self.y)
        score_positive_edges = torch.from_numpy(np.array(self.test_positive_edges, dtype=np.int64).T).type(torch.long).to(self.device)
        score_negative_edges = torch.from_numpy(np.array(self.test_negative_edges, dtype=np.int64).T).type(torch.long).to(self.device)
        test_positive_z = torch.cat((self.train_z[score_positive_edges[0,:],:], self.train_z[score_positive_edges[1,:],:]),1)
        test_negative_z = torch.cat((self.train_z[score_negative_edges[0,:],:], self.train_z[score_negative_edges[1,:],:]),1)
        scores = torch.mm(torch.cat((test_positive_z, test_negative_z),0), self.model.regression_weights.to(self.device))
        probability_scores = torch.exp(F.softmax(scores, dim=1))
        predictions = probability_scores[:,0]/probability_scores[:,0:2].sum(1)
        predictions = predictions.detach().numpy()
        targets = [0]*len(self.test_positive_edges) + [1]*len(self.test_negative_edges)
        auc, f1 = calculate_auc(targets, predictions, self.edges)
        self.logs["performance"].append([epoch+1, auc, f1])
        

    def save_model(self):
        """

        """
        print("\nEmbedding being saved.\n")
        self.train_z = self.train_z.detach().numpy()
        embedding_header = ["id"] + ["x_" + str(x) for x in range(self.train_z.shape[1])]
        self.train_z = np.concatenate([np.array(range(self.train_z.shape[0])).reshape(-1,1),self.train_z],axis=1)
        self.train_z = pd.DataFrame(self.train_z, columns = embedding_header)
        self.train_z.to_csv(self.args.embedding_path, index = None)
        print("\nWeights being saved.\n")
        self.regression_weights = self.model.regression_weights.detach().numpy().T
        regression_header = ["x_" + str(x) for x in range(self.regression_weights.shape[1])]
        self.regression_weights = pd.DataFrame(self.regression_weights, columns = regression_header)
        self.regression_weights.to_csv(self.args.regression_weights_path, index = None)     
