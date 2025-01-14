import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv

import networkx as nx
import numpy as np
from utils.edge_reader import read_edges_from_file
from subgraphs.top_k import top_k_overlapping_densest_subgraphs
from utils.plot.plot import plot_save_graph, plot_save_subgraphs
from utils.plot.hypergraph_plot import plot_save_hypergraph
from hypergraph.hypergraph import graph_to_hypergraph

import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from tqdm import tqdm
from sklearn import svm
import warnings


class HyperGraphConvNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, alpha):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.input = nn.Linear(self.input_size, self.hidden_size[0], bias=False)
        self.alpha = alpha
        self.hiddens = nn.ModuleList([
            HypergraphConv(self.hidden_size[h], self.hidden_size[h + 1])
            for h in range(len(self.hidden_size) - 1)
        ])
        self.output = nn.Linear(hidden_size[-1], output_size)

    def forward(self, x, hyperedge_index):
        """
        Forward pass for the hypergraph convolution network.

        Args:
            x (torch.Tensor): Input feature matrix.
            hyperedge_index (torch.Tensor): Hyperedge index tensor.

        Returns:
            torch.Tensor: Output predictions.
        """
        # Step 1: Input transformation
        x = self.input(x)
        x = self.relu(x)

        # Step 2: Pass through hidden layers with hypergraph convolution
        for hidden in self.hiddens:
            x = hidden(x, hyperedge_index)
            x = self.relu(x)

        # Step 3: Output layer
        x = self.output(x)
        x = self.softmax(x)

        return x


    def create_hyperedges(self, x):
        # Create the similarity graph
        similarity_matrix = torch.abs(F.cosine_similarity(x[..., None, :, :], x[..., :, None, :], dim=-1))
        similarity = torch.sort(similarity_matrix.view(-1))[0]
        eps = torch.quantile(similarity, self.alpha, interpolation='nearest')
        adj_matrix = similarity_matrix >= eps

        # Use your own algorithm to convert similarity graph into a hypergraph
        G = nx.from_numpy_array(adj_matrix.numpy())
        
        # Parameters
        k = 3
        lambda_param = 0.5
        min_subset_size = 10
        max_subset_size = 25
        k_hop = 3

        # Find the top-k overlapping densest subgraphs
        subgraphs = top_k_overlapping_densest_subgraphs(G, k, lambda_param, min_subset_size, max_subset_size, k_hop)
    
        # Instantiating a dictionary for assing nodes to each subgraph
        hyper_dic = {}

        print(f"Found {len(subgraphs)} subgraphs:")
        for i, sg in enumerate(subgraphs, 1):
            print(f"Subgraph {i}: Nodes = {sg.nodes()}, Edges = {sg.edges()}")
            hyper_dic[f"Subgraph {i}"] = set(sg.nodes())
   
        hyperedges = []
        # Iterate over each subgraph in hyper_dic
        for subgraph_name, nodes in hyper_dic.items():
            # Convert the set of nodes to a list and append as a hyperedge
            hyperedges.append(list(nodes))
        
        hyperedges = self.similarity_to_hypergraph(adj_matrix)
        return hyperedges

    def similarity_to_hypergraph(self, adj_matrix):
        # Example: Treat each row of adj_matrix as a hyperedge
        row_indices = torch.where(adj_matrix)[0].unique()
        hyperedges = []
        num_nodes = adj_matrix.size(0)  # Total number of nodes

        for row in row_indices:
            members = torch.where(adj_matrix[row])[0]
            if all(member < num_nodes for member in members):
                hyperedges.append(members)

        # Convert hyperedges into PyTorch Geometric format
        # Each column in hyperedge_index corresponds to one hyperedge
        hyperedge_index = torch.cat([torch.stack((members, torch.full_like(members, row))) for row, members in enumerate(hyperedges)], dim=1)
        return hyperedge_index


    

class GRACES:
    def __init__(self, n_features, hidden_size=None, q=2, n_dropouts=10, dropout_prob=0.5, batch_size=16,
                 learning_rate=0.001, epochs=50, alpha=0.95, sigma=0, f_correct=0):
        self.n_features = n_features
        self.q = q
        if hidden_size is None:
            self.hidden_size = [64, 32]
        else:
            self.hidden_size = hidden_size
        self.n_dropouts = n_dropouts
        self.dropout_prob = dropout_prob
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.sigma = sigma
        self.f_correct = f_correct
        self.S = None
        self.new = None
        self.model = None
        self.last_model = None
        self.loss_fn = None
        self.f_scores = None

    @staticmethod
    def bias(x):
        if not all(x[:, 0] == 1):
            x = torch.cat((torch.ones(x.shape[0], 1), x.float()), dim=1)
        return x

    def xavier_initialization(self):
        if self.last_model is not None:
            weight = torch.zeros(self.hidden_size[0], len(self.S))
            nn.init.xavier_normal_(weight, gain=nn.init.calculate_gain('relu'))
            old_s = self.S.copy()
            if self.new in old_s:
                old_s.remove(self.new)
            for i in self.S:
                if i != self.new:
                    weight[:, self.S.index(i)] = self.last_model.input.weight.data[:, old_s.index(i)]
            self.model.input.weight.data = weight
            for h in range(len(self.hidden_size) - 1):
                self.model.hiddens[h].weight.data = self.last_model.hiddens[h].weight.data
            self.model.output.weight.data = self.last_model.output.weight.data

    def train(self, x, y):
        input_size = len(self.S)
        output_size = len(torch.unique(y))
        self.model = HyperGraphConvNet(input_size, output_size, self.hidden_size, self.alpha)
        self.xavier_initialization()
        x = x[:, self.S]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Generate hyperedges for training
        hyperedge_index = self.model.create_hyperedges(x)

        train_set = []
        for i in range(x.shape[0]):
            train_set.append([x[i, :], y[i]])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        for e in range(self.epochs):
            for data, label in train_loader:
                input_0 = data.view(data.shape[0], -1)
                optimizer.zero_grad()
                output = self.model(input_0.float(), hyperedge_index)
                loss = self.loss_fn(output, label)
                loss.backward()
                optimizer.step()
        self.last_model = copy.deepcopy(self.model)
        
    def f_test(self, x, y):
        """
        Perform an F-test for feature selection.

        Args:
            x (numpy.ndarray or torch.Tensor): Feature matrix.
            y (numpy.ndarray or torch.Tensor): Target labels.

        Returns:
            numpy.ndarray: F-scores for each feature.
        """
        if isinstance(x, torch.Tensor):
            x = x.numpy()  # Convert to numpy if input is a torch.Tensor
        if isinstance(y, torch.Tensor):
            y = y.numpy()  # Convert to numpy if input is a torch.Tensor
        
        # Use SelectKBest with f_classif to compute F-scores
        slc = SelectKBest(f_classif, k=x.shape[1])  # k is set to the number of features
        slc.fit(x, y)  # Fit the selector to the data
        
        # Return the computed F-scores
        return getattr(slc, 'scores_')

    def dropout(self):
        model_dp = copy.deepcopy(self.model)
        for h in range(len(self.hidden_size) - 1):
            h_size = self.hidden_size[h]
            dropout_index = np.random.choice(range(h_size), int(h_size * self.dropout_prob), replace=False)
            model_dp.hiddens[h].weight.data[:, dropout_index] = torch.zeros(model_dp.hiddens[h].weight[:, dropout_index].shape)
        dropout_index = np.random.choice(range(self.hidden_size[-1]), int(self.hidden_size[-1] * self.dropout_prob), replace=False)
        model_dp.output.weight.data[:, dropout_index] = torch.zeros(model_dp.output.weight[:, dropout_index].shape)
        return model_dp

    def gradient(self, x, y, model):
        model_gr = HyperGraphConvNet(x.shape[1], len(torch.unique(y)), self.hidden_size, self.alpha)
        temp = torch.zeros(model_gr.input.weight.shape)
        temp[:, self.S] = model.input.weight
        model_gr.input.weight.data = temp
        for h in range(len(self.hidden_size) - 1):
            model_gr.hiddens[h].weight.data = model.hiddens[h].weight + self.sigma * torch.randn(model.hiddens[h].weight.shape)
        model_gr.output.weight.data = model.output.weight
        hyperedge_index = model_gr.create_hyperedges(x)
        output_gr = model_gr(x.float(), hyperedge_index)
        loss_gr = self.loss_fn(output_gr, y)
        loss_gr.backward()
        input_gradient = model_gr.input.weight.grad
        return input_gradient

    def average(self, x, y, n_average):
        grad_cache = None
        for num in range(n_average):
            model = self.dropout()
            input_grad = self.gradient(x, y, model)
            if grad_cache is None:
                grad_cache = input_grad
            else:
                grad_cache += input_grad
        return grad_cache / n_average

    def find(self, input_gradient):
        gradient_norm = input_gradient.norm(p=self.q, dim=0)
        gradient_norm = gradient_norm / gradient_norm.norm(p=2)
        gradient_norm[1:] = (1 - self.f_correct) * gradient_norm[1:] + self.f_correct * self.f_scores
        gradient_norm[self.S] = 0
        max_index = torch.argmax(gradient_norm)
        return max_index.item()
    
    def select(self, x, y):
        x = torch.tensor(x)
        y = torch.tensor(y)
        self.f_scores = torch.tensor(self.f_test(x, y))
        self.f_scores[torch.isnan(self.f_scores)] = 0
        self.f_scores = self.f_scores / self.f_scores.norm(p=2)
        x = self.bias(x)
        self.S = [0]
        self.loss_fn = nn.CrossEntropyLoss()
        while len(self.S) < self.n_features + 1:
            self.train(x, y)
            input_gradient = self.average(x, y, self.n_dropouts)
            self.new = self.find(input_gradient)
            self.S.append(self.new)
        selection = self.S
        selection.remove(0)
        selection = [s - 1 for s in selection]
        return selection

    
    
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn import svm
import warnings

warnings.filterwarnings("ignore")

# Grid search range on key hyperparameters
dropout_prob = [0.1, 0.5, 0.75]
f_correct = [0, 0.1, 0.5, 0.9]

def main(name, n_features, n_iters, n_repeats):
    np.random.seed(0)  # For reproducibility
    data = scipy.io.loadmat('data/' + name)
    x = data['X'].astype(float)
    if name == 'colon' or name == 'leukemia':
        y = np.int64(data['Y'])
        y[y == -1] = 0
    else:
        y = np.int64(data['Y']) - 1
    y = y.reshape(-1)

    auc_test = np.zeros(n_iters)
    seeds = np.random.choice(range(100), n_iters, replace=False)  # For reproducibility
    for iter in tqdm(range(n_iters)):
        # Split data into training, validation, and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=seeds[iter], stratify=y)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=2/7, random_state=seeds[iter], stratify=y_train)
        
        auc_grid = np.zeros((len(dropout_prob), len(f_correct)))
        loss_grid = np.zeros((len(dropout_prob), len(f_correct)))
        
        for i in range(len(dropout_prob)):
            for j in range(len(f_correct)):
                for r in range(n_repeats):
                    # Initialize GRACES with hypergraph support
                    slc_g = GRACES(n_features=n_features, dropout_prob=dropout_prob[i], f_correct=f_correct[j])
                    selection_g = slc_g.select(x_train, y_train)  # Hypergraph-based feature selection
                    
                    # Reduce features for training and validation
                    x_train_red_g = x_train[:, selection_g]
                    x_val_red = x_val[:, selection_g]
                    
                    # Train and evaluate an SVM classifier
                    clf_g = svm.SVC(probability=True)
                    clf_g.fit(x_train_red_g, y_train)
                    y_val_pred = clf_g.predict_proba(x_val_red)
                    
                    # Compute AUROC and cross-entropy loss
                    auc_grid[i, j] += roc_auc_score(y_val, y_val_pred[:, 1])
                    loss_grid[i, j] += -np.sum(y_val * np.log(y_val_pred[:, 1]))
        
        # Find the best hyperparameters
        index_i, index_j = np.where(auc_grid == np.max(auc_grid))
        best_index = np.argmin(loss_grid[index_i, index_j])  # Break tie based on cross-entropy loss
        best_prob, best_f_correct = dropout_prob[int(index_i[best_index])], f_correct[int(index_j[best_index])]
        
        for r in range(n_repeats):
            # Final selection and testing
            slc = GRACES(n_features=n_features, dropout_prob=best_prob, f_correct=best_f_correct)
            selection = slc.select(x_train, y_train)
            x_train_red = x_train[:, selection]
            x_test_red = x_test[:, selection]
            
            # Train and test SVM classifier
            clf = svm.SVC(probability=True)
            clf.fit(x_train_red, y_train)
            y_test_pred = clf.predict_proba(x_test_red)
            auc_test[iter] += roc_auc_score(y_test, y_test_pred[:, 1])
    
    return auc_test / n_repeats

if __name__ == "__main__":
    name = 'Prostate_GE'
    max_features = 10
    n_iters = 20
    n_repeats = 3
    results = np.zeros((max_features, n_iters))
    
    for p in range(max_features):
        results[p, :] = main(name=name, n_features=p+1, n_iters=n_iters, n_repeats=n_repeats)
    
    print('Average test AUROC:', np.mean(results, axis=1))

