from argparse import ArgumentParser
import os
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import networkx as nx
from torch.utils.data import random_split, DataLoader, Dataset
from scipy.spatial import Delaunay
from torch_geometric.datasets import QM9
from rdkit import Chem
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_networkx

from util.eval_helper import degree_stats, orbit_stats_all, clustering_stats, spectral_stats, eigval_stats, compute_list_eigh, spectral_filter_stats
from util.molecular_eval import BasicMolecularMetrics

N_MAX = 36 # This is only used as a default values for args to keep the default consistent. Commandline flag overwrites it.

class TreeDataset(Dataset):

    def __init__(self, n_nodes, n_graphs, k, same_sample=False, SON=False, ignore_first_eigv=False):
        filename = f'data/trees_{n_nodes}_{n_graphs}{"_same_sample" if same_sample else ""}.pt'
        self.k = k
        self.ignore_first_eigv = ignore_first_eigv
        if os.path.isfile(filename):
            self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample , self.n_max = torch.load(filename)
            print(f'Dataset {filename} loaded from file')
        else:
            self.adjs = []
            self.eigvals = []
            self.eigvecs = []
            self.n_nodes = []
            self.max_eigval = 0
            self.min_eigval = 0
            self.same_sample = same_sample
            for i in range(n_graphs):
                G = nx.random_tree(n_nodes)
                adj = torch.from_numpy(nx.adjacency_matrix(G).toarray()).float()
                L = nx.normalized_laplacian_matrix(G).toarray()
                L = torch.from_numpy(L).float()
                eigval, eigvec = torch.linalg.eigh(L)
                self.eigvals.append(eigval)
                self.eigvecs.append(eigvec)
                self.adjs.append(adj)
                self.n_nodes.append(len(G.nodes()))
                max_eigval = torch.max(eigval)
                if max_eigval > self.max_eigval:
                    self.max_eigval = max_eigval
                min_eigval = torch.min(eigval)
                if min_eigval < self.min_eigval:
                    self.min_eigval = min_eigval
            self.n_max = n_nodes
            torch.save([self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max], filename)
            print(f'Dataset {filename} saved')

        self.max_k_eigval = 0
        for eigv in self.eigvals:
            if eigv[self.k] > self.max_k_eigval:
                self.max_k_eigval = eigv[self.k].item()

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        if self.same_sample:
            idx = self.__len__() - 1
        graph = {}
        graph["n_nodes"] = self.n_nodes[idx]
        size_diff = self.n_max - graph["n_nodes"]
        graph["adj"] = F.pad(self.adjs[idx], [0, size_diff, 0, size_diff])
        eigvals = self.eigvals[idx]
        eigvecs = self.eigvecs[idx]
        if self.ignore_first_eigv:
            eigvals = eigvals[1:]
            eigvecs = eigvecs[:,1:]
            size_diff += 1
        graph["eigval"] = F.pad(eigvals, [0, max(0, self.n_max - eigvals.size(0))])
        graph["eigvec"] = F.pad(eigvecs, [0, size_diff, 0, size_diff])
        graph["mask"] = F.pad(torch.ones_like(self.adjs[idx]), [0, size_diff, 0, size_diff]).long()
        return graph

class GridDataset(Dataset):

    def __init__(self, grid_start, grid_end, k, same_sample=False, SON=False, ignore_first_eigv=False):
        filename = f'data/grids_{grid_start}_{grid_end}{"_same_sample" if same_sample else ""}.pt'
        self.k = k
        self.ignore_first_eigv = ignore_first_eigv
        if os.path.isfile(filename):
            self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max = torch.load(filename)
            print(f'Dataset {filename} loaded from file')
        else:
            self.adjs = []
            self.eigvals = []
            self.eigvecs = []
            self.n_nodes = []
            self.max_eigval = 0
            self.min_eigval = 0
            self.same_sample = same_sample
            for i in range(grid_start, grid_end):
                for j in range(grid_start, grid_end):
                    G = nx.grid_2d_graph(i, j)
                    adj = torch.from_numpy(nx.adjacency_matrix(G).toarray()).float()
                    L = nx.normalized_laplacian_matrix(G).toarray()
                    L = torch.from_numpy(L).float()
                    eigval, eigvec = torch.linalg.eigh(L)
                    self.eigvals.append(eigval)
                    self.eigvecs.append(eigvec)
                    self.adjs.append(adj)
                    self.n_nodes.append(len(G.nodes()))
                    max_eigval = torch.max(eigval)
                    if max_eigval > self.max_eigval:
                        self.max_eigval = max_eigval
                    min_eigval = torch.min(eigval)
                    if min_eigval < self.min_eigval:
                        self.min_eigval = min_eigval
            self.n_max = (grid_end - 1) * (grid_end - 1)
            torch.save([self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max], filename)
            print(f'Dataset {filename} saved')

        self.max_k_eigval = 0
        for eigv in self.eigvals:
            if eigv[self.k] > self.max_k_eigval:
                self.max_k_eigval = eigv[self.k].item()

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        if self.same_sample:
            idx = self.__len__() - 1
        graph = {}
        graph["n_nodes"] = self.n_nodes[idx]
        size_diff = self.n_max - graph["n_nodes"]
        graph["adj"] = F.pad(self.adjs[idx], [0, size_diff, 0, size_diff])
        eigvals = self.eigvals[idx]
        eigvecs = self.eigvecs[idx]
        if self.ignore_first_eigv:
            eigvals = eigvals[1:]
            eigvecs = eigvecs[:,1:]
            size_diff += 1
        graph["eigval"] = F.pad(eigvals, [0, max(0, self.n_max - eigvals.size(0))])
        graph["eigvec"] = F.pad(eigvecs, [0, size_diff, 0, size_diff])
        graph["mask"] = F.pad(torch.ones_like(self.adjs[idx]), [0, size_diff, 0, size_diff]).long()
        return graph

class GridDatasetNonIso(Dataset):

    def __init__(self, grid_start, grid_end, k, same_sample=False, SON=False, ignore_first_eigv=False):
        filename = f'data/grids_non_isomorphic_{grid_start}_{grid_end}{"_same_sample" if same_sample else ""}.pt'
        self.k = k
        self.ignore_first_eigv = ignore_first_eigv
        if os.path.isfile(filename):
            self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max = torch.load(filename)
            print(f'Dataset {filename} loaded from file')
        else:
            self.adjs = []
            self.eigvals = []
            self.eigvecs = []
            self.n_nodes = []
            self.max_eigval = 0
            self.min_eigval = 0
            self.same_sample = same_sample
            for i in range(grid_start, grid_end):
                for j in range(i, grid_end): # Do not include isomorphic grids (unlike GraphRNN and GRAN)
                    G = nx.grid_2d_graph(i, j)
                    adj = torch.from_numpy(nx.adjacency_matrix(G).toarray()).float()
                    L = nx.normalized_laplacian_matrix(G).toarray()
                    L = torch.from_numpy(L).float()
                    eigval, eigvec = torch.linalg.eigh(L)
                    
                    self.eigvals.append(eigval)
                    self.eigvecs.append(eigvec)
                    self.adjs.append(adj)
                    self.n_nodes.append(len(G.nodes()))
                    max_eigval = torch.max(eigval)
                    if max_eigval > self.max_eigval:
                        self.max_eigval = max_eigval
                    min_eigval = torch.min(eigval)
                    if min_eigval < self.min_eigval:
                        self.min_eigval = min_eigval
            self.n_max = (grid_end - 1) * (grid_end - 1)
            torch.save([self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max], filename)
            print(f'Dataset {filename} saved')
        
        self.max_k_eigval = 0
        for eigv in self.eigvals:
            if eigv[self.k] > self.max_k_eigval:
                self.max_k_eigval = eigv[self.k].item()

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        if self.same_sample:
            idx = self.__len__() - 1
        graph = {}
        graph["n_nodes"] = self.n_nodes[idx]
        size_diff = self.n_max - graph["n_nodes"]
        graph["adj"] = F.pad(self.adjs[idx], [0, size_diff, 0, size_diff])
        eigvals = self.eigvals[idx]
        eigvecs = self.eigvecs[idx]
        if self.ignore_first_eigv:
            eigvals = eigvals[1:]
            eigvecs = eigvecs[:,1:]
            size_diff += 1
        graph["eigval"] = F.pad(eigvals, [0, max(0, self.n_max - eigvals.size(0))])
        graph["eigvec"] = F.pad(eigvecs, [0, size_diff, 0, size_diff])
        graph["mask"] = F.pad(torch.ones_like(self.adjs[idx]), [0, size_diff, 0, size_diff]).long()
        return graph

class SBMDataset(Dataset):

    def __init__(self, n_graphs, k, same_sample=False, SON=False, ignore_first_eigv=False):
        filename = f'data/sbm_{n_graphs}{"_same_sample" if same_sample else ""}.pt'
        self.k = k
        self.ignore_first_eigv = ignore_first_eigv
        if os.path.isfile(filename):
            self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max = torch.load(filename)
            print(f'Dataset {filename} loaded from file')
        else:
            self.adjs = []
            self.eigvals = []
            self.eigvecs = []
            self.n_nodes = []
            self.max_eigval = 0
            self.min_eigval = 0
            self.same_sample = same_sample
            for seed in range(n_graphs):
                n_comunities = np.random.random_integers(2, 5)
                comunity_sizes = np.random.random_integers(20, 40, size=n_comunities)
                probs = np.ones([n_comunities, n_comunities]) * 0.005
                probs[np.arange(n_comunities), np.arange(n_comunities)] = 0.3
                G = nx.stochastic_block_model(comunity_sizes, probs, seed=seed)
                adj = torch.from_numpy(nx.adjacency_matrix(G).toarray()).float()
                L = nx.normalized_laplacian_matrix(G).toarray()
                L = torch.from_numpy(L).float()
                eigval, eigvec = torch.linalg.eigh(L)
                
                self.eigvals.append(eigval)
                self.eigvecs.append(eigvec)
                self.adjs.append(adj)
                self.n_nodes.append(len(G.nodes()))
                max_eigval = torch.max(eigval)
                if max_eigval > self.max_eigval:
                    self.max_eigval = max_eigval
                min_eigval = torch.min(eigval)
                if min_eigval < self.min_eigval:
                    self.min_eigval = min_eigval
            self.n_max = max(self.n_nodes)
            torch.save([self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max], filename)
            print(f'Dataset {filename} saved')

        self.max_k_eigval = 0
        for eigv in self.eigvals:
            if eigv[self.k] > self.max_k_eigval:
                self.max_k_eigval = eigv[self.k].item()

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        if self.same_sample:
            idx = self.__len__() - 1
        graph = {}
        graph["n_nodes"] = self.n_nodes[idx]
        size_diff = self.n_max - graph["n_nodes"]
        graph["adj"] = F.pad(self.adjs[idx], [0, size_diff, 0, size_diff])
        eigvals = self.eigvals[idx]
        eigvecs = self.eigvecs[idx]
        if self.ignore_first_eigv:
            eigvals = eigvals[1:]
            eigvecs = eigvecs[:,1:]
            size_diff += 1
        graph["eigval"] = F.pad(eigvals, [0, max(0, self.n_max - eigvals.size(0))])
        graph["eigvec"] = F.pad(eigvecs, [0, size_diff, 0, size_diff])
        graph["mask"] = F.pad(torch.ones_like(self.adjs[idx]), [0, size_diff, 0, size_diff]).long()
        return graph

class LobsterDataset(Dataset):
    """ From https://github.com/lrjconan/GRAN/blob/master/utils/data_helper.py#L169 """

    def __init__(self, n_graphs, k, same_sample=False, SON=False, ignore_first_eigv=False):
        filename = f'data/lobsters_{n_graphs}{"_same_sample" if same_sample else ""}.pt'
        self.k = k
        self.ignore_first_eigv = ignore_first_eigv
        if os.path.isfile(filename):
            self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max = torch.load(filename)
            print(f'Dataset {filename} loaded from file')
        else:
            self.adjs = []
            self.eigvals = []
            self.eigvecs = []
            self.n_nodes = []
            self.max_eigval = 0
            self.min_eigval = 0
            self.same_sample = same_sample

            p1 = 0.7
            p2 = 0.7
            count = 0
            min_node = 10
            max_node = 100
            mean_node = 80

            seed = 1234
            while count < n_graphs:
                G = nx.random_lobster(mean_node, p1, p2, seed=seed)
                if len(G.nodes()) >= min_node and len(G.nodes()) <= max_node:
                    adj = torch.from_numpy(nx.adjacency_matrix(G).toarray()).float()
                    L = nx.normalized_laplacian_matrix(G).toarray()
                    L = torch.from_numpy(L).float()
                    eigval, eigvec = torch.linalg.eigh(L)
                    self.eigvals.append(eigval)
                    self.eigvecs.append(eigvec)
                    self.adjs.append(adj)
                    self.n_nodes.append(len(G.nodes()))
                    max_eigval = torch.max(eigval)
                    if max_eigval > self.max_eigval:
                        self.max_eigval = max_eigval
                    min_eigval = torch.min(eigval)
                    if min_eigval < self.min_eigval:
                        self.min_eigval = min_eigval
                    count += 1
                seed += 1
            self.n_max = max_node
            torch.save([self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max], filename)
            print(f'Dataset {filename} saved')

        self.max_k_eigval = 0
        for eigv in self.eigvals:
            last_idx = self.k if self.k < len(eigv) else len(eigv) - 1
            if eigv[last_idx] > self.max_k_eigval:
                self.max_k_eigval = eigv[last_idx].item()

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        if self.same_sample:
            idx = self.__len__() - 1
        graph = {}
        graph["n_nodes"] = self.n_nodes[idx]
        size_diff = self.n_max - graph["n_nodes"]
        graph["adj"] = F.pad(self.adjs[idx], [0, size_diff, 0, size_diff])
        eigvals = self.eigvals[idx]
        eigvecs = self.eigvecs[idx]
        if self.ignore_first_eigv:
            eigvals = eigvals[1:]
            eigvecs = eigvecs[:,1:]
            size_diff += 1
        graph["eigval"] = F.pad(eigvals, [0, max(0, self.n_max - eigvals.size(0))])
        graph["eigvec"] = F.pad(eigvecs, [0, size_diff, 0, size_diff])
        graph["mask"] = F.pad(torch.ones_like(self.adjs[idx]), [0, size_diff, 0, size_diff]).long()
        return graph

class ProteinDataset(Dataset):
    """ Based on https://github.com/lrjconan/GRAN/blob/master/utils/data_helper.py#L192 """

    def __init__(self, k, same_sample=False, SON=False, ignore_first_eigv=False):
        min_num_nodes=100
        max_num_nodes=500
        filename = f'data/proteins_{min_num_nodes}_{max_num_nodes}{"_same_sample" if same_sample else ""}.pt'
        self.k = k
        self.ignore_first_eigv = ignore_first_eigv
        if os.path.isfile(filename):
            self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max = torch.load(filename)
            print(f'Dataset {filename} loaded from file')
        else:
            self.adjs = []
            self.eigvals = []
            self.eigvecs = []
            self.n_nodes = []
            self.n_max = 0
            self.max_eigval = 0
            self.min_eigval = 0
            self.same_sample = same_sample

            G = nx.Graph()
            data_dir = 'data'
            # Load data
            path = os.path.join(data_dir, 'DD')
            data_adj = np.loadtxt(os.path.join(path, 'DD_A.txt'), delimiter=',').astype(int)
            data_node_label = np.loadtxt(os.path.join(path, 'DD_node_labels.txt'), delimiter=',').astype(int)
            data_graph_indicator = np.loadtxt(os.path.join(path, 'DD_graph_indicator.txt'), delimiter=',').astype(int)
            data_graph_types = np.loadtxt(os.path.join(path, 'DD_graph_labels.txt'), delimiter=',').astype(int)

            data_tuple = list(map(tuple, data_adj))

            # Add edges
            G.add_edges_from(data_tuple)
            G.remove_nodes_from(list(nx.isolates(G)))

            # remove self-loop
            G.remove_edges_from(nx.selfloop_edges(G))

            # Split into graphs
            graph_num = data_graph_indicator.max()
            node_list = np.arange(data_graph_indicator.shape[0]) + 1

            for i in range(graph_num):
                # Find the nodes for each graph
                nodes = node_list[data_graph_indicator == i + 1]
                G_sub = G.subgraph(nodes)
                G_sub.graph['label'] = data_graph_types[i]
                if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes() <= max_num_nodes:
                    adj = torch.from_numpy(nx.adjacency_matrix(G_sub).toarray()).float()
                    L = nx.normalized_laplacian_matrix(G_sub).toarray()
                    L = torch.from_numpy(L).float()
                    eigval, eigvec = torch.linalg.eigh(L)
                    
                    self.eigvals.append(eigval)
                    self.eigvecs.append(eigvec)
                    self.adjs.append(adj)
                    self.n_nodes.append(G_sub.number_of_nodes())
                    if G_sub.number_of_nodes() > self.n_max:
                        self.n_max = G_sub.number_of_nodes()
                    max_eigval = torch.max(eigval)
                    if max_eigval > self.max_eigval:
                        self.max_eigval = max_eigval
                    min_eigval = torch.min(eigval)
                    if min_eigval < self.min_eigval:
                        self.min_eigval = min_eigval

            torch.save([self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max], filename)
            print(f'Dataset {filename} saved')

        self.max_k_eigval = 0
        for eigv in self.eigvals:
            last_idx = self.k if self.k < len(eigv) else len(eigv) - 1
            if eigv[last_idx] > self.max_k_eigval:
                self.max_k_eigval = eigv[last_idx].item()

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        if self.same_sample:
            idx = self.__len__() - 1
        graph = {}
        graph["n_nodes"] = self.n_nodes[idx]
        size_diff = self.n_max - graph["n_nodes"]
        graph["adj"] = F.pad(self.adjs[idx], [0, size_diff, 0, size_diff])
        eigvals = self.eigvals[idx]
        eigvecs = self.eigvecs[idx]
        if self.ignore_first_eigv:
            eigvals = eigvals[1:]
            eigvecs = eigvecs[:,1:]
            size_diff += 1
        graph["eigval"] = F.pad(eigvals, [0, max(0, self.n_max - eigvals.size(0))])
        graph["eigvec"] = F.pad(eigvecs, [0, size_diff, 0, size_diff])

        graph["mask"] = F.pad(torch.ones_like(self.adjs[idx]), [0, size_diff, 0, size_diff]).long()

        return graph

class RemoveHydrogens(object):
    """ Based on code from https://openreview.net/forum?id=-Gk_IPJWvk"""

    def __init__(self):
        pass

    def __call__(self, data):
        if hasattr(data, 'pos'):
            del data.pos
        if hasattr(data, 'z'):
            del data.z
        if hasattr(data, 'y'):
            del data.y
        E = to_dense_adj(data.edge_index, edge_attr=data.edge_attr, max_num_nodes=data.x.shape[0])  # 1, n, n, e_types

        non_hydrogens = data.x[:, 0] == 0
        data.x = data.x[non_hydrogens, 1:5]

        E = E.squeeze(0)
        E = E[non_hydrogens, :, :]
        E = E[:, non_hydrogens, :]      # N, N, e_types

        A = torch.sum(E * torch.arange(1, E.shape[-1] + 1)[None,  None, :], dim=2)


        data.edge_index, edge_attr = dense_to_sparse(A)


        edge_attr = edge_attr.long().unsqueeze(-1) - 1

        data.edge_attr = torch.zeros(edge_attr.shape[0], 3)
        data.edge_attr.scatter_(1, edge_attr, 1)

        if data.edge_index.numel() > 0:
            assert data.edge_index.max() < len(data.x), f"{data.x}, {data.edge_index}"
        return data

class QM9Dataset(Dataset):

    """ Based on code from https://openreview.net/forum?id=-Gk_IPJWvk"""

    def __init__(self, n_graphs, k, same_sample=False, SON=False, ignore_first_eigv=False):
        filename = f'data/qm9_{"full" if n_graphs == -1 else n_graphs}{"_same_sample" if same_sample else ""}.pt'
        self.k = k
        self.ignore_first_eigv = ignore_first_eigv
        self.atom_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F'}       #  Warning: hydrogens have been removed
        self.bond_dict = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]

        if os.path.isfile(filename):
            self.edge_features, self.adjs, self.node_features, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max = torch.load(filename)
            print(f'Dataset {filename} loaded from file')
        else:
            self.adjs = []
            self.node_features = []
            self.edge_features = []
            self.eigvals = []
            self.eigvecs = []
            self.n_nodes = []
            self.max_eigval = 0
            self.min_eigval = 0
            self.same_sample = same_sample

            remove_hydrogens = RemoveHydrogens()
            qm9 = QM9('data', transform=None, pre_transform=remove_hydrogens, pre_filter=None)

            if n_graphs == -1:
                n_graphs = len(qm9)
            for i in range(n_graphs):
                data = qm9[i]
                if data.x.size(0) == 1:
                    # Skip graphs with only 1 heavy atom (3 such graphs in qm9)
                    continue
                G = to_networkx(data, to_undirected=True)
                adj = torch.from_numpy(nx.adjacency_matrix(G).toarray()).float()
                L = nx.normalized_laplacian_matrix(G).toarray()
                L = torch.from_numpy(L).float()
                eigval, eigvec = torch.linalg.eigh(L)
                # Get dense adj feature matrix
                edge_feat = to_dense_adj(data.edge_index, edge_attr=data.edge_attr, max_num_nodes=data.x.shape[0]).squeeze(0)
                node_feat = data.x
                
                self.eigvals.append(eigval)
                self.eigvecs.append(eigvec)
                self.adjs.append(adj)
                self.node_features.append(node_feat)
                self.edge_features.append(edge_feat)
                self.n_nodes.append(len(G.nodes()))
                max_eigval = torch.max(eigval)
                if max_eigval > self.max_eigval:
                    self.max_eigval = max_eigval
                min_eigval = torch.min(eigval)
                if min_eigval < self.min_eigval:
                    self.min_eigval = min_eigval
            self.n_max = 9
            torch.save([self.edge_features, self.adjs, self.node_features, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max], filename)
            print(f'Dataset {filename} saved')

        self.max_k_eigval = 0
        for eigv in self.eigvals:
            if eigv.max() > self.max_k_eigval:
                self.max_k_eigval = eigv.max().item()

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        if self.same_sample:
            idx = self.__len__() - 1
        graph = {}
        graph["n_nodes"] = self.n_nodes[idx]
        size_diff = self.n_max - graph["n_nodes"]
        graph["adj"] = F.pad(self.adjs[idx], [0, size_diff, 0, size_diff])
        graph["edge_features"] = F.pad(self.edge_features[idx], [0, 0, 0, size_diff, 0, size_diff])
        eigvals = self.eigvals[idx]
        eigvecs = self.eigvecs[idx]
        if self.ignore_first_eigv:
            eigvals = eigvals[1:]
            eigvecs = eigvecs[:,1:]
            size_diff += 1
        graph["eigval"] = F.pad(eigvals, [0, max(0, self.n_max - eigvals.size(0))])
        graph["eigvec"] = F.pad(eigvecs, [0, size_diff, 0, size_diff])
        graph["mask"] = F.pad(torch.ones_like(self.adjs[idx]), [0, size_diff, 0, size_diff]).long()
        graph["node_features"] = F.pad(self.node_features[idx], [0, 0, 0, size_diff])
        return graph

class PlanarDataset(Dataset):

    def __init__(self, n_nodes, n_graphs, k, same_sample=False, SON=False, ignore_first_eigv=False):
        filename = f'data/planar_{n_nodes}_{n_graphs}{"_same_sample" if same_sample else ""}.pt'
        self.k = k
        self.ignore_first_eigv = ignore_first_eigv
        if os.path.isfile(filename):
            self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample , self.n_max = torch.load(filename)
            print(f'Dataset {filename} loaded from file')
        else:
            self.adjs = []
            self.eigvals = []
            self.eigvecs = []
            self.n_nodes = []
            self.max_eigval = 0
            self.min_eigval = 0
            self.same_sample = same_sample
            for i in range(n_graphs):
                # Generate planar graphs using Delauney traingulation
                points = np.random.rand(n_nodes,2)
                tri = Delaunay(points)
                adj = np.zeros([n_nodes,n_nodes])
                for t in tri.simplices:
                    adj[t[0], t[1]] = 1
                    adj[t[1], t[2]] = 1
                    adj[t[2], t[0]] = 1
                    adj[t[1], t[0]] = 1
                    adj[t[2], t[1]] = 1
                    adj[t[0], t[2]] = 1
                G = nx.from_numpy_array(adj)
                adj = torch.from_numpy(adj).float()
                L = nx.normalized_laplacian_matrix(G).toarray()
                L = torch.from_numpy(L).float()
                eigval, eigvec = torch.linalg.eigh(L)
                
                self.eigvals.append(eigval)
                self.eigvecs.append(eigvec)
                self.adjs.append(adj)
                self.n_nodes.append(len(G.nodes()))
                max_eigval = torch.max(eigval)
                if max_eigval > self.max_eigval:
                    self.max_eigval = max_eigval
                min_eigval = torch.min(eigval)
                if min_eigval < self.min_eigval:
                    self.min_eigval = min_eigval
            self.n_max = n_nodes
            torch.save([self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max], filename)
            print(f'Dataset {filename} saved')

        self.max_k_eigval = 0
        for eigv in self.eigvals:
            if eigv[self.k] > self.max_k_eigval:
                self.max_k_eigval = eigv[self.k].item()

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        if self.same_sample:
            idx = self.__len__() - 1
        graph = {}
        graph["n_nodes"] = self.n_nodes[idx]
        size_diff = self.n_max - graph["n_nodes"]
        graph["adj"] = F.pad(self.adjs[idx], [0, size_diff, 0, size_diff])
        eigvals = self.eigvals[idx]
        eigvecs = self.eigvecs[idx]
        if self.ignore_first_eigv:
            eigvals = eigvals[1:]
            eigvecs = eigvecs[:,1:]
            size_diff += 1
        graph["eigval"] = F.pad(eigvals, [0, max(0, self.n_max - eigvals.size(0))])
        graph["eigvec"] = F.pad(eigvecs, [0, size_diff, 0, size_diff])
        graph["mask"] = F.pad(torch.ones_like(self.adjs[idx]), [0, size_diff, 0, size_diff]).long()
        return graph


def n_community(num_communities, max_nodes, p_inter=0.05):
    assert num_communities > 1
    
    one_community_size = max_nodes // num_communities
    c_sizes = [one_community_size] * num_communities
    total_nodes = one_community_size * num_communities
    
    """ 
    Community graph construction from https://github.com/ermongroup/GraphScoreMatching/blob/master/utils/data_generators.py#L10

    here we calculate `p_make_a_bridge` so that `p_inter = \mathbb{E}(Number_of_bridge_edges) / Total_number_of_nodes `
    
    To make it more clear: 
    let `M = num_communities` and `N = one_community_size`, then
    
    ```
    p_inter
    = \mathbb{E}(Number_of_bridge_edges) / Total_number_of_nodes
    = (p_make_a_bridge * C_M^2 * N^2) / (MN)  # see the code below for this derivation
    = p_make_a_bridge * (M-1) * N / 2
    ```
    
    so we have:
    """
    p_make_a_bridge = p_inter * 2 / ((num_communities - 1) * one_community_size)
    
    print(num_communities, total_nodes, end=' ')
    graphs = [nx.gnp_random_graph(c_sizes[i], 0.7, seed=i) for i in range(len(c_sizes))]

    G = nx.disjoint_union_all(graphs)
    communities = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    add_edge = 0
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1.nodes())
        for j in range(i + 1, len(communities)):  # loop for C_M^2 times
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            has_inter_edge = False
            for n1 in nodes1:  # loop for N times
                for n2 in nodes2:  # loop for N times
                    if np.random.rand() < p_make_a_bridge:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
                        add_edge += 1
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])
                add_edge += 1
    print('connected comp: ', len([G.subgraph(c).copy() for c in nx.connected_components(G)]),
          'add edges: ', add_edge)
    print(G.number_of_edges())
    return G

class CommunityDataset(Dataset):
    """
    The dataset from https://github.com/ermongroup/GraphScoreMatching/blob/master/gen_data.py#L8
    params should be:
    n_start = 12
    n_end = 21
    n_graphs = 100
    """

    def __init__(self, n_start, n_end, n_graphs, k, same_sample=False, SON=False, ignore_first_eigv=False):
        filename = f'data/community_{n_start}_{n_end}_{n_graphs}{"_same_sample" if same_sample else ""}.pt'
        self.k = k
        self.ignore_first_eigv = ignore_first_eigv
        if os.path.isfile(filename):
            self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample , self.n_max = torch.load(filename)
            print(f'Dataset {filename} loaded from file')
        else:
            self.adjs = []
            self.eigvals = []
            self.eigvecs = []
            self.n_nodes = []
            self.max_eigval = 0
            self.min_eigval = 0
            self.same_sample = same_sample
            num_communities = 2
            for p in range(n_graphs):
                n_max = np.random.choice(np.arange(n_start, n_end).tolist())
                G = n_community(num_communities, n_max, p_inter=0.05)
                adj = torch.from_numpy(nx.adjacency_matrix(G).toarray()).float()
                L = nx.normalized_laplacian_matrix(G).toarray()
                L = torch.from_numpy(L).float()
                eigval, eigvec = torch.linalg.eigh(L)
                
                self.eigvals.append(eigval)
                self.eigvecs.append(eigvec)
                self.adjs.append(adj)
                self.n_nodes.append(len(G.nodes()))
                max_eigval = torch.max(eigval)
                if max_eigval > self.max_eigval:
                    self.max_eigval = max_eigval
                min_eigval = torch.min(eigval)
                if min_eigval < self.min_eigval:
                    self.min_eigval = min_eigval
            self.n_max = max(self.n_nodes)
            torch.save([self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max], filename)
            print(f'Dataset {filename} saved')

        self.max_k_eigval = 0
        for eigv in self.eigvals:
            if eigv[self.k] > self.max_k_eigval:
                self.max_k_eigval = eigv[self.k].item()

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        if self.same_sample:
            idx = self.__len__() - 1
        graph = {}
        graph["n_nodes"] = self.n_nodes[idx]
        size_diff = self.n_max - graph["n_nodes"]
        graph["adj"] = F.pad(self.adjs[idx], [0, size_diff, 0, size_diff])
        eigvals = self.eigvals[idx]
        eigvecs = self.eigvecs[idx]
        if self.ignore_first_eigv:
            eigvals = eigvals[1:]
            eigvecs = eigvecs[:,1:]
            size_diff += 1
        graph["eigval"] = F.pad(eigvals, [0, max(0, self.n_max - eigvals.size(0))])
        graph["eigvec"] = F.pad(eigvecs, [0, size_diff, 0, size_diff])
        graph["mask"] = F.pad(torch.ones_like(self.adjs[idx]), [0, size_diff, 0, size_diff]).long()
        return graph


class GraphDataModule(pl.LightningDataModule):

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', default=10, type=int)
        parser.add_argument('--n_nodes', default=N_MAX, type=int)
        parser.add_argument('--n_graphs', default=200, type=int)
        parser.add_argument('--n_data_workers', default=2, type=int)
        parser.add_argument('--same_sample', default=False, action="store_true")
        parser.add_argument('--n_start', default=10, type=int)
        parser.add_argument('--n_end', default=20, type=int)
        parser.add_argument('--dataset', default='tree', type=str)
        parser.add_argument('--SON', default=False, action="store_true")
        parser.add_argument('--validate_on_train_cond', default=False, action="store_true")
        parser.add_argument('--ignore_first_eigv', default=False, action="store_true")
        parser.add_argument('--qm9_strict_eval', default=False, action="store_true")
        
        return parser

    def __init__(self, data_dir: str = './data', batch_size: int = 10, k = N_MAX,
                n_nodes: int = N_MAX, n_graphs: int = 200, n_data_workers: int = 4,
                same_sample: bool = False, n_start: int = 10, n_end: int = 20,
                dataset: str = 'tree', validate_on_train_cond: bool = False,
                ignore_first_eigv: bool = False, eval_MMD: bool = False,
                compute_emd: bool = False, qm9_strict_eval: bool = False):
        super().__init__()
        self.batch_size = batch_size

        self.dataset = dataset
        self.n_nodes = n_nodes
        self.n_graphs = n_graphs
        self.n_start = n_start
        self.n_end = n_end
        self.validate_on_train_cond = validate_on_train_cond
        self.ignore_first_eigv = ignore_first_eigv
        self.eval_MMD = eval_MMD
        self.compute_emd = compute_emd
        
        self.k = k
        self.n_data_workers = n_data_workers
        self.same_sample = same_sample

        if self.dataset == 'tree':
            self.dataset_string = f'tree_{self.n_nodes}-{self.n_graphs}'
        elif self.dataset == 'grid':
            self.dataset_string = f'grid_{self.n_start}-{self.n_end}'
        elif self.dataset == 'grid_non_iso':
            self.dataset_string = f'grid_non_iso_{self.n_start}-{self.n_end}'
        elif self.dataset == 'sbm':
            self.dataset_string = f'sbm_{self.n_graphs}'
        elif self.dataset == 'lobster':
            self.dataset_string = f'lobster_{self.n_graphs}'
        elif self.dataset == 'protein':
            self.dataset_string = f'protein'
        elif self.dataset == 'qm9':
            self.dataset_string = f'qm9_{self.n_graphs}'
        elif self.dataset == 'planar':
            self.dataset_string = f'planar_{self.n_nodes}-{self.n_graphs}'
        elif self.dataset == 'community':
            self.dataset_string = f'community_{self.n_start}-{self.n_end}-{self.n_graphs}'
        else:
            raise ValueError

        if self.same_sample:
            self.dataset_string = 'same_sample_' + self.dataset_string

    def setup(self, stage=None):
        if self.dataset == 'tree':
            graphs = TreeDataset(self.n_nodes, self.n_graphs, self.k, same_sample=self.same_sample, ignore_first_eigv=self.ignore_first_eigv)
        elif self.dataset == 'grid':
            graphs = GridDataset(self.n_start, self.n_end, self.k, same_sample=self.same_sample, ignore_first_eigv=self.ignore_first_eigv)
        elif self.dataset == 'grid_non_iso':
            graphs = GridDatasetNonIso(self.n_start, self.n_end, self.k, same_sample=self.same_sample, ignore_first_eigv=self.ignore_first_eigv)
        elif self.dataset == 'sbm':
            graphs = SBMDataset(self.n_graphs, self.k, same_sample=self.same_sample, ignore_first_eigv=self.ignore_first_eigv)
        elif self.dataset == 'lobster':
            graphs = LobsterDataset(self.n_graphs, self.k, same_sample=self.same_sample, ignore_first_eigv=self.ignore_first_eigv)
        elif self.dataset == 'protein':
            graphs = ProteinDataset(self.k, same_sample=self.same_sample, ignore_first_eigv=self.ignore_first_eigv)
        elif self.dataset == 'planar':
            graphs = PlanarDataset(self.n_nodes, self.n_graphs, self.k, same_sample=self.same_sample, ignore_first_eigv=self.ignore_first_eigv)
        elif self.dataset == 'community':
            graphs = CommunityDataset(self.n_start, self.n_end, self.n_graphs, self.k, same_sample=self.same_sample, ignore_first_eigv=self.ignore_first_eigv)
        elif self.dataset == 'qm9':
            graphs = QM9Dataset(self.n_graphs, self.k, same_sample=self.same_sample, ignore_first_eigv=self.ignore_first_eigv)
            self.atom_dict = graphs.atom_dict
            self.bond_dict = graphs.bond_dict
        else:
            raise ValueError

        self.n_max = graphs.n_max
        self.max_k_eigval = graphs.max_k_eigval
        
        if self.same_sample:
            self.train = graphs
            self.val = graphs
            self.test = graphs
        else:            
            if self.dataset == 'qm9':
                # Split sizes used by GraphVAE and subsequent methods
                test_len = 10000
                val_len = 10000
                train_len = len(graphs) - val_len - test_len
            else:
                # GRAN-like splits for all other datasets
                test_len = int(round(len(graphs)*0.2))
                train_len = int(round((len(graphs) - test_len)*0.8))
                val_len = len(graphs) - train_len - test_len
            print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
            self.train, self.val, self.test = random_split(graphs, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(1234))

            if self.validate_on_train_cond: # Check how well the model does when conditioned on true training spectra
                self.train_test, self.train_val, _ = random_split(self.train, [test_len, val_len, len(self.train) - val_len - test_len], generator=torch.Generator().manual_seed(1234))

        if self.dataset == 'qm9':
            self.molecular_metrics = BasicMolecularMetrics(self.atom_dict, self.bond_dict, self.train, strict=self.qm9_strict_eval)
        else:
            if self.eval_MMD:
                val_graphs = [nx.from_numpy_matrix(g['adj'][:g['n_nodes'], :g['n_nodes']].cpu().detach().numpy()) for g in self.test]
                val_eigvals = [graph["eigval"][1:self.k+1].cpu().detach().numpy() for graph in self.test]
            else:  
                val_graphs = [nx.from_numpy_matrix(g['adj'][:g['n_nodes'], :g['n_nodes']].cpu().detach().numpy()) for g in self.val]
                val_eigvals = [graph["eigval"][1:self.k+1].cpu().detach().numpy() for graph in self.val]
            train_graphs = [nx.from_numpy_matrix(g['adj'][:g['n_nodes'], :g['n_nodes']].cpu().detach().numpy()) for g in self.train]
            train_eigvals = [graph["eigval"][1:self.k+1].cpu().detach().numpy() for graph in self.train]
            # Get training set vs validation set MMD measures
            if self.compute_emd: 
                metric_type = 'EMD' # Use EMD kernel (slow, only used for community graphs)
            else:
                metric_type = 'MMD' # Use Gaussian TV kernel
            self.train_mmd_degree = degree_stats(val_graphs, train_graphs, compute_emd=(metric_type=='EMD'))
            self.train_mmd_4orbits = orbit_stats_all(val_graphs, train_graphs, compute_emd=(metric_type=='EMD'))
            self.train_mmd_clustering = clustering_stats(val_graphs, train_graphs, compute_emd=(metric_type=='EMD'))    
            self.train_mmd_spectral = spectral_stats(val_graphs, train_graphs, compute_emd=(metric_type=='EMD'))
            mmd_eigval = eigval_stats(val_eigvals, train_eigvals, max_eig=self.max_k_eigval, compute_emd=(metric_type=='EMD'))
            true_graph_eigvals, true_graph_eigvecs = compute_list_eigh(val_graphs)
            fake_graph_eigvals, fake_graph_eigvecs = compute_list_eigh(train_graphs)
            self.train_mmd_spectral_filter_graph = spectral_filter_stats(true_graph_eigvecs, true_graph_eigvals, fake_graph_eigvecs, fake_graph_eigvals, compute_emd=(metric_type=='EMD'))
            print(f'{metric_type} measures of Training set vs Validation set: degree {self.train_mmd_degree}, 4orbits {self.train_mmd_4orbits}, clustering {self.train_mmd_clustering}, spectral {self.train_mmd_spectral}, mmd_eigval {mmd_eigval}, mmd_spectral_filter_graph {self.train_mmd_spectral_filter_graph}')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_data_workers, pin_memory=False)

    def val_dataloader(self):
        if self.validate_on_train_cond:
            return [DataLoader(self.val, batch_size=self.batch_size, num_workers=self.n_data_workers, pin_memory=False), DataLoader(self.train_val, batch_size=self.batch_size, num_workers=self.n_data_workers, pin_memory=False)]
        else:
            return [DataLoader(self.val, batch_size=self.batch_size, num_workers=self.n_data_workers, pin_memory=False)]

    def test_dataloader(self):
        if self.validate_on_train_cond:
            return [DataLoader(self.test, batch_size=self.batch_size, num_workers=self.n_data_workers, pin_memory=False), DataLoader(self.train_test, batch_size=self.batch_size, num_workers=self.n_data_workers, pin_memory=False)]
        else:
            return [DataLoader(self.test, batch_size=self.batch_size, num_workers=self.n_data_workers, pin_memory=False)]
