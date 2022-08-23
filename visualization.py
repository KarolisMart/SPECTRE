import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def plot_samples(graph_files, output_filename, column_names, row_names=None, n_samples=1, node_size=100, iterations=100, largest_component=False):
    """
    Plots first n_samples from the given datafiles in each column.

    graph_files: list of path strings. Can be a list of lists to plot different files in different rows.
    output_filename: filename or path of the output file (str). Do not include file type
    column_names: list of strings used for column names
    row_names: (Optinal) list of row names
    n_samples: number of samples to plot per graph file (int) 
    node_size: size of nodes in the plot (float)
    iterations: number of spring embedding iterations to use (decrease for speed) (int)
    largest_component: only the largest connected component is ploted (bool)

    """
    graph_files = [graph_files] if not isinstance(graph_files[0], list) else graph_files
    
    num_rows = n_samples * len(graph_files)
    fig, axs = plt.subplots(num_rows, len(graph_files), figsize=(4 * len(column_names),4 * num_rows)
    
    for row, col_graph_files in enumerate(graph_files):
        for col, graph_file in enumerate(col_graph_files):
            graphs = torch.load(graph_file)
            axs[0, col].set_title(column_names[col], fontsize=22)
                            
            for j in range(n_samples):
                row = row * n_samples + j
                G = graphs[j]

                if largest_component:
                    CGs = [G.subgraph(c) for c in nx.connected_components(G)]
                    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
                    G = CGs[0]

                pos = nx.spring_layout(G, iterations=iterations)
                w, U = np.linalg.eigh(nx.normalized_laplacian_matrix(G).toarray())

                vmin, vmax = np.min(U[:, 1]), np.max(U[:, 1])
                m = max(np.abs(vmin), vmax)
                vmin, vmax = -m, m

                nx.draw(G, pos, font_size=5, node_size=node_size, with_labels=False, node_color=U[:, 1], cmap=plt.cm.coolwarm,
                        vmin=vmin, vmax=vmax, edge_color='grey', ax=axs[row, col])

                if col == 0 and row_names is not None:
                    axs[j, 0].set_ylabel(row_names[row], fontsize=22)
                    axs[j, 0].axis('on')
                    axs[j, 0].spines['top'].set_visible(False)
                    axs[j, 0].spines['right'].set_visible(False)
                    axs[j, 0].spines['bottom'].set_visible(False)
                    axs[j, 0].spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_filename}.png', format='png', dpi=100)
    plt.savefig(f'{output_filename}.pdf', format='pdf')
