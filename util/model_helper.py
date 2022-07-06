import torch
import torch.distributions as td
from torch_geometric.data import Data


def print_mem():
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Allocated MAX:', round(torch.cuda.max_memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    print('Cached MAX:   ', round(torch.cuda.max_memory_reserved(0)/1024**3,1), 'GB')
    torch.cuda.reset_peak_memory_stats(0)


def zero_diag(matrix: torch.Tensor):
    matrix = matrix * (1 - torch.eye(matrix.size(-2), matrix.size(-1), device=matrix.device))
    return matrix

def discretize(A: torch.Tensor, temperature: float = 2.0/3.0, hard: bool = False): # temp = 2/3 suggested by http://www.stats.ox.ac.uk/~cmaddis/pubs/concrete.pdf in Gumbel-softmax temp=0.01 is common (lower temp-> more discrete)
    """Discretize the continuous adjacency matrix. We use a Bernoulli

    :param A:adjacency matrix
    :return A:discretized adj matrix

    From GG-GAN
    """
    # Force adjacency to 0-1 range
    A = A.clamp(0.0, 1.0)
    relaxedA = td.RelaxedBernoulli(temperature, probs=A).rsample()
    if temperature < 1e-4:
      hard = True
    if hard:
      Ar = relaxedA.round() - relaxedA.detach() + relaxedA
    else:
      Ar = relaxedA
    # rezero and resymmetrize
    Az = zero_diag(Ar)
    Atriu = torch.triu(Az)
    A = Atriu + Atriu.permute(0, 2, 1)

    return A

def rand_rot(matrix: torch.Tensor, mask: torch.Tensor = torch.tensor([]), variance: float = 0.1, right_noise: bool = False): # variance=0.025 # 0.01, exp needs larger variance
    rand_matrix = torch.randn(*matrix.shape[:-1], matrix.shape[-2], device=matrix.device, dtype=matrix.dtype)*variance
    rand_matrix = torch.triu(rand_matrix, diagonal=1)
    rand_matrix = rand_matrix - torch.transpose(rand_matrix, -2, -1)
    if len(mask) > 0:
        rand_matrix = rand_matrix * mask
    rand_rot = torch.matrix_exp(rand_matrix)
    matrix = rand_rot @ matrix
    if right_noise:
        rand_matrix = torch.randn(*matrix.shape[:-2], matrix.shape[-1], matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)*variance
        rand_matrix = torch.triu(rand_matrix, diagonal=1)
        rand_matrix = rand_matrix - torch.transpose(rand_matrix, -2, -1)
        if len(mask) > 0:
            rand_matrix = rand_matrix * mask[:, :matrix.shape[-1], :matrix.shape[-1]]
        rand_rot = torch.matrix_exp(rand_matrix)
        matrix = matrix @ rand_rot
    if len(mask) > 0:
        return matrix * mask[:, :mask.size(1), :matrix.size(2)]
    else:
        return matrix


def eigval_noise(eigval: torch.Tensor, variance: float = 0.1):
    deltas = eigval.clone()
    deltas[:, 1:] = deltas.diff(dim=-1) # Get deltas from eigvalues
    deltas[deltas == 0] = deltas[deltas == 0] + 1e-5 # Don't leave deltas as exactly 0
    deltas = deltas * (1 + (torch.randn_like(deltas) * variance))
    eigval = deltas.cumsum(-1)
    return eigval


def reorder_adj(adj: torch.Tensor, indices: torch.Tensor, edge_features: torch.Tensor = None):
    i_ids_1 = indices.unsqueeze(-1).expand(-1, -1, adj.size(-1)).clone()
    j_ids_1 = indices.unsqueeze(1).expand(-1, adj.size(1), -1).clone()
    adj = adj.gather(1, i_ids_1).gather(2, j_ids_1)
    if edge_features is not None:
        i_ids_1 = i_ids_1.unsqueeze(-1).expand(-1, -1, -1, edge_features.size(-1)).clone()
        j_ids_1 = j_ids_1.unsqueeze(-1).expand(-1, -1, -1, edge_features.size(-1)).clone()
        edge_features = edge_features.gather(1, i_ids_1).gather(2, j_ids_1)
        return adj, edge_features
    else:
        return adj


def categorical_permute(matrix: torch.Tensor, mask: torch.Tensor = torch.tensor([]), fraction: float = 0.3, noise: float = 0.0):
    if not len(mask) > 0:
        mask = torch.ones_like(matrix)
    n = mask.size(1)
    p = fraction #* 1/((n * (n-1)) / 2)

    permuted_matrix = matrix.clone()

    rand_indices = torch.argsort((torch.rand(permuted_matrix.size(0), permuted_matrix.size(1), permuted_matrix.size(2), device=permuted_matrix.device) + 1e-6), dim=-1, descending=True)

    permuted_matrix = permuted_matrix.gather(-1, rand_indices)

    sampled_rows = ((torch.ones_like(permuted_matrix[:,:,0])*p).bernoulli().unsqueeze(-1).expand_as(permuted_matrix) == 1)   

    # Shufle selected rows
    permuted_matrix[~sampled_rows] = matrix[~sampled_rows]

    # Add noise
    permuted_matrix = torch.clamp(permuted_matrix + (torch.randn_like(permuted_matrix) * noise), min=0, max=1)

    permuted_matrix = permuted_matrix * mask
    return permuted_matrix


def rand_rewire(adj: torch.Tensor, mask: torch.Tensor = torch.tensor([]), fraction: float = 0.3, noise: float = 0.0, edge_features = None):
    if not len(mask) > 0:
        mask = torch.ones_like(adj)
    n = mask.size(1)
    p = fraction #* 1/((n * (n-1)) / 2)

    rewired_adj = adj.clone()

    rand_indices = torch.argsort((torch.rand(rewired_adj.size(0), rewired_adj.size(1), device=rewired_adj.device) + 1e-6) * mask[:,:,0], dim=-1, descending=True)

    sampled_edges = (torch.ones_like(adj)*p).bernoulli()
    sampled_edges = torch.tril(sampled_edges, diagonal=-1)
    sampled_edges = sampled_edges + sampled_edges.transpose(-2,-1)
    sampled_edges = (sampled_edges == 1)

    if edge_features is not None:
        rewired_edge_features = edge_features.clone()
        shufled_adj, shufled_edge_features = reorder_adj(rewired_adj, rand_indices, rewired_edge_features)
    else:
        shufled_adj = reorder_adj(rewired_adj, rand_indices)

    # In expectation the graph density will be the same
    rewired_adj[sampled_edges] = shufled_adj[sampled_edges]

    # Add noise
    rewired_adj = torch.clamp(rewired_adj + (torch.randn_like(rewired_adj) * noise), min=0, max=1)

    rewired_adj = mask * rewired_adj

    if edge_features is not None:
        sampled_edges = sampled_edges.unsqueeze(-1).expand_as(rewired_edge_features)
        rewired_edge_features[sampled_edges] = shufled_edge_features[sampled_edges]

        # Add categorical noise (assume symetric matrix):
        tril_mask = (torch.tril(torch.ones_like(adj), diagonal=-1).unsqueeze(-1).expand_as(rewired_edge_features) == 1)
        tril_rewired_edge_features = torch.zeros_like(rewired_edge_features)
        tril_rewired_edge_features[tril_mask] = categorical_permute(rewired_edge_features[tril_mask] .view(rewired_edge_features.size(0), -1, rewired_edge_features.size(-1)), mask=mask[tril_mask[:,:,:,0]].view(mask.size(0), -1, 1), fraction=fraction, noise=noise).view(-1)
        rewired_edge_features = tril_rewired_edge_features + tril_rewired_edge_features.transpose(1, 2)
        
        rewired_edge_features = rewired_edge_features * mask.unsqueeze(-1).expand_as(rewired_edge_features)
        return rewired_adj, rewired_edge_features
    else:
        return rewired_adj    


def masked_instance_norm(x: torch.Tensor, mask: torch.Tensor, eps:float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), 1]
    """
    mean = (torch.sum(x * mask, 1) / torch.sum(mask, 1))   # (N,C)
    var_term = ((x - mean.unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,C)
    var = (torch.sum(var_term, 1) / torch.sum(mask, 1))  # (N,C)
    mean = mean.unsqueeze(1).expand_as(x)  # (N, L, C)
    var = var.unsqueeze(1).expand_as(x)    # (N, L, C)
    instance_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, C)
    instance_norm = instance_norm * mask
    return instance_norm


def masked_instance_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = (torch.sum(x * mask, dim=[1,2]) / torch.sum(mask, dim=[1,2]))   # (N,C)
    var_term = ((x - mean.unsqueeze(1).unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[1,2]) / torch.sum(mask, dim=[1,2]))  # (N,C)
    mean = mean.unsqueeze(1).unsqueeze(1).expand_as(x)  # (N, L, L, C)
    var = var.unsqueeze(1).unsqueeze(1).expand_as(x)    # (N, L, L, C)
    instance_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    instance_norm = instance_norm * mask
    return instance_norm


def masked_layer_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = torch.sum(x * mask, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1])   # (N)
    var_term = ((x - mean.view(-1,1,1,1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1]))  # (N)
    mean = mean.view(-1,1,1,1).expand_as(x)  # (N, L, L, C)
    var = var.view(-1,1,1,1).expand_as(x)    # (N, L, L, C)
    layer_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    layer_norm = layer_norm * mask
    return layer_norm


def trace(matrix: torch.Tensor):
    """ Compute trace for batched matrices
    """
    return torch.diagonal(matrix, dim1=-2, dim2=-1).sum(dim=-1) 

def dense_batch_to_sparse(x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor):
    num_nodes = mask.sum(-1)[:,0].sum()

    node_ids = torch.arange(0, mask.size(1), device=mask.device)
    i_ids, j_ids = torch.meshgrid(node_ids, node_ids)
    i_ids = i_ids[torch.tril(torch.ones_like(i_ids), diagonal=-1) == 1]
    j_ids = j_ids[torch.tril(torch.ones_like(j_ids), diagonal=-1) == 1]
    edge_mask = mask[:,i_ids,j_ids].bool()
    edge_attr = adj[:,i_ids,j_ids][edge_mask].view(-1,1)
    edge_mask = edge_mask.view(1,-1).expand(2,-1).contiguous()
    edge_index = torch.cat([i_ids.view(1,-1).expand(mask.size(0), -1).contiguous().view(1,-1), j_ids.view(1,-1).expand(mask.size(0), -1).contiguous().view(1,-1)], dim=0) + (torch.arange(0, mask.size(0), device=x.device) * mask[:,0].sum(-1)).view(-1,1).expand(-1, i_ids.size(0)).contiguous().view(1,-1).expand(2,-1)
    edge_index = edge_index[edge_mask].view(2,-1)
    batch = torch.arange(0, mask.size(0), device=x.device).view(-1,1).expand(-1, mask.size(1)).contiguous()[mask[:,0].bool()]
    x = x.view(-1,x.size(-1))
    data = Data(x=x, edge_index=edge_index.long(), edge_attr=edge_attr, num_nodes=num_nodes.long(), batch=batch.long(), num_graphs=mask.size(0))
    return data


def stiefel_metric(A: torch.Tensor, B: torch.Tensor, manifold: torch.Tensor):
    """ Canonical Stiefel manifold metric - equation 2.39 in https://arxiv.org/pdf/physics/9806030.pdf
    """ 
    metric = trace(A.transpose(-2, -1) @ (torch.eye(manifold.size(-2), device=manifold.device, dtype=manifold.dtype) - 0.5 * manifold @ manifold.transpose(-2, -1)) @ B)
    return metric


def deterministic_vector_sign_flip(U: torch.Tensor):
    max_abs_rows = torch.argmax(torch.abs(U), dim=-2)
    signs = torch.sign(torch.gather(U, -2, max_abs_rows.unsqueeze(-2).expand_as(U)))
    U = U * signs
    return U


def sort_eigvecs(U: torch.Tensor, mask: torch.Tensor, sign_flip: bool = True):
    """ Canonical Ordering of Eigenvecotrs (direction is flipped such that max absolute value is made positive and node order is picked by sorting by eigenvectors in order)
    x: [batch_size (N), num_objects (L), num_eigvecs(C)]
    mask: [batch_size (N), num_objects (L), 1]
    """ 
    mask = mask.squeeze(-1) # Ensure mask is N x L, not N x L x 1
    if sign_flip:
        U = deterministic_vector_sign_flip(U)
    indices = torch.arange(U.size(1), device=U.device).unsqueeze(0).expand(U.size(0), U.size(1))
    # Sort nodes by eigenvector embeddings, break ties using larger eigenvector
    for i in range(U.size(-1)-1,-1,-1): # Iterate backwards over the eigenvectors (largest to smallest)
        index = torch.sort(U[:,:,i], dim=-1, descending=True, stable=True)[1].unsqueeze(-1).expand_as(U)
        U = torch.gather(U, -2, index)
        mask = torch.gather(mask, -1, index[:,:,0])
        indices = torch.gather(indices, -1, index[:,:,0])
    # Make sure masked elements are last
    mask_index = torch.sort(mask, dim=-1, descending=True, stable=True)[1].unsqueeze(-1).expand_as(U)
    U = torch.gather(U, -2, mask_index)
    indices = torch.gather(indices, -1, mask_index[:,:,0])
    return U, indices


def gram_schmidt(V: torch.Tensor):
    """
    Make columns of batched matrices orthonormal
    V: [batch_size (N), num_objects (L), num_eigvecs(C)] - assumed to be sorted with sort_eigvecs
    mask: [batch_size (N), num_objects (L), 1]
    """
    k = V.size(-1)
    U = V.clone()
    for i in range(0, k):
        U[:,:,i] = U[:,:,i] / torch.linalg.vector_norm(U[:,:,i], dim=-1, keepdim=True)
        # Correct all subsequent vectors:
        U_i = U[:,:,i].unsqueeze(-1).expand_as(U[:,:,i+1:k])
        U[:,:,i+1:k] = U[:,:,i+1:k] - (U_i * (U[:,:,i+1:k] * U_i).sum(dim=-2, keepdim=True))
    return U


def interpolate_eigvecs(U_1: torch.Tensor, U_2: torch.Tensor, mask: torch.Tensor, alpha: torch.Tensor):
    """ Canonical Ordering of Eigenvecotrs (direction is flipped such that max absolute value is made positive and node order is picked by sorting by eigenvectors in order)
    U_1 and U_2: [batch_size (N), num_objects (L), num_eigvecs(C)] - assumed to be sorted with sort_eigvecs
    mask: [batch_size (N), num_objects (L), 1]
    alpha: [batch_size (N), num_objects (L), 1] - Value of 1.0 returns U_1, while value of 0.0 retursn U_2
    """ 
    U = U_2 + ((U_1 - U_2) * alpha) # if alpha=1 take first matrix (U_1)
    U = torch.linalg.qr(U, mode='reduced')[0]
    # U = gram_schmidt(U)
    U = deterministic_vector_sign_flip(U)
    U = U * mask
    return U
