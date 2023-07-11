# Chebyshev approximation for Laplacian powers

import torch
import torch.nn as nn

def create_adj(size):
    a = torch.rand(size, size)
    a[a > 0.5] = 1
    a[a <= 0.5] = 0

    # For illustration we set the diagonal elements to zero
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if i == j:
                a[i,j] = 0
    
    return a

def calc_degree_matrix(a):
    return torch.diag(a.sum(dim=-1))

def create_graph_laplacian(a):
    return calc_degree_matrix(a) - a

def calc_degree_matrix_norm(a):
    return torch.diag(torch.pow(a.sum(dim=-1), -0.5))

def create_graph_laplacian_norm(a):
    size = a.shape[-1]
    D_norm = calc_degree_matrix_norm(a)
    L_norm = torch.ones(size) - (D_norm @ a @ D_norm)

    return L_norm

def find_eigmax(L):
    with torch.no_grad():
        eigen_values, _ = torch.linalg.eig(L)
        return torch.max(eigen_values.real)

def chebyshev_laplacian(X, L, thetas, order):
    """
    Parameters
    ----------
    X
    L
    thetas
    order: Number of hops
    """
    list_powers = []
    nodes = L.shape[0]

    T0 = X.float()

    eigmax = find_eigmax(L=L)
    L_rescaled = 2 * L/eigmax - torch.eye(n=nodes)

    y = T0*thetas[0]
    list_powers.append(y)
    T1 = torch.matmul(L_rescaled, T0)
    list_powers.append(T1 * thetas[1])

    # Notation: i=k-2, j=k-1
    Ti = T0
    Tj = T1

    # T_k = 2*L_rescaled*T_{k-1} - T_{k-2}
    for k in range(2, order):
        Tk = 2 * torch.matmul(L_rescaled, Tj) - Ti
        list_powers.append(Tk * thetas[k])
        # Update for next iteration
        Ti, Tj = Tj, Tk
    
    y_out = torch.stack(list_powers, dim=-1)

    # The powers may be summed or concatenated.
    # Here we do concatenation.
    y_out = y_out.view(nodes, -1)  # -1 = order * features_of_signal

    return y_out

if __name__ == "__main__":
    features = 3
    out_features = 50
    nodes = 10

    a = create_adj(size=nodes)
    L = create_graph_laplacian_norm(a)

    x = torch.rand(nodes, features)
    power_order = 4  # p-hops
    thetas = nn.Parameter(torch.rand(power_order))

    out = chebyshev_laplacian(X=x, L=L, thetas=thetas, order=power_order)

    print("Chebyshev approximate out powers concatenated: ", out.shape)

    linear = nn.Linear(in_features=power_order*features, out_features=out_features)
    layer_out = linear(out)
    print("Layers output: ", layer_out.shape)
