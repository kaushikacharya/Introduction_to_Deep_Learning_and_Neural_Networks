import torch

# random binary Adj matrix
a = torch.rand(5, 5)
a[a > 0.5] = 1
a[a <= 0.5] = 0

def calc_degree_matrix(a):
    return torch.diag(a.sum(dim=-1))

def create_graph_laplacian(a):
    return calc_degree_matrix(a) - a

print("A: ", a)
print("L: ", create_graph_laplacian(a))
