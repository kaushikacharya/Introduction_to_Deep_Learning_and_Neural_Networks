import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest

seed = 172
torch.manual_seed(seed)


class Attention(nn.Module):

    def __init__(self, y_dim: int, h_dim: int):
        super().__init__()
        #1. Define the vector dimensions and the trainable parameters
        self.y_dim = y_dim
        self.h_dim = h_dim

        self.W = nn.Parameter(torch.FloatTensor(self.y_dim, self.h_dim))
        # print(f"W: {self.W}")

    def forward(self,
                y: torch.Tensor, # y.size() = (1, y_dim)
                h: torch.Tensor # h.size() = (1, h_dim)
                ):
        #2. Define the forward pass
        # score = y.T * h
        score = torch.matmul(torch.matmul(y, self.W), h.T)
        print(f"score: {score}")
        z = F.softmax(score, dim=0)
        print(f"z: {z}")
        a = torch.matmul(z, h)
        print(f"attention: {a}")
        return a

class TestAttention(unittest.TestCase):
    def _setup(self, y_dim, h_dim):
        self.attention = Attention(y_dim=y_dim, h_dim=h_dim)
    
    def test_case_1(self):
        y = torch.Tensor([[0.2469, 0.2080, 0.8997, 0.9753, 0.8461]])
        h = torch.Tensor([[0.7945, 0.9302, 0.0049]])
        self._setup(y_dim=y.size()[1], h_dim=h.size()[1])
        output = self.attention(y,h)
        print(f"output: {output}")
        expected = torch.Tensor([[0.0489, 0.3148, 0.4832]])
        self.assertEqual(output, expected)

if __name__ == "__main__":
    unittest.main()
