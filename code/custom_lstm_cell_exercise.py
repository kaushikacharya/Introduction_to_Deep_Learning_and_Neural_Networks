import torch
from torch import nn

class My_LSTM_cell(torch.nn.Module):
    """
    A simple LSTM cell network for educational AI-summer purposes
    """
    def __init__(self, input_length=10, hidden_length=20):
        super(My_LSTM_cell, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        # forget gate components
        # 1. DEFINE FORGET GATE COMPONENTS
        self.linear_gate_w1 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_forget_gate = nn.Sigmoid()

        # input gate components
        self.linear_gate_w2 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_gate = nn.Sigmoid()

        # cell memory components
        # 2. DEFINE CELL MEMORY COMPONENTS
        self.linear_gate_w3 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r3 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.tanh_gate = nn.Tanh()

        # out gate components
        # 3. DEFINE OUT GATE COMPONENTS
        self.linear_gate_w4 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r4 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        # self.linear_gate_c4 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_out_gate = nn.Sigmoid()

        # final output
        # 4. DEFINE OUTPUT
        self.activation_final = nn.Tanh()

    def forget(self, x, h):
        # 5. FORGET GATE
        x_temp = self.linear_gate_w1(x)
        h_temp = self.linear_gate_r1(h)
        f = self.sigmoid_forget_gate(x_temp + h_temp)
        return f

    def input_gate(self, x, h):

        # Equation 1. input gate
        x_temp = self.linear_gate_w2(x)
        h_temp = self.linear_gate_r2(h)
        i = self.sigmoid_gate(x_temp + h_temp)
        return i

    def cell_memory_gate(self, i, f, x, h, c_prev):
        # 6. CELL MEMORY GATE
        x_temp = self.linear_gate_w3(x)
        h_temp = self.linear_gate_r3(h)
        c_next = torch.mul(f, c_prev) + torch.mul(i, self.tanh_gate(x_temp + h_temp))
        # c_next = f * c_prev + i * self.tanh_gate(x_temp + h_temp)
        return c_next

    def out_gate(self, x, h):
        # 7. OUT GATE
        x_temp = self.linear_gate_w4(x)
        h_temp = self.linear_gate_r4(h)
        # c_temp = self.linear_gate_c4(c)
        # o = self.sigmoid_out_gate(x_temp + h_temp + c_temp)
        o = self.sigmoid_out_gate(x_temp + h_temp)
        return o

    def forward(self, x, tuple_in ):
        (h, c_prev) = tuple_in
        # Equation 1. input gate
        i = self.input_gate(x, h)

        # Equation 2. forget gate
        f = self.forget(x, h)

        # Equation 3. updating the cell memory
        c_next = self.cell_memory_gate(i, f, x, h,c_prev)

        # Equation 4. calculate the main output gate
        o = self.out_gate(x, h)


        # Equation 5. produce next hidden output
        h_next = o * self.activation_final(c_next)

        return h_next, c_next
