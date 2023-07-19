import torch
import torch.nn as nn

def train():
    model = nn.Linear(in_features=4, out_features=2)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

    for epoch in range(10):
        # Converting inputs and labels to Variable
        inputs = torch.Tensor([0.8,0.4,0.4,0.2])
        labels = torch.Tensor([1, 0])

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward
        optimizer.zero_grad()

        # Get output from the model, given the inputs
        outputs = model(inputs)

        # Get loss for the predicted output
        loss = criterion(outputs, labels)
        print("loss: ", loss)

        # Get gradients w.r.t. to parameters
        loss.backward()

        # Update parameters
        optimizer.step()

        print("epoch {} :: loss {}".format(epoch, loss.item()))

if __name__ == "__main__":
    train()
