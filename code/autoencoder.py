import torch
import torch.nn as nn

seed = 172
torch.manual_seed(seed)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]

        # Layer #1: Conv2d, Output shape: [-1, 12, 16, 16], Param #: 588
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=17)

        # Layer #3: Conv2d, Output shape: [-1, 24, 8, 8], Param #: 4,632
        self.cnn2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=9)

        # Layer #5: Conv2d, Output shape: [-1, 48, 4, 4], Param #: 18,480
        self.cnn3 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5)

        # Layer #7: Conv2d, Output shape: [-1, 96, 2, 2], Param #: 73,824
        self.cnn4 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3)

        # Layer #9: ConvTranspose2d, Output shape: [-1, 48, 4, 4], Param #: 73,776
        self.cnnt1 = nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=3)

        # Layer #11: ConvTranspose2d, Output shape: [-1, 24, 8, 8], Param #: 18,456
        self.cnnt2 = nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=5)

        # Layer #13: ConvTranspose2d, Output shape: [-1, 12, 16, 16], Param #: 4,620
        self.cnnt3 = nn.ConvTranspose2d(in_channels=24, out_channels=12, kernel_size=9)

        # Layer #15: ConvTranspose2d, Output shape: [-1, 3, 32, 32], Param #: 579
        self.cnnt4 = nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=17)

    def forward(self, x):
       # don't forget to return the output
       x = torch.relu(self.cnn1(x))
       x = torch.relu(self.cnn2(x))
       x = torch.relu(self.cnn3(x))
       x = torch.relu(self.cnn4(x))
       x = torch.relu(self.cnnt1(x))
       x = torch.relu(self.cnnt2(x))
       x = torch.relu(self.cnnt3(x))
       x = torch.sigmoid(self.cnnt4(x))

       return x
