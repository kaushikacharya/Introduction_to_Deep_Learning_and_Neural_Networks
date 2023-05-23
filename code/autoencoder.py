import torch
import torch.nn as nn

seed = 172
torch.manual_seed(seed)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]

        # ------------ Encoder ------------
        # Layer #1: Conv2d, Output shape: [-1, 12, 16, 16], Param #: 588
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=4, stride=2, padding=1, dilation=1)

        # Layer #2: ReLU
        self.relu1 = nn.ReLU()

        # Layer #3: Conv2d, Output shape: [-1, 24, 8, 8], Param #: 4,632
        self.cnn2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=4, stride=2, padding=1, dilation=1)

        # Layer #4: ReLU
        self.relu2 = nn.ReLU()

        # Layer #5: Conv2d, Output shape: [-1, 48, 4, 4], Param #: 18,480
        self.cnn3 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=4, stride=2, padding=1, dilation=1)

        # Layer #6: ReLU
        self.relu3 = nn.ReLU()

        # Layer #7: Conv2d, Output shape: [-1, 96, 2, 2], Param #: 73,824
        self.cnn4 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=4, stride=2, padding=1, dilation=1)

        # Layer #8: ReLU
        self.relu4 = nn.ReLU()

        # ------------ Decoder ------------
        # Layer #9: ConvTranspose2d, Output shape: [-1, 48, 4, 4], Param #: 73,776
        self.cnnt1 = nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=4, stride=2, padding=1, output_padding=0, dilation=1)

        # Layer #10: ReLU
        self.relu5 = nn.ReLU()

        # Layer #11: ConvTranspose2d, Output shape: [-1, 24, 8, 8], Param #: 18,456
        self.cnnt2 = nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=4, stride=2, padding=1, output_padding=0, dilation=1)

        # Layer #12: ReLU
        self.relu6 = nn.ReLU()

        # Layer #13: ConvTranspose2d, Output shape: [-1, 12, 16, 16], Param #: 4,620
        self.cnnt3 = nn.ConvTranspose2d(in_channels=24, out_channels=12, kernel_size=4, stride=2, padding=1, output_padding=0, dilation=1)

        # Layer #14: ReLU
        self.relu7 = nn.ReLU()

        # Layer #15: ConvTranspose2d, Output shape: [-1, 3, 32, 32], Param #: 579
        self.cnnt4 = nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=4, stride=2, padding=1, output_padding=0, dilation=1)

        # Layer #16
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
       # don't forget to return the output
       x = self.relu1(self.cnn1(x))
       x = self.relu2(self.cnn2(x))
       x = self.relu3(self.cnn3(x))
       x = self.relu4(self.cnn4(x))
       x = self.relu5(self.cnnt1(x))
       x = self.relu6(self.cnnt2(x))
       x = self.relu7(self.cnnt3(x))
       x = self.sigmoid(self.cnnt4(x))

       return x
