
class Model(nn.Sequential):
    def __init__(self):
        super(Model, self).__init__()

        self.Mfcc = MFCC_layer()
        self.Conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding=1)
        self.Relu_1 = nn.ReLU()
        self.BatchNorm_1 = nn.BatchNorm2d(num_features=32)
        self. MaxPool_1 = nn.MaxPool2d(kernel_size=2, padding=1)

        self.Conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1)
        self.Relu_2 = nn.ReLU()
        self.BatchNorm_2 = nn.BatchNorm2d(num_features=64)
        self.MaxPool_2 = nn.MaxPool2d(kernel_size=2, padding=1)

        self.Conv_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1)
        self.Relu_3 = nn.ReLU()
        self.BatchNorm_3 = nn.BatchNorm2d(num_features=64)

        self.Flatten = nn.Flatten()
        # in_features for linear layer should be the multiplication of the last 3 digits of the ReLU-8 output shape (64*11*51)
        # which is equal to the last number in the Flatten output shape (35904)
        self.Linear_1 = nn.Linear(in_features=64*11*51, out_features=64, bias=True)
        self.Relu_L_1 = nn.ReLU()
        self.BatchNorm_4 = nn.BatchNorm1d(num_features=64)
        self.Linear_2 = nn.Linear(in_features=64, out_features=1, bias=True)
        self.BatchNorm_5 = nn.BatchNorm1d(num_features=1)
        self.Out_activation = nn.Sigmoid()
    
    class MFCC_layer(self, nn.Module):
        def __init__(self):
            super(MFCC_layer, self).__init__()
            self.sample_rate = 40000/5
            self.transform_MFCC = transforms.MFCC(sample_rate = self.sample_rate)

        def forward(self, x):
            with torch.no_grad():
                x = self.transform_MFCC(x)
            return x
    
    def normalize(self, inputs):
        mf = self.Mfcc

    def forward(self, mf):
        c1 = self.Conv_1(inputs)
        b1 = self.BatchNorm_1(c1)
        r1 = self.Relu_1(b1)
        m1 = self.MaxPool_1(r1)

        c2 = self.Conv_2(m1)
        b2 = self.BatchNorm_1(c2)
        r2 = self.Relu_1(b2)
        m2 = self.MaxPool_2(r2)

        c3 = self.Conv_3(m2)
        b3 = self.BatchNorm_1(c3)
        r3 = self.Relu_1(b3)
        x = self.Flatten(r3)

        x = self.Linear_1(x)
        x = self.BatchNorm_1(x)
        x = self.Relu_1(x)
        x = self.Linear_2(x)
        x = self.BatchNorm_5(x)
        x = self.Out_activation(x)

        return x
