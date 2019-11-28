import numpy as np
import torch
import torch.nn as nn

def train_cnn(training_set, labels):
    train_data = torch.from_numpy(np.array(training_set))
    train_labels = torch.from_numpy(np.array(labels))

    # Hyper Parameters
    EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
    LR = 0.001              # learning rate

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Sequential(         # input shape (1, 1, 6)
                nn.Conv2d(
                    in_channels=1,              # input height
                    out_channels=16,            # n_filters
                    kernel_size=5,              # filter size
                    stride=1,                   # filter movement/step
                    padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
                ),                              # output shape (16, 28, 28)
                nn.ReLU(),                      # activation
                nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
            )
            self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
                nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
                nn.ReLU(),                      # activation
                nn.MaxPool2d(2),                # output shape (32, 7, 7)
            )
            self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
            output = self.out(x)
            return output, x    # return x for visualization


    cnn = CNN()
    cnn.cuda()
    print(cnn)  # net architecture

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

    # training and testing
    for epoch in range(EPOCH):
        for b_x, b_y in zip(train_data, train_labels):   # gives batch data, normalize x when iterate train_loader

            b_x = b_x.cuda()
            b_y = b_y.cuda()

            output = cnn(b_x)[0]               # cnn output
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients


def train(training_set, labels):
    train_data = torch.from_numpy(np.array(training_set)).float()
    train_labels = torch.from_numpy(np.array(labels)).float()
    print(train_data.shape)
    print(train_labels.shape)

    LR = 0.001              # learning rate

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # self.hidden = torch.nn.Linear(n_feature, n_hidden)
            # self.predict = torch.nn.Linear(n_hidden, n_output)
            self.network = nn.Sequential(nn.Linear(train_data.shape[1], 10), nn.ReLU(), nn.Linear(10, 1))

        def forward(self, x):
            x = self.network(x)
            return x

    net = Net()
    print(net)

    optimizer = torch.optim.SGD(net.parameters(), lr=LR)
    loss_fnc = torch.nn.MSELoss()

    # plt.ion()

    for t in range(200):
        predict = net(train_data)
        loss = loss_fnc(predict, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()