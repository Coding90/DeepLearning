import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Parameter setting
batch_size = 64
learning_rate = 0.0002
epochs = 15



def check_data(mnist_train, mnist_test):
    # test len train_dataset and test_datset
    print("---------------check data-------------")
    print("train_size: ", len(mnist_train))
    print("test_size:  ", len(mnist_test))

    # test front 10 images in train dataset
    for i in range(9):
        img = mnist_train[i][0].numpy()
        label = mnist_train[i][1]
        plt.imshow(img[0], cmap='gray')
        plt.title(label)
        plt.show()


# train model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1,16,5),
            nn.ReLU(),
            nn.Conv2d(16,32,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(0.25),
            nn.Conv2d(32,64,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(64*3*3, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.view(batch_size, -1)
        out = self.fc_layer(out)
        return out


def main():


    # data load https://www.aiworkbox.com/lessons/load-mnist-dataset-from-pytorch-torchvision
    mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
    mnist_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

    #Check data
    # check_data(mnist_train, mnist_test)


    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True, pin_memory=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    loss_arr = []


    # train start
    for epoch in range(epochs):
        print("epoch: ", epoch)
        for batch_idx, [image, label] in enumerate(train_loader):
            x = image.to(device)
            y_ = label.to(device)

            optimizer.zero_grad()
            output = model.forward(x)
            loss = loss_func(output, y_)
            loss.backward()
            optimizer.step()
            # print("batch_idx: ", batch_idx)
            if batch_idx % 10 == 0:
                print('Train Epoch: {}    {}/{}  {:.0f}%  loss: {:.6f}'.format(epoch, batch_idx * len(image), len(train_loader.dataset),
                           100 * batch_idx / len(train_loader), loss.item()))
                loss_arr.append(loss.cpu().detach().numpy())
    correct = 0
    total = 0

    with torch.no_grad():
        for image, label in test_loader:
            x = image.to(device)
            y_ = label.to(device)

            output = model.forward(x)
            _, output_index = torch.max(output, 1)
            total += label.size(0)
            correct += (output_index == y_).sum().float()
        print("accuracy data: {}", format(100*correct/total))








if __name__ == '__main__':
    main()