import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F



transform = transforms.ToTensor()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

print('initial trainset length')
print(len(trainset))
print('testset length')
print(len(testset))

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Your split codes comes here.
# You need to random select the 5000 validation images 



# found here: https://pytorch.org/docs/stable/data.html
valset, trainset = torch.utils.data.random_split(trainset, [5000,45000])



def evaluate_validation_set(net, valloader, criterion):
    correct = 0
    total = 0
    running_loss = 0.0
    count = 0
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            count += 1

    accuracy = correct / total
    loss = running_loss/count
    return accuracy, loss
# besides, you can also make modifications for faster training 
# by selecting a subset of the original dataset.


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




import torch.optim as optim

def train(num_epochs, btch_size, learn_rate, trainset, valset):

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=btch_size,
                                          shuffle=True, num_workers=0)

    valloader = torch.utils.data.DataLoader(valset, batch_size=btch_size,
                                         shuffle=False, num_workers=0)

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learn_rate, momentum=0.9)

    iteration_log = []
    loss_log = []
    accuracy_log = []
    validation_loss_log = []
    validation_accuracy_log = []
    visualizer_iteration = 1000
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        running_loss2 = 0.0
        running_total = 0
        running_correct = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #get accuracy 
            _, predicted = torch.max(outputs.data, 1)
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()

            # print statistics
            running_loss += loss.item()
            running_loss2 += loss.item()

            if i % visualizer_iteration == 0 and i != 0:
                accuracy, loss = evaluate_validation_set(net, valloader, criterion)
                validation_accuracy_log.append(accuracy)
                validation_loss_log.append(loss)

                iteration_log.append(len(trainloader)*(epoch) + i)
                if i != 0:
                    accuracy_log.append(running_correct/running_total)
                    loss_log.append(running_loss2/visualizer_iteration)
                else:
                    accuracy_log.append(running_correct)
                    loss_log.append(running_loss2)
            
                running_correct = 0
                running_total = 0
                running_loss2 = 0.0
                # print('iteration_log:')
                # print(iteration_log)
                # print('loss_log:')
                # print(loss_log)
                # print('accuracy_log:')
                # print(accuracy_log)

            if i % 2000 == 1999:    # print every 2000 mini-batches

                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    return iteration_log, loss_log, validation_loss_log, accuracy_log, validation_accuracy_log

btch_size = 4
learn_rate = 0.001
num_epochs = 2

iteration_log, loss_log, validation_loss_log, accuracy_log, validation_accuracy_log = train(num_epochs, btch_size, learn_rate, trainset, valset)

# print('iteration_log:')
# print(iteration_log)
# print('loss_log:')
# print(loss_log)


import matplotlib.pyplot as plt
import numpy as np

# functions to show the loss
# *** remember to use %matplotlib inline in .ipynb, otherwise, you cannot see the output.
# plt.figure(1)

# plt.subplot(211)
# plt.plot(iteration_log, loss_log, 'r--', iteration_log, validation_loss_log, 'b-.')
# plt.title('Training loss')
# plt.subplot(212)
# plt.plot(iteration_log, accuracy_log, 'r--', iteration_log, validation_accuracy_log, 'b-.')
# plt.title('Training accuracy')


plt.figure(1)

plt.subplot(211)
plt.plot(iteration_log, loss_log, 'r--', label = 'training loss')
plt.plot(iteration_log, validation_loss_log, 'b-.', label = 'validation loss')
plt.title('Loss for training and validation')
plt.xlabel('training iterations')
legend = plt.legend(loc='upper right', shadow=True)
plt.subplot(212)
plt.plot(iteration_log, accuracy_log, 'r--', label = 'training accuracy')
plt.plot(iteration_log, validation_accuracy_log, 'b-.', label = 'validation accuracy')
plt.title('Accuracy for training and validation')
plt.xlabel('training iterations')
legend = plt.legend(loc='lower right', shadow=True)

plt.tight_layout()
plt.show()


# plt.figure(2)
# plt.subplot(211)
# plt.plot(iteration_log, validation_loss_log, color='red', linestyle='--')
# plt.title('Validation loss')
# plt.subplot(212)
# plt.plot(iteration_log, validation_accuracy_log, color='blue', linestyle='-.')
# plt.title('Validation accuracy')


# fig.savefig("test.png")


def eval_net(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

eval_net(net, testloader)