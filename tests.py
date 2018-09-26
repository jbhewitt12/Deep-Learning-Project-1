import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


transform = transforms.ToTensor()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

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

valloader = torch.utils.data.DataLoader(valset, batch_size=4,
                                         shuffle=False, num_workers=0)

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


net = Net()


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

iteration_log = []
loss_log = []
accuracy_log = []
visualizer_iteration = 100
for epoch in range(2):  # loop over the dataset multiple times

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

# print('iteration_log:')
# print(iteration_log)
# print('loss_log:')
# print(loss_log)


import matplotlib.pyplot as plt
import numpy as np

# functions to show the loss
# *** remember to use %matplotlib inline in .ipynb, otherwise, you cannot see the output.
fig, ax = plt.subplots()
ax.plot(iteration_log, loss_log, color='red', linestyle='--')
# ax.plot([1,1000,7000], [0.1, 0.2, 0.15], color='blue', linestyle='-.')
ax.plot(iteration_log, accuracy_log, color='blue', linestyle='-.')

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

# fig.savefig("test.png")
plt.show()


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