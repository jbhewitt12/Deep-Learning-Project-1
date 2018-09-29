import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from itertools import product
import sys 
import time

# -------------------------------------------
# count = 1
# epochs = 2
# accuracy_log = [3,5,6]
# iteration_log = np.linspace(0,epochs,len(accuracy_log))
# btch_size = 4
# learn_rate = 0.04
# num_epochs = 5
# top_title =  'batch size: %d, learn rate %f, epochs: %d' % (count, btch_size, learn_rate, num_epochs)

# plt.figure(count+1)
# plt.suptitle(top_title)
# plt.subplot(211)
# plt.plot(iteration_log, [3,5,6], 'r--', label = 'training loss')
# plt.plot(iteration_log, [2,5,7], 'b-.', label = 'validation loss')
# plt.title('Loss for training and validation')
# plt.xlabel('training iterations')
# legend = plt.legend(loc='upper right', shadow=True)
# plt.subplot(212)
# plt.plot(iteration_log, [3,7,7], 'r--', label = 'training accuracy')
# plt.plot(iteration_log, [3,5,4], 'b-.', label = 'validation accuracy')
# plt.title('Accuracy for training and validation')
# plt.xlabel('training iterations')
# legend = plt.legend(loc='lower right', shadow=True)
# plt.tight_layout()
# plt.subplots_adjust(top=0.88)
# string = '%d.png' % (count)
# plt.show()
# # plt.savefig(string, bbox_inches='tight')
# sys.exit() 

# -------------------------------------------
# batch_opts = [2,4,8]
# lr_opts = [0.001,0.0001,0.00001]
# epoch_opts = [1,2,4,8,16]

# all_options = list(product(batch_opts,lr_opts,epoch_opts))
# # print(all_options)

# count = 0
# btch_size = 4
# learn_rate = 0.001
# num_epochs = 2
# option_labels = ['batches', 'learning rate','epochs']
# for option in all_options:

    
#     with open("Output.txt", "a") as text_file:
        
#         for i in range(3):
#             text_file.write(option_labels[i]+": %s\n" % option[i])
    
#     sys.exit()  
# -------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
            images, labels = images.to(device), labels.to(device)
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


def train(num_epochs, btch_size, learn_rate, trainset, valset):
    start_time = time.time()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=btch_size,
                                          shuffle=True, num_workers=0)

    valloader = torch.utils.data.DataLoader(valset, batch_size=btch_size,
                                         shuffle=False, num_workers=0)

    net = Net()
    net = net.to(device) 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learn_rate, momentum=0.9)

    #initialize variables for logging results
    iteration_log = []
    loss_log = []
    accuracy_log = []
    validation_loss_log = []
    validation_accuracy_log = []
    visualizer_iteration = 1000 #set how many batches are passed before avg accuracy and loss are saved

    for epoch in range(num_epochs):  # loop over the dataset for num_epochs
        elapsed_time = time.time() - start_time
        if elapsed_time > 60 * 20: #If training takes longer than 20 mins then end early
            print('Ended early')
            return net, iteration_log, loss_log, validation_loss_log, accuracy_log, validation_accuracy_log
        running_loss = 0.0
        log_running_loss = 0.0
        running_total = 0
        running_correct = 0

        for i, data in enumerate(trainloader, 0):
            
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
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

            # running losses
            running_loss += loss.item()
            log_running_loss += loss.item()

            if i % visualizer_iteration == 0 and i != 0: #Get avg accuracy and loss for training and validation
                accuracy, loss = evaluate_validation_set(net, valloader, criterion) #find the avg accuracy and loss of the current net on the validation set
                validation_accuracy_log.append(accuracy)
                validation_loss_log.append(loss)

                iteration_log.append(len(trainloader)*(epoch) + i)
                if i != 0:
                    accuracy_log.append(running_correct/running_total)
                    loss_log.append(log_running_loss/visualizer_iteration)
                else:
                    accuracy_log.append(running_correct)
                    loss_log.append(log_running_loss)
            
                running_correct = 0
                running_total = 0
                log_running_loss = 0.0

            if i % 4000 == 3999:    # print every 2000 mini-batches

                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 4000))
                running_loss = 0.0

    return net, iteration_log, loss_log, validation_loss_log, accuracy_log, validation_accuracy_log

def plot_results(option, count, iteration_log, loss_log, validation_loss_log, accuracy_log, validation_accuracy_log):
    btch_size = option[0]
    learn_rate = option[1]
    num_epochs = option[2]
    iteration_log = np.linspace(0,num_epochs,len(iteration_log))
    top_title =  'batch size: %d, learn rate: %f, epochs: %d' % (btch_size, learn_rate, num_epochs)
    plt.figure(count+1)

    plt.suptitle(top_title)
    plt.subplot(211)
    plt.plot(iteration_log, loss_log, 'r--', label = 'training loss')
    plt.plot(iteration_log, validation_loss_log, 'b-.', label = 'validation loss')
    plt.title('Loss for training and validation')
    plt.xlabel('epochs')
    legend = plt.legend(loc='upper right', shadow=True)
    plt.subplot(212)
    plt.plot(iteration_log, accuracy_log, 'r--', label = 'training accuracy')
    plt.plot(iteration_log, validation_accuracy_log, 'b-.', label = 'validation accuracy')
    plt.title('Accuracy for training and validation')
    plt.xlabel('epochs')
    legend = plt.legend(loc='lower right', shadow=True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    string = '%d.png' % (count)
    plt.savefig(string, bbox_inches='tight')
    
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
    accuracy = correct / total
    return accuracy

# _________________________________________

batch_opts = [8]
lr_opts = [0.001]
epoch_opts = [16]

all_options = list(product(batch_opts,lr_opts,epoch_opts))
option_labels = ['batches', 'learning rate','epochs']
# print(all_options)
max_accuracy_list = []
test_accuracy_list = []

count = 0
btch_size = 4
learn_rate = 0.001
num_epochs = 2


for option in all_options:
    btch_size = option[0]
    learn_rate = option[1]
    num_epochs = option[2]
    # print(btch_size,learn_rate, num_epochs)
    start_time = time.time()
    net, iteration_log, loss_log, validation_loss_log, accuracy_log, validation_accuracy_log = train(num_epochs, btch_size, learn_rate, trainset, valset)
    elapsed_time = (time.time() - start_time)/60
    print('** evaluating **')
    print('batch size: %d, learn rate: %f, epochs: %d' % (btch_size, learn_rate, num_epochs))
    test_accuracy = eval_net(net, testloader)
    test_accuracy_list.append(test_accuracy)

    max_accuracy = max(validation_accuracy_log)
    max_accuracy_list.append(test_accuracy)
    min_loss = min(validation_loss_log)
    with open("Output.txt", "a") as text_file:
        text_file.write("\n \nCount: %s, Time: %f\n" % (count, elapsed_time))
        text_file.write("Accuracy of the network on the 10000 test images: %f\n" % (test_accuracy))
        text_file.write("Max Validation accuracy: %f, Min Validation loss: %f\n" % (max_accuracy, min_loss))
        for i in range(3):
            text_file.write(option_labels[i]+": %s\n" % option[i])
        
    print('plotting')    
    plot_results(option, count, iteration_log, loss_log, validation_loss_log, accuracy_log, validation_accuracy_log)
    # plt.show()
    count += 1
    # sys.exit()
total_best_accuracy = max(test_accuracy_list)
best_accuracy_index = max_accuracy_list.index(total_best_accuracy)
print('Final best test accuracy:')
print(total_best_accuracy)
print('At count:')
print(best_accuracy_index)

with open("Output.txt", "a") as text_file:
    text_file.write("\n\n*****-------*****FINAL RESULT*****-------*****\n")
    text_file.write("\n \nCount number of best accuracy: %s, Best test accuracy: %f\n" % (best_accuracy_index, total_best_accuracy))


# plt.show()

# print('iteration_log:')
# print(iteration_log)
# print('loss_log:')
# print(loss_log)






# functions to show the loss
# *** remember to use %matplotlib inline in .ipynb, otherwise, you cannot see the output.
# plt.figure(1)

# plt.subplot(211)
# plt.plot(iteration_log, loss_log, 'r--', iteration_log, validation_loss_log, 'b-.')
# plt.title('Training loss')
# plt.subplot(212)
# plt.plot(iteration_log, accuracy_log, 'r--', iteration_log, validation_accuracy_log, 'b-.')
# plt.title('Training accuracy')





# plt.figure(2)
# plt.subplot(211)
# plt.plot(iteration_log, validation_loss_log, color='red', linestyle='--')
# plt.title('Validation loss')
# plt.subplot(212)
# plt.plot(iteration_log, validation_accuracy_log, color='blue', linestyle='-.')
# plt.title('Validation accuracy')


# fig.savefig("test.png")


