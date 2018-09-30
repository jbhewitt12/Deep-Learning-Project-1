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

def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad: 
            continue # frozen weights 
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list: 
            no_decay.append(param)
        else: 
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]

class Net(nn.Module):
    
    def __init__(self):
        
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 28, 5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(28, 42, 5, padding=2)
        self.conv3 = nn.Conv2d(42, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def train(num_epochs, btch_size, learn_rate, trainset, valset):
    use_weight_decay = True

    start_time = time.time()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=btch_size,
                                          shuffle=True, num_workers=0)

    valloader = torch.utils.data.DataLoader(valset, batch_size=btch_size,
                                         shuffle=False, num_workers=0)

    net = Net()
    net = net.to(device) 

    criterion = nn.CrossEntropyLoss()

    if use_weight_decay:
        params = add_weight_decay(net, 2e-5) # adding weight decay to network
        optimizer = optim.SGD(params, lr=learn_rate, momentum=0.95)
    else:
        optimizer = optim.SGD(net.parameters(), lr=learn_rate, momentum=0.9)

    #initialize variables for logging results
    iteration_log = []
    loss_log = []
    accuracy_log = []
    validation_loss_log = []
    validation_accuracy_log = []
    visualizer_iteration = 2000 #set how many batches are passed before avg accuracy and loss are saved
    last_epoch = False

    for epoch in range(num_epochs):  # loop over the dataset for num_epochs
        # scheduler.step()
        
        #Simple learning rate scheduler 
        if epoch < 1:
            lr = 0.005
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        elif epoch < 3:
            lr = 0.001 
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = 0.0001 
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

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
                print('validation accuracy: %f' % accuracy)

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

    
    # when you load the model back again via state_dict method, remember to do MyModel.eval(), otherwise the results will differ.
    torch.save(net.state_dict(), './modified.pth')
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
    
def eval_net(testing, net, testloader):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    accuracy = correct / total
    return accuracy

# ************** MAIN ************** #

testing = True #Change this to False to train the network.

test_baseline = False #change this to False to test modified.pth 

horizontal_flip = False #Change this to true to add horizontal flipping transform

device = 'cuda' if torch.cuda.is_available() else 'cpu' #If cuda is available then the program will use it, otherwise it will use the cpu. 

if horizontal_flip:
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         # transforms.RandomVerticalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
else:
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

# found here: https://pytorch.org/docs/stable/data.html
valset, trainset = torch.utils.data.random_split(trainset, [5000,45000])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if testing: #TESTING ------------------------------------------------
    
    net = Net()
    net = net.to(device) 
    net.eval()
    if test_baseline:
        print('Testing baseline.pth')
        net.load_state_dict(torch.load('baseline.pth'))
    else:
        print('Testing modified.pth')
        net.load_state_dict(torch.load('modified.pth'))
    accuracy = eval_net(testing, net, testloader)

else: #TRAINING ------------------------------------------------
    
    #This is for running through different training options for batch size, learning rate and number of epochs 
    batch_opts = [10]
    lr_opts = [0.01]
    epoch_opts = [5]

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
        test_accuracy = eval_net(testing, net, testloader)
        test_accuracy_list.append(test_accuracy)

        max_accuracy = max(validation_accuracy_log)
        max_accuracy_list.append(test_accuracy)
        min_loss = min(validation_loss_log)
        with open("Output.txt", "a") as text_file:
            text_file.write("\n \nTime: %f mins  \n" % (elapsed_time))
            text_file.write("Accuracy of the network on the 10000 test images: %f  \n" % (test_accuracy))
            text_file.write("Max Validation accuracy: %f, Min Validation loss: %f  \n" % (max_accuracy, min_loss))
            
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
