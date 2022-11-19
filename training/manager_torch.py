import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
#from model_torch import Net
import torch.nn as nn
import neptune
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)


#device = "cpu"
class NetworkManager:
    '''
    Helper class to manage the generation of subnetwork training given a dataset
    '''
    def __init__(self, dataset, epochs=5, child_batchsize=128, acc_beta=0.8, clip_cifar10s=0.0):
        '''
        Manager which is tasked with creating subnetworks, training them on a dataset, and retrieving
        cifar10s in the term of accuracy, which is passed to the controller RNN.
        Args:
            dataset: a tuple of 4 arrays (X_train, y_train, X_val, y_val)
            epochs: number of epochs to train the subnetworks
            child_batchsize: batchsize of training the subnetworks
            acc_beta: exponential weight for the accuracy
            clip_cifar10s: float - to clip cifar10s in [-range, range] to prevent
                large weight updates. Use when training is highly unstable.
        '''
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        self.epochs = epochs
        self.batchsize = child_batchsize
        
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batchsize,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batchsize,
                                                 shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.clip_cifar10s = clip_cifar10s

        self.beta = acc_beta
        self.beta_bias = acc_beta
        self.moving_acc = 0.0

    def get_cifar10s(self, model_fn, actions):
        '''
        Creates a subnetwork given the actions predicted by the controller RNN,
        trains it on the provided dataset, and then returns a cifar10.
        Args:
            model_fn: a function which accepts one argument, a list of
                parsed actions, obtained via an inverse mapping from the
                StateSpace.
            actions: a list of parsed actions obtained via an inverse mapping
                from the StateSpace. It is in a specific order as given below:
                Consider 4 states were added to the StateSpace via the `add_state`
                method. Then the `actions` array will be of length 4, with the
                values of those states in the order that they were added.
                If number of layers is greater than one, then the `actions` array
                will be of length `4 * number of layers` (in the above scenario).
                The index from [0:4] will be for layer 0, from [4:8] for layer 1,
                etc for the number of layers.
                These action values are for direct use in the construction of models.
        Returns:
            a cifar10 for training a model with the given actions
        '''
        net = model_fn(actions,10).to(self.device)
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.Adam(net.parameters(), lr=0.001,betas=(0.9, 0.999))

        for epoch in range(self.epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs.to(self.device))
                loss = criterion(outputs, labels.to(self.device))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 200 == 199:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for data in self.testloader:
                    images, labels = data
                    outputs = net(images.to(self.device))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted.cpu() == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))
        
        acc=correct/total
        cifar10 = (acc - self.moving_acc)

        # if cifar10s are clipped, clip them in the range -0.05 to 0.05
        if self.clip_cifar10s:
            cifar10 = np.clip(cifar10, -0.05, 0.05)

        # update moving accuracy with bias correction for 1st update
        if self.beta > 0.0 and self.beta < 1.0:
            self.moving_acc = self.beta * self.moving_acc + (1 - self.beta) * acc
            self.moving_acc = self.moving_acc / (1 - self.beta_bias)
            self.beta_bias = 0
            cifar10 = np.clip(cifar10, -0.1, 0.1)

        print()
        print("Manager: EWA Accuracy = ", self.moving_acc)
        neptune.log_metric("EWA",  self.moving_acc)
        return cifar10, acc
    
    
    def get_batch_jacobian(self,net, x, target):
        net.zero_grad()
        x.requires_grad_(True)
        y = net(x)
        y.backward(torch.ones_like(y))
        jacob = x.grad.detach()

        return jacob, target.detach()


    def eval_score(self,jacob, labels=None):
        corrs = np.corrcoef(jacob)
        v, _  = np.linalg.eig(corrs)
        k = 1e-5
        return -np.sum(np.log(v + k) + 1./(v + k))


    
    
    
    def get_cifar10s_wt(self, model_fn, actions):
        
        net = model_fn(actions,1).to(self.device)
        scores=[]
        batches_average=2
        count_batches =0 
        for i, data in enumerate(self.trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            jacobs, labels= self.get_batch_jacobian(net, inputs, labels)
            jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
            try:
                s = self.eval_score(jacobs, labels)
            except Exception as e:
                print(e)
                s = np.nan
            scores.append(s)
            count_batches+=1
            if(count_batches>batches_average):
                break
            
            
        cifar10=1+np.mean(scores)/1000
        
        cifar10 = np.clip(cifar10, 0, 1)

        
        cifar10 = (cifar10 - self.moving_acc)

        # if cifar10s are clipped, clip them in the range -0.05 to 0.05
        
        # update moving accuracy with bias correction for 1st update
        if self.beta > 0.0 and self.beta < 1.0:
            self.moving_acc = self.beta * self.moving_acc + (1 - self.beta) * cifar10
            self.moving_acc = self.moving_acc / (1 - self.beta_bias)
            self.beta_bias = 0
            #cifar10 = np.clip(cifar10, -0.1, 0.1)

        print()
        return cifar10, 0.0