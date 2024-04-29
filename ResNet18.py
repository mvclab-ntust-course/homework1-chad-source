# import wandb

# wandb.login()
# wandb.init(
    
#     project="hw1",
    
#     config={
#     "learning_rate": 0.001,
#     "architecture": "CNN",
#     "dataset": "CIFAR-100",
#     "epochs": 10,
#     }
# )

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class BasicBlock(nn.Module):

    def __init__(self, input, output, s):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input, out_channels=output, kernel_size=3, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(output)
        self.conv2 = nn.Conv2d(in_channels=output, out_channels=output, kernel_size=3, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(output)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        return F.relu(out)

class NotBasicBlock(nn.Module):

  def __init__(self, input, output, s):
        super(NotBasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input, out_channels=output, kernel_size=3, stride=s[0], padding=1)
        self.bn1 = nn.BatchNorm2d(output)
        self.conv2 = nn.Conv2d(in_channels=output, out_channels=output, kernel_size=3, stride=s[1], padding=1)
        self.bn2 = nn.BatchNorm2d(output)
        self.shortcut = nn.Sequential(nn.Conv2d(input, output, kernel_size=1, stride=s[0], padding=0),nn.BatchNorm2d(output))
    
  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = out + self.shortcut(x)

    return F.relu(out)
    
class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = nn.Sequential(BasicBlock(64, 64, 1),BasicBlock(64, 64, 1),NotBasicBlock(64, 128, [2, 1]))

        self.layer2 = nn.Sequential(BasicBlock(128, 128, 1),NotBasicBlock(128, 256, [2, 1]))

        self.layer3 = nn.Sequential(BasicBlock(256, 256, 1),NotBasicBlock(256, 512, [2, 1]))

        self.layer4 = BasicBlock(512, 512, 1)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, 100)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset, validset = torch.utils.data.random_split(
    torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform = transform),
    lengths=[40000, 10000])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validset, batch_size=16, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)
    classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')

    net = ResNet18().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(1):  # epoch
        running_loss = 0.0
        for i, data in (enumerate(trainloader, 0)):

            inputs, labels = data[0].to(device), data[1].to(device)

            # zero gradient
            optimizer.zero_grad()

            # forward, backward, optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # wandb.log({"loss": loss})
            running_loss += loss.item()
            if i % 250 == 0:
                # validation
                net.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in validloader:
                        images, labels = data[0].to(device), data[1].to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                # wandb.log({"accuracy": correct/total})
            running_loss = 0.0
    print('Finished Training')
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    
    # calculate accuracy
    class_correct = [0] * 100
    class_total = [0] * 100
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    for i in range(100):
        if class_total[i] > 0:
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
        else:
            print('Accuracy of %5s : N/A (no training examples)' % (classes[i]))


    # wandb.finish()