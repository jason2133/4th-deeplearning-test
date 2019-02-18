# Import
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch import nn
from torchvision import datasets, transforms

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


num_classes = 10 # Class는 총 10개 - 'bear', 'bird', 'butterfly', 'car', 'cat', 'deer', 'dog', 'horse', 'sheep', 'tiger'
num_epochs = 30
batch_size = 100
learning_rate = 0.001

# Transform
transform = transforms.Compose(
                   [transforms.Resize((32,32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])

# Transform : 32 * 32 -> Tensor -> Normalize 정규화 0.5

# Load Data - Train & Test
# 다른 컴퓨터에서는 경로는 당연히 수정해야 됨
# Train 경로 : "C:/Users/jason/Desktop/prography_dl/datasets/train"
# Test 경로 : "C:/Users/jason/Desktop/prography_dl/datasets/test"

train_dataset = dset.ImageFolder(root = "C:/Users/jason/Desktop/prography_dl/datasets/train", transform=transform)
test_dataset = dset.ImageFolder(root = "C:/Users/jason/Desktop/prography_dl/datasets/test",transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True)

# Class들
classes=('bear', 'bird', 'butterfly', 'car', 'cat', 'deer', 'dog', 'horse', 'sheep', 'tiger')


# 이미지 보여주기 위한 함수 정의
def imshow(image):
    image = image / 2 + 0.5
    numpy_image = image.numpy()
    plt.imshow(np.transpose(numpy_image, (1, 2, 0)))
    plt.show()

# CNN 모델 정의하기
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, 3),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(576, 120), nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84), nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(84, 21), nn.ReLU()
        )
        self.fc4 = nn.Linear(21, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out

if __name__ == '__main__':
    model = CNNModel().to(device)
    print(model)

    # Loss
    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    for epoch in range(num_epochs):
        average_cost = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 최적화 -> Zero_gradient 활용
            optimizer.zero_grad()

            # Forward, Backward, Optimizer
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Compute average cost
            average_cost += loss.item() / batch_size

        if epoch % 2 == 0:
            print("Epoch [%d / %d], Loss: %.8f" % (epoch + 1, num_epochs, average_cost))

    print("END")


    model.eval()  # Evaluate

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Predict
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        print("Accuracy : ", 100 * (float(correct)) / total)