import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import precision_score, recall_score

from mobileNetV2 import mobilenetv2
from effnetv2 import effnetv2_l, effnetv2_m, effnetv2_s,effnetv2_xl

transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),        
    transforms.Normalize(mean=[0.5], std=[0.5])  
])

trainFile = 'YOUR_FILE_PATH'
testFile = 'YOUR_FILE_PATH'


train_dataset = ImageFolder(root=trainFile, transform=transform)
test_dataset = ImageFolder(root=testFile, transform=transform)
print("Train Class to index mapping:", train_dataset.class_to_idx)
print("Test Class to index mapping:", test_dataset.class_to_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# model = mobilenetv2()
model = effnetv2_l()
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)  
        self.pool = nn.MaxPool2d(2, 2)    
        self.conv2 = nn.Conv2d(16, 32, 3) 
        self.fc1 = nn.Linear(32 * 14 * 14, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)   
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 14 * 14)  
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(num_epochs):
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)


                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())


        accuracy = 100 * sum([1 for p, t in zip(all_predictions, all_labels) if p == t]) / len(all_labels)
        precision = precision_score(all_labels, all_predictions, average='binary')
        recall = recall_score(all_labels, all_labels, average='binary')

        print(f'Test Metrics - Accuracy: {accuracy:.2f}%, '
              f'Precision: {precision:.4f}, '
              f'Recall: {recall:.4f}')

        TP = sum([1 for p, t in zip(all_predictions, all_labels) if p == 1 and t == 1])
        FP = sum([1 for p, t in zip(all_predictions, all_labels) if p == 1 and t == 0])
        TN = sum([1 for p, t in zip(all_predictions, all_labels) if p == 0 and t == 0])
        FN = sum([1 for p, t in zip(all_predictions, all_labels) if p == 0 and t == 1])

        accuracy = 100 * (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        print(f'HTest Metrics - Accuracy: {accuracy:.2f}%, '
              f'HPrecision: {precision:.4f}, '
              f'HRecall: {recall:.4f}')
        print(f'HTP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}')  # 可选：打印混淆矩阵值进行验证

if __name__ == '__main__':
    train_model(num_epochs=50)
    
    torch.save(model.state_dict(), 'cnn_model.pth')
