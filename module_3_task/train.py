import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import ResNet18
from sklearn.metrics import classification_report
from utils import save_model

device = ("cuda" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training

seed = torch.initial_seed()
# create model
model = ResNet18()
model = model.to(device)

# Data set up

transform = transforms.Compose([
          transforms.Resize((90, 90)),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.5, 0.5,0.5], std=[0.5,0.5,0.5])
          ])

batch_size = 64
image_folder = './Images'
data = datasets.ImageFolder(image_folder, transform=transform)

# Splitting the data into train and test sets
train_set, test_set = torch.utils.data.random_split(data, [int(len(data) * 0.90), len(data) - int(len(data) * 0.90)])
train_data_size = len(train_set)
test_data_size = len(test_set)

# Data loaders
trainLoader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True)
testLoader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# Training the model
epochs = 30
train_loss = []

for epoch in range(epochs):
   
    total_train_loss = 0
    
    # training our model
    for idx, (image, label) in enumerate(trainLoader):
        image, label = image.to(device), label.to(device)

        optimizer.zero_grad()
        pred = model(image)

        loss = criterion(pred, label)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()

    total_train_loss = total_train_loss / (idx + 1)
    train_loss.append(total_train_loss)

    print(f'Epoch: {epoch} | Train Loss: {total_train_loss}')


# Model Evaluation

# Set the model to evaluation mode
model.eval()

true_labels_train = []
predicted_labels_train = []

# Iterate through the training data
with torch.no_grad():
    for images, labels in trainLoader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Get predicted labels
        _, predicted = torch.max(outputs, 1)
        
        # Append true and predicted labels
        true_labels_train.extend(labels.cpu().numpy())
        predicted_labels_train.extend(predicted.cpu().numpy())

# Generate classification report for training set
report_train = classification_report(true_labels_train, predicted_labels_train)

print("Classification Report for Training Set:")
print(report_train)

# Set the model to evaluation mode
model.eval()

true_labels_test = []
predicted_labels_test = []

# Iterate through the test data
with torch.no_grad():
    for images, labels in testLoader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Get predicted labels
        _, predicted = torch.max(outputs, 1)
        
        # Append true and predicted labels
        true_labels_test.extend(labels.cpu().numpy())
        predicted_labels_test.extend(predicted.cpu().numpy())

# Generate classification report for test set
report_test = classification_report(true_labels_test, predicted_labels_test)

print("Classification Report for Test Set:")
print(report_test)

# Save the model
save_model(model, "resnet18_model.pth")