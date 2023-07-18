# %%
import torch
import os
from torch import nn
from torch.utils.data import DataLoader,  Subset, random_split, Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
from PIL import Image
import glob

print(os.getcwd())
if os.getcwd().split('/')[-1] != 'Dogs-Cats':
    os.chdir('supervised/Dogs-Cats')
# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

# %%
class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        label = str.split(img_name, '.')[0]  # Convert label to integer
        if label == "dog":
            label = torch.tensor([1, 0])  # One-hot encode as [1, 0] for dog
        elif label == "cat":
            label = torch.tensor([0, 1])  # One-hot encode as [0, 1] for cat
        
        try:
            return image, label.float()
        except Exception as e:
            print(e)
            print(img_name, label)
            breakpoint()        

# %%
train_path_dir = 'data/train'
test_path_dir = 'data/test1'

batch_size=64
image_size = (256, 256)  # Specify the desired image size
learning_rate = 1e-3
epochs = 300
val_ratio = 0.2

transform = transforms.Compose([
    Resize(image_size),
    ToTensor()
])

# %%
train_data = CatDogDataset(train_path_dir, transform=transform)
test_data = CatDogDataset(test_path_dir, transform=transform)
val_ratio = 0.2

# Determine the sizes of the training and validation sets
dataset_size = len(train_data)
val_size = int(val_ratio * dataset_size)
train_size = dataset_size - val_size

# Split the dataset into training and validation sets
train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

# Create data loaders for training and validation sets
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# %%
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate the input size to the fully connected layers based on the image size
        self.fc_input_size = 64 * (image_size[0] // 8) * (image_size[1] // 8)

        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

model = CNNModel().to(device)

# %%
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# %%
def train_model(model, train_dataloader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_dataloader:
        print(next(model.parameters()).device)
        print(images.device, labels.device)
        print(type(images), type(labels))
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return model


# %%
def test_model(model, val_dataloader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get the predicted labels

            # Convert one-hot encoded labels back to their original format
            _, labels = torch.max(labels, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# %%
def test_highest_accuracy_model(accuracy):
    model_path = f"../models/dogs-cats/dogs-cats-cnn.pth-accurracy-{accuracy:.2f}"
    filenames = os.listdir('../models/dogs-cats')
    highest_accuracy = max([float(str.split(filename, '-')[-1]) for filename in filenames])
    filepath_with_highest_accuracy = f"../models/dogs-cats/dogs-cats-cnn.pth-accurracy-{highest_accuracy}"
    if accuracy > highest_accuracy:
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")
        #delete previous model
        os.remove(filepath_with_highest_accuracy) 

# %%
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    model = train_model(model, train_dataloader, loss_fn, optimizer, device)
    accuracy = test_model(model, val_dataloader, device)
    test_highest_accuracy_model(accuracy)
    print(f"Epoch [{epoch+1}/{epochs}], Test Accuracy: {accuracy:.2f}%")
print("Done!")

# %%
model_path = f"../models/dogs-cats/dogs-cats-cnn.pth-accurracy-{accuracy:.2f}"
torch.save(model.state_dict(), model_path)
print(f"Saved PyTorch Model State to {model_path}")

# %%
model = CNNModel().to(device)
filenames = os.listdir('../models/dogs-cats')
highest_accuracy = max([float(str.split(filename, '-')[-1]) for filename in filenames])
filepath_with_highest_accuracy = f"../models/dogs-cats/dogs-cats-cnn.pth-accurracy-{highest_accuracy}"
model.load_state_dict(torch.load(filepath_with_highest_accuracy, map_location=device))


