import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import rasterio
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import models as models
import os
from sklearn.metrics import fbeta_score

class RockDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        with rasterio.open(self.image_paths[idx]) as src:
            image = src.read()  # Shape: (4, height, width)
            image = torch.FloatTensor(image)
        
        # Normalize the image
        image = (image - image.mean()) / (image.std() + 1e-6)
        
        if self.transform:
            image = self.transform(image)
        label = torch.FloatTensor([self.labels[idx]])
        return image, label

class ResNet_kelan(nn.Module):
    def __init__(self, input_channel_dim: int = 4, num_classes: int = 1):
        super(ResNet_kelan, self).__init__()
        self.model = models.resnet50(num_classes=num_classes)
        self.model.conv1 = torch.nn.Conv2d(input_channel_dim, 64, kernel_size=(7, 7), 
                                         stride=(2, 2), padding=(3, 3), bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x.squeeze(1)


class RockDetectionCNN(nn.Module):
    def __init__(self):
        super(RockDetectionCNN, self).__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def evaluate_model(model, data_loader, device, threshold=0.3):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs.squeeze()
            predicted = (outputs > threshold).float()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    true_positives = np.sum((all_preds == 1) & (all_labels == 1))
    false_negatives = np.sum((all_preds == 0) & (all_labels == 1))
    false_positives = np.sum((all_preds == 1) & (all_labels == 0))
    
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    f2_score = 5 * (precision * recall) / (4 * precision + recall)
    accuracy = np.mean(all_preds == all_labels)
    
    return {
        'recall': recall,
        'precision': precision,
        'f2': f2_score,
        'accuracy': accuracy
    }

def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    f2_history = []
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    model = model.to(device)
    best_val_f2 = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.squeeze()
            labels = labels.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        val_metrics = evaluate_model(model, val_loader, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Metrics: Recall: {val_metrics["recall"]:.4f}, '
              f'Precision: {val_metrics["precision"]:.4f}, '
              f'Accuracy: {val_metrics["accuracy"]:.4f}, '
              f'F2 Score: {val_metrics["f2"]:.4f}\n')
        
        f2_history.append(val_metrics["f2"])
        
        if val_metrics['f2'] > best_val_f2:
            best_val_f2 = val_metrics['f2']
            best_model_state = model.state_dict().copy()
    
    print(f"F2 Score History: {f2_history}")
    model.load_state_dict(best_model_state)
    return model

def prepare_data(label_file_path, image_dir, batch_size=32):
    """
    Prepare the data loaders for training, validation, and test sets
    """
    # Read the label file
    df = pd.read_csv(label_file_path, header=None, names=['filename', 'label'])
    
    # Filter out non-tif files
    df = df[df['filename'].str.endswith('.tif')]
  
    # Add full path and verify files exist
    df['full_path'] = df['filename'].apply(lambda x: os.path.join(image_dir, x))
    existing_files = df['full_path'].apply(os.path.exists)
    
    if not all(existing_files):
        missing_files = df[~existing_files]['filename'].tolist()
        print(f"Warning: {len(missing_files)} files not found:")
        print("\n".join(missing_files[:5]))
        if len(missing_files) > 5:
            print(f"... and {len(missing_files) - 5} more")
        df = df[existing_files]
    
    # Convert boolean labels to integers
    df['label'] = df['label'].map({True: 1, False: 0})
    
    # First split: 80% train+val, 20% test
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Second split: Split remaining 80% into 80% train, 20% val (64% and 16% of total)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)
    
    print(f"Total images: {len(df)}")
    print(f"Training samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation samples: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test samples: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Create datasets
    train_dataset = RockDataset(train_df['full_path'].values, train_df['label'].values)
    val_dataset = RockDataset(val_df['full_path'].values, val_df['label'].values)
    test_dataset = RockDataset(test_df['full_path'].values, test_df['label'].values)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


def train_and_validate(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    """
    Train the model and return the best model state based on F2 score
    """
    f2_history = []
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model = model.to(device)
    best_val_f2 = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.squeeze()
            labels = labels.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        val_metrics = evaluate_model(model, val_loader, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Metrics: Recall: {val_metrics["recall"]:.4f}, '
              f'Precision: {val_metrics["precision"]:.4f}, '
              f'Accuracy: {val_metrics["accuracy"]:.4f}, '
              f'F2 Score: {val_metrics["f2"]:.4f}\n')
        
        f2_history.append(val_metrics["f2"])
        
        # Save model if it has better F2 score
        if val_metrics['f2'] > best_val_f2:
            best_val_f2 = val_metrics['f2']
            best_model_state = model.state_dict().copy()
            print(f"New best model saved! F2 Score: {best_val_f2:.4f}")
    
    print(f"F2 Score History: {f2_history}")
    print(f"Best Validation F2 Score: {best_val_f2:.4f}")
    return best_model_state, best_val_f2

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set paths
    LABEL_FILE = './datasets/data_all_4b_good/labels.txt'
    IMAGE_DIR = './datasets/data_all_4b_good/'
    
    # Prepare data with test set
    train_loader, val_loader, test_loader = prepare_data(LABEL_FILE, IMAGE_DIR, batch_size=32)
    
    # Initialize model (choose which model to use)
    model = RockDetectionCNN()  # Original CNN
    # model = ResNet_kelan_multiclass()  # Original ResNet with 2 classes
    # model = ResNet_kelan()  # Current ResNet with binary classification
    
    # Train and get best model state
    best_model_state, best_val_f2 = train_and_validate(model, train_loader, val_loader, num_epochs=300, device=device)
    
    # Load the best model state before testing
    model.load_state_dict(best_model_state)
    
    # Evaluate best model on test set
    print("\nEvaluating best model on test set...")
    print(f"(Best validation F2 score was: {best_val_f2:.4f})")
    test_metrics = evaluate_model(model, test_loader, device)
    print("\nTest Set Metrics:")
    print(f"F2 Score: {test_metrics['f2']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Save the best model
    torch.save(best_model_state, 'rock_detection_best_CNN_300e.pth')
    print("\nBest model saved as 'rock_detection_best_CNN_300e.pth'")
