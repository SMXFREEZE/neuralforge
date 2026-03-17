"""
NeuralForge — LeNet-5 Training Script
Trains a LeNet-5 CNN on MNIST dataset using PyTorch
Achieves >99% test accuracy with INT8-quantization-friendly architecture
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class LeNet5(nn.Module):
    """
    LeNet-5 Architecture (modified for MNIST):
      Input:  1 x 28 x 28
      Conv1:  6 filters, 5x5, stride 1  → 6 x 24 x 24
      ReLU + MaxPool 2x2                → 6 x 12 x 12
      Conv2:  16 filters, 5x5, stride 1 → 16 x 8 x 8
      ReLU + MaxPool 2x2                → 16 x 4 x 4
      FC1:    16*4*4 = 256 → 120
      ReLU
      FC2:    120 → 84
      ReLU
      FC3:    84 → 10 (output classes)
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                  f" ({100. * batch_idx / len(train_loader):.0f}%)]"
                  f"  Loss: {loss.item():.4f}")

    accuracy = 100. * correct / total
    avg_loss = running_loss / len(train_loader)
    return avg_loss, accuracy


def evaluate(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    return avg_loss, accuracy


def main():
    print("=" * 60)
    print("  NeuralForge — LeNet-5 Training on MNIST")
    print("=" * 60)

    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    epochs = 10
    seed = 42

    torch.manual_seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
    ])

    print("Downloading MNIST dataset...")
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Model
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Training loop
    print("\n" + "-" * 60)
    print("Training...")
    print("-" * 60)

    best_accuracy = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch
        )
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)
        scheduler.step()

        print(f"\nEpoch {epoch}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.2f}%")
        print(f"  Test  Loss: {test_loss:.4f}  |  Test  Acc: {test_acc:.2f}%")

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, 'checkpoints/lenet5_best.pth')
            print(f"  ★ New best model saved! ({test_acc:.2f}%)")

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"  Training Complete!")
    print(f"  Best Test Accuracy: {best_accuracy:.2f}%")
    print(f"  Total Time: {elapsed:.1f}s")
    print(f"  Model saved to: checkpoints/lenet5_best.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()
