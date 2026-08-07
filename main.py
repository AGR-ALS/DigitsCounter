import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm

from CNN import CNN

MODEL_FILE = "digit_classifier.pth"
DIGITS_FOLDER = "digits"
OUTPUT_FILE = "output.txt"

def load_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Train", leave=True)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / len(train_loader)


def validate(model, test_loader, device, epoch, epochs):
    model.eval()
    correct, total = 0, 0
    val_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} Val", leave=False)
    with torch.no_grad():
        for images, labels in val_pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def save_best_model(model, model_path, acc, best_val_acc):
    if acc > best_val_acc:
        torch.save(model.state_dict(), model_path)
        print(f"New best model saved with val_acc: {acc:.4f}")
        return acc
    return best_val_acc


def train_model(model_path="mnist_digit_classifier.pth", epochs=10, batch_size=128, num_classes=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_loader, test_loader = load_data(batch_size)

    model = CNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs)
        acc = validate(model, test_loader, device, epoch, epochs)
        print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}%")
        best_val_acc = save_best_model(model, model_path, acc, best_val_acc)

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded best model with val_acc: {best_val_acc:.4f}")

    return model


def count_digits_in_folder(folder_path, model_path, num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    counts = np.zeros(num_classes, dtype=int)

    files = [f for f in os.listdir(folder_path)]
    if not files:
        print(f"No images found in {folder_path}")
        return counts.tolist()

    print(f"Processing {len(files)} images from {folder_path}")

    with torch.no_grad():
        for filename in tqdm(files, desc="Counting digits", unit="img"):
            img_path = os.path.join(folder_path, filename)
            try:
                image = Image.open(img_path).convert('L')
                tensor = transform(image).unsqueeze(0).to(device)

                output = model(tensor)
                predicted_digit = torch.argmax(output, dim=1).item()

                counts[predicted_digit] += 1
            except Exception as e:
                print(f"\nError processing {filename}: {e}")

    return counts.tolist()


if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE):
        print("Model not found. Training a new model")
        train_model(model_path=MODEL_FILE, num_classes=10, epochs=10, batch_size=128, lr=0.001)
    else:
        print(f"Found existing model: {MODEL_FILE}")

    if os.path.exists(DIGITS_FOLDER):
        result_array = count_digits_in_folder(DIGITS_FOLDER, num_classes=10, model_path=MODEL_FILE)

        with open(OUTPUT_FILE, 'w') as f:
            f.write(str(result_array))
    else:
        print(f"Folder {DIGITS_FOLDER} not found.")