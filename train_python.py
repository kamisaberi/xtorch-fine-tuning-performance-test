import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
import os

# --- 1. Configuration ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = '/home/kami/Documents/datasets' # Directory where food-101 folder exists
MODEL_PATH = 'resnet50_for_food101_finetuning.pt'
NUM_EPOCHS = 3 # Food-101 is large, 3 epochs is a good start
BATCH_SIZE = 64 # Use a larger batch size if your GPU allows
LEARNING_RATE = 0.001

# --- 2. Data Loading (The easy way) ---
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Torchvision handles downloading and parsing automatically
train_dataset = datasets.Food101(root=DATA_DIR, split='train', transform=data_transforms, download=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
print(f"Training data loaded: {len(train_dataset)} images.")

# --- 3. Model Loading and Setup ---
model = torch.jit.load(MODEL_PATH)

# Freeze all layers except the final one ('fc')
for name, param in model.named_parameters():
    if "fc" not in name:
        param.requires_grad = False

params_to_update = [p for p in model.parameters() if p.requires_grad]
print(f"Number of parameter groups to fine-tune: {len(params_to_update)}")
model.to(device)

# --- 4. Training Loop ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params_to_update, lr=LEARNING_RATE)

print("\nStarting Python fine-tuning on Food-101...")
start_time = time.time()
model.train()

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        if (i+1) % 100 == 0:
            print(f'  Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch {epoch+1} Summary -> Loss: {epoch_loss:.4f}')

end_time = time.time()
print(f"\nPython fine-tuning finished in {end_time - start_time:.2f} seconds.")
torch.jit.save(model, "food101_finetuned_from_python.pt")
