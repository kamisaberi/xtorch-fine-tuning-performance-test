import torch
import torchvision.models as models
import torch.nn as nn

# --- Constants ---
NUM_CLASSES = 101
OUTPUT_PATH = "resnet50_for_food101_finetuning.pt"

# --- 1. Load pre-trained ResNet50 ---
print("Loading pre-trained ResNet50 model...")
model = models.resnet50(pretrained=True)

# --- 2. Modify the final layer for 101 classes ---
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
print(f"Model final layer replaced for {NUM_CLASSES} classes.")

# --- 3. Save as a TorchScript file ---
model.train() # Set to train mode before scripting
scripted_model = torch.jit.script(model)
scripted_model.save(OUTPUT_PATH)

print(f"Model ready for fine-tuning has been saved to: {OUTPUT_PATH}")
