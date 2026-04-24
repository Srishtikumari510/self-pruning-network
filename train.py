from model import PrunableNet
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# dataset
transform = transforms.ToTensor()

train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# model
model = PrunableNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# sparsity loss
def sparsity_loss(model):
    loss = 0
    for m in model.modules():
        if hasattr(m, "gate_scores"):
            gates = torch.sigmoid(m.gate_scores)
            loss += gates.sum()
    return loss

# training
lambda_val = 0.01

for epoch in range(5):
    model.train()
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        ce_loss = F.cross_entropy(outputs, labels)
        sp_loss = sparsity_loss(model)
        
        loss = ce_loss + lambda_val * sp_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch} done")
