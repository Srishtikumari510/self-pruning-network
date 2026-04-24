with open("app.py", "w") as f:
    f.write("""
import streamlit as st
import torch
from model import PrunableNet
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load model
@st.cache_resource
def load_model():
    model = PrunableNet()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

st.title("🧠 Self-Pruning Neural Network")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        pred = output.argmax(1).item()

    st.write(f"Prediction: {pred}")

# Sparsity function
def calculate_sparsity(model):
    total = 0
    zero = 0

    for m in model.modules():
        if hasattr(m, "gate_scores"):
            gates = torch.sigmoid(m.gate_scores)
            total += gates.numel()
            zero += (gates < 1e-2).sum().item()

    return 100 * zero / total

# Show sparsity
sparsity = calculate_sparsity(model)
st.write(f"Sparsity: {sparsity:.2f}%")

# Plot gate distribution
def plot_gates(model):
    all_gates = []

    for m in model.modules():
        if hasattr(m, "gate_scores"):
            gates = torch.sigmoid(m.gate_scores).detach().numpy().flatten()
            all_gates.extend(gates)

    fig, ax = plt.subplots()
    ax.hist(all_gates, bins=50)
    ax.set_title("Gate Distribution")

    st.pyplot(fig)

plot_gates(model)
""")
