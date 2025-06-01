import torch
import requests
from PIL import Image
from torchvision import transforms
import gradio as gr

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).eval()

# Download human-readable labels for ImageNet
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

def predict(inp):
    inp = transforms.ToTensor()(inp).unsqueeze(0)  # Convert to tensor and add batch dimension
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp), dim=0)
        confidence = {labels[i]: prediction[0][i].item() for i in range(1000)}
    return confidence


gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"),
             outputs = gr.Label(num_top_classes=5),
             examples=["deer.jpg", "leopard.jpeg"]).launch()