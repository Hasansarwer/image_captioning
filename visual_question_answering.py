import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
# Initialize the processor and model from Hugging Face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def vqa(image_url, question):
    # Load the image from the URL
    raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    
    # Preprocess the image and question
    inputs = processor(raw_image, question, return_tensors="pt")

    # Generate the answer
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer
if __name__ == "__main__":
    image_folder = r"F:\Marriage Photo\edit photos\reception"
    question = "What is dog doing?"
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    answer = vqa(img_url, question)
    print(f"Answer: {answer}")
