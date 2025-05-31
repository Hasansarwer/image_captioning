from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os


# Initialize the processor and model from Hugging Face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")



def generate_caption(image_path):
    # load an image
    image = Image.open(image_path)
    # preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # generate caption
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    image_folder = r"F:\Marriage Photo\edit photos\reception"
    captions = []

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(image_folder, filename)
            caption = generate_caption(img_path)
            captions.append((filename, caption))
    for filename, caption in captions:
        print(f"{filename}: {caption}")
    with open("captions.txt", "w") as f:
        for filename, caption in captions:
            f.write(f"{filename}: {caption}\n")
