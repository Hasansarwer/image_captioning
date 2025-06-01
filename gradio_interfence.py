import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")
    
    # Generate caption
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

def caption_image(image):
    """
    Takes a PIL Image input and returns a caption.
    """   
    try:
        caption = generate_caption(image)
        return caption
    except Exception as e:
        return f"Error generating caption: {str(e)}"
    
# Create a Gradio interface
iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs="text",
    title="Image Captioning with BLIP",
    description="Upload an image to generate a caption using the BLIP model."
)

iface.launch(server_name="127.0.0.1", server_port=7860)  # Set share=True to allow public access