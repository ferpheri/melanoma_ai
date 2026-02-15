# app.py
from model import predict
from PIL import Image
import gradio as gr

WARNING_MSG = "⚠️ This is a research tool. Predictions may be incorrect. Do not rely on this 100%."

def analyze(image, model_choice):
    if image is None:
        return WARNING_MSG, None
    prob = predict(image, model_choice)
    percentage = round(prob * 100, 2)
    result_text = "⚠️ Melanoma Risk Detected" if prob > 0.5 else "✅ Likely Benign"
    result_text += f" ({percentage}%)"
    return result_text, image

model_options = ["mobilenet", "resnet", "efficientnet"]

iface = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Image(type="pil", label="Upload Skin Lesion"),
        gr.Dropdown(model_options, label="Choose Model", value="mobilenet")
    ],
    outputs=[
        gr.Textbox(label="Result"),
        gr.Image(label="Uploaded Image")
    ],
    title="Skin Lesion Analyzer (Melanoma AI)",
    description=WARNING_MSG,
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()