import base64
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from typing import Optional
from PIL import Image
import io
from model import predict

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": None,
        "image_data": None,
        "model_choice": "mobilenet" # Default
    })

@app.post("/", response_class=HTMLResponse)
async def predict_image(
    request: Request,
    file: Optional[UploadFile] = File(None),
    model_choice: str = Form(...),
    existing_image: Optional[str] = Form(None)
):
    if file and file.filename:
        contents = await file.read()
    elif existing_image:
        # Extract base64 part
        header, encoded = existing_image.split(",", 1)
        contents = base64.b64decode(encoded)
    else:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": None,
            "image_data": None,
            "model_choice": model_choice
        })

    encoded_image = base64.b64encode(contents).decode("utf-8")
    image_data = f"data:image/jpeg;base64,{encoded_image}"

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    prob = predict(image, model_choice)
    percentage = prob * 100

    result_text = "⚠️ Melanoma Risk Detected" if prob > 0.5 else "✅ Likely Benign"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result_text,
            "probability": percentage,
            "image_data": image_data,
            "model_choice": model_choice
        }
    )
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=10000)