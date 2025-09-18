"""FastAPI serving application for YOLO3D model."""

import io
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.yolo3d_module import MultiHeadCNN, YOLO3DModule

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="YOLO3D API", description="3D Object Detection API using YOLO3D model", version="1.0.0"
)

# Global variables for model
model: Optional[YOLO3DModule] = None
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform: transforms.Compose = None

# KITTI class names
CLASS_NAMES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]


def load_model(checkpoint_path: str = None) -> YOLO3DModule:
    """Load the YOLO3D model from checkpoint or create new."""
    global model, transform

    try:
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info(f"Loading model from checkpoint: {checkpoint_path}")
            model = YOLO3DModule.load_from_checkpoint(checkpoint_path)
        else:
            logger.info("Creating new model with pretrained weights")
            # Create model with pretrained backbone
            net = MultiHeadCNN(input_channels=3, num_classes=8, backbone="resnet", pretrained=True)

            # Wrap in Lightning module (dummy optimizer for serving)
            model = YOLO3DModule(
                model=net,
                optimizer=torch.optim.Adam,
                scheduler=None,
            )

        model.eval()
        model.to(device)
        logger.info(f"Model loaded successfully on device: {device}")

        # Setup image preprocessing
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        return model

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model input."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Apply transforms and add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor.to(device)


def postprocess_predictions(outputs: Dict[str, torch.Tensor]) -> Dict:
    """Convert model outputs to human-readable format."""
    # Move tensors to CPU and convert to numpy
    results = {}

    with torch.no_grad():
        # Classification
        class_probs = torch.softmax(outputs["classification"], dim=1)
        class_id = torch.argmax(class_probs, dim=1).cpu().numpy()[0]
        class_confidence = class_probs.max().cpu().item()

        results["class"] = {
            "id": int(class_id),
            "name": CLASS_NAMES[class_id],
            "confidence": float(class_confidence),
        }

        # Object confidence
        results["object_confidence"] = float(outputs["confidence"].cpu().item())

        # 3D Orientation (convert sin/cos back to angle)
        orientation = outputs["orientation"].cpu().numpy()[0]
        angle = np.arctan2(orientation[0], orientation[1])  # atan2(sin, cos)
        results["orientation"] = {
            "angle_rad": float(angle),
            "angle_deg": float(np.degrees(angle)),
            "sin": float(orientation[0]),
            "cos": float(orientation[1]),
        }

        # 3D Location
        location = outputs["location"].cpu().numpy()[0]
        results["location"] = {
            "x": float(location[0]),
            "y": float(location[1]),
            "z": float(location[2]),
        }

        # 3D Dimensions
        dimension = outputs["dimension"].cpu().numpy()[0]
        results["dimension"] = {
            "height": float(dimension[0]),
            "width": float(dimension[1]),
            "length": float(dimension[2]),
        }

        # Calculate 3D bounding box corners (simplified)
        # This is a basic implementation - can be extended for full 3D bbox
        results["3d_bbox"] = {
            "center": results["location"],
            "size": results["dimension"],
            "rotation_y": results["orientation"]["angle_rad"],
        }

    return results


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    logger.info("Starting YOLO3D API server...")
    load_model()
    logger.info("YOLO3D API server ready!")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "YOLO3D API is running",
        "model_loaded": model is not None,
        "device": str(device),
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "classes": CLASS_NAMES,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict 3D object detection on uploaded image.

    Args:
        file: Image file (JPEG, PNG, etc.)

    Returns:
        JSON with predicted class, 3D bounding box, orientation, location, and dimensions
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Preprocess image
        input_tensor = preprocess_image(image)

        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)

        # Postprocess results
        results = postprocess_predictions(outputs)

        # Add metadata
        results["metadata"] = {
            "image_size": image.size,
            "model_input_size": [224, 224],
            "device": str(device),
        }

        return JSONResponse(content=results)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict 3D object detection on multiple images.

    Args:
        files: List of image files

    Returns:
        JSON with predictions for each image
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")

    results = []

    for i, file in enumerate(files):
        try:
            # Validate file type
            if not file.content_type.startswith("image/"):
                results.append(
                    {"index": i, "filename": file.filename, "error": "File must be an image"}
                )
                continue

            # Read and process image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))

            # Preprocess image
            input_tensor = preprocess_image(image)

            # Make prediction
            with torch.no_grad():
                outputs = model(input_tensor)

            # Postprocess results
            prediction = postprocess_predictions(outputs)
            prediction["metadata"] = {
                "index": i,
                "filename": file.filename,
                "image_size": image.size,
                "model_input_size": [224, 224],
            }

            results.append(prediction)

        except Exception as e:
            logger.error(f"Batch prediction error for image {i}: {str(e)}")
            results.append({"index": i, "filename": file.filename, "error": str(e)})

    return JSONResponse(content={"predictions": results})


@app.post("/load_model")
async def load_model_endpoint(checkpoint_path: Optional[str] = None):
    """
    Load or reload model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint file
    """
    try:
        load_model(checkpoint_path)
        return {"message": "Model loaded successfully", "checkpoint_path": checkpoint_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Load model
    load_model()

    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
