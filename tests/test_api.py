"""Client script to test YOLO3D API."""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import aiohttp


async def test_health_check(base_url: str = "http://localhost:8000"):
    """Test health check endpoint."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{base_url}/health") as response:
                result = await response.json()
                print("Health Check:")
                print(json.dumps(result, indent=2))
                return response.status == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False


async def predict_single_image(
    image_path: str, base_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """Predict on a single image."""
    async with aiohttp.ClientSession() as session:
        try:
            with open(image_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field(
                    "file", f, filename=Path(image_path).name, content_type="image/jpeg"
                )

                async with session.post(f"{base_url}/predict", data=data) as response:
                    result = await response.json()

                    if response.status == 200:
                        print(f"\nPrediction for {image_path}:")
                        print(json.dumps(result, indent=2))
                        return result
                    else:
                        print(f"Error: {result}")
                        return {}

        except Exception as e:
            print(f"Prediction failed: {e}")
            return {}


async def predict_batch_images(
    image_paths: List[str], base_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """Predict on multiple images."""
    async with aiohttp.ClientSession() as session:
        try:
            data = aiohttp.FormData()

            for image_path in image_paths:
                with open(image_path, "rb") as f:
                    data.add_field(
                        "files", f, filename=Path(image_path).name, content_type="image/jpeg"
                    )

            async with session.post(f"{base_url}/predict_batch", data=data) as response:
                result = await response.json()

                if response.status == 200:
                    print("\nBatch predictions:")
                    print(json.dumps(result, indent=2))
                    return result
                else:
                    print(f"Error: {result}")
                    return {}

        except Exception as e:
            print(f"Batch prediction failed: {e}")
            return {}


def create_dummy_image(output_path: str = "dummy_test_image.jpg"):
    """Create a dummy image for testing."""
    import numpy as np
    from PIL import Image

    # Create a dummy RGB image
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(dummy_img)
    image.save(output_path)
    print(f"Created dummy test image: {output_path}")
    return output_path


async def main():
    parser = argparse.ArgumentParser(description="Test YOLO3D API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--images", nargs="+", help="Paths to multiple test images")
    parser.add_argument("--create-dummy", action="store_true", help="Create dummy test image")

    args = parser.parse_args()

    base_url = args.url

    # Test health check
    print("Testing YOLO3D API...")
    healthy = await test_health_check(base_url)

    if not healthy:
        print("API is not healthy. Make sure the server is running.")
        return

    # Create dummy image if requested
    if args.create_dummy:
        dummy_path = create_dummy_image()
        if not args.image:
            args.image = dummy_path

    # Single image prediction
    if args.image:
        if Path(args.image).exists():
            await predict_single_image(args.image, base_url)
        else:
            print(f"Image not found: {args.image}")

    # Batch prediction
    if args.images:
        valid_images = [img for img in args.images if Path(img).exists()]
        if valid_images:
            await predict_batch_images(valid_images, base_url)
        else:
            print("No valid images found for batch prediction")

    # If no images specified and dummy was created, use it
    if not args.image and not args.images and args.create_dummy:
        await predict_single_image(dummy_path, base_url)


if __name__ == "__main__":
    asyncio.run(main())
