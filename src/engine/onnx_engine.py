"""Common ONNX engine."""

import rootutils

ROOT = rootutils.autosetup()

from pathlib import Path
from typing import List, Union

import onnxruntime as ort

from src.schema.onnx_schema import OnnxMetadataSchema
from src.utils.logger import get_logger

log = get_logger()


class CommonOnnxEngine:
    """Common ONNX runtime engine module."""

    def __init__(self, engine_path: str, provider: str = "cpu") -> None:
        """Initialize ONNX runtime common engine."""
        self.engine_path = Path(engine_path)
        self.provider = provider
        self.decrypt_key = None
        self.provider = self.check_providers(provider)

    def setup(self) -> None:
        """Setup ONNX runtime engine."""
        log.info(f"Setup YOLO3D ONNX engine...")
        if self.decrypt_key:
            with open(str(self.engine_path), "rb") as f:
                engine_data = f.read()
            self.engine = ort.InferenceSession(engine_data, providers=self.provider)
        else:
            self.engine = ort.InferenceSession(
                str(self.engine_path), providers=self.provider
            )
        self.metadata = self.get_metadata()
        self.img_shape = self.metadata[0].input_shape[2:]

        log.info(f"YOLO3D ONNX is ready!")

    def get_metadata(self) -> List[OnnxMetadataSchema]:
        """Get model metadata."""
        inputs = self.engine.get_inputs()
        outputs = self.engine.get_outputs()

        result: List[OnnxMetadataSchema] = []
        for inp, out in zip(inputs, outputs):
            result.append(
                OnnxMetadataSchema(
                    input_name=inp.name,
                    input_shape=inp.shape,
                    output_name=out.name,
                    output_shape=out.shape,
                )
            )

        return result

    def check_providers(self, provider: Union[str, list]) -> list:
        """Check available providers. If provider is not available, use CPU instead."""
        assert provider in ["cpu", "gpu"], "Invalid provider"
        available_providers = ort.get_available_providers()
        log.debug(f"Available providers: {available_providers}")
        if provider == "cpu" and "OpenVINOExecutionProvider" in available_providers:
            provider = ["CPUExecutionProvider", "OpenVINOExecutionProvider"]
        elif provider == "gpu" and "CUDAExecutionProvider" in available_providers:
            provider = ["CUDAExecutionProvider"]
        else:
            provider = ["CPUExecutionProvider"]

        return provider
