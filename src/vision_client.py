import os
from typing import Dict, Any, List

from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

load_dotenv()


class VisionService:
    def __init__(self) -> None:
        endpoint = os.getenv("VISION_ENDPOINT")
        key = os.getenv("VISION_KEY")

        if not endpoint or not key:
            raise ValueError("VISION_ENDPOINT or VISION_KEY not set in environment")

        self.client = ImageAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
        )

        # Tags + caption are enough to start
        self._features: List[VisualFeatures] = [
            VisualFeatures.TAGS,
            VisualFeatures.CAPTION,
        ]

    def _to_dict(self, result) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "tags": [],
            "caption": None,
            "metadata": {
                "width": getattr(result.metadata, "width", None),
                "height": getattr(result.metadata, "height", None),
                "format": getattr(result.metadata, "format", None),
                "model_version": getattr(result, "model_version", None),
            },
        }

        # tags
        if result.tags:
            out["tags"] = [
                {"name": t.name, "confidence": t.confidence}
                for t in result.tags.list
            ]

        # caption (Image Analysis 4.0, region must support captions)
        caption = getattr(result, "caption", None)
        if caption is not None and getattr(caption, "text", None):
            out["caption"] = {
                "text": caption.text,
                "confidence": caption.confidence,
            }

        return out

    def analyze_file(self, image_path: str) -> Dict[str, Any]:
        with open(image_path, "rb") as f:
            image_data = f.read()

        result = self.client.analyze(
            image_data=image_data,
            visual_features=self._features,
        )
        d = self._to_dict(result)
        d["source"] = {"type": "file", "value": image_path}
        return d
