import boto3
import json
import io
import base64
from PIL import Image
from config import Config


def _make_bedrock_client(region: str):
    kwargs = {"service_name": "bedrock-runtime", "region_name": region}
    if Config.AWS_ACCESS_KEY_ID and Config.AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"]     = Config.AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = Config.AWS_SECRET_ACCESS_KEY
    return boto3.client(**kwargs)


def _is_nova(model_id: str) -> bool:
    return "nova" in model_id.lower()


class BedrockModel:
    _instance = None

    def __init__(
        self,
        model_id:   str = Config.BEDROCK_MODEL_ID,
        region:     str = Config.AWS_REGION,
        max_tokens: int = 2048,
    ):
        self.client     = _make_bedrock_client(region)
        self.model_id   = model_id
        self.max_tokens = max_tokens
        self.is_nova    = _is_nova(model_id)
        print(f"[BedrockModel] Using model: {self.model_id} ({'Nova' if self.is_nova else 'Claude'})")

    @classmethod
    def get_instance(cls) -> "BedrockModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _build_body_text(self, system_prompt: str, user_message: str) -> dict:
        """Build request body for text-only call — handles Claude vs Nova format."""
        if self.is_nova:
            # Nova uses "system" as a list of objects, not a string
            return {
                "system":     [{"text": system_prompt}],
                "messages":   [{"role": "user", "content": [{"text": user_message}]}],
                "inferenceConfig": {"maxTokens": self.max_tokens}
            }
        else:
            # Claude format
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens":        self.max_tokens,
                "system":            system_prompt,
                "messages":          [{"role": "user", "content": user_message}]
            }

    def _build_body_vision(self, prompt: str, img_b64: str) -> dict:
        """Build request body for vision call — handles Claude vs Nova format."""
        if self.is_nova:
            return {
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "image": {
                                "format": "jpeg",
                                "source": {"bytes": img_b64}
                            }
                        },
                        {"text": prompt}
                    ]
                }],
                "inferenceConfig": {"maxTokens": self.max_tokens}
            }
        else:
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens":        self.max_tokens,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type":       "base64",
                                "media_type": "image/jpeg",
                                "data":       img_b64
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }]
            }

    def _extract_text(self, response: dict) -> str:
        """Extract text from response — handles Claude vs Nova response format."""
        if self.is_nova:
            # Nova: output.message.content[0].text
            return response["output"]["message"]["content"][0]["text"].strip()
        else:
            # Claude: content[0].text
            return response["content"][0]["text"].strip()

    def _clean(self, text: str) -> str:
        """Strip markdown code fences if present."""
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return text.strip()

    def invoke(self, system_prompt: str, user_message: str) -> str:
        """Text-only call. Returns clean string."""
        body     = self._build_body_text(system_prompt, user_message)
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body)
        )
        result = json.loads(response["body"].read())
        return self._clean(self._extract_text(result))

    def invoke_json(self, system_prompt: str, user_message: str) -> dict:
        """Text-only call, returns parsed JSON."""
        return json.loads(self.invoke(system_prompt, user_message))

    def invoke_vision(self, prompt: str, image: Image.Image, max_dim: int = 1280) -> str:
        """Vision call with image + text prompt. Returns clean string."""
        # Resize
        w, h = image.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        body     = self._build_body_vision(prompt, img_b64)
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body)
        )
        result = json.loads(response["body"].read())
        return self._clean(self._extract_text(result))

    def invoke_vision_json(self, prompt: str, image: Image.Image) -> list | dict:
        """Vision call, returns parsed JSON."""
        return json.loads(self.invoke_vision(prompt, image))
