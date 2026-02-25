import boto3
import json

class BedrockModel:
    _instance = None  # singleton so model is only initialized once

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-haiku-20241022-v1:0",
        region:   str = "us-east-1",
        max_tokens: int = 1024,
    ):
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region
        )
        self.model_id   = model_id
        self.max_tokens = max_tokens

    @classmethod
    def get_instance(cls) -> "BedrockModel":
        """Return shared singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def invoke(self, system_prompt: str, user_message: str) -> str:
        """
        Call the model and return the raw text response.
        Strips markdown code fences if present.
        """
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens":        self.max_tokens,
                "system":            system_prompt,
                "messages": [
                    {"role": "user", "content": user_message}
                ]
            })
        )

        result  = json.loads(response["body"].read())
        content = result["content"][0]["text"].strip()

        # Strip markdown fences if model wraps JSON in ```json ... ```
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        return content.strip()

    def invoke_json(self, system_prompt: str, user_message: str) -> dict:
        """Call the model and parse response as JSON directly."""
        content = self.invoke(system_prompt, user_message)
        return json.loads(content)