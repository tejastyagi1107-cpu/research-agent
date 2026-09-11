import os
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenChatParamsMetaNames as ChatParams


class WatsonxClient:
    """Singleton wrapper for IBM watsonx.ai ModelInference (chat API)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        api_key = os.environ.get("WATSONX_API_KEY")
        url = os.environ.get("WATSONX_URL", "https://au-syd.ml.cloud.ibm.com")
        project_id = os.environ.get("WATSONX_PROJECT_ID")
        model_id = os.environ.get("WATSONX_MODEL_ID", "meta-llama/llama-3-3-70b-instruct")

        credentials = Credentials(api_key=api_key, url=url)
        self.model = ModelInference(
            model_id=model_id,
            credentials=credentials,
            project_id=project_id,
        )
        self._default_params = {
            ChatParams.MAX_TOKENS: 1024,
            ChatParams.TEMPERATURE: 0.3,
            ChatParams.REPETITION_PENALTY: 1.1,
        }

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        params = {**self._default_params, ChatParams.MAX_TOKENS: max_tokens}
        response = self.model.chat(
            messages=[{"role": "user", "content": prompt}],
            params=params,
        )
        # chat() returns a dict with choices[0].message.content
        return response["choices"][0]["message"]["content"].strip()


def get_watsonx_client() -> WatsonxClient:
    return WatsonxClient()
