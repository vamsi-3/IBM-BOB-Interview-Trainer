"""
WatsonX LLM Client
-------------------
Thin wrapper around the IBM WatsonX text generation REST API.
Uses the meta-llama/llama-3-3-70b-instruct model.
"""

import os
import json
import requests
from typing import Optional


class WatsonXClient:
    """Handles authentication and text generation calls to WatsonX."""

    IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"
    GENERATION_URL = (
        "https://jp-tok.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
    )
    MODEL_ID = "meta-llama/llama-3-3-70b-instruct"
    PROJECT_ID = "418fcea5-6fa2-4bc7-be2f-e191ec8e34f7"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("IBM_CLOUD_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "IBM Cloud API key is required. "
                "Set the IBM_CLOUD_API_KEY environment variable or pass it directly."
            )
        self._iam_token: Optional[str] = None

    # ── IAM Token ────────────────────────────────────────────────────────────

    def _get_iam_token(self) -> str:
        """Fetch a fresh IAM bearer token using the API key."""
        response = requests.post(
            self.IAM_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": self.api_key,
            },
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"IAM token fetch failed [{response.status_code}]: {response.text}"
            )
        return response.json()["access_token"]

    def _ensure_token(self) -> str:
        """Return cached token or fetch a new one."""
        if not self._iam_token:
            self._iam_token = self._get_iam_token()
        return self._iam_token

    def _refresh_token(self) -> str:
        """Force-refresh the IAM token."""
        self._iam_token = self._get_iam_token()
        return self._iam_token

    # ── Text Generation ───────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
    ) -> str:
        """
        Call the WatsonX text generation API and return the generated text.

        Parameters
        ----------
        prompt : str
            The full formatted prompt string.
        max_new_tokens : int
            Maximum tokens to generate.
        temperature : float
            Sampling temperature (0 = deterministic, 1 = creative).
        top_p : float
            Nucleus sampling probability.
        repetition_penalty : float
            Penalty for repeating tokens.

        Returns
        -------
        str
            The generated text from the model.
        """
        token = self._ensure_token()

        payload = {
            "model_id": self.MODEL_ID,
            "project_id": self.PROJECT_ID,
            "input": prompt,
            "parameters": {
                "decoding_method": "sample",
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
            },
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        response = requests.post(
            self.GENERATION_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=60,
        )

        # Retry once on 401 (expired token)
        if response.status_code == 401:
            token = self._refresh_token()
            headers["Authorization"] = f"Bearer {token}"
            response = requests.post(
                self.GENERATION_URL,
                headers=headers,
                data=json.dumps(payload),
                timeout=60,
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"WatsonX API error [{response.status_code}]: {response.text}"
            )

        result = response.json()
        try:
            return result["results"][0]["generated_text"].strip()
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Unexpected WatsonX response format: {result}"
            ) from exc
