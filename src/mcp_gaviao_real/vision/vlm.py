"""Optional async OpenAI-compatible VLM image captioning client."""

from __future__ import annotations

import asyncio
import base64

import httpx

from mcp_gaviao_real.config import GaviaoSettings


class VlmCaptioner:
    """Async client calling an OpenAI-compatible /v1/chat/completions vision endpoint."""

    def __init__(self, settings: GaviaoSettings) -> None:
        self.settings = settings
        self._semaphore = asyncio.Semaphore(max(1, settings.vlm_concurrency))

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.vlm_base_url and self.settings.vlm_model)

    async def describe_image(
        self,
        webp_bytes: bytes,
        context_hint: str = "",
    ) -> tuple[str | None, str | None]:
        """Return (description, warning) for a WebP image."""
        if not self.is_configured:
            return (
                None,
                "vlm_captions=True solicitado, mas GAVIAO_VLM_BASE_URL e GAVIAO_VLM_MODEL não estão configurados.",
            )

        b64 = base64.b64encode(webp_bytes).decode("ascii")
        endpoint = f"{self.settings.vlm_base_url.rstrip('/')}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.settings.vlm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.vlm_api_key}"

        prompt = (
            "Descreva de forma objetiva e densa em dados o conteúdo desta imagem extraída de um documento. "
            "Se for gráfico, informe tipo, eixos, séries e tendências/valores principais. "
            "Se for tabela, transcreva colunas e valores-chave. "
            "Se for diagrama, descreva os nós e o fluxo."
        )
        if context_hint:
            prompt += f" Contexto da página: {context_hint}"

        payload = {
            "model": self.settings.vlm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/webp;base64,{b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 450,
            "temperature": 0.1,
        }

        async with self._semaphore:
            try:
                async with httpx.AsyncClient(timeout=self.settings.vlm_timeout_seconds) as client:
                    resp = await client.post(endpoint, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                return (text or None), None
            except Exception as exc:
                return None, f"Falha ao gerar legenda VLM ({exc})."
