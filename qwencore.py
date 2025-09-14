# qwen_provider_standalone.py
# -*- coding: utf-8 -*-
"""
Полностью самостоятельный провайдер для chat.qwen.ai без gpt4free.

- НЕТ внутренних импортов (errors/typing/requests/...).
- Есть совместимые заглушки: AsyncGeneratorProvider, ProviderModelMixin, JsonConversation, Reasoning, Usage.
- Потоковая генерация через SSE, обработка midtoken (bx-umidtoken) с авто-обновлением.
- Поведение аналогично исходному: сначала yield JsonConversation, далее Reasoning(...) / текст кусочками,
  в конце yield Usage(...).

Установка:
    pip install aiohttp

Пример использования:
    import asyncio
    from qwen_provider_standalone import Qwen

    async def demo():
        messages = [{"role": "user", "content": "Привет! Суммируй квантовые компьютеры в 3 пунктах."}]
        async for chunk in Qwen.create_async_generator(
            model="qwen3-235b-a22b",
            messages=messages,
            stream=True,
            enable_thinking=True,
        ):
            if isinstance(chunk, Reasoning):
                # Размышления модели (think-фаза)
                pass
            elif isinstance(chunk, JsonConversation):
                # созданный чат
                print("CHAT:", chunk.chat_id)
            elif isinstance(chunk, Usage):
                print("\nUSAGE:", dict(chunk))
            else:
                # Текст ответа (answer-фаза)
                print(chunk, end="", flush=True)

    if __name__ == "__main__":
        asyncio.run(demo())
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass
from time import time
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Union

import aiohttp

# ---------- Лёгкий логгер ----------
logger = logging.getLogger("qwen_provider")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---------- Совместимые заглушки интерфейса ----------
class RateLimitError(Exception):
    """Выбрасывается после нескольких неудачных попыток из-за лимитов."""

# Тип для совместимости с AsyncResult (итерация по yield)
AsyncResult = AsyncIterator[Union[str, "Reasoning", "Usage", "JsonConversation"]]

@dataclass
class JsonConversation:
    chat_id: str
    cookies: Dict[str, str]
    parent_id: Optional[str] = None

class Reasoning:
    """Обёртка для think-фазы (как в исходном коде)."""
    def __init__(self, content: str):
        self.content = content
    def __str__(self) -> str:
        return self.content

class Usage(dict):
    """Совместимая обёртка: Usage(**usage_dict)."""
    pass

class AsyncGeneratorProvider:
    """Маркерный базовый класс для совместимости."""
    pass

class ProviderModelMixin:
    """Миксин с get_model для списка models/default_model."""
    @classmethod
    def get_model(cls, model: Optional[str]) -> str:
        if model and hasattr(cls, "models") and model in getattr(cls, "models"):
            return model
        return getattr(cls, "default_model")

# ---------- Вспомогательные функции ----------
def get_last_user_message(messages: Iterable[Dict[str, str]]) -> str:
    """Берём последний user-месседж, иначе конкатенируем все user."""
    messages = list(messages)
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return "\n\n".join(m.get("content", "") for m in messages if m.get("role") == "user")

async def sse_stream(resp: aiohttp.ClientResponse) -> AsyncIterator[Dict[str, Any]]:
    """
    Простой SSE-парсер. Собираем 'data:' блоки до пустой строки, парсим JSON.
    """
    buf: List[str] = []
    async for raw in resp.content:
        line = raw.decode("utf-8", errors="ignore").rstrip("\n")
        if line.startswith("data:"):
            buf.append(line[len("data:"):].strip())
        elif not line.strip():
            if buf:
                payload = "\n".join(buf).strip()
                buf.clear()
                if not payload:
                    continue
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue
    # слить остатки
    if buf:
        try:
            yield json.loads("\n".join(buf).strip())
        except json.JSONDecodeError:
            pass

# ---------- Основной провайдер ----------
class Qwen(AsyncGeneratorProvider, ProviderModelMixin):
    """
    Провайдер для Qwen (chat.qwen.ai), без зависимостей gpt4free.
    Совместим по интерфейсу с вашим исходным провайдером:
    - сначала yield JsonConversation
    - затем Reasoning(...) для think-фазы
    - обычные текстовые чанки для answer-фазы
    - в конце yield Usage(...)
    """
    url = "https://chat.qwen.ai"
    working = True
    active_by_default = True
    supports_stream = True
    supports_message_history = False

    models = [
        "qwen3-max-preview",
        "qwen3-235b-a22b",
        "qwen3-coder-plus",
        "qwen3-30b-a3b",
        "qwen3-coder-30b-a3b-instruct",
        "qwen-max-latest",
        "qwen-plus-2025-01-25",
        "qwq-32b",
        "qwen-turbo-2025-02-11",
        "qwen2.5-omni-7b",
        "qvq-72b-preview-0310",
        "qwen2.5-vl-32b-instruct",
        "qwen2.5-14b-instruct-1m",
        "qwen2.5-coder-32b-instruct",
        "qwen2.5-72b-instruct",
    ]
    default_model = "qwen3-235b-a22b"

    _midtoken: Optional[str] = None
    _midtoken_uses: int = 0

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: List[Dict[str, str]],
        conversation: Optional[JsonConversation] = None,
        proxy: Optional[str] = None,
        timeout: int = 120,
        stream: bool = True,
        enable_thinking: bool = True,
        **kwargs,
    ) -> AsyncResult:

        model_name = cls.get_model(model)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.5",
            "Origin": cls.url,
            "Referer": f"{cls.url}/",
            "Content-Type": "application/json",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Connection": "keep-alive",
            "Authorization": "Bearer",
            "Source": "web",
        }

        prompt = get_last_user_message(messages)

        async with aiohttp.ClientSession(headers=headers) as session:
            for attempt in range(5):
                try:
                    # --- Получение/обновление midtoken ---
                    if not cls._midtoken:
                        logger.info("[Qwen] Нет активного midtoken. Получаем новый…")
                        async with session.get("https://sg-wum.alibaba.com/w/wu.json", proxy=proxy) as r:
                            r.raise_for_status()
                            text = await r.text()
                            match = re.search(r"(?:umx\.wu|__fycb)\('([^']+)'\)", text)
                            if not match:
                                raise RuntimeError("Failed to extract bx-umidtoken.")
                            cls._midtoken = match.group(1)
                            cls._midtoken_uses = 1
                            logger.info(f"[Qwen] Новый midtoken получен. uses={cls._midtoken_uses}")
                    else:
                        cls._midtoken_uses += 1
                        logger.info(f"[Qwen] Используем кэш midtoken. uses={cls._midtoken_uses}")

                    req_headers = session.headers.copy()
                    req_headers["bx-umidtoken"] = cls._midtoken
                    req_headers["bx-v"] = "2.5.31"

                    # --- Создаём чат, если нет ---
                    if conversation is None:
                        chat_payload = {
                            "title": "New Chat",
                            "models": [model_name],
                            "chat_mode": "normal",
                            "chat_type": "t2t",
                            "timestamp": int(time() * 1000),
                        }
                        async with session.post(
                            f"{cls.url}/api/v2/chats/new",
                            json=chat_payload,
                            headers=req_headers,
                            proxy=proxy,
                        ) as resp:
                            resp.raise_for_status()
                            data = await resp.json()
                            if not (data.get("success") and data["data"].get("id")):
                                raise RuntimeError(f"Failed to create chat: {data}")
                            conversation = JsonConversation(
                                chat_id=data["data"]["id"],
                                cookies={k: v.value for k, v in resp.cookies.items()},
                                parent_id=None,
                            )
                    # Сообщаем наружу, что чат готов
                    yield conversation

                    # --- Формируем сообщение пользователя ---
                    message_id = str(uuid.uuid4())
                    msg_payload = {
                        "stream": stream,
                        "incremental_output": stream,
                        "chat_id": conversation.chat_id,
                        "chat_mode": "normal",
                        "model": model_name,
                        "parent_id": conversation.parent_id,
                        "messages": [
                            {
                                "fid": message_id,
                                "parentId": conversation.parent_id,
                                "childrenIds": [],
                                "role": "user",
                                "content": prompt,
                                "user_action": "chat",
                                "files": [],
                                "models": [model_name],
                                "chat_type": "t2t",
                                "feature_config": {
                                    "thinking_enabled": enable_thinking,
                                    "output_schema": "phase",
                                    "thinking_budget": 81920,
                                },
                                "extra": {"meta": {"subChatType": "t2t"}},
                                "sub_chat_type": "t2t",
                                "parent_id": None,
                            }
                        ],
                    }

                    # --- Отправляем и читаем поток ---
                    async with session.post(
                        f"{cls.url}/api/v2/chat/completions?chat_id={conversation.chat_id}",
                        json=msg_payload,
                        headers=req_headers,
                        proxy=proxy,
                        timeout=timeout,
                        cookies=conversation.cookies,
                    ) as resp:
                        # Первую строку часто присылают JSON'ом (response_id и т.п.)
                        first_line = await resp.content.readline()
                        try:
                            line_str = first_line.decode().strip()
                        except Exception:
                            line_str = ""
                        if line_str.startswith("{"):
                            try:
                                head = json.loads(line_str)
                                # Ошибка со стороны сервера (редко)
                                if head.get("data", {}).get("code"):
                                    raise RuntimeError(f"Response: {head}")
                                conversation.parent_id = head.get("response.created", {}).get("response_id")
                            except Exception:
                                pass

                        thinking_started = False
                        usage = None
                        async for chunk in sse_stream(resp):
                            try:
                                usage = chunk.get("usage", usage)
                                choices = chunk.get("choices", [])
                                if not choices:
                                    continue
                                delta = choices[0].get("delta", {})
                                phase = delta.get("phase")
                                content = delta.get("content")

                                if phase == "think" and not thinking_started:
                                    thinking_started = True
                                elif phase == "answer" and thinking_started:
                                    thinking_started = False

                                if content:
                                    if thinking_started:
                                        yield Reasoning(content)
                                    else:
                                        yield content
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue

                        if usage:
                            yield Usage(**usage)
                        return

                except (aiohttp.ClientResponseError, RuntimeError) as e:
                    is_rate_limit = (isinstance(e, aiohttp.ClientResponseError) and e.status == 429) or \
                                    ("RateLimited" in str(e))
                    if is_rate_limit:
                        logger.warning(f"[Qwen] Rate limit (attempt {attempt+1}/5). Сбрасываю midtoken.")
                        cls._midtoken = None
                        cls._midtoken_uses = 0
                        await asyncio.sleep(2)
                        continue
                    else:
                        raise e

            # Если не вышли раньше — превысили количество попыток
            raise RateLimitError("The Qwen provider reached the request limit after 5 attempts.")
