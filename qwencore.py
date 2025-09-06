from __future__ import annotations

import asyncio
import json
import uuid
import re
import time
import aiohttp
from dataclasses import dataclass
from typing import AsyncGenerator, List, Dict, Optional

class QwenError(Exception):
    """Базовое исключение для ошибок клиента Qwen."""
    pass

class RateLimitError(QwenError):
    """Исключение при превышении лимита запросов."""
    pass

@dataclass
class QwenConversation:
    """Хранит состояние диалога с Qwen."""
    chat_id: str
    cookies: Dict
    parent_id: Optional[str] = None

class QwenClient:
    """
    Асинхронный клиент для взаимодействия с веб-сервисом Qwen (chat.qwen.ai).
    """
    BASE_URL = "https://chat.qwen.ai"
    
    # Список моделей с исправленным синтаксисом (добавлены запятые)
    MODELS = [
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
    DEFAULT_MODEL = "qwen3-max-preview"

    def __init__(self, model: str = None, timeout: int = 120, proxy: str = None, debug: bool = False):
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        self.proxy = proxy
        self.debug = debug
        self._session: Optional[aiohttp.ClientSession] = None
        self._midtoken: Optional[str] = None
        self._conversation: Optional[QwenConversation] = None

    def _log(self, message: str):
        if self.debug:
            print(f"[QwenClient] {message}")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self._session

    async def _get_midtoken(self, session: aiohttp.ClientSession) -> str:
        if self._midtoken:
            return self._midtoken

        self._log("Токен 'midtoken' отсутствует, получаю новый...")
        async with session.get('https://sg-wum.alibaba.com/w/wu.json', proxy=self.proxy) as r:
            r.raise_for_status()
            text = await r.text()
            match = re.search(r"'(.*?)'", text)
            if not match:
                raise QwenError("Не удалось извлечь 'bx-umidtoken' со страницы.")
            self._midtoken = match.group(1)
            self._log(f"Новый 'midtoken' получен: {self._midtoken[:15]}...")
            return self._midtoken

    async def _get_conversation(self, session: aiohttp.ClientSession, headers: Dict) -> QwenConversation:
        if self._conversation:
            return self._conversation

        self._log("Активный диалог отсутствует, создаю новый...")
        payload = {
            "title": "New Chat",
            "models": [self.model],
            "chat_mode": "normal",
            "chat_type": "t2t",
            "timestamp": int(time.time() * 1000)
        }
        async with session.post(f'{self.BASE_URL}/api/v2/chats/new', json=payload, headers=headers, proxy=self.proxy) as r:
            r.raise_for_status()
            data = await r.json()
            if not (data.get('success') and data.get('data', {}).get('id')):
                raise QwenError(f"Не удалось создать новый диалог: {data}")
            
            chat_id = data['data']['id']
            cookies = {key: value for key, value in r.cookies.items()}
            self._conversation = QwenConversation(chat_id=chat_id, cookies=cookies)
            self._log(f"Новый диалог создан: chat_id={chat_id}")
            return self._conversation

    async def _parse_stream(self, response: aiohttp.ClientResponse) -> AsyncGenerator[str, None]:
        async for line in response.content:
            line = line.decode('utf-8').strip()
            if line.startswith('data:'):
                data_str = line[len('data:'):].strip()
                if data_str and data_str != "[DONE]":
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices", [])
                        if choices:
                            content = choices[0].get("delta", {}).get("content")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        self._log(f"Не удалось декодировать JSON из потока: {data_str}")

    async def chat(self, prompt: str) -> str:
        """Отправляет промпт и возвращает полный ответ в виде строки."""
        full_response = ""
        async for chunk in self.chat_stream(prompt):
            full_response += chunk
        return full_response

    async def chat_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Отправляет промпт и возвращает асинхронный генератор частей ответа."""
        session = await self._get_session()
        
        for attempt in range(3):
            try:
                midtoken = await self._get_midtoken(session)
                req_headers = {
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
                    'Accept': 'application/json, text/plain, */*',
                    'Origin': self.BASE_URL, 'Referer': f'{self.BASE_URL}/', 'Content-Type': 'application/json',
                    'Authorization': 'Bearer', 'Source': 'web',
                    'bx-umidtoken': midtoken, 'bx-v': '2.5.31'
                }
                
                conversation = await self._get_conversation(session, req_headers)

                # УПРОЩЕННЫЙ PAYLOAD
                msg_payload = {
                    "stream": True,
                    "incremental_output": True,
                    "chat_id": conversation.chat_id,
                    "chat_mode": "normal",
                    "model": self.model,
                    "parent_id": conversation.parent_id,
                    "messages": [
                        {
                            "fid": str(uuid.uuid4()),
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }

                url = f'{self.BASE_URL}/api/v2/chat/completions?chat_id={conversation.chat_id}'
                async with session.post(url, json=msg_payload, headers=req_headers, proxy=self.proxy, cookies=conversation.cookies) as r:
                    r.raise_for_status()
                    async for chunk in self._parse_stream(r):
                        yield chunk
                return # Успешное завершение

            except aiohttp.ClientResponseError as e:
                if e.status == 429 and attempt < 2:
                    self._log(f"Превышен лимит запросов (попытка {attempt + 1}/3). Сбрасываю токен и жду...")
                    self._midtoken = None # Сброс токена для его пересоздания
                    self._conversation = None # Сброс диалога
                    await asyncio.sleep(2)
                    continue
                raise QwenError(f"Ошибка API: {e.status} - {e.message}") from e
            except Exception as e:
                raise QwenError(f"Произошла непредвиденная ошибка: {e}") from e
        
        raise RateLimitError("Не удалось получить ответ от Qwen после нескольких попыток.")

    async def close(self):
        """Закрывает сессию aiohttp."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._log("Сессия закрыта.")

# --- Пример использования ---
async def main():
    print("--- Тест полного ответа ---")
    client = QwenClient(debug=True)
    try:
        response = await client.chat("Привет! Расскажи короткий факт о космосе.")
        print("\nПолный ответ:", response)
    except QwenError as e:
        print(f"\nОшибка: {e}")
    finally:
        await client.close()

    print("\n" + "="*30 + "\n")

    print("--- Тест потокового ответа ---")
    client = QwenClient(debug=True)
    try:
        print("Потоковый ответ: ", end="", flush=True)
        async for chunk in client.chat_stream("Напиши короткий стих о программировании."):
            print(chunk, end="", flush=True)
        print()
    except QwenError as e:
        print(f"\nОшибка: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
