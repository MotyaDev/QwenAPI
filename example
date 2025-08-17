import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qwencore import Qwen, JsonConversation

async def main():
    messages = [{"role": "user", "content": "Дай мне список всех моделей qwen кодовые имена"}]
    
    try:
        response_generator = Qwen.create_async_generator(
            model=Qwen.default_model,
            messages=messages,
            stream=True,
            enable_thinking=False,
        )
        
        full_response = ""

        async for chunk in response_generator:
            if isinstance(chunk, str):
                full_response += chunk
                print(chunk, end="", flush=True)
            elif hasattr(chunk, 'content'):  # reasoning
                print(f"[Reasoning: {chunk.content}]", end="", flush=True)
        
        print("\n\full response:", full_response.strip())
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
