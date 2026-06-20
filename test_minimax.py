#!/usr/bin/env python3
"""MiniMax API bağlantı testi - app.py'nin kullandığı Anthropic-uyumlu endpoint"""

import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MINIMAX_API_KEY")
if not api_key:
    raise SystemExit("MINIMAX_API_KEY bulunamadı (.env dosyasını kontrol edin)")

client = anthropic.Anthropic(
    api_key=api_key,
    base_url="https://api.minimax.io/anthropic"
)

print("--- MiniMax Anthropic-uyumlu API testi ---")
try:
    message = client.messages.create(
        model="MiniMax-M2.7",
        max_tokens=200,
        messages=[{"role": "user", "content": [{"type": "text", "text": "Merhaba"}]}]
    )
    text_found = False
    for block in message.content:
        if block.type == "text":
            print("Yanıt:", block.text)
            text_found = True
    if not text_found:
        print("Metin bloğu yok (max_tokens'ı artırmayı deneyin). Bloklar:", message.content)
except anthropic.APIStatusError as e:
    print(f"API hatası ({e.status_code}): {e.message}")
