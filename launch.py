"""
Run this INSTEAD of `uvicorn main:app` during local testing.
It starts the FastAPI server AND creates a public ngrok tunnel
so Telegram can reach your local machine.
"""
import uvicorn
import requests
from pyngrok import ngrok
from core.config import settings

if __name__ == "__main__":
    # Set the ngrok auth token from .env
    if not settings.NGROK_AUTH_TOKEN:
        print("❌ NGROK_AUTH_TOKEN is not set in your .env file.")
        print("   Sign up free at https://ngrok.com, get your token, and add it to .env")
        exit(1)

    ngrok.set_auth_token(settings.NGROK_AUTH_TOKEN)
    
    # Kill any existing tunnels to avoid ERR_NGROK_334 on restarts
    try:
        for t in ngrok.get_tunnels():
            ngrok.disconnect(t.public_url)
            print(f"   Disconnected stale tunnel: {t.public_url}")
    except Exception:
        pass  # No existing tunnels, carry on
    
    tunnel = ngrok.connect(8000)
    public_url = tunnel.public_url
    webhook_url = f"{public_url}/telegram-webhook"
    
    print("=" * 60)
    print(f"✅ ngrok tunnel active!")
    print(f"   Public URL  : {public_url}")
    print(f"   Webhook URL : {webhook_url}")
    print("=" * 60)
    
    # Automatically register the webhook with Telegram
    if settings.TELEGRAM_BOT_TOKEN:
        resp = requests.post(
            f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/setWebhook",
            json={"url": webhook_url}
        )
        result = resp.json()
        if result.get("ok"):
            print("✅ Telegram Webhook registered successfully!")
        else:
            print(f"❌ Telegram Webhook registration failed: {result}")
    else:
        print("⚠️  TELEGRAM_BOT_TOKEN not set — webhook not registered.")
    
    print("\nStarting FastAPI server... Press Ctrl+C to quit.\n")
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
