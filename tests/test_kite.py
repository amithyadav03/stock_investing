from core.config import settings
from kiteconnect import KiteConnect

def test_connection():
    print("Testing Kite Connect Keys...")
    if not settings.KITE_API_KEY or not settings.KITE_ACCESS_TOKEN:
        print("❌ Error: KITE_API_KEY or KITE_ACCESS_TOKEN are missing/empty in .env")
        return

    try:
        kite = KiteConnect(api_key=settings.KITE_API_KEY)
        kite.set_access_token(settings.KITE_ACCESS_TOKEN)
        
        # Test basic connection parsing
        profile = kite.profile()
        print("✅ SUCCESS! Connected to Zerodha Kite!")
        print(f"User Name: {profile.get('user_name', 'Unknown')}")
        print(f"User ID: {profile.get('user_id', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ KITE ERROR: Failed to authenticate. Details: {e}")

if __name__ == "__main__":
    test_connection()
