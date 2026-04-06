from kiteconnect import KiteConnect
from core.config import settings

def print_login_url():
    """
    Step 1: Run this function every morning. It returns a URL.
    You must click the URL, log in to Zerodha on your browser, and it will redirect you 
    to your callback URL with a `request_token` in the URL bar.
    """
    kite = KiteConnect(api_key=settings.KITE_API_KEY)
    print("Please click this URL to login to Kite:")
    print(kite.login_url())
    print("\nAfter logging in, look at the URL you are redirected to.")
    print("It will look like: http://127.0.0.1:8000/?action=login&status=success&request_token=YOUR_TOKEN_HERE")

def generate_access_token(request_token: str) -> str:
    """
    Step 2: Copy the `request_token` from the URL bar and pass it here.
    This fetches the daily `access_token` which you paste into your `.env` file.
    """
    kite = KiteConnect(api_key=settings.KITE_API_KEY)
    data = kite.generate_session(request_token, api_secret=settings.KITE_API_SECRET)
    
    access_token = data["access_token"]
    print("\nSUCCESS! Your Daily Access Token is:")
    print("====================================")
    print(access_token)
    print("====================================")
    print("Paste this into your .env file as KITE_ACCESS_TOKEN")
    return access_token

if __name__ == "__main__":
    if not settings.KITE_API_KEY:
        print("Please enter your KITE_API_KEY and SECRET in the .env file first.")
    else:
        print_login_url()
        
        # Make the script interactive
        req_token = input("\nPaste the request_token from the URL bar here and press Enter: ").strip()
        
        if req_token:
            try:
                generate_access_token(req_token)
            except Exception as e:
                print(f"Failed to generate access token. Error: {e}")
        else:
            print("No token provided. Exiting.")
