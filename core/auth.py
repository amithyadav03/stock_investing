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
        # Example Usage:
        # 1. First run this script with print_login_url() uncommented:
        print_login_url()
        
        # 2. Then paste your token below and run the script again with this uncommented:
        # my_request_token = "PASTE_YOUR_REQUEST_TOKEN_HERE"
        # generate_access_token(my_request_token)
