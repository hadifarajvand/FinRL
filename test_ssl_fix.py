import yfinance as yf
import ssl
import os

# Set environment variables for SSL
os.environ['CURL_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'
os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'

# Configure yfinance with different settings
yf.set_config(proxy=None)

# Test download
try:
    df = yf.download('AAPL', start='2024-01-01', end='2024-01-10', progress=False)
    print("SSL fix successful! Downloaded data shape:", df.shape)
    print("Sample data:")
    print(df.head())
except Exception as e:
    print(f"SSL fix failed: {e}")
    print("Trying alternative approach...")
    
    # Try with different SSL settings
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        df = yf.download('AAPL', start='2024-01-01', end='2024-01-10', progress=False, verify=False)
        print("Alternative SSL fix successful! Downloaded data shape:", df.shape)
    except Exception as e2:
        print(f"Alternative SSL fix also failed: {e2}") 