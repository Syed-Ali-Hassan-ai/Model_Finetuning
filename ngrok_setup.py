"""
Ngrok Setup Script
Creates a public HTTPS tunnel to the local FastAPI server.

SETUP INSTRUCTIONS (First Time):
1. Sign up for free account at https://ngrok.com/
2. Install ngrok: https://ngrok.com/download
3. Get your auth token from: https://dashboard.ngrok.com/get-started/your-authtoken
4. Configure ngrok: ngrok config add-authtoken YOUR_AUTH_TOKEN
5. Run this script: python ngrok_setup.py
"""

import subprocess
import time
import sys
import os
from pyngrok import ngrok, conf
import threading
import uvicorn


def check_ngrok_installed():
    """
    Check if ngrok is installed and configured.
    """
    try:
        result = subprocess.run(['ngrok', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Ngrok installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass

    print("\n" + "="*70)
    print("❌ NGROK NOT FOUND")
    print("="*70)
    print("\nPlease install ngrok:")
    print("\n1. Download from: https://ngrok.com/download")
    print("2. Extract and move to your PATH")
    print("\nOR install via package manager:")
    print("  - macOS:   brew install ngrok")
    print("  - Linux:   snap install ngrok")
    print("  - Windows: choco install ngrok")
    print("\n3. Sign up at: https://ngrok.com/")
    print("4. Get auth token: https://dashboard.ngrok.com/get-started/your-authtoken")
    print("5. Configure: ngrok config add-authtoken YOUR_AUTH_TOKEN")
    print("="*70 + "\n")
    return False


def setup_ngrok_authtoken():
    """
    Guide user to set up ngrok auth token.
    """
    print("\n" + "="*70)
    print("NGROK SETUP")
    print("="*70)
    print("\nTo use ngrok, you need to:")
    print("\n1. Sign up for a FREE account:")
    print("   → https://ngrok.com/")
    print("\n2. Get your auth token:")
    print("   → https://dashboard.ngrok.com/get-started/your-authtoken")
    print("\n3. Configure ngrok with your token:")
    print("   → ngrok config add-authtoken YOUR_AUTH_TOKEN")
    print("\n" + "="*70)

    choice = input("\nHave you completed the setup? (y/n): ").strip().lower()
    if choice != 'y':
        print("\nPlease complete the setup and run this script again.")
        sys.exit(0)


def start_fastapi_server(port=8000):
    """
    Start the FastAPI server in a separate thread.
    """
    from app import app

    print(f"\n[1/3] Starting FastAPI server on port {port}...")

    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait for server to start
    time.sleep(3)
    print(f"✓ FastAPI server started on http://localhost:{port}")


def create_ngrok_tunnel(port=8000):
    """
    Create ngrok tunnel.
    """
    print(f"\n[2/3] Creating ngrok tunnel...")

    try:
        # Create tunnel
        public_url = ngrok.connect(port, bind_tls=True)
        print(f"✓ Ngrok tunnel created!")
        return public_url

    except Exception as e:
        print(f"\n❌ Error creating tunnel: {e}")
        print("\nPossible issues:")
        print("  - Auth token not configured")
        print("  - Port already in use")
        print("  - Network connectivity issues")
        return None


def display_tunnel_info(public_url):
    """
    Display tunnel information.
    """
    print("\n" + "="*70)
    print("🌐 PUBLIC TUNNEL ACTIVE")
    print("="*70)
    print(f"\nPublic URL: {public_url}")
    print(f"Local URL:  http://localhost:8000")
    print(f"\nAPI Documentation: {public_url}/docs")
    print("\nTest endpoints:")
    print(f"  Health:    {public_url}/health")
    print(f"  Metrics:   {public_url}/metrics")
    print(f"  Predict:   {public_url}/predict (POST)")
    print(f"  Keywords:  {public_url}/keywords (POST)")
    print("\n" + "="*70)
    print("\nTunnel will remain active until you press Ctrl+C")
    print("="*70 + "\n")


def test_endpoints(public_url):
    """
    Test the API endpoints.
    """
    import requests

    print("\n[3/3] Testing API endpoints...")

    try:
        # Test health endpoint
        response = requests.get(f"{public_url}/health", timeout=10)
        if response.status_code == 200:
            print("✓ Health check: OK")
        else:
            print(f"⚠ Health check: {response.status_code}")

        # Test sentiment prediction
        test_text = "This is a great product! I love it!"
        response = requests.post(
            f"{public_url}/predict",
            json={"text": test_text},
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Sentiment prediction: {result['label']} ({result['confidence']:.2f})")
        else:
            print(f"⚠ Sentiment prediction failed: {response.status_code}")

    except Exception as e:
        print(f"⚠ Endpoint testing error: {e}")


def main():
    """
    Main function.
    """
    print("\n" + "="*70)
    print("AI LAB ML SERVICE - NGROK TUNNEL SETUP")
    print("="*70)

    # Check if ngrok is installed
    if not check_ngrok_installed():
        setup_ngrok_authtoken()
        if not check_ngrok_installed():
            print("\n❌ Ngrok still not found. Please install and configure it first.")
            sys.exit(1)

    # Check if models exist
    from pathlib import Path
    if not Path("models/sentiment_model_best").exists():
        print("\n⚠ WARNING: Trained models not found!")
        print("Please run 'python lora_trainer.py' first to train the models.")
        choice = input("\nContinue anyway? (y/n): ").strip().lower()
        if choice != 'y':
            sys.exit(0)

    # Start FastAPI server
    start_fastapi_server(port=8000)

    # Create ngrok tunnel
    public_url = create_ngrok_tunnel(port=8000)

    if public_url:
        # Display tunnel info
        display_tunnel_info(public_url)

        # Test endpoints
        test_endpoints(public_url)

        # Keep tunnel alive
        try:
            print("\n✓ Tunnel is ready! Press Ctrl+C to stop.\n")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n✓ Shutting down tunnel...")
            ngrok.disconnect(public_url)
            print("✓ Tunnel closed.")
    else:
        print("\n❌ Failed to create ngrok tunnel.")
        sys.exit(1)


if __name__ == "__main__":
    main()
