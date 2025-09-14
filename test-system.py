#!/usr/bin/env python3
"""
Test script for AI-Driven Public Health Chatbot Action Server
"""

import os
import sys
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gemini_api():
    """Test Gemini API connectivity"""
    print("🔍 Testing Gemini API...")
    try:
        import google.generativeai as genai

        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ GEMINI_API_KEY not found in environment")
            return False

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content('Hello, test message for health chatbot')

        print("✅ Gemini API working!")
        print(f"Response: {response.text[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return False

def test_server_health():
    """Test simple server health endpoint"""
    print("\n🔍 Testing Simple Server...")
    try:
        response = requests.get('http://localhost:8000/health-check', timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Simple server is healthy!")
            print(f"Service: {data.get('service', 'Unknown')}")
            print(f"Version: {data.get('version', 'Unknown')}")
            print(f"Gemini Configured: {data.get('gemini_configured', False)}")
            return True
        else:
            print(f"❌ Health check failed with status: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to simple server: {e}")
        return False

def test_generate_endpoint():
    """Test the generate endpoint"""
    print("\n🔍 Testing Generate Endpoint...")
    try:
        payload = {
            "text": "I have fever and headache",
            "lang": "en"
        }
        response = requests.post('http://localhost:8000/generate', json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print("✅ Generate endpoint working!")
            print(f"Response: {data.get('response', '')[:200]}...")
            return True
        else:
            print(f"❌ Generate endpoint failed with status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to generate endpoint: {e}")
        return False

def main():
    print("🚀 AI-Driven Public Health Chatbot - System Test")
    print("=" * 50)

    # Test Gemini API
    gemini_ok = test_gemini_api()

    # Test server (assuming it's running)
    server_ok = test_server_health()

    if server_ok:
        generate_ok = test_generate_endpoint()
    else:
        generate_ok = False
        print("⚠️  Skipping generate test - server not running")

    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"Gemini API: {'✅ PASS' if gemini_ok else '❌ FAIL'}")
    print(f"Action Server: {'✅ PASS' if server_ok else '❌ FAIL'}")
    print(f"Generate Endpoint: {'✅ PASS' if generate_ok else '❌ FAIL'}")

    if gemini_ok and server_ok and generate_ok:
        print("\n🎉 All tests passed! Your chatbot is ready!")
        print("\n📋 Next Steps:")
        print("1. Deploy to Railway: Follow RAILWAY_DEPLOYMENT.md")
        print("2. Test with Rasa: rasa shell (in Linux/WSL)")
        print("3. Configure Twilio for WhatsApp/SMS")
        print("4. Set up web dashboard monitoring")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()