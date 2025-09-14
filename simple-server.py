#!/usr/bin/env python3
"""
Simple test server for AI-Driven Public Health Chatbot
"""

import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Gemini API
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    genai.configure(api_key=gemini_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    print("✅ Gemini API configured")
else:
    gemini_model = None
    print("❌ Gemini API key not found")

@app.route('/health-check')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "AI-Driven Public Health Chatbot",
        "version": "1.0.0",
        "gemini_configured": gemini_model is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/generate', methods=['POST'])
def generate():
    """Generate AI response"""
    try:
        data = request.get_json()
        user_text = data.get('text', '')
        user_lang = data.get('lang', 'en')

        if not gemini_model:
            return jsonify({"error": "Gemini API not configured"}), 500

        # Create health-focused prompt
        prompt = f"""
        You are a qualified public health AI assistant for populations in India.

        User query: {user_text}
        Language: {user_lang}

        Provide accurate, safe health guidance following these guidelines:
        1. Always recommend consulting healthcare professionals for serious symptoms
        2. Provide preventive healthcare advice
        3. Include vaccination information when relevant
        4. Give culturally sensitive advice for Indian populations
        5. Be concise but comprehensive (under 200 words)
        6. Include emergency contact information if symptoms are severe (104 for emergency)

        Ensure response is medically accurate and culturally appropriate.
        """

        response = gemini_model.generate_content(prompt)
        print(f"DEBUG: Gemini response received: {response.text[:100]}...")

        return jsonify({
            "response": response.text,
            "language": user_lang,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"DEBUG: Error in generate: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/test')
def test():
    """Simple test endpoint"""
    return jsonify({"message": "Chatbot server is running!", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    print("🚀 Starting Simple Chatbot Server...")
    print("📍 Health Check: http://localhost:8000/health-check")
    print("🤖 Generate: http://localhost:8000/generate (POST)")
    print("🧪 Test: http://localhost:8000/test")
    app.run(host='0.0.0.0', port=8000, debug=True)