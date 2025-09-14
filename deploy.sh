#!/bin/bash

# SIH25049 - AI-Driven Public Health Chatbot Deployment Script for Railway

echo "🚀 Starting deployment of AI-Driven Public Health Chatbot..."

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Please install it first:"
    echo "npm install -g @railway/cli"
    echo "or visit: https://docs.railway.app/develop/cli"
    exit 1
fi

# Login to Railway (if not already logged in)
echo "🔐 Checking Railway authentication..."
railway login

# Link to Railway project
echo "🔗 Linking to Railway project..."
railway link

# Set environment variables
echo "⚙️ Setting up environment variables..."

# Required environment variables
echo "Please set the following environment variables in Railway dashboard:"
echo "1. GEMINI_API_KEY - Your Google Gemini API key"
echo "2. GOOGLE_TRANSLATE_KEY - Your Google Translate API key"
echo "3. TWILIO_ACCOUNT_SID - Your Twilio Account SID"
echo "4. TWILIO_AUTH_TOKEN - Your Twilio Auth Token"
echo "5. TWILIO_WHATSAPP_NUMBER - Your Twilio WhatsApp number"
echo "6. DATABASE_URL - PostgreSQL database URL (will be auto-generated)"

# Deploy action server
echo "🚀 Deploying action server..."
cd action-server
railway up --service action-server

# Deploy Rasa server
echo "🚀 Deploying Rasa server..."
cd ../rasa
railway up --service rasa-server

# Deploy web dashboard
echo "🚀 Deploying web dashboard..."
cd ../web-dashboard
railway up --service web-dashboard

echo "✅ Deployment completed!"
echo "🌐 Your AI-Driven Public Health Chatbot is now live on Railway!"
echo ""
echo "Next steps:"
echo "1. Check deployment status: railway status"
echo "2. View logs: railway logs"
echo "3. Configure your domain: railway domain"
echo "4. Set up PostgreSQL database: railway add postgresql"
echo ""
echo "Important: Make sure to set all required environment variables in Railway dashboard"