#!/bin/bash

# Railway Local Development Script
# This script helps test your services locally before deploying to Railway

echo "🚀 Starting AI-Driven Public Health Chatbot (Local Development)"
echo "============================================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not available. Please install docker-compose."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.railway .env
    echo "⚠️  Please edit .env file with your actual API keys!"
fi

echo "🏗️  Building Docker images..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

echo "⏳ Waiting for services to be ready..."
sleep 30

echo "🔍 Checking service health..."
echo "Action Server: http://localhost:8000/health-check"
echo "Rasa Server: http://localhost:5005/status"
echo "Web Dashboard: http://localhost:3000"

echo ""
echo "✅ Services started successfully!"
echo "📖 Check RAILWAY_DEPLOYMENT.md for deployment instructions"
echo "🛑 To stop services: docker-compose down"