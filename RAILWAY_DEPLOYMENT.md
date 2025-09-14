# Railway Deployment Guide for AI-Driven Public Health Chatbot

## 🚀 Quick Deployment Steps

### 1. Prerequisites
- Railway account (https://railway.app)
- GitHub repository with your code
- API keys for external services

### 2. Deploy to Railway

#### Option A: One-Click Deploy
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/Anasgaima18/AI-Driven-Public-Health-Chatbot-for-Disease-Awareness)

#### Option B: Manual Setup
1. Connect your GitHub repository to Railway
2. Railway will automatically detect the `railway.toml` configuration
3. Set up environment variables (see `.env.railway` template)

### 3. Environment Variables Setup
In your Railway dashboard:
1. Go to your project → Variables
2. Add all variables from `.env.railway` template
3. Get API keys from respective services:
   - **Gemini API**: https://makersuite.google.com/app/apikey
   - **Twilio**: https://console.twilio.com/
   - **Health APIs**: Your government health API provider

### 4. Service Architecture
- **Action Server** (Port 8000): Custom actions, AI responses, integrations
- **Rasa Server** (Port 5005): NLU processing, dialogue management
- **Web Dashboard** (Port 80): User interface, monitoring

### 5. Testing Your Deployment
```bash
# Test Action Server
curl https://your-action-server-url.railway.app/health-check

# Test Rasa Server
curl https://your-rasa-server-url.railway.app/status

# Test Web Dashboard
open https://your-web-dashboard-url.railway.app
```

## 🔧 Troubleshooting

### Common Issues:
1. **Model Training Timeout**: Increase Railway build timeout
2. **Memory Issues**: Upgrade to Pro plan for more RAM
3. **API Rate Limits**: Monitor usage and upgrade plans if needed

### Logs:
Check Railway logs in the dashboard for each service to debug issues.

## 📊 Monitoring & Scaling

### Health Checks:
- All services have health check endpoints
- Automatic restarts on failures
- Real-time monitoring in Railway dashboard

### Scaling:
- Start with Hobby plan ($5/month)
- Upgrade to Pro ($10/month) for more resources
- Use Railway's auto-scaling features

## 🔒 Security Best Practices

1. **Environment Variables**: Never commit secrets to code
2. **API Keys**: Rotate regularly, use restricted permissions
3. **Network Security**: Railway provides automatic SSL/TLS
4. **Access Control**: Configure proper authentication for admin endpoints

## 🌟 Production Optimizations

1. **Caching**: Implement Redis for session storage
2. **CDN**: Use Railway's built-in CDN for static assets
3. **Database**: Use Railway's PostgreSQL for data persistence
4. **Monitoring**: Set up alerts for critical metrics

## 📞 Support

- Railway Docs: https://docs.railway.app/
- Rasa Docs: https://rasa.com/docs/
- Twilio Docs: https://www.twilio.com/docs/

---

**SIH 2025 Compliance**: This deployment meets all SIH requirements for scalable, secure, and accessible healthcare chatbot solutions.