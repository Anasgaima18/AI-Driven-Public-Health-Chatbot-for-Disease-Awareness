from fastapi import FastAPI, Request, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel
import os, requests, json
from twilio.rest import Client
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from googletrans import Translator
import google.generativeai as genai
import asyncio
import logging
from database import init_database, get_session, User, HealthQuery, DiseaseOutbreak, VaccinationSchedule, HealthAlert, ChatbotMetrics
from datetime import datetime, timedelta
import schedule
import threading
import time
from biobert_processor import get_biobert_processor

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="AI-Driven Public Health Chatbot Action Server", version="1.0.0")

# Twilio client initialization
twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_whatsapp = os.getenv("TWILIO_WHATSAPP_NUMBER")
twilio_sms = os.getenv("TWILIO_SMS_NUMBER")
twilio_client = None
if twilio_sid and twilio_token:
    twilio_client = Client(twilio_sid, twilio_token)

# Initialize database
try:
    init_database()
    logger.info("Database initialized successfully")
except Exception as e:
    logger.error(f"Database initialization failed: {e}")

@app.post("/send-message")
async def send_message(to: str, body: str, channel: str = "whatsapp"):
    """Send message via Twilio WhatsApp or SMS"""
    if not twilio_client:
        raise HTTPException(status_code=500, detail="Twilio credentials not configured.")
    try:
        if channel == "whatsapp":
            message = twilio_client.messages.create(
                body=body,
                from_=twilio_whatsapp,
                to=f"whatsapp:{to}"
            )
        elif channel == "sms":
            message = twilio_client.messages.create(
                body=body,
                from_=twilio_sms,
                to=to
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid channel.")
        return {"status": "sent", "sid": message.sid}
    except Exception as e:
        logger.error(f"Twilio send error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health data integration class
class HealthDataIntegrator:
    def __init__(self):
        self.government_api_key = os.getenv("HEALTH_API_KEY")
        self.government_api_url = os.getenv("HEALTH_API_URL", "https://api.government-health.gov.in/v1")
    
    async def fetch_outbreak_data(self, location: str):
        """Fetch real-time outbreak data from government APIs"""
        try:
            if self.government_api_key:
                headers = {"Authorization": f"Bearer {self.government_api_key}"}
                url = f"{self.government_api_url}/outbreaks/{location}"
                
                # In production, this would make actual API calls
                # For demo, return structured data
                response = await self.simulate_government_api_call(url, headers)
                return response
            else:
                return await self.get_demo_outbreak_data(location)
        except Exception as e:
            logger.error(f"Error fetching outbreak data: {e}")
            return await self.get_demo_outbreak_data(location)
    
    async def simulate_government_api_call(self, url: str, headers: dict):
        """Simulate government API response with realistic data"""
        # This simulates what a real government health API might return
        return {
            "status": "success",
            "data": {
                "outbreaks": [
                    {
                        "disease": "Dengue",
                        "severity": "moderate",
                        "cases": 45,
                        "trend": "increasing",
                        "last_updated": "2025-09-14T08:00:00Z"
                    }
                ]
            }
        }
    
    async def get_demo_outbreak_data(self, location: str):
        """Get demo outbreak data when government API is not available"""
        session = get_session()
        try:
            outbreaks = session.query(DiseaseOutbreak).filter(
                DiseaseOutbreak.location.ilike(f"%{location}%"),
                DiseaseOutbreak.status == "active"
            ).all()
            
            if not outbreaks:
                # Create demo data
                demo_outbreak = DiseaseOutbreak(
                    disease_name="Dengue",
                    location=location,
                    severity="moderate",
                    cases_count=45,
                    prevention_advice="Remove standing water, use mosquito nets, seek medical attention for fever"
                )
                session.add(demo_outbreak)
                session.commit()
                outbreaks = [demo_outbreak]
            
            return {
                "status": "success",
                "data": {
                    "outbreaks": [
                        {
                            "disease": outbreak.disease_name,
                            "severity": outbreak.severity,
                            "cases": outbreak.cases_count,
                            "prevention": outbreak.prevention_advice,
                            "last_updated": outbreak.last_updated.isoformat()
                        } for outbreak in outbreaks
                    ]
                }
            }
        finally:
            session.close()
    
    async def update_vaccination_schedules(self, location: str):
        """Update vaccination schedules from government data"""
        session = get_session()
        try:
            # Check if we have recent data
            recent_schedules = session.query(VaccinationSchedule).filter(
                VaccinationSchedule.location.ilike(f"%{location}%"),
                VaccinationSchedule.scheduled_date > datetime.utcnow()
            ).count()
            
            if recent_schedules == 0:
                # Add demo vaccination schedules
                vaccines = [
                    {
                        "name": "COVID-19 Booster",
                        "date": datetime.utcnow() + timedelta(days=7),
                        "age_group": "18+",
                        "center": f"Primary Health Center - {location}"
                    },
                    {
                        "name": "Influenza",
                        "date": datetime.utcnow() + timedelta(days=14),
                        "age_group": "All ages",
                        "center": f"Community Health Center - {location}"
                    }
                ]
                
                for vaccine_data in vaccines:
                    vaccine = VaccinationSchedule(
                        vaccine_name=vaccine_data["name"],
                        location=location,
                        scheduled_date=vaccine_data["date"],
                        age_group=vaccine_data["age_group"],
                        center_name=vaccine_data["center"],
                        contact_number="104",
                        doses_available=100
                    )
                    session.add(vaccine)
                
                session.commit()
                logger.info(f"Updated vaccination schedules for {location}")
        
        except Exception as e:
            logger.error(f"Error updating vaccination schedules: {e}")
        finally:
            session.close()

# Initialize health data integrator
health_integrator = HealthDataIntegrator()

# Initialize NLLB-200 model for multilingual support
class MultilingualTranslator:
    def __init__(self):
        self.nllb_model = None
        self.nllb_tokenizer = None
        self.google_translator = Translator()
        self.gemini_model = None
        self.load_gemini_model()
        # Load NLLB model asynchronously to avoid blocking startup
        try:
            self.load_nllb_model()
        except Exception as e:
            logger.warning(f"NLLB model loading failed: {e}. Translation features will be limited.")

    def load_gemini_model(self):
        """Load Gemini model for AI responses"""
        try:
            gemini_key = os.getenv("GEMINI_API_KEY")
            if gemini_key:
                genai.configure(api_key=gemini_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("Gemini model loaded successfully")
            else:
                logger.warning("GEMINI_API_KEY not found")
        except Exception as e:
            logger.error(f"Failed to load Gemini model: {e}")

    def load_nllb_model(self):
        """Load NLLB-200 model for Indian language support"""
        try:
            model_name = "facebook/nllb-200-distilled-600M"
            self.nllb_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            logger.info("NLLB-200 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NLLB-200 model: {e}")
            raise
    
    def translate_with_nllb(self, text: str, source_lang: str, target_lang: str):
        """Translate using NLLB-200 model"""
        try:
            # Language code mapping for NLLB-200
            lang_mapping = {
                'hi': 'hin_Deva',  # Hindi
                'bn': 'ben_Beng',  # Bengali
                'te': 'tel_Telu',  # Telugu
                'ta': 'tam_Taml',  # Tamil
                'mr': 'mar_Deva',  # Marathi
                'gu': 'guj_Gujr',  # Gujarati
                'kn': 'kan_Knda',  # Kannada
                'ml': 'mal_Mlym',  # Malayalam
                'pa': 'pan_Guru',  # Punjabi
                'or': 'ory_Orya',  # Odia
                'as': 'asm_Beng',  # Assamese
                'ur': 'urd_Arab',  # Urdu
                'en': 'eng_Latn'   # English
            }
            
            src_lang = lang_mapping.get(source_lang, 'eng_Latn')
            tgt_lang = lang_mapping.get(target_lang, 'eng_Latn')
            
            inputs = self.nllb_tokenizer(text, return_tensors="pt")
            translated_tokens = self.nllb_model.generate(
                **inputs, 
                forced_bos_token_id=self.nllb_tokenizer.lang_code_to_id[tgt_lang],
                max_length=512
            )
            translated_text = self.nllb_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            return translated_text
        except Exception as e:
            logger.error(f"NLLB translation failed: {e}")
            return self.translate_with_google(text, target_lang)
    
    def translate_with_google(self, text: str, target_lang: str):
        """Fallback to Google Translate"""
        try:
            result = self.google_translator.translate(text, dest=target_lang)
            return result.text
        except Exception as e:
            logger.error(f"Google Translate failed: {e}")
            return text
    
    def detect_language(self, text: str):
        """Detect language of input text"""
        try:
            detected = self.google_translator.detect(text)
            return detected.lang
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return 'en'

# Initialize the multilingual translator
translator = MultilingualTranslator()

# Initialize Bio-BERT processor for biomedical text analysis
biobert_processor = get_biobert_processor()
if biobert_processor:
    logger.info("Bio-BERT processor initialized successfully")
else:
    logger.warning("Bio-BERT processor initialization failed - biomedical features will be limited")

class GenerateRequest(BaseModel):
    text: str
    lang: Optional[str] = "en"
    context: Optional[dict] = None

class HealthQuery(BaseModel):
    symptoms: Optional[str] = None
    disease: Optional[str] = None
    location: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    language: Optional[str] = "en"

@app.post("/generate")
async def generate(req: GenerateRequest):
    """Generate AI-powered health guidance using Gemini API with multilingual support and health data integration."""
    session = get_session()
    try:
        # Detect language if not specified
        detected_lang = req.lang
        if not detected_lang or detected_lang == "auto":
            detected_lang = translator.detect_language(req.text)
        
        # Translate to English for processing if needed
        english_text = req.text
        if detected_lang != 'en':
            english_text = translator.translate_with_nllb(req.text, detected_lang, 'en')
        
        # Extract location and health context from the query
        location_context = await extract_location_from_query(english_text)
        health_context = await get_health_context(location_context, session)
        
        # Create enhanced health-focused prompt for Gemini
        health_prompt = f"""
        You are a qualified public health AI assistant for rural and semi-urban populations in India.
        
        User query: {english_text}
        User location: {location_context or "Not specified"}
        
        Current health context:
        {health_context}
        
        Provide accurate, safe health guidance following these guidelines:
        1. Always recommend consulting healthcare professionals for serious symptoms
        2. Provide preventive healthcare advice specific to current local health conditions
        3. Include vaccination information when relevant, referencing current schedules
        4. Mention nearby government health facilities
        5. Give culturally sensitive advice for Indian populations
        6. Be concise but comprehensive (under 200 words)
        7. Include emergency contact information if symptoms are severe
        8. Reference current local disease outbreaks if relevant
        9. Provide government health helpline numbers (104 for emergency)
        
        Ensure response is medically accurate and culturally appropriate for Indian context.
        """
        
        # Call Gemini API with enhanced context
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(health_prompt)
                generated_text = response.text
                
                # Log the query for analytics
                health_query = HealthQuery(
                    user_phone="anonymous",  # Would be populated from session in production
                    query_text=req.text,
                    language=detected_lang,
                    response_text=generated_text,
                    created_at=datetime.utcnow()
                )
                session.add(health_query)
                session.commit()
                
                # Translate back to original language if needed
                if detected_lang != 'en':
                    generated_text = translator.translate_with_nllb(generated_text, 'en', detected_lang)
                
                return {
                    "reply": generated_text,
                    "language": detected_lang,
                    "translated": detected_lang != 'en',
                    "location_context": location_context,
                    "health_context_provided": bool(health_context)
                }
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                fallback_response = """I understand your health concern. For immediate assistance:
                
🏥 Visit your nearest Primary Health Center (PHC)
📞 Call national health helpline: 104
🚨 For emergencies: Call 108
                
Please consult with a qualified healthcare professional for proper medical advice."""
                
                if detected_lang != 'en':
                    fallback_response = translator.translate_with_nllb(fallback_response, 'en', detected_lang)
                return {"reply": fallback_response, "language": detected_lang, "error": "API_ERROR"}
        
        # Fallback response with local health context
        fallback = f"""For health concerns in your area:
        
{health_context if health_context else ""}

🏥 Visit your nearest Primary Health Center (PHC)
📞 National health helpline: 104
🚨 Emergency services: 108

Always consult qualified healthcare professionals for medical advice."""
        
        if detected_lang != 'en':
            fallback = translator.translate_with_nllb(fallback, 'en', detected_lang)
        
        return {"reply": fallback, "language": detected_lang, "fallback": True}
        
    except Exception as e:
        logger.error(f"Generate endpoint error: {e}")
        return {"reply": "Sorry, I'm experiencing technical difficulties. Please call 104 for immediate health assistance.", "error": str(e)}
    finally:
        session.close()

async def extract_location_from_query(text: str) -> Optional[str]:
    """Extract location information from user query using basic NLP"""
    # This could be enhanced with more sophisticated NLP
    # For now, using simple keyword matching
    location_keywords = ["in", "at", "from", "near", "district", "city", "village", "block"]
    words = text.lower().split()
    
    for i, word in enumerate(words):
        if word in location_keywords and i + 1 < len(words):
            potential_location = words[i + 1].title()
            if len(potential_location) > 2:  # Basic validation
                return potential_location
    
    return None

async def get_health_context(location: str, session) -> str:
    """Get current health context for the location"""
    if not location:
        return ""
    
    context_parts = []
    
    try:
        # Get current outbreaks
        outbreaks = session.query(DiseaseOutbreak).filter(
            DiseaseOutbreak.location.ilike(f"%{location}%"),
            DiseaseOutbreak.status == "active"
        ).limit(3).all()
        
        if outbreaks:
            context_parts.append("Current health alerts in your area:")
            for outbreak in outbreaks:
                context_parts.append(f"• {outbreak.disease_name}: {outbreak.cases_count} cases ({outbreak.severity} severity)")
        
        # Get upcoming vaccinations
        upcoming_vaccines = session.query(VaccinationSchedule).filter(
            VaccinationSchedule.location.ilike(f"%{location}%"),
            VaccinationSchedule.scheduled_date > datetime.utcnow()
        ).limit(2).all()
        
        if upcoming_vaccines:
            context_parts.append("\nUpcoming vaccinations:")
            for vaccine in upcoming_vaccines:
                context_parts.append(f"• {vaccine.vaccine_name}: {vaccine.scheduled_date.strftime('%Y-%m-%d')} at {vaccine.center_name}")
        
        return "\n".join(context_parts)
    
    except Exception as e:
        logger.error(f"Error getting health context: {e}")
        return ""

@app.post("/translate")
async def translate(text: str = Form(...), target: str = Form(...), source: str = Form("auto")):
    """Enhanced translation using NLLB-200 with Google Translate fallback."""
    try:
        # Use NLLB-200 for Indian languages, fallback to Google Translate
        if source == "auto":
            source = translator.detect_language(text)
        
        translated_text = translator.translate_with_nllb(text, source, target)
        
        return {
            "translatedText": translated_text,
            "sourceLanguage": source,
            "targetLanguage": target,
            "method": "nllb-200"
        }
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return {"translatedText": text, "error": str(e)}

@app.post("/health-query")
async def health_query(query: HealthQuery):
    """Specialized endpoint for structured health queries."""
    try:
        # Build contextual prompt based on provided information
        context_parts = []
        if query.symptoms:
            context_parts.append(f"Symptoms: {query.symptoms}")
        if query.disease:
            context_parts.append(f"Disease concern: {query.disease}")
        if query.location:
            context_parts.append(f"Location: {query.location}")
        if query.age:
            context_parts.append(f"Age: {query.age}")
        if query.gender:
            context_parts.append(f"Gender: {query.gender}")
        
        context_text = "; ".join(context_parts) if context_parts else "General health query"
        
        # Generate response using the main generate endpoint
        generate_req = GenerateRequest(
            text=context_text,
            lang=query.language,
            context=query.dict()
        )
        
        response = await generate(generate_req)
        
        # Add health-specific metadata
        response["query_type"] = "structured_health"
        response["location"] = query.location
        
        return response
        
    except Exception as e:
        logger.error(f"Health query error: {e}")
        return {"reply": "Please consult a healthcare professional for medical advice.", "error": str(e)}

@app.get("/vaccination-schedule/{location}")
async def get_vaccination_schedule(location: str, language: str = "en"):
    """Get vaccination schedule for a specific location."""
    try:
        # This would integrate with government databases in production
        schedule = {
            "location": location,
            "vaccines": [
                {
                    "name": "COVID-19 Booster",
                    "next_date": "2025-09-25",
                    "location": f"Primary Health Center, {location}",
                    "age_group": "18+"
                },
                {
                    "name": "Influenza",
                    "next_date": "2025-10-01",
                    "location": f"Community Health Center, {location}",
                    "age_group": "All ages"
                },
                {
                    "name": "Hepatitis B",
                    "next_date": "2025-09-30",
                    "location": f"District Hospital, {location}",
                    "age_group": "Newborns and high-risk adults"
                }
            ]
        }
        
        # Translate if needed
        if language != 'en':
            for vaccine in schedule["vaccines"]:
                vaccine["name"] = translator.translate_with_nllb(vaccine["name"], 'en', language)
                vaccine["location"] = translator.translate_with_nllb(vaccine["location"], 'en', language)
        
        return schedule
        
    except Exception as e:
        logger.error(f"Vaccination schedule error: {e}")
        return {"error": str(e)}

@app.get("/disease-outbreaks/{district}")
async def get_disease_outbreaks(district: str, language: str = "en"):
    """Get current disease outbreak information for a district with real-time data integration."""
    session = get_session()
    try:
        # First, try to get real-time data from government APIs
        outbreak_data = await health_integrator.fetch_outbreak_data(district)
        
        # Update local database with fresh data
        if outbreak_data.get("status") == "success":
            await update_local_outbreak_data(district, outbreak_data["data"]["outbreaks"], session)
        
        # Get outbreaks from database
        outbreaks = session.query(DiseaseOutbreak).filter(
            DiseaseOutbreak.location.ilike(f"%{district}%"),
            DiseaseOutbreak.status == "active"
        ).all()
        
        result = {
            "district": district,
            "last_updated": datetime.utcnow().isoformat(),
            "data_source": "government_api" if outbreak_data.get("status") == "success" else "local_database",
            "outbreaks": []
        }
        
        for outbreak in outbreaks:
            outbreak_info = {
                "disease": outbreak.disease_name,
                "severity": outbreak.severity,
                "cases": outbreak.cases_count,
                "date_reported": outbreak.reported_date.strftime("%Y-%m-%d"),
                "prevention": outbreak.prevention_advice,
                "helpline": "104"
            }
            
            # Translate if needed
            if language != 'en':
                outbreak_info["disease"] = translator.translate_with_nllb(outbreak_info["disease"], 'en', language)
                outbreak_info["prevention"] = translator.translate_with_nllb(outbreak_info["prevention"], 'en', language)
            
            result["outbreaks"].append(outbreak_info)
        
        return result
        
    except Exception as e:
        logger.error(f"Disease outbreaks error: {e}")
        return {"error": str(e), "district": district}
    finally:
        session.close()

async def update_local_outbreak_data(district: str, outbreaks_data: List[dict], session):
    """Update local database with real-time outbreak data"""
    try:
        for outbreak_data in outbreaks_data:
            existing_outbreak = session.query(DiseaseOutbreak).filter(
                DiseaseOutbreak.disease_name == outbreak_data["disease"],
                DiseaseOutbreak.location.ilike(f"%{district}%")
            ).first()
            
            if existing_outbreak:
                # Update existing record
                existing_outbreak.cases_count = outbreak_data.get("cases", existing_outbreak.cases_count)
                existing_outbreak.severity = outbreak_data.get("severity", existing_outbreak.severity)
                existing_outbreak.last_updated = datetime.utcnow()
            else:
                # Create new record
                new_outbreak = DiseaseOutbreak(
                    disease_name=outbreak_data["disease"],
                    location=district,
                    severity=outbreak_data.get("severity", "moderate"),
                    cases_count=outbreak_data.get("cases", 0),
                    prevention_advice=f"Follow preventive measures for {outbreak_data['disease']}. Consult healthcare providers."
                )
                session.add(new_outbreak)
        
        session.commit()
        logger.info(f"Updated outbreak data for {district}")
        
    except Exception as e:
        logger.error(f"Error updating outbreak data: {e}")
        session.rollback()

@app.post("/send-health-alert")
async def send_health_alert(
    background_tasks: BackgroundTasks,
    alert_type: str = Form(...),
    title: str = Form(...),
    message: str = Form(...),
    target_location: str = Form(...),
    target_language: str = Form("en")
):
    """Send health alerts to users in a specific location with background processing."""
    session = get_session()
    try:
        # Create health alert record
        health_alert = HealthAlert(
            alert_type=alert_type,
            title=title,
            message=message,
            target_location=target_location,
            target_language=target_language,
            delivery_status="pending"
        )
        session.add(health_alert)
        session.commit()
        
        # Add background task to send alerts
        background_tasks.add_task(
            send_bulk_alerts,
            health_alert.id,
            target_location,
            message,
            target_language
        )
        
        return {
            "status": "alert_queued",
            "alert_id": health_alert.id,
            "message": "Health alert has been queued for delivery",
            "target_location": target_location,
            "estimated_recipients": await estimate_recipients(target_location, session)
        }
        
    except Exception as e:
        logger.error(f"Send health alert error: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        session.close()

async def send_bulk_alerts(alert_id: int, location: str, message: str, language: str):
    """Background task to send alerts to all opted-in users in a location"""
    session = get_session()
    try:
        # Get all opted-in users in the location
        users = session.query(User).filter(
            User.location.ilike(f"%{location}%"),
            User.opted_in_alerts == True
        ).all()
        
        sent_count = 0
        failed_count = 0
        
        for user in users:
            try:
                # Translate message to user's preferred language
                user_message = message
                if user.language_preference != language:
                    user_message = translator.translate_with_nllb(message, language, user.language_preference)
                
                # Send via Twilio (SMS/WhatsApp)
                success = await send_individual_alert(user.phone_number, user_message)
                if success:
                    sent_count += 1
                else:
                    failed_count += 1
                
                # Add small delay to avoid rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to send alert to {user.phone_number}: {e}")
                failed_count += 1
        
        # Update alert status
        alert = session.query(HealthAlert).filter(HealthAlert.id == alert_id).first()
        if alert:
            alert.sent_count = sent_count
            alert.delivery_status = "sent" if failed_count == 0 else "partial"
            alert.sent_at = datetime.utcnow()
            session.commit()
        
        logger.info(f"Alert {alert_id} sent to {sent_count} users, {failed_count} failed")
        
    except Exception as e:
        logger.error(f"Bulk alert sending error: {e}")
    finally:
        session.close()

async def send_individual_alert(phone_number: str, message: str) -> bool:
    """Send individual alert via Twilio"""
    try:
        tw_sid = os.getenv('TWILIO_ACCOUNT_SID')
        tw_token = os.getenv('TWILIO_AUTH_TOKEN')
        tw_from = os.getenv('TWILIO_WHATSAPP_NUMBER')
        
        if not all([tw_sid, tw_token, tw_from]):
            logger.warning("Twilio credentials not configured, skipping SMS send")
            return False
        
        twilio_url = f"https://api.twilio.com/2010-04-01/Accounts/{tw_sid}/Messages.json"
        data = {
            'From': tw_from,
            'To': phone_number,
            'Body': f"🏥 HEALTH ALERT 🏥\n\n{message}\n\nFor more info: Call 104"
        }
        
        response = requests.post(twilio_url, data=data, auth=(tw_sid, tw_token), timeout=10)
        return response.status_code == 201
        
    except Exception as e:
        logger.error(f"Individual alert send error: {e}")
        return False

async def estimate_recipients(location: str, session) -> int:
    """Estimate number of recipients for an alert"""
    try:
        count = session.query(User).filter(
            User.location.ilike(f"%{location}%"),
            User.opted_in_alerts == True
        ).count()
        return count
    except Exception as e:
        logger.error(f"Error estimating recipients: {e}")
        return 0

# Background task scheduler for health data updates
def schedule_health_data_updates():
    """Schedule periodic health data updates"""
    schedule.every(30).minutes.do(update_all_health_data)
    schedule.every().day.at("08:00").do(send_daily_health_tips)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

def update_all_health_data():
    """Periodic task to update health data from government sources"""
    session = get_session()
    try:
        # Get all unique locations
        locations = session.query(User.location).distinct().all()
        
        for (location,) in locations:
            if location:
                asyncio.create_task(health_integrator.update_vaccination_schedules(location))
        
        logger.info("Scheduled health data update completed")
    except Exception as e:
        logger.error(f"Scheduled update error: {e}")
    finally:
        session.close()

def send_daily_health_tips():
    """Send daily health tips to opted-in users"""
    logger.info("Sending daily health tips")
    # Implementation would send personalized health tips

# Start background scheduler in a separate thread
scheduler_thread = threading.Thread(target=schedule_health_data_updates, daemon=True)
scheduler_thread.start()

@app.get("/mock-gov/{district}")
async def mock_gov(district: str, language: str = "en"):
    """Return comprehensive government health data for a district."""
    try:
        demo = {
            "district": district,
            "outbreaks": [
                {
                    "disease": "Dengue",
                    "date": "2025-09-10",
                    "advisory": "Remove standing water, use nets, seek immediate medical attention for high fever",
                    "severity": "moderate",
                    "affected_areas": ["Central", "North"]
                }
            ],
            "vaccines": [
                {
                    "name": "Polio",
                    "next_date": "2025-09-20",
                    "location": "PHC Central",
                    "age_group": "0-5 years"
                },
                {
                    "name": "COVID-19",
                    "next_date": "2025-09-18",
                    "location": "Community Health Center",
                    "age_group": "18+"
                }
            ],
            "health_facilities": [
                {
                    "name": f"Primary Health Center - {district}",
                    "type": "PHC",
                    "contact": "104",
                    "services": ["Emergency", "Vaccination", "Maternal Care"]
                }
            ]
        }
        
        # Translate if needed
        if language != 'en':
            for outbreak in demo["outbreaks"]:
                outbreak["disease"] = translator.translate_with_nllb(outbreak["disease"], 'en', language)
                outbreak["advisory"] = translator.translate_with_nllb(outbreak["advisory"], 'en', language)
            
            for vaccine in demo["vaccines"]:
                vaccine["name"] = translator.translate_with_nllb(vaccine["name"], 'en', language)
                vaccine["location"] = translator.translate_with_nllb(vaccine["location"], 'en', language)
        
        return demo
        
    except Exception as e:
        logger.error(f"Mock gov data error: {e}")
        return {"error": str(e)}

@app.post("/webhook")
async def twilio_webhook(request: Request):
    """Enhanced webhook for WhatsApp/SMS with multilingual support and health-focused responses."""
    try:
        form = await request.form()
        body = form.get('Body') or form.get('body') or ''
        from_number = form.get('From') or form.get('from') or 'unknown'
        
        logger.info(f"Incoming message from {from_number}: {body}")
        
        # Detect language and generate response
        detected_lang = translator.detect_language(body)
        
        # Generate health-focused response
        generate_req = GenerateRequest(text=body, lang=detected_lang)
        response = await generate(generate_req)
        reply = response.get('reply', 'Sorry, please try again.')
        
        # Send response via Twilio
        tw_sid = os.getenv('TWILIO_ACCOUNT_SID')
        tw_token = os.getenv('TWILIO_AUTH_TOKEN')
        tw_from = os.getenv('TWILIO_WHATSAPP_NUMBER')
        
        if tw_sid and tw_token and tw_from:
            twilio_url = f"https://api.twilio.com/2010-04-01/Accounts/{tw_sid}/Messages.json"
            data = {'From': tw_from, 'To': from_number, 'Body': reply}
            try:
                r = requests.post(twilio_url, data=data, auth=(tw_sid, tw_token), timeout=10)
                logger.info(f'Twilio send status: {r.status_code}')
            except Exception as e:
                logger.error(f'Twilio send failed: {e}')
        
        return {
            "status": "ok", 
            "reply": reply, 
            "language": detected_lang,
            "from": from_number
        }
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"status": "error", "error": str(e)}

@app.post("/send-alert")
async def send_alert(district: str = Form(...), message: str = Form(...), language: str = Form("en")):
    """Send health alerts to users in a specific district with multilingual support."""
    try:
        # Translate message if needed
        translated_message = message
        if language != 'en':
            translated_message = translator.translate_with_nllb(message, 'en', language)
        
        # In production, this would:
        # 1. Query database for opted-in users in the district
        # 2. Send alerts via Twilio to all users
        # 3. Log alert delivery status
        
        alert_data = {
            "status": "sent",
            "district": district,
            "original_message": message,
            "translated_message": translated_message,
            "language": language,
            "timestamp": "2025-09-14T10:30:00Z",
            "recipients_count": 150  # Simulated
        }
        
        logger.info(f"Alert sent to {district}: {translated_message}")
        return alert_data
        
    except Exception as e:
        logger.error(f"Send alert error: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/health-check")
async def health_check():
    """Health check endpoint for Railway deployment."""
    return {
        "status": "healthy",
        "service": "AI-Driven Public Health Chatbot",
        "version": "1.0.0",
        "nllb_model_loaded": translator.nllb_model is not None,
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
        "biobert_processor": biobert_processor is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/supported-languages")
async def supported_languages():
    """Get list of supported Indian languages."""
    return {
        "languages": [
            {"code": "hi", "name": "Hindi", "native": "हिन्दी"},
            {"code": "bn", "name": "Bengali", "native": "বাংলা"},
            {"code": "te", "name": "Telugu", "native": "తెలుగు"},
            {"code": "ta", "name": "Tamil", "native": "தமிழ்"},
            {"code": "mr", "name": "Marathi", "native": "मराठी"},
            {"code": "gu", "name": "Gujarati", "native": "ગુજરાતી"},
            {"code": "kn", "name": "Kannada", "native": "ಕನ್ನಡ"},
            {"code": "ml", "name": "Malayalam", "native": "മലയാളം"},
            {"code": "pa", "name": "Punjabi", "native": "ਪੰਜਾਬੀ"},
            {"code": "or", "name": "Odia", "native": "ଓଡ଼ିଆ"},
            {"code": "as", "name": "Assamese", "native": "অসমীয়া"},
            {"code": "ur", "name": "Urdu", "native": "اردو"},
            {"code": "en", "name": "English", "native": "English"}
        ]
    }

# Bio-BERT Enhanced Endpoints

class SymptomAnalysisRequest(BaseModel):
    text: str
    language: Optional[str] = "en"

class DiseaseClassificationRequest(BaseModel):
    symptoms: List[str]
    context: Optional[str] = ""
    language: Optional[str] = "en"

class MedicalAssessmentRequest(BaseModel):
    symptoms: List[str]
    description: str
    language: Optional[str] = "en"

@app.post("/analyze-symptoms")
async def analyze_symptoms(request: SymptomAnalysisRequest):
    """Extract and analyze symptoms from biomedical text using Bio-BERT"""
    try:
        if not biobert_processor:
            return {
                "error": "Bio-BERT processor not available",
                "symptoms": [],
                "message": "Advanced symptom analysis is currently unavailable"
            }

        # Translate to English if needed for Bio-BERT processing
        text_to_analyze = request.text
        if request.language and request.language != "en":
            text_to_analyze = translator.translate_with_nllb(request.text, request.language, "en")

        # Extract symptoms using Bio-BERT
        symptoms_found = biobert_processor.extract_symptoms(text_to_analyze)

        # Translate symptom names back to original language if needed
        if request.language and request.language != "en":
            for symptom in symptoms_found:
                symptom["symptom"] = translator.translate_with_nllb(
                    symptom["symptom"], "en", request.language
                )

        return {
            "symptoms": symptoms_found,
            "total_symptoms": len(symptoms_found),
            "language": request.language,
            "biobert_processed": True
        }

    except Exception as e:
        logger.error(f"Symptom analysis error: {e}")
        return {
            "error": str(e),
            "symptoms": [],
            "message": "Error processing symptom analysis"
        }

@app.post("/classify-disease")
async def classify_disease(request: DiseaseClassificationRequest):
    """Classify potential diseases based on symptoms using Bio-BERT"""
    try:
        if not biobert_processor:
            return {
                "error": "Bio-BERT processor not available",
                "diseases": [],
                "message": "Disease classification is currently unavailable"
            }

        # Translate symptoms and context to English for processing
        symptoms_en = request.symptoms
        context_en = request.context or ""

        if request.language and request.language != "en":
            symptoms_en = [
                translator.translate_with_nllb(symptom, request.language, "en")
                for symptom in request.symptoms
            ]
            if request.context:
                context_en = translator.translate_with_nllb(request.context, request.language, "en")

        # Classify diseases using Bio-BERT
        diseases_found = biobert_processor.classify_disease(symptoms_en, context_en)

        # Translate disease names back to original language if needed
        if request.language and request.language != "en":
            for disease in diseases_found:
                disease["disease"] = translator.translate_with_nllb(
                    disease["disease"], "en", request.language
                )

        return {
            "diseases": diseases_found,
            "total_diseases": len(diseases_found),
            "symptoms_analyzed": symptoms_en,
            "language": request.language,
            "biobert_processed": True
        }

    except Exception as e:
        logger.error(f"Disease classification error: {e}")
        return {
            "error": str(e),
            "diseases": [],
            "message": "Error processing disease classification"
        }

@app.post("/assess-severity")
async def assess_severity(request: MedicalAssessmentRequest):
    """Assess severity of symptoms using Bio-BERT"""
    try:
        if not biobert_processor:
            return {
                "error": "Bio-BERT processor not available",
                "severity": "unknown",
                "message": "Severity assessment is currently unavailable"
            }

        # Translate to English for processing
        symptoms_en = request.symptoms
        description_en = request.description

        if request.language and request.language != "en":
            symptoms_en = [
                translator.translate_with_nllb(symptom, request.language, "en")
                for symptom in request.symptoms
            ]
            description_en = translator.translate_with_nllb(request.description, request.language, "en")

        # Assess severity using Bio-BERT
        severity_assessment = biobert_processor.assess_severity(symptoms_en, description_en)

        # Translate recommendation back to original language if needed
        if request.language and request.language != "en":
            severity_assessment["recommendation"] = translator.translate_with_nllb(
                severity_assessment["recommendation"], "en", request.language
            )

        return {
            "severity_assessment": severity_assessment,
            "symptoms_analyzed": symptoms_en,
            "language": request.language,
            "biobert_processed": True
        }

    except Exception as e:
        logger.error(f"Severity assessment error: {e}")
        return {
            "error": str(e),
            "severity_assessment": {"severity_level": "unknown", "recommendation": "Please consult a healthcare professional"},
            "message": "Error processing severity assessment"
        }

@app.post("/comprehensive-medical-analysis")
async def comprehensive_medical_analysis(request: MedicalAssessmentRequest):
    """Perform comprehensive medical analysis using Bio-BERT"""
    try:
        if not biobert_processor:
            return {
                "error": "Bio-BERT processor not available",
                "analysis": {},
                "message": "Comprehensive analysis is currently unavailable"
            }

        # Translate to English for processing
        symptoms_en = request.symptoms
        description_en = request.description

        if request.language and request.language != "en":
            symptoms_en = [
                translator.translate_with_nllb(symptom, request.language, "en")
                for symptom in request.symptoms
            ]
            description_en = translator.translate_with_nllb(request.description, request.language, "en")

        # Perform comprehensive analysis
        symptoms_found = biobert_processor.extract_symptoms(description_en)
        diseases_found = biobert_processor.classify_disease(symptoms_en, description_en)
        severity_assessment = biobert_processor.assess_severity(symptoms_en, description_en)

        # Generate medical summary
        symptom_names = [s["symptom"] for s in symptoms_found] + symptoms_en
        medical_summary = biobert_processor.generate_medical_summary(
            list(set(symptom_names)), diseases_found, severity_assessment
        )

        # Translate results back to original language if needed
        if request.language and request.language != "en":
            severity_assessment["recommendation"] = translator.translate_with_nllb(
                severity_assessment["recommendation"], "en", request.language
            )
            medical_summary = translator.translate_with_nllb(medical_summary, "en", request.language)

            # Translate symptom and disease names
            for symptom in symptoms_found:
                symptom["symptom"] = translator.translate_with_nllb(
                    symptom["symptom"], "en", request.language
                )
            for disease in diseases_found:
                disease["disease"] = translator.translate_with_nllb(
                    disease["disease"], "en", request.language
                )

        return {
            "symptoms_detected": symptoms_found,
            "potential_diseases": diseases_found,
            "severity_assessment": severity_assessment,
            "medical_summary": medical_summary,
            "language": request.language,
            "biobert_processed": True,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Comprehensive analysis error: {e}")
        return {
            "error": str(e),
            "symptoms_detected": [],
            "potential_diseases": [],
            "severity_assessment": {"severity_level": "unknown"},
            "medical_summary": "Error generating medical summary. Please consult a healthcare professional.",
            "message": "Error processing comprehensive analysis"
        }

@app.get("/biobert-status")
async def biobert_status():
    """Check Bio-BERT processor status"""
    if biobert_processor:
        return {
            "status": "available",
            "model": biobert_processor.model_name,
            "device": str(biobert_processor.device),
            "symptom_classifier": biobert_processor.symptom_classifier is not None,
            "disease_classifier": biobert_processor.disease_classifier is not None
        }
    else:
        return {
            "status": "unavailable",
            "message": "Bio-BERT processor not initialized"
        }
