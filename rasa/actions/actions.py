from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, FollowupAction
import requests, os, json
import logging

logger = logging.getLogger(__name__)

class ActionGenerateAnswer(Action):
    def name(self) -> str:
        return "action_generate_answer"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict) -> list:
        user_text = tracker.latest_message.get('text') or ""
        lang = tracker.get_slot('language') or 'en'
        
        payload = {"text": user_text, "lang": lang}
        action_server_url = os.getenv("ACTION_SERVER_URL", "http://localhost:8000")
        
        try:
            r = requests.post(f"{action_server_url}/generate", json=payload, timeout=15)
            response_data = r.json()
            reply = response_data.get("reply", "Sorry, I couldn't fetch an answer.")
            detected_lang = response_data.get("language", lang)
            
            # Update language slot if language was detected
            events = []
            if detected_lang != lang:
                events.append(SlotSet("language", detected_lang))
            
            dispatcher.utter_message(text=reply)
            return events
            
        except Exception as e:
            logger.error(f"Action server error: {e}")
            fallback_reply = "I apologize for the technical difficulty. Please call 104 for immediate health assistance or visit your nearest health center."
            dispatcher.utter_message(text=fallback_reply)
            return []

class ActionGetVaccinationInfo(Action):
    def name(self) -> str:
        return "action_get_vaccination_info"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict) -> list:
        location = tracker.get_slot('location') or "your area"
        lang = tracker.get_slot('language') or 'en'
        
        action_server_url = os.getenv("ACTION_SERVER_URL", "http://localhost:8000")
        
        try:
            r = requests.get(f"{action_server_url}/vaccination-schedule/{location}", 
                           params={"language": lang}, timeout=10)
            data = r.json()
            
            if "vaccines" in data:
                message = f"Vaccination schedule for {location}:\n\n"
                for vaccine in data["vaccines"]:
                    message += f"🩹 {vaccine['name']}\n"
                    message += f"📅 Date: {vaccine['next_date']}\n"
                    message += f"📍 Location: {vaccine['location']}\n"
                    message += f"👥 Age Group: {vaccine['age_group']}\n\n"
                message += "For more information, call 104 or visit your nearest health center."
            else:
                message = "Sorry, vaccination information is not available for your area right now. Please contact your local health center."
            
            dispatcher.utter_message(text=message)
            return []
            
        except Exception as e:
            logger.error(f"Vaccination info error: {e}")
            dispatcher.utter_message(text="Please contact your local health center for vaccination information or call 104.")
            return []

class ActionGetDiseaseOutbreaks(Action):
    def name(self) -> str:
        return "action_get_disease_outbreaks"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict) -> list:
        location = tracker.get_slot('location') or "your district"
        lang = tracker.get_slot('language') or 'en'
        
        action_server_url = os.getenv("ACTION_SERVER_URL", "http://localhost:8000")
        
        try:
            r = requests.get(f"{action_server_url}/disease-outbreaks/{location}", 
                           params={"language": lang}, timeout=10)
            data = r.json()
            
            if "outbreaks" in data and data["outbreaks"]:
                message = f"⚠️ Current health alerts for {location}:\n\n"
                for outbreak in data["outbreaks"]:
                    severity_emoji = "🔴" if outbreak.get("severity") == "high" else "🟡" if outbreak.get("severity") == "moderate" else "🟢"
                    message += f"{severity_emoji} {outbreak['disease']}\n"
                    message += f"📊 Cases: {outbreak.get('cases', 'Unknown')}\n"
                    message += f"🛡️ Prevention: {outbreak['prevention']}\n"
                    message += f"📞 Helpline: {outbreak.get('helpline', '104')}\n\n"
                message += "Stay safe and follow prevention guidelines!"
            else:
                message = f"✅ No current disease outbreaks reported in {location}. Continue following basic health precautions."
            
            dispatcher.utter_message(text=message)
            return []
            
        except Exception as e:
            logger.error(f"Disease outbreaks error: {e}")
            dispatcher.utter_message(text="Please check with local health authorities for current health alerts or call 104.")
            return []

class ActionSetLanguage(Action):
    def name(self) -> str:
        return "action_set_language"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict) -> list:
        # Extract language from entities or intent
        entities = tracker.latest_message.get('entities', [])
        language = None
        
        for entity in entities:
            if entity['entity'] == 'language':
                language = entity['value']
                break
        
        if not language:
            language = 'en'  # Default to English
        
        dispatcher.utter_message(text=f"Language set to {language}. I'll respond in this language from now on.")
        return [SlotSet("language", language)]

class ActionProvideHealthTips(Action):
    def name(self) -> str:
        return "action_provide_health_tips"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict) -> list:
        lang = tracker.get_slot('language') or 'en'
        disease = tracker.get_slot('disease') or None
        
        # Create health query for action server
        query_text = f"Provide preventive health tips"
        if disease:
            query_text = f"Provide preventive health tips for {disease}"
        
        payload = {"text": query_text, "lang": lang}
        action_server_url = os.getenv("ACTION_SERVER_URL", "http://localhost:8000")
        
        try:
            r = requests.post(f"{action_server_url}/generate", json=payload, timeout=15)
            response_data = r.json()
            reply = response_data.get("reply", "Remember to maintain good hygiene, eat healthy food, exercise regularly, and get enough sleep.")
            
            dispatcher.utter_message(text=reply)
            return []
            
        except Exception as e:
            logger.error(f"Health tips error: {e}")
            basic_tips = """
🌟 Basic Health Tips:
• Wash hands frequently with soap
• Drink clean, boiled water
• Eat fresh fruits and vegetables
• Get adequate sleep (7-8 hours)
• Exercise regularly
• Visit health center for regular check-ups
• Call 104 for health emergencies
            """
            dispatcher.utter_message(text=basic_tips.strip())
            return []

class ActionSymptomChecker(Action):
    def name(self) -> str:
        return "action_symptom_checker"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict) -> list:
        symptoms = tracker.get_slot('symptoms') or ""
        lang = tracker.get_slot('language') or 'en'
        age = tracker.get_slot('age') or None
        
        # Build query for symptom checking
        query_data = {
            "symptoms": symptoms,
            "age": age,
            "language": lang
        }
        
        action_server_url = os.getenv("ACTION_SERVER_URL", "http://localhost:8000")
        
        try:
            r = requests.post(f"{action_server_url}/health-query", json=query_data, timeout=15)
            response_data = r.json()
            reply = response_data.get("reply", "Please consult a healthcare professional for proper diagnosis.")
            
            # Add disclaimer
            disclaimer = "\n\n⚠️ This is not a medical diagnosis. Please consult a qualified healthcare professional for proper medical advice."
            
            dispatcher.utter_message(text=reply + disclaimer)
            return []
            
        except Exception as e:
            logger.error(f"Symptom checker error: {e}")
            emergency_response = """
⚠️ For any concerning symptoms:
• Call 104 immediately for emergency
• Visit nearest Primary Health Center
• Do not delay seeking professional medical help
• This chatbot cannot replace medical diagnosis
            """
            dispatcher.utter_message(text=emergency_response.strip())
            return []
