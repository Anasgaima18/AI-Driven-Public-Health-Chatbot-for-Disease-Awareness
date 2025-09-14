from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests, os, json

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
            r = requests.post(f"{action_server_url}/generate", json=payload, timeout=10)
            reply = r.json().get("reply", "Sorry, I couldn't fetch an answer.")
        except Exception as e:
            reply = f"Demo reply: couldn't reach action server ({e})"
        dispatcher.utter_message(text=reply)
        return []
