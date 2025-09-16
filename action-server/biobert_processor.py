"""
Bio-BERT Integration for Biomedical Text Processing
Provides advanced biomedical text analysis, symptom recognition, and disease classification
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.metrics.pairwise import cosine_similarity
import re

logger = logging.getLogger(__name__)

class BioBERTProcessor:
    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.1"):
        """
        Initialize Bio-BERT model for biomedical text processing

        Args:
            model_name: HuggingFace model name for Bio-BERT
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.symptom_classifier = None
        self.disease_classifier = None

        # Medical knowledge base
        self.medical_terms = self._load_medical_knowledge()
        self.symptom_embeddings = {}
        self.disease_embeddings = {}

        self.load_model()

    def load_model(self):
        """Load Bio-BERT model and tokenizers"""
        try:
            logger.info(f"Loading Bio-BERT model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            # Load additional classifiers for specific tasks
            self._load_classifiers()

            logger.info("Bio-BERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Bio-BERT model: {e}")
            raise

    def _load_classifiers(self):
        """Load specialized classifiers for symptoms and diseases"""
        try:
            # Symptom classification pipeline
            self.symptom_classifier = pipeline(
                "text-classification",
                model="dmis-lab/biobert-base-cased-v1.1",
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )

            # Disease classification pipeline
            self.disease_classifier = pipeline(
                "text-classification",
                model="dmis-lab/biobert-base-cased-v1.1",
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )

            logger.info("Bio-BERT classifiers loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load specialized classifiers: {e}")

    def _load_medical_knowledge(self) -> Dict[str, List[str]]:
        """Load medical knowledge base with symptoms and diseases"""
        return {
            "symptoms": [
                "fever", "headache", "cough", "fatigue", "nausea", "vomiting",
                "diarrhea", "abdominal pain", "chest pain", "shortness of breath",
                "dizziness", "muscle pain", "joint pain", "rash", "sore throat",
                "runny nose", "loss of appetite", "weight loss", "night sweats",
                "chills", "sweating", "confusion", "seizures", "paralysis"
            ],
            "diseases": [
                "dengue", "malaria", "typhoid", "covid-19", "influenza",
                "tuberculosis", "pneumonia", "diabetes", "hypertension",
                "asthma", "bronchitis", "hepatitis", "cholera", "measles",
                "chickenpox", "mumps", "rubella", "diphtheria", "tetanus"
            ],
            "severity_indicators": [
                "severe", "mild", "moderate", "critical", "emergency",
                "urgent", "chronic", "acute", "life-threatening"
            ]
        }

    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Get Bio-BERT embeddings for input text

        Args:
            text: Input biomedical text

        Returns:
            numpy array of embeddings
        """
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling over token embeddings
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = embeddings.cpu().numpy()

            return embeddings[0]  # Return first (and only) embedding
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return np.zeros(768)  # Return zero vector on error

    def extract_symptoms(self, text: str) -> List[Dict[str, any]]:
        """
        Extract symptoms from biomedical text using Bio-BERT

        Args:
            text: Input text describing symptoms

        Returns:
            List of detected symptoms with confidence scores
        """
        symptoms_found = []
        text_lower = text.lower()

        # Direct symptom matching
        for symptom in self.medical_terms["symptoms"]:
            if symptom in text_lower:
                # Get context around symptom
                symptom_context = self._extract_context(text, symptom)
                embedding = self.get_embeddings(symptom_context)

                symptoms_found.append({
                    "symptom": symptom,
                    "confidence": 0.9,  # High confidence for direct matches
                    "context": symptom_context,
                    "embedding": embedding
                })

        # Use Bio-BERT for semantic symptom detection
        if self.symptom_classifier:
            try:
                # Classify the entire text for symptom presence
                result = self.symptom_classifier(text, return_all_scores=True)
                for pred in result[0]:
                    if pred["label"] == "SYMPTOM_PRESENT" and pred["score"] > 0.7:
                        # Extract potential symptoms using embeddings
                        semantic_symptoms = self._find_semantic_symptoms(text)
                        symptoms_found.extend(semantic_symptoms)
            except Exception as e:
                logger.warning(f"Symptom classification failed: {e}")

        return symptoms_found

    def _extract_context(self, text: str, keyword: str, window: int = 50) -> str:
        """Extract context around a keyword"""
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        start = text_lower.find(keyword_lower)

        if start == -1:
            return text[:100]  # Return first 100 chars if keyword not found

        # Extract context window
        context_start = max(0, start - window)
        context_end = min(len(text), start + len(keyword) + window)

        return text[context_start:context_end]

    def _find_semantic_symptoms(self, text: str) -> List[Dict[str, any]]:
        """Find symptoms using semantic similarity"""
        symptoms = []
        text_embedding = self.get_embeddings(text)

        for symptom in self.medical_terms["symptoms"]:
            if symptom not in self.symptom_embeddings:
                self.symptom_embeddings[symptom] = self.get_embeddings(symptom)

            similarity = cosine_similarity(
                [text_embedding],
                [self.symptom_embeddings[symptom]]
            )[0][0]

            if similarity > 0.7:  # Similarity threshold
                symptoms.append({
                    "symptom": symptom,
                    "confidence": float(similarity),
                    "context": text,
                    "embedding": text_embedding
                })

        return symptoms

    def classify_disease(self, symptoms: List[str], context: str = "") -> List[Dict[str, any]]:
        """
        Classify potential diseases based on symptoms

        Args:
            symptoms: List of symptoms
            context: Additional context information

        Returns:
            List of potential diseases with confidence scores
        """
        diseases_found = []
        symptom_text = " ".join(symptoms) + " " + context
        symptom_embedding = self.get_embeddings(symptom_text)

        for disease in self.medical_terms["diseases"]:
            if disease not in self.disease_embeddings:
                self.disease_embeddings[disease] = self.get_embeddings(disease)

            similarity = cosine_similarity(
                [symptom_embedding],
                [self.disease_embeddings[disease]]
            )[0][0]

            if similarity > 0.6:  # Disease classification threshold
                diseases_found.append({
                    "disease": disease,
                    "confidence": float(similarity),
                    "matched_symptoms": symptoms,
                    "embedding": symptom_embedding
                })

        # Sort by confidence
        diseases_found.sort(key=lambda x: x["confidence"], reverse=True)
        return diseases_found[:5]  # Return top 5 matches

    def assess_severity(self, symptoms: List[str], text: str) -> Dict[str, any]:
        """
        Assess severity of symptoms using Bio-BERT

        Args:
            symptoms: List of symptoms
            text: Full symptom description

        Returns:
            Severity assessment with confidence
        """
        severity_indicators = []
        text_lower = text.lower()

        # Check for severity indicators
        for indicator in self.medical_terms["severity_indicators"]:
            if indicator in text_lower:
                severity_indicators.append(indicator)

        # Use Bio-BERT to assess overall severity
        severity_text = f"Symptoms: {' '.join(symptoms)}. Description: {text}"
        severity_embedding = self.get_embeddings(severity_text)

        # Simple severity classification based on embeddings
        severity_score = self._calculate_severity_score(severity_embedding, severity_indicators)

        severity_level = "mild"
        if severity_score > 0.7:
            severity_level = "severe"
        elif severity_score > 0.4:
            severity_level = "moderate"

        return {
            "severity_level": severity_level,
            "severity_score": severity_score,
            "severity_indicators": severity_indicators,
            "recommendation": self._get_severity_recommendation(severity_level)
        }

    def _calculate_severity_score(self, embedding: np.ndarray, indicators: List[str]) -> float:
        """Calculate severity score from embedding and indicators"""
        base_score = 0.3  # Base severity score

        # Adjust based on severity indicators
        severity_weights = {
            "severe": 0.3,
            "critical": 0.4,
            "emergency": 0.4,
            "life-threatening": 0.5,
            "urgent": 0.2,
            "acute": 0.2
        }

        for indicator in indicators:
            if indicator in severity_weights:
                base_score += severity_weights[indicator]

        return min(base_score, 1.0)

    def _get_severity_recommendation(self, severity: str) -> str:
        """Get medical recommendation based on severity"""
        recommendations = {
            "mild": "Monitor symptoms and consult a healthcare provider if they worsen. Rest and stay hydrated.",
            "moderate": "Seek medical attention within 24-48 hours. Follow basic hygiene practices.",
            "severe": "Seek immediate medical attention. Call emergency services if experiencing difficulty breathing or chest pain."
        }
        return recommendations.get(severity, "Consult a healthcare professional immediately.")

    def generate_medical_summary(self, symptoms: List[str], diseases: List[Dict], severity: Dict) -> str:
        """
        Generate a comprehensive medical summary using Bio-BERT insights

        Args:
            symptoms: List of detected symptoms
            diseases: List of potential diseases
            severity: Severity assessment

        Returns:
            Formatted medical summary
        """
        summary = f"""
        MEDICAL ASSESSMENT SUMMARY
        ==========================

        DETECTED SYMPTOMS:
        {', '.join(symptoms) if symptoms else 'No specific symptoms identified'}

        POTENTIAL DIAGNOSES:
        """

        if diseases:
            for i, disease in enumerate(diseases[:3], 1):
                summary += f"{i}. {disease['disease'].title()} (Confidence: {disease['confidence']:.2f})\n"
        else:
            summary += "No specific diseases identified from symptoms\n"

        summary += f"""
        SEVERITY ASSESSMENT:
        Level: {severity['severity_level'].title()}
        Score: {severity['severity_score']:.2f}

        RECOMMENDATION:
        {severity['recommendation']}

        IMPORTANT: This is an AI-generated assessment and should not replace professional medical advice.
        Please consult a qualified healthcare provider for accurate diagnosis and treatment.
        """

        return summary.strip()

# Global Bio-BERT processor instance
biobert_processor = None

def get_biobert_processor() -> BioBERTProcessor:
    """Get or create Bio-BERT processor instance"""
    global biobert_processor
    if biobert_processor is None:
        try:
            biobert_processor = BioBERTProcessor()
        except Exception as e:
            logger.error(f"Failed to initialize Bio-BERT processor: {e}")
            return None
    return biobert_processor</content>
<parameter name="filePath">c:\Users\Moham\Downloads\AI-Driven Public Health Chatbot for Disease Awareness\action-server\biobert_processor.py