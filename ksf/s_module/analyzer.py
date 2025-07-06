import logging
import spacy
from transformers import pipeline, Pipeline
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class IntentAnalyzer:
    """
    Analyzes user queries using advanced NLP models to understand intent,
    extract entities, and assess relevance. This is the new brain of the S-Module.
    """
    def __init__(self, manifest: Dict[str, Any]):
        """
        Initializes the analyzer by loading necessary models and the knowledge manifest.
        """
        self.manifest = manifest
        self.nlp_ner: Optional[spacy.language.Language] = None
        self.classifier: Optional[Pipeline] = None

        try:
            # 1. Load SpaCy for Named Entity Recognition (NER)
            logger.info("Loading SpaCy model for NER...")
            self.nlp_ner = spacy.load("zh_core_web_md")
            logger.info("✓ SpaCy model loaded.")

            # 2. Load a Zero-Shot a NLI model for Intent Classification
            logger.info("Loading Zero-Shot classifier model...")
            # We use a model suitable for NLI, which powers zero-shot classification
            self.classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33")
            logger.info("✓ Zero-Shot classifier loaded.")

        except Exception as e:
            logger.error(f"Failed to load NLP models: {e}", exc_info=True)
            logger.warning("IntentAnalyzer will operate in a degraded, rule-based mode.")
            # Here you could implement a fallback to the old simple logic if needed

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Performs a full analysis of the user's query.
        """
        if not self.nlp_ner or not self.classifier:
            logger.warning("Analysis is degraded due to model loading failure.")
            return {"intent": "UNKNOWN", "entities": [], "error": "Models not loaded"}
            
        # 1. Extract named entities
        entities = self._enhanced_ner(text)
        
        # 2. Classify intent
        intent_result = self._advanced_intent_classification(text)

        return {
            "intent": intent_result['intent'],
            "intent_score": intent_result['score'],
            "entities": entities
        }

    def _enhanced_ner(self, text: str) -> List[Dict[str, str]]:
        """Extracts entities using SpaCy."""
        doc = self.nlp_ner(text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        logger.debug(f"Extracted entities: {entities}")
        return entities

    def _advanced_intent_classification(self, text: str) -> Dict[str, Any]:
        """
        Classifies intent using a zero-shot model, guided by the knowledge manifest.
        """
        # Define candidate intents, some generic, some from the manifest
        candidate_labels = ["查询信息", "寻求建议", "进行比较", "规划行程", "闲聊"]
        hypothesis_template = "这句用户提问的意图是关于{}。"
        
        result = self.classifier(text, candidate_labels, hypothesis_template=hypothesis_template)
        
        top_intent = result['labels'][0]
        top_score = result['scores'][0]
        
        logger.debug(f"Intent classification for '{text}': {top_intent} (Score: {top_score:.2f})")
        return {"intent": top_intent, "score": top_score}

    def assess_relevance(self, query: str, knowledge_content: str) -> float:
        """
        Assesses the relevance of a piece of knowledge to the query using the NLI model.
        Returns a score between 0 and 1.
        """
        if not self.classifier:
            return 0.5 # Default relevance if model isn't loaded
            
        # We frame this as a zero-shot classification problem where the knowledge is the text
        # and the query is the single "class" we want to check for entailment.
        hypothesis_template = "这段知识可以回答关于'{}'的提问。"
        result = self.classifier(knowledge_content, [query], hypothesis_template=hypothesis_template)
        
        relevance_score = result['scores'][0]
        logger.debug(f"Relevance of knowledge to '{query}': {relevance_score:.2f}")
        return relevance_score 