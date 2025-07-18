"""
Query Classifier - Intelligent query classification and routing
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math

from ..utils.logger import log_info, log_debug, log_warning


class QueryType(Enum):
    """Types of queries the system can handle."""
    FACTUAL = "factual"          # Encyclopedic, factual information
    MATHEMATICAL = "mathematical" # Math problems, calculations
    CURRENT_EVENTS = "current"   # Recent news, current information
    CONVERSATIONAL = "conversational" # Chat, greetings, opinions
    DEFINITIONAL = "definitional" # Word definitions, explanations
    COMPARATIVE = "comparative"   # Comparisons between things
    PROCEDURAL = "procedural"     # How-to, step-by-step instructions
    BIOGRAPHICAL = "biographical" # Information about people
    GEOGRAPHICAL = "geographical" # Places, locations, geography
    HISTORICAL = "historical"     # Past events, history
    UNKNOWN = "unknown"          # Unclear or unclassifiable


@dataclass
class QueryClassification:
    """Result of query classification."""
    query_type: QueryType
    confidence: float
    keywords: List[str]
    entities: List[str]
    intent: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate classification data."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")


class QueryClassifier:
    """Intelligent query classification system."""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.keywords = self._initialize_keywords()
        self.entities_cache = {}
        
    def _initialize_patterns(self) -> Dict[QueryType, List[str]]:
        """Initialize regex patterns for query classification."""
        return {
            QueryType.MATHEMATICAL: [
                r'(?:calculate|solve|compute|find|what is)\s+.*[\+\-\*\/\=\^\(\)]',
                r'\b(?:equation|formula|derivative|integral|root|square|cube)\b',
                r'\b(?:sin|cos|tan|log|ln|exp|sqrt|abs|factorial)\b',
                r'\d+\s*[\+\-\*\/\^\(\)]\s*\d+',
                r'(?:what is|calculate|solve)\s+\d+.*\d+',
                r'\b(?:percentage|percent|ratio|proportion|fraction)\b',
                r'\b(?:graph|plot|function|variable|x|y|z)\s*[\=\(\)]'
            ],
            QueryType.DEFINITIONAL: [
                r'(?:what is|what does|define|definition of|meaning of)\s+\w+',
                r'(?:explain|tell me about|describe)\s+(?:the\s+)?(?:concept|term|word)\s+\w+',
                r'(?:what does)\s+\w+\s+(?:mean|stand for)',
                r'(?:definition|meaning|explanation)\s+of\s+\w+'
            ],
            QueryType.BIOGRAPHICAL: [
                r'(?:who is|who was|tell me about|biography of)\s+[A-Z][a-z]+\s+[A-Z][a-z]+',
                r'(?:when was|where was)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:born|died)',
                r'(?:how old is|age of)\s+[A-Z][a-z]+\s+[A-Z][a-z]+',
                r'(?:famous|known for|achievements of)\s+[A-Z][a-z]+\s+[A-Z][a-z]+'
            ],
            QueryType.HISTORICAL: [
                r'(?:what happened|when did|when was|history of)\s+.*(?:in|during)\s+\d{4}',
                r'(?:world war|civil war|revolution|battle of|empire|dynasty)',
                r'(?:ancient|medieval|renaissance|industrial|modern)\s+(?:era|period|times)',
                r'(?:BC|AD|century|decade|year)\s+\d+',
                r'(?:historical|historic|past|former|old|ancient)\s+(?:event|time|period)'
            ],
            QueryType.GEOGRAPHICAL: [
                r'(?:where is|capital of|location of|geography of)\s+[A-Z][a-z]+',
                r'(?:country|city|state|province|continent|ocean|mountain|river)',
                r'(?:population of|area of|climate of|language of)\s+[A-Z][a-z]+',
                r'(?:map|coordinates|latitude|longitude|timezone)'
            ],
            QueryType.COMPARATIVE: [
                r'(?:compare|difference between|vs|versus|better than)',
                r'(?:what is the difference|how does)\s+\w+\s+(?:differ|compare)',
                r'(?:similarities|differences|pros and cons|advantages|disadvantages)',
                r'(?:which is better|which is faster|which is bigger)'
            ],
            QueryType.PROCEDURAL: [
                r'(?:how to|how do|how can|steps to|instructions for)',
                r'(?:tutorial|guide|manual|process|procedure|method)',
                r'(?:step by step|walkthrough|instructions|directions)',
                r'(?:make|create|build|install|setup|configure)'
            ],
            QueryType.CURRENT_EVENTS: [
                r'(?:latest|recent|current|today|yesterday|this week|this month)',
                r'(?:news|breaking|update|happening|events|developments)',
                r'(?:stock price|market|economy|politics|election|covid|pandemic)',
                r'(?:what is happening|current situation|recent events)'
            ],
            QueryType.CONVERSATIONAL: [
                r'(?:hello|hi|hey|good morning|good afternoon|good evening)',
                r'(?:how are you|how do you feel|what do you think)',
                r'(?:thank you|thanks|please|sorry|excuse me)',
                r'(?:bye|goodbye|see you|farewell|talk to you later)',
                r'(?:can you help|please help|i need help)'
            ]
        }
    
    def _initialize_keywords(self) -> Dict[QueryType, List[str]]:
        """Initialize keyword lists for query classification."""
        return {
            QueryType.FACTUAL: [
                "fact", "information", "data", "knowledge", "truth", "reality",
                "encyclopedia", "reference", "source", "evidence", "proof"
            ],
            QueryType.MATHEMATICAL: [
                "math", "mathematics", "calculation", "equation", "formula",
                "algebra", "geometry", "calculus", "statistics", "probability",
                "number", "digit", "integer", "decimal", "fraction"
            ],
            QueryType.DEFINITIONAL: [
                "definition", "meaning", "explanation", "concept", "term",
                "word", "phrase", "vocabulary", "dictionary", "glossary"
            ],
            QueryType.BIOGRAPHICAL: [
                "biography", "life", "birth", "death", "career", "achievement",
                "famous", "celebrity", "politician", "scientist", "artist"
            ],
            QueryType.HISTORICAL: [
                "history", "historical", "past", "ancient", "medieval",
                "war", "battle", "empire", "civilization", "era", "period"
            ],
            QueryType.GEOGRAPHICAL: [
                "geography", "location", "place", "country", "city", "state",
                "continent", "ocean", "mountain", "river", "capital", "population"
            ],
            QueryType.COMPARATIVE: [
                "compare", "comparison", "difference", "similarity", "versus",
                "better", "worse", "advantage", "disadvantage", "pros", "cons"
            ],
            QueryType.PROCEDURAL: [
                "how", "tutorial", "guide", "instruction", "step", "process",
                "procedure", "method", "technique", "way", "approach"
            ],
            QueryType.CURRENT_EVENTS: [
                "news", "current", "recent", "latest", "today", "now",
                "breaking", "update", "happening", "event", "development"
            ],
            QueryType.CONVERSATIONAL: [
                "hello", "hi", "thanks", "please", "sorry", "opinion",
                "feel", "think", "believe", "chat", "talk", "conversation"
            ]
        }

    def classify(self, query: str) -> QueryClassification:
        """
        Classify a query into its most likely type.
        
        Args:
            query: The input query string
            
        Returns:
            QueryClassification object with type, confidence, and metadata
        """
        log_debug(f"Classifying query: {query[:50]}...")
        
        query_lower = query.lower().strip()
        
        # Calculate scores for each query type
        scores = {}
        for query_type in QueryType:
            if query_type == QueryType.UNKNOWN:
                continue
            scores[query_type] = self._calculate_score(query_lower, query_type)
        
        # Find the best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        # If no strong match, classify as UNKNOWN
        if best_score < 0.3:
            best_type = QueryType.UNKNOWN
            best_score = 0.0
        
        # Extract additional metadata
        keywords = self._extract_keywords(query_lower, best_type)
        entities = self._extract_entities(query)
        intent = self._determine_intent(query_lower, best_type)
        
        metadata = {
            "query_length": len(query),
            "word_count": len(query.split()),
            "has_numbers": bool(re.search(r'\d', query)),
            "has_punctuation": bool(re.search(r'[?!]', query)),
            "scores": scores
        }
        
        classification = QueryClassification(
            query_type=best_type,
            confidence=best_score,
            keywords=keywords,
            entities=entities,
            intent=intent,
            metadata=metadata
        )
        
        log_info(f"Query classified as {best_type.value} with confidence {best_score:.2f}")
        return classification

    def _calculate_score(self, query: str, query_type: QueryType) -> float:
        """Calculate similarity score between query and query type."""
        pattern_score = self._calculate_pattern_score(query, query_type)
        keyword_score = self._calculate_keyword_score(query, query_type)
        
        # Weighted combination
        total_score = (0.6 * pattern_score) + (0.4 * keyword_score)
        return min(total_score, 1.0)

    def _calculate_pattern_score(self, query: str, query_type: QueryType) -> float:
        """Calculate score based on regex pattern matching."""
        patterns = self.patterns.get(query_type, [])
        if not patterns:
            return 0.0
            
        matches = 0
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                matches += 1
        
        return matches / len(patterns) if patterns else 0.0

    def _calculate_keyword_score(self, query: str, query_type: QueryType) -> float:
        """Calculate score based on keyword presence."""
        keywords = self.keywords.get(query_type, [])
        if not keywords:
            return 0.0
            
        query_words = set(query.split())
        keyword_matches = len(query_words.intersection(keywords))
        
        return keyword_matches / len(keywords) if keywords else 0.0

    def _extract_keywords(self, query: str, query_type: QueryType) -> List[str]:
        """Extract relevant keywords from the query."""
        query_words = set(query.split())
        type_keywords = self.keywords.get(query_type, [])
        
        # Find intersection of query words and type keywords
        relevant_keywords = list(query_words.intersection(type_keywords))
        
        # Add high-value words that aren't in type keywords
        high_value_words = [word for word in query_words 
                          if len(word) > 3 and word not in ['what', 'where', 'when', 'how', 'why']]
        
        return list(set(relevant_keywords + high_value_words[:3]))

    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from the query."""
        # Simple entity extraction - can be enhanced with NLP libraries
        entities = []
        
        # Extract proper nouns (capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        entities.extend(proper_nouns)
        
        # Extract years
        years = re.findall(r'\b(?:19|20)\d{2}\b', query)
        entities.extend(years)
        
        # Extract numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', query)
        entities.extend(numbers)
        
        return list(set(entities))

    def _determine_intent(self, query: str, query_type: QueryType) -> str:
        """Determine the user's intent based on query type and content."""
        intent_map = {
            QueryType.FACTUAL: "seek_information",
            QueryType.MATHEMATICAL: "solve_problem",
            QueryType.CURRENT_EVENTS: "get_updates",
            QueryType.CONVERSATIONAL: "interact_socially",
            QueryType.DEFINITIONAL: "understand_meaning",
            QueryType.COMPARATIVE: "make_comparison",
            QueryType.PROCEDURAL: "learn_process",
            QueryType.BIOGRAPHICAL: "learn_about_person",
            QueryType.GEOGRAPHICAL: "get_location_info",
            QueryType.HISTORICAL: "learn_history",
            QueryType.UNKNOWN: "unclear_intent"
        }
        
        base_intent = intent_map.get(query_type, "unknown")
        
        # Refine intent based on query markers
        if any(word in query for word in ['how', 'steps', 'tutorial']):
            return f"{base_intent}_instructional"
        elif any(word in query for word in ['why', 'reason', 'cause']):
            return f"{base_intent}_explanatory"
        elif any(word in query for word in ['best', 'recommend', 'suggest']):
            return f"{base_intent}_advisory"
        
        return base_intent

    def get_routing_recommendation(self, classification: QueryClassification) -> Dict[str, Any]:
        """
        Get routing recommendation based on classification.
        
        Args:
            classification: QueryClassification object
            
        Returns:
            Dictionary with routing recommendations
        """
        routing_map = {
            QueryType.FACTUAL: {"handler": "knowledge_base", "priority": "medium"},
            QueryType.MATHEMATICAL: {"handler": "math_solver", "priority": "high"},
            QueryType.CURRENT_EVENTS: {"handler": "news_api", "priority": "high"},
            QueryType.CONVERSATIONAL: {"handler": "chat_bot", "priority": "low"},
            QueryType.DEFINITIONAL: {"handler": "dictionary_api", "priority": "medium"},
            QueryType.COMPARATIVE: {"handler": "comparison_engine", "priority": "medium"},
            QueryType.PROCEDURAL: {"handler": "instruction_generator", "priority": "medium"},
            QueryType.BIOGRAPHICAL: {"handler": "biography_db", "priority": "medium"},
            QueryType.GEOGRAPHICAL: {"handler": "geography_api", "priority": "medium"},
            QueryType.HISTORICAL: {"handler": "history_db", "priority": "medium"},
            QueryType.UNKNOWN: {"handler": "fallback", "priority": "low"}
        }
        
        base_routing = routing_map.get(classification.query_type, routing_map[QueryType.UNKNOWN])
        
        # Adjust priority based on confidence
        if classification.confidence > 0.8:
            priority = "high"
        elif classification.confidence > 0.6:
            priority = "medium"
        else:
            priority = "low"
            
        return {
            "primary_handler": base_routing["handler"],
            "priority": priority,
            "confidence": classification.confidence,
            "fallback_handlers": self._get_fallback_handlers(classification),
            "special_requirements": self._get_special_requirements(classification)
        }

    def _get_fallback_handlers(self, classification: QueryClassification) -> List[str]:
        """Get fallback handlers based on classification."""
        fallbacks = ["general_search", "fallback"]
        
        # Add specific fallbacks based on query type
        if classification.query_type == QueryType.CURRENT_EVENTS:
            fallbacks.insert(0, "web_search")
        elif classification.query_type == QueryType.MATHEMATICAL:
            fallbacks.insert(0, "calculator")
        elif classification.query_type == QueryType.DEFINITIONAL:
            fallbacks.insert(0, "web_search")
            
        return fallbacks

    def _get_special_requirements(self, classification: QueryClassification) -> Dict[str, Any]:
        """Get special requirements for handling the query."""
        requirements = {}
        
        if classification.query_type == QueryType.CURRENT_EVENTS:
            requirements["requires_real_time"] = True
        elif classification.query_type == QueryType.MATHEMATICAL:
            requirements["requires_computation"] = True
        elif classification.query_type == QueryType.PROCEDURAL:
            requirements["requires_structured_output"] = True
        elif classification.query_type == QueryType.COMPARATIVE:
            requirements["requires_multi_source"] = True
            
        return requirements

    def batch_classify(self, queries: List[str]) -> List[QueryClassification]:
        """
        Classify multiple queries in batch.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of QueryClassification objects
        """
        log_info(f"Batch classifying {len(queries)} queries")
        
        results = []
        for query in queries:
            try:
                classification = self.classify(query)
                results.append(classification)
            except Exception as e:
                log_warning(f"Error classifying query '{query}': {str(e)}")
                # Return unknown classification for failed queries
                results.append(QueryClassification(
                    query_type=QueryType.UNKNOWN,
                    confidence=0.0,
                    keywords=[],
                    entities=[],
                    intent="error",
                    metadata={"error": str(e)}
                ))
        
        return results

    def get_classification_stats(self, classifications: List[QueryClassification]) -> Dict[str, Any]:
        """
        Get statistics about a batch of classifications.
        
        Args:
            classifications: List of QueryClassification objects
            
        Returns:
            Dictionary with statistics
        """
        if not classifications:
            return {}
            
        type_counts = {}
        confidence_sum = 0
        
        for classification in classifications:
            query_type = classification.query_type.value
            type_counts[query_type] = type_counts.get(query_type, 0) + 1
            confidence_sum += classification.confidence
        
        avg_confidence = confidence_sum / len(classifications)
        
        return {
            "total_queries": len(classifications),
            "type_distribution": type_counts,
            "average_confidence": avg_confidence,
            "high_confidence_queries": len([c for c in classifications if c.confidence > 0.8]),
            "unknown_queries": len([c for c in classifications if c.query_type == QueryType.UNKNOWN])
        }