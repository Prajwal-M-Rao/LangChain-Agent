"""
Wikipedia Tool - Wikipedia search and content retrieval
"""

import re
import requests
from typing import Dict, Any, List, Optional
from urllib.parse import quote
import time

from .base_tool import BaseTool, ToolResult, ToolConfig, ToolType, ToolStatus
from ..utils.logger import log_info, log_error, log_warning


class WikipediaTool(BaseTool):
    """Tool for searching and retrieving Wikipedia content."""
    
    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self.tool_type = ToolType.ENCYCLOPEDIA
        self.base_url = "https://en.wikipedia.org/api/rest_v1"
        self.search_url = "https://en.wikipedia.org/w/api.php"
    
    def _initialize(self):
        """Initialize Wikipedia tool."""
        try:
            # Test API connectivity
            response = requests.get(f"{self.base_url}/page/summary/Test", timeout=5)
            if response.status_code == 200:
                self.status = ToolStatus.AVAILABLE
                log_info("Wikipedia tool initialized successfully")
            else:
                self.status = ToolStatus.UNAVAILABLE
                log_warning("Wikipedia API not accessible")
        except Exception as e:
            self.status = ToolStatus.ERROR
            log_error(f"Wikipedia tool initialization failed: {e}")
    
    @property
    def name(self) -> str:
        return "Wikipedia"
    
    @property
    def description(self) -> str:
        return "Search Wikipedia for factual information, biographies, and general knowledge"
    
    def _search_impl(self, query: str, **kwargs) -> ToolResult:
        """Implementation-specific Wikipedia search."""
        try:
            # Search for relevant pages
            search_results = self._search_pages(query)
            
            if not search_results:
                return ToolResult(
                    success=False,
                    content="",
                    metadata={"query": query, "search_results": []},
                    source=self.name,
                    confidence=0.0,
                    processing_time=0.0,
                    error_message="No Wikipedia pages found for query"
                )
            
            # Get the best matching page
            best_page = search_results[0]
            page_content = self._get_page_content(best_page["title"])
            
            if not page_content:
                return ToolResult(
                    success=False,
                    content="",
                    metadata={"query": query, "page_title": best_page["title"]},
                    source=self.name,
                    confidence=0.0,
                    processing_time=0.0,
                    error_message="Could not retrieve page content"
                )
            
            # Calculate confidence based on query match
            confidence = self._calculate_confidence(query, page_content, best_page)
            
            return ToolResult(
                success=True,
                content=page_content["extract"],
                metadata={
                    "query": query,
                    "page_title": page_content["title"],
                    "page_url": page_content["url"],
                    "search_results": search_results,
                    "page_id": page_content.get("pageid"),
                    "thumbnail": page_content.get("thumbnail"),
                    "coordinates": page_content.get("coordinates")
                },
                source=self.name,
                confidence=confidence,
                processing_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                metadata={"query": query, "error": str(e)},
                source=self.name,
                confidence=0.0,
                processing_time=0.0,
                error_message=f"Wikipedia search error: {str(e)}"
            )
    
    def _search_pages(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for Wikipedia pages matching the query."""
        try:
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "srinfo": "totalhits",
                "srprop": "size|snippet|titlesnippet|timestamp"
            }
            
            response = requests.get(
                self.search_url,
                params=params,
                timeout=self.config.timeout,
                headers={"User-Agent": "Enhanced-Wikipedia-Agent/1.0"}
            )
            response.raise_for_status()
            
            data = response.json()
            
            if "query" in data and "search" in data["query"]:
                return data["query"]["search"]
            
            return []
            
        except Exception as e:
            log_error(f"Wikipedia search failed: {e}")
            return []
    
    def _get_page_content(self, title: str) -> Optional[Dict[str, Any]]:
        """Get detailed content for a Wikipedia page."""
        try:
            # URL encode the title
            encoded_title = quote(title.replace(" ", "_"))
            
            # Get page summary
            summary_url = f"{self.base_url}/page/summary/{encoded_title}"
            response = requests.get(
                summary_url,
                timeout=self.config.timeout,
                headers={"User-Agent": "Enhanced-Wikipedia-Agent/1.0"}
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract relevant information
            content = {
                "title": data.get("title", title),
                "extract": data.get("extract", ""),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "pageid": data.get("pageid"),
                "thumbnail": data.get("thumbnail"),
                "coordinates": data.get("coordinates")
            }
            
            # If extract is too short, try to get more content
            if len(content["extract"]) < 100:
                extended_content = self._get_extended_content(title)
                if extended_content:
                    content["extract"] = extended_content
            
            return content
            
        except Exception as e:
            log_error(f"Failed to get Wikipedia page content for '{title}': {e}")
            return None
    
    def _get_extended_content(self, title: str) -> Optional[str]:
        """Get extended content using the Wikipedia API."""
        try:
            params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "exsectionformat": "plain"
            }
            
            response = requests.get(
                self.search_url,
                params=params,
                timeout=self.config.timeout,
                headers={"User-Agent": "Enhanced-Wikipedia-Agent/1.0"}
            )
            response.raise_for_status()
            
            data = response.json()
            
            if "query" in data and "pages" in data["query"]:
                pages = data["query"]["pages"]
                for page_id, page_data in pages.items():
                    if page_id != "-1":  # Page exists
                        return page_data.get("extract", "")
            
            return None
            
        except Exception as e:
            log_error(f"Failed to get extended content for '{title}': {e}")
            return None
    
    def _calculate_confidence(self, query: str, page_content: Dict[str, Any], search_result: Dict[str, Any]) -> float:
        """Calculate confidence score for the Wikipedia result."""
        confidence = 0.0
        
        # Base confidence from search ranking
        confidence += 0.3  # Base score for finding a result
        
        # Title match
        title = page_content.get("title", "").lower()
        query_lower = query.lower()
        
        if query_lower in title:
            confidence += 0.3
        elif any(word in title for word in query_lower.split()):
            confidence += 0.2
        
        # Content relevance
        extract = page_content.get("extract", "").lower()
        query_words = set(query_lower.split())
        extract_words = set(extract.split())
        
        if query_words & extract_words:
            word_overlap = len(query_words & extract_words) / len(query_words)
            confidence += 0.2 * word_overlap
        
        # Search result quality
        if search_result.get("size", 0) > 1000:  # Substantial article
            confidence += 0.1
        
        # Snippet match
        snippet = search_result.get("snippet", "").lower()
        if query_lower in snippet:
            confidence += 0.1
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def can_handle_query(self, query: str) -> bool:
        """Check if Wikipedia can handle this query."""
        if not super().can_handle_query(query):
            return False
        
        # Wikipedia is good for factual, encyclopedic queries
        factual_indicators = [
            "what is", "who is", "where is", "when was", "when did",
            "tell me about", "explain", "definition", "biography",
            "history of", "facts about"
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in factual_indicators)
    
    def estimate_confidence(self, query: str) -> float:
        """Estimate confidence for handling this query."""
        if not self.can_handle_query(query):
            return 0.0
        
        # Higher confidence for encyclopedic queries
        encyclopedia_terms = [
            "biography", "history", "definition", "facts", "information",
            "what is", "who is", "tell me about", "explain"
        ]
        
        query_lower = query.lower()
        confidence = 0.5  # Base confidence
        
        for term in encyclopedia_terms:
            if term in query_lower:
                confidence += 0.1
        
        return min(confidence, 0.9)  # Cap at 0.9
    
    def get_random_article(self) -> ToolResult:
        """Get a random Wikipedia article."""
        try:
            params = {
                "action": "query",
                "format": "json",
                "list": "random",
                "rnnamespace": 0,
                "rnlimit": 1
            }
            
            response = requests.get(
                self.search_url,
                params=params,
                timeout=self.config.timeout,
                headers={"User-Agent": "Enhanced-Wikipedia-Agent/1.0"}
            )
            response.raise_for_status()
            
            data = response.json()
            
            if "query" in data and "random" in data["query"] and data["query"]["random"]:
                title = data["query"]["random"][0]["title"]
                return self.search(title)
            
            return self._create_error_result("Could not get random article", "random")
            
        except Exception as e:
            return self._create_error_result(f"Random article error: {str(e)}", "random")
    
    def get_page_categories(self, title: str) -> List[str]:
        """Get categories for a Wikipedia page."""
        try:
            params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "categories",
                "cllimit": 50
            }
            
            response = requests.get(
                self.search_url,
                params=params,
                timeout=self.config.timeout,
                headers={"User-Agent": "Enhanced-Wikipedia-Agent/1.0"}
            )
            response.raise_for_status()
            
            data = response.json()
            
            categories = []
            if "query" in data and "pages" in data["query"]:
                for page_id, page_data in data["query"]["pages"].items():
                    if "categories" in page_data:
                        categories = [cat["title"].replace("Category:", "") 
                                    for cat in page_data["categories"]]
            
            return categories
            
        except Exception as e:
            log_error(f"Failed to get categories for '{title}': {e}")
            return []