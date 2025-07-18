"""
DuckDuckGo search tool for web search capabilities.
Provides safe, privacy-focused web search functionality.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from .base_tool import BaseTool, ToolResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Individual search result."""
    title: str
    url: str
    snippet: str
    source: str = "DuckDuckGo"


class DuckDuckGoTool(BaseTool):
    """DuckDuckGo search tool implementation."""
    
    name = "duckduckgo_search"
    description = "Search the web using DuckDuckGo. Useful for current events, recent information, and general web search."
    
    def __init__(self, max_results: int = 5, safe_search: str = "moderate"):
        super().__init__()
        self.max_results = max_results
        self.safe_search = safe_search
        self._session = None
    
    async def _get_session(self):
        """Get or create HTTP session."""
        if self._session is None:
            import httpx
            self._session = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
            )
        return self._session
    
    async def execute(self, query: str) -> str:
        """Execute DuckDuckGo search."""
        try:
            # Import duckduckgo_search if available
            try:
                from duckduckgo_search import DDGS
                return await self._search_with_ddgs(query)
            except ImportError:
                # Fallback to direct API approach
                return await self._search_direct_api(query)
                
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return f"Error performing search: {str(e)}"
    
    async def _search_with_ddgs(self, query: str) -> str:
        """Search using duckduckgo_search library."""
        try:
            from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                search_results = ddgs.text(
                    query,
                    max_results=self.max_results,
                    safesearch=self.safe_search
                )
                
                for result in search_results:
                    results.append(SearchResult(
                        title=result.get("title", ""),
                        url=result.get("href", ""),
                        snippet=result.get("body", ""),
                        source="DuckDuckGo"
                    ))
            
            return self._format_results(results, query)
            
        except Exception as e:
            logger.error(f"DDGS search error: {e}")
            raise
    
    async def _search_direct_api(self, query: str) -> str:
        """Search using direct API calls (fallback method)."""
        try:
            session = await self._get_session()
            
            # DuckDuckGo instant answer API
            params = {
                "q": query,
                "format": "json",
                "no_redirect": "1",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            response = await session.get(
                "https://api.duckduckgo.com/",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Process results
            results = []
            
            # Check for instant answer
            if data.get("Abstract"):
                results.append(SearchResult(
                    title=data.get("Heading", "Instant Answer"),
                    url=data.get("AbstractURL", ""),
                    snippet=data.get("Abstract", ""),
                    source="DuckDuckGo Instant Answer"
                ))
            
            # Check for related topics
            for topic in data.get("RelatedTopics", [])[:3]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append(SearchResult(
                        title=topic.get("Text", "").split(" - ")[0],
                        url=topic.get("FirstURL", ""),
                        snippet=topic.get("Text", ""),
                        source="DuckDuckGo Related"
                    ))
            
            # If no results from instant answer, try web search
            if not results:
                return await self._fallback_search(query)
            
            return self._format_results(results, query)
            
        except Exception as e:
            logger.error(f"Direct API search error: {e}")
            return await self._fallback_search(query)
    
    async def _fallback_search(self, query: str) -> str:
        """Fallback search method."""
        try:
            # Simple web scraping approach (use with caution)
            session = await self._get_session()
            
            # Search DuckDuckGo HTML
            params = {
                "q": query,
                "ia": "web"
            }
            
            response = await session.get(
                "https://html.duckduckgo.com/html/",
                params=params
            )
            response.raise_for_status()
            
            # Parse HTML for basic results
            html_content = response.text
            results = self._parse_html_results(html_content)
            
            if results:
                return self._format_results(results, query)
            else:
                return f"No search results found for: {query}"
                
        except Exception as e:
            logger.error(f"Fallback search error: {e}")
            return f"Search unavailable. Error: {str(e)}"
    
    def _parse_html_results(self, html: str) -> List[SearchResult]:
        """Parse HTML search results (basic implementation)."""
        results = []
        
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find result divs
            result_divs = soup.find_all('div', class_='result')
            
            for div in result_divs[:self.max_results]:
                title_elem = div.find('a', class_='result__a')
                snippet_elem = div.find('a', class_='result__snippet')
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    results.append(SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source="DuckDuckGo Web"
                    ))
            
        except ImportError:
            logger.warning("BeautifulSoup not available for HTML parsing")
        except Exception as e:
            logger.error(f"HTML parsing error: {e}")
        
        return results
    
    def _format_results(self, results: List[SearchResult], query: str) -> str:
        """Format search results into a readable string."""
        if not results:
            return f"No search results found for: {query}"
        
        formatted_results = []
        formatted_results.append(f"DuckDuckGo Search Results for: {query}")
        formatted_results.append("=" * 50)
        
        for i, result in enumerate(results, 1):
            formatted_results.append(f"\n{i}. {result.title}")
            formatted_results.append(f"   URL: {result.url}")
            formatted_results.append(f"   {result.snippet}")
            formatted_results.append(f"   Source: {result.source}")
        
        formatted_results.append(f"\nSearch completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(formatted_results)
    
    async def search_news(self, query: str, max_results: int = 3) -> str:
        """Search for news articles specifically."""
        try:
            from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                news_results = ddgs.news(
                    query,
                    max_results=max_results,
                    safesearch=self.safe_search
                )
                
                for result in news_results:
                    results.append(SearchResult(
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        snippet=result.get("body", ""),
                        source=f"DuckDuckGo News - {result.get('source', '')}"
                    ))
            
            return self._format_results(results, f"News: {query}")
            
        except ImportError:
            # Fallback to regular search with news keywords
            return await self.execute(f"news {query}")
        except Exception as e:
            logger.error(f"News search error: {e}")
            return f"Error searching news: {str(e)}"
    
    async def search_images(self, query: str, max_results: int = 3) -> str:
        """Search for images."""
        try:
            from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                image_results = ddgs.images(
                    query,
                    max_results=max_results,
                    safesearch=self.safe_search
                )
                
                for result in image_results:
                    results.append(SearchResult(
                        title=result.get("title", ""),
                        url=result.get("image", ""),
                        snippet=f"Size: {result.get('width', 'Unknown')}x{result.get('height', 'Unknown')}",
                        source=f"DuckDuckGo Images - {result.get('source', '')}"
                    ))
            
            return self._format_results(results, f"Images: {query}")
            
        except ImportError:
            return "Image search requires duckduckgo_search library"
        except Exception as e:
            logger.error(f"Image search error: {e}")
            return f"Error searching images: {str(e)}"
    
    def get_search_suggestions(self, query: str) -> List[str]:
        """Get search suggestions for a query."""
        try:
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                suggestions = ddgs.suggestions(query)
                return [suggestion.get("phrase", "") for suggestion in suggestions]
                
        except ImportError:
            return []
        except Exception as e:
            logger.error(f"Suggestions error: {e}")
            return []
    
    async def validate_query(self, query: str) -> bool:
        """Validate if query is suitable for web search."""
        if not query or len(query.strip()) < 2:
            return False
        
        # Check for potentially harmful queries
        harmful_keywords = [
            "illegal", "hack", "exploit", "malware", "virus",
            "piracy", "crack", "torrent", "drugs", "weapons"
        ]
        
        query_lower = query.lower()
        for keyword in harmful_keywords:
            if keyword in query_lower:
                logger.warning(f"Potentially harmful query detected: {query}")
                return False
        
        return True
    
    async def close(self):
        """Close HTTP session."""
        if self._session:
            await self._session.aclose()
            self._session = None
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get tool information."""
        return {
            "name": self.name,
            "description": self.description,
            "max_results": self.max_results,
            "safe_search": self.safe_search,
            "capabilities": [
                "web_search",
                "news_search", 
                "image_search",
                "search_suggestions"
            ]
        }