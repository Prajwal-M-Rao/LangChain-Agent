"""
Base Tool - Abstract base class for all knowledge source tools
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time
import asyncio
from enum import Enum

from ..utils.logger import log_info, log_error, log_warning
from ..utils.metrics import metrics


class ToolType(Enum):
    """Tool types for classification."""
    ENCYCLOPEDIA = "encyclopedia"
    SEARCH = "search"
    COMPUTATION = "computation"
    GENERAL = "general"


class ToolStatus(Enum):
    """Tool status indicators."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"


@dataclass
class ToolResult:
    """Standard result format for all tools."""
    success: bool
    content: str
    metadata: Dict[str, Any]
    source: str
    confidence: float
    processing_time: float
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Validate result data."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class ToolConfig:
    """Configuration for tools."""
    enabled: bool = True
    timeout: int = 30
    max_retries: int = 3
    rate_limit_per_minute: int = 60
    cache_ttl: int = 3600  # 1 hour
    api_key: Optional[str] = None
    custom_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}


class BaseTool(ABC):
    """Abstract base class for all knowledge source tools."""
    
    def __init__(self, config: ToolConfig):
        self.config = config
        self.tool_type = ToolType.GENERAL
        self.status = ToolStatus.AVAILABLE
        self.last_used = 0
        self.usage_count = 0
        self.error_count = 0
        self.rate_limit_tracker = []
        
        # Initialize tool-specific setup
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """Initialize tool-specific components."""
        pass
    
    @abstractmethod
    def _search_impl(self, query: str, **kwargs) -> ToolResult:
        """Implementation-specific search logic."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass
    
    def is_available(self) -> bool:
        """Check if tool is currently available."""
        return (
            self.config.enabled and 
            self.status == ToolStatus.AVAILABLE and
            not self._is_rate_limited()
        )
    
    def _is_rate_limited(self) -> bool:
        """Check if tool is rate limited."""
        now = time.time()
        # Remove old entries
        self.rate_limit_tracker = [
            timestamp for timestamp in self.rate_limit_tracker 
            if now - timestamp < 60  # Within last minute
        ]
        
        return len(self.rate_limit_tracker) >= self.config.rate_limit_per_minute
    
    def _update_rate_limit(self):
        """Update rate limit tracking."""
        self.rate_limit_tracker.append(time.time())
    
    def _create_error_result(self, error_msg: str, query: str) -> ToolResult:
        """Create standardized error result."""
        return ToolResult(
            success=False,
            content="",
            metadata={"query": query, "tool": self.name},
            source=self.name,
            confidence=0.0,
            processing_time=0.0,
            error_message=error_msg
        )
    
    def search(self, query: str, **kwargs) -> ToolResult:
        """
        Main search method with error handling and metrics.
        
        Args:
            query: Search query string
            **kwargs: Additional search parameters
            
        Returns:
            ToolResult: Standardized result object
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if not query or not query.strip():
                return self._create_error_result("Empty query provided", query)
            
            # Check availability
            if not self.is_available():
                if not self.config.enabled:
                    error_msg = f"Tool {self.name} is disabled"
                elif self.status != ToolStatus.AVAILABLE:
                    error_msg = f"Tool {self.name} is {self.status.value}"
                else:
                    error_msg = f"Tool {self.name} is rate limited"
                
                return self._create_error_result(error_msg, query)
            
            # Update tracking
            self._update_rate_limit()
            self.usage_count += 1
            self.last_used = time.time()
            
            # Log search attempt
            log_info(f"Searching with {self.name}: {query}")
            
            # Perform search with retries
            result = self._search_with_retries(query, **kwargs)
            
            # Update metrics
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            if result.success:
                metrics.increment_counter(f"tool_{self.name}_success")
                metrics.record_timing(f"tool_{self.name}_response_time", processing_time)
                log_info(f"Tool {self.name} search successful: {len(result.content)} chars")
            else:
                metrics.increment_counter(f"tool_{self.name}_failure")
                self.error_count += 1
                log_error(f"Tool {self.name} search failed: {result.error_message}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Unexpected error in {self.name}: {str(e)}"
            
            log_error(error_msg)
            metrics.increment_counter(f"tool_{self.name}_error")
            self.error_count += 1
            
            result = self._create_error_result(error_msg, query)
            result.processing_time = processing_time
            return result
    
    def _search_with_retries(self, query: str, **kwargs) -> ToolResult:
        """Search with retry logic."""
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    # Wait before retry
                    time.sleep(min(2 ** attempt, 10))  # Exponential backoff
                    log_info(f"Retrying {self.name} search (attempt {attempt + 1})")
                
                result = self._search_impl(query, **kwargs)
                
                if result.success:
                    return result
                else:
                    last_error = result.error_message
                    
            except Exception as e:
                last_error = str(e)
                log_warning(f"Attempt {attempt + 1} failed for {self.name}: {e}")
        
        # All retries failed
        return self._create_error_result(
            f"All {self.config.max_retries + 1} attempts failed. Last error: {last_error}",
            query
        )
    
    async def search_async(self, query: str, **kwargs) -> ToolResult:
        """Asynchronous search method."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search, query, **kwargs)
    
    def get_status(self) -> Dict[str, Any]:
        """Get tool status information."""
        return {
            "name": self.name,
            "type": self.tool_type.value,
            "status": self.status.value,
            "enabled": self.config.enabled,
            "is_available": self.is_available(),
            "usage_count": self.usage_count,
            "error_count": self.error_count,
            "error_rate": (self.error_count / max(self.usage_count, 1)) * 100,
            "last_used": self.last_used,
            "rate_limited": self._is_rate_limited(),
            "rate_limit_remaining": max(0, self.config.rate_limit_per_minute - len(self.rate_limit_tracker))
        }
    
    def reset_stats(self):
        """Reset usage statistics."""
        self.usage_count = 0
        self.error_count = 0
        self.rate_limit_tracker = []
        self.last_used = 0
    
    def can_handle_query(self, query: str) -> bool:
        """
        Check if this tool can handle the given query.
        Override in subclasses for more specific logic.
        """
        return self.is_available()
    
    def estimate_confidence(self, query: str) -> float:
        """
        Estimate confidence for handling this query.
        Override in subclasses for more specific logic.
        """
        if not self.can_handle_query(query):
            return 0.0
        return 0.5  # Default neutral confidence
    
    def __str__(self) -> str:
        return f"{self.name} ({self.tool_type.value})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', status='{self.status.value}')>"