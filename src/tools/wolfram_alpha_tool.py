"""
Wolfram Alpha tool for computational queries and mathematical operations.
Provides access to Wolfram Alpha's computational knowledge engine.
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import re

from .base_tool import BaseTool, ToolResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WolframResult:
    """Wolfram Alpha result structure."""
    query: str
    primary_result: str
    subpods: List[Dict[str, Any]]
    success: bool
    error: Optional[str] = None
    assumptions: Optional[List[str]] = None
    warnings: Optional[List[str]] = None


class WolframAlphaTool(BaseTool):
    """Wolfram Alpha computational tool."""
    
    name = "wolfram_alpha"
    description = "Query Wolfram Alpha for mathematical calculations, scientific data, conversions, and factual information."
    
    def __init__(self, app_id: Optional[str] = None):
        super().__init__()
        self.app_id = app_id or os.getenv("WOLFRAM_ALPHA_APP_ID")
        self.base_url = "https://api.wolframalpha.com/v2/query"
        self._session = None
        
        if not self.app_id:
            logger.warning("Wolfram Alpha App ID not provided. Tool will use fallback methods.")
    
    async def _get_session(self):
        """Get or create HTTP session."""
        if self._session is None:
            import httpx
            self._session = httpx.AsyncClient(timeout=30.0)
        return self._session
    
    async def execute(self, query: str) -> str:
        """Execute Wolfram Alpha query."""
        try:
            if self.app_id:
                return await self._query_wolfram_api(query)
            else:
                return await self._fallback_computation(query)
                
        except Exception as e:
            logger.error(f"Wolfram Alpha error: {e}")
            return f"Error querying Wolfram Alpha: {str(e)}"
    
    async def _query_wolfram_api(self, query: str) -> str:
        """Query Wolfram Alpha API."""
        try:
            session = await self._get_session()
            
            params = {
                "appid": self.app_id,
                "input": query,
                "format": "plaintext",
                "output": "json",
                "includepodid": "Result,DecimalApproximation,Solution,Plot",
                "excludepodid": "Input"
            }
            
            response = await session.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get("queryresult", {}).get("success", False):
                error_msg = data.get("queryresult", {}).get("error", {}).get("msg", "Unknown error")
                return f"Wolfram Alpha could not process the query: {error_msg}"
            
            return self._format_wolfram_response(data, query)
            
        except Exception as e:
            logger.error(f"Wolfram API error: {e}")
            return await self._fallback_computation(query)
    
    def _format_wolfram_response(self, data: Dict, query: str) -> str:
        """Format Wolfram Alpha response."""
        try:
            query_result = data.get("queryresult", {})
            pods = query_result.get("pods", [])
            
            if not pods:
                return f"No results found for: {query}"
            
            formatted_results = []
            formatted_results.append(f"Wolfram Alpha Results for: {query}")
            formatted_results.append("=" * 50)
            
            # Extract main result
            result_pod = None
            for pod in pods:
                if pod.get("id") in ["Result", "DecimalApproximation", "Solution"]:
                    result_pod = pod
                    break
            
            if result_pod:
                subpods = result_pod.get("subpods", [])
                if subpods:
                    main_result = subpods[0].get("plaintext", "")
                    formatted_results.append(f"\nMain Result: {main_result}")
            
            # Extract other relevant information
            for pod in pods:
                title = pod.get("title", "")
                if title in ["Input", "Input interpretation"]:
                    continue
                
                subpods = pod.get("subpods", [])
                if subpods:
                    formatted_results.append(f"\n{title}:")
                    for subpod in subpods:
                        text = subpod.get("plaintext", "")
                        if text:
                            formatted_results.append(f"  {text}")
            
            # Add assumptions if present
            assumptions = query_result.get("assumptions", [])
            if assumptions:
                formatted_results.append("\nAssumptions:")
                for assumption in assumptions:
                    if isinstance(assumption, dict):
                        desc = assumption.get("description", "")
                        if desc:
                            formatted_results.append(f"  {desc}")
            
            # Add warnings if present
            warnings = query_result.get("warnings", [])
            if warnings:
                formatted_results.append("\nWarnings:")
                for warning in warnings:
                    if isinstance(warning, dict):
                        text = warning.get("text", "")
                        if text:
                            formatted_results.append(f"  {text}")
            
            formatted_results.append(f"\nQuery completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Error formatting Wolfram response: {e}")
            return f"Error processing Wolfram Alpha results: {str(e)}"
    
    async def _fallback_computation(self, query: str) -> str:
        """Fallback computation for when Wolfram Alpha API is unavailable."""
        try:
            # Try to handle basic mathematical operations
            if self._is_math_query(query):
                return await self._compute_math(query)
            
            # Try to handle unit conversions
            if self._is_conversion_query(query):
                return await self._compute_conversion(query)
            
            # Try to handle basic factual queries
            if self._is_factual_query(query):
                return await self._compute_factual(query)
            
            return f"Wolfram Alpha API not available. Query '{query}' requires computational engine access."
            
        except Exception as e:
            logger.error(f"Fallback computation error: {e}")
            return f"Computation failed: {str(e)}"
    
    def _is_math_query(self, query: str) -> bool:
        """Check if query is a mathematical expression."""
        math_patterns = [
            r'\d+[\+\-\*/\^]\d+',  # Basic arithmetic
            r'sqrt\(',  # Square root
            r'sin\(|cos\(|tan\(',  # Trigonometric functions
            r'log\(',  # Logarithm
            r'derivative|integral|limit',  # Calculus
            r'solve|equation',  # Equation solving
        ]
        
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in math_patterns)
    
    def _is_conversion_query(self, query: str) -> bool:
        """Check if query is a unit conversion."""
        conversion_patterns = [
            r'\d+\s*(km|miles|feet|meters|inches|cm|mm)',
            r'\d+\s*(kg|pounds|grams|ounces)',
            r'\d+\s*(celsius|fahrenheit|kelvin)',
            r'\d+\s*(dollars|euros|pounds|yen)',
            r'convert|to|in'
        ]
        
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in conversion_patterns)
    
    def _is_factual_query(self, query: str) -> bool:
        """Check if query is asking for factual information."""
        factual_patterns = [
            r'what is|what are',
            r'population of|capital of',
            r'distance between',
            r'when was|when did',
            r'how many|how much',
            r'molecular weight|atomic number',
            r'speed of light|gravitational constant'
        ]
        
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in factual_patterns)
    
    async def _compute_math(self, query: str) -> str:
        """Compute basic mathematical expressions."""
        try:
            # Use sympy for mathematical computations if available
            try:
                import sympy as sp
                
                # Clean the query for mathematical processing
                cleaned_query = self._clean_math_query(query)
                
                # Try to evaluate the expression
                result = sp.sympify(cleaned_query)
                evaluated = result.evalf()
                
                return f"Mathematical Result for: {query}\n" \
                       f"Expression: {result}\n" \
                       f"Numerical Value: {evaluated}"
                
            except ImportError:
                # Fallback to basic eval (use with caution)
                return await self._basic_math_eval(query)
                
        except Exception as e:
            logger.error(f"Math computation error: {e}")
            return f"Could not compute mathematical expression: {str(e)}"
    
    def _clean_math_query(self, query: str) -> str:
        """Clean mathematical query for processing."""
        # Replace common mathematical terms
        replacements = {
            'sqrt': 'sqrt',
            'sin': 'sin',
            'cos': 'cos',
            'tan': 'tan',
            'log': 'log',
            'ln': 'log',
            'pi': 'pi',
            'e': 'E',
            '^': '**',
            'x': '*'
        }
        
        cleaned = query.lower()
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        return cleaned
    
    async def _basic_math_eval(self, query: str) -> str:
        """Basic mathematical evaluation (limited and safe)."""
        try:
            # Extract numerical expression
            import re
            
            # Simple pattern matching for basic operations
            pattern = r'(\d+(?:\.\d+)?)\s*([\+\-\*/])\s*(\d+(?:\.\d+)?)'
            match = re.search(pattern, query)
            
            if match:
                num1, op, num2 = match.groups()
                num1, num2 = float(num1), float(num2)
                
                if op == '+':
                    result = num1 + num2
                elif op == '-':
                    result = num1 - num2
                elif op == '*':
                    result = num1 * num2
                elif op == '/':
                    if num2 != 0:
                        result = num1 / num2
                    else:
                        return "Error: Division by zero"
                else:
                    return f"Unsupported operation: {op}"
                
                return f"Basic Calculation: {num1} {op} {num2} = {result}"
            
            return f"Could not parse mathematical expression: {query}"
            
        except Exception as e:
            return f"Basic math evaluation error: {str(e)}"
    
    async def _compute_conversion(self, query: str) -> str:
        """Compute unit conversions."""
        try:
            # Try to use pint for unit conversions if available
            try:
                import pint
                
                ureg = pint.UnitRegistry()
                
                # Extract conversion pattern
                import re
                pattern = r'(\d+(?:\.\d+)?)\s*(\w+)\s*(?:to|in)\s*(\w+)'
                match = re.search(pattern, query.lower())
                
                if match:
                    value, from_unit, to_unit = match.groups()
                    value = float(value)
                    
                    # Create quantity and convert
                    quantity = value * ureg[from_unit]
                    converted = quantity.to(to_unit)
                    
                    return f"Unit Conversion: {value} {from_unit} = {converted}"
                
                return f"Could not parse conversion: {query}"
                
            except ImportError:
                return await self._basic_conversion(query)
                
        except Exception as e:
            logger.error(f"Conversion error: {e}")
            return f"Unit conversion failed: {str(e)}"
    
    async def _basic_conversion(self, query: str) -> str:
        """Basic unit conversions without external libraries."""
        # Simple conversion factors
        conversions = {
            'km_to_miles': 0.621371,
            'miles_to_km': 1.60934,
            'kg_to_pounds': 2.20462,
            'pounds_to_kg': 0.453592,
            'celsius_to_fahrenheit': lambda c: (c * 9/5) + 32,
            'fahrenheit_to_celsius': lambda f: (f - 32) * 5/9,
        }
        
        query_lower = query.lower()
        
        # Try to match basic conversions
        import re
        
        # Temperature conversions
        if 'celsius' in query_lower and 'fahrenheit' in query_lower:
            temp_match = re.search(r'(\d+(?:\.\d+)?)', query)
            if temp_match:
                temp = float(temp_match.group(1))
                if 'celsius' in query_lower.split('fahrenheit')[0]:
                    result = conversions['celsius_to_fahrenheit'](temp)
                    return f"Temperature Conversion: {temp}°C = {result:.2f}°F"
                else:
                    result = conversions['fahrenheit_to_celsius'](temp)
                    return f"Temperature Conversion: {temp}°F = {result:.2f}°C"
        
        return f"Basic conversion not available for: {query}"
    
    async def _compute_factual(self, query: str) -> str:
        """Compute factual information."""
        # Basic constants and facts
        constants = {
            'speed of light': '299,792,458 m/s',
            'gravitational constant': '6.67430 × 10^-11 m³/kg⋅s²',
            'planck constant': '6.62607015 × 10^-34 J⋅s',
            'avogadro number': '6.02214076 × 10^23 mol^-1',
            'pi': '3.14159265359',
            'e': '2.71828182846'
        }
        
        query_lower = query.lower()
        
        for key, value in constants.items():
            if key in query_lower:
                return f"Constant: {key} = {value}"
        
        return f"Factual information not available for: {query}"
    
    async def validate_query(self, query: str) -> bool:
        """Validate Wolfram Alpha query."""
        if not query or len(query.strip()) < 2:
            return False
        
        # Check for potentially problematic queries
        if len(query) > 500:
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
            "has_api_key": bool(self.app_id),
            "capabilities": [
                "mathematical_computation",
                "unit_conversion",
                "scientific_constants",
                "equation_solving",
                "statistical_analysis",
                "factual_queries"
            ]
        }