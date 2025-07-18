"""
Configuration loader for the Enhanced Wikipedia Agent.
Handles loading and validation of configuration from YAML files and environment variables.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

class ConfigLoader:
    """Handles loading and managing configuration settings."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to the main configuration file
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load_environment()
        self._load_config()
        
    def _load_environment(self):
        """Load environment variables from .env file."""
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
            
    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
            
        # Override with environment variables
        self._override_with_env()
        
    def _override_with_env(self):
        """Override configuration values with environment variables."""
        env_mappings = {
            'WOLFRAM_ALPHA_API_KEY': ['sources', 'wolfram_alpha', 'api_key'],
            'HUGGINGFACE_API_KEY': ['llm', 'api_key'],
            'LOG_LEVEL': ['logging', 'level'],
            'LOG_DIR': ['logging', 'directory'],
            'CACHE_DIR': ['cache', 'directory'],
            'CACHE_TYPE': ['cache', 'type'],
            'CACHE_MAX_SIZE': ['cache', 'max_size'],
            'DEBUG_MODE': ['development', 'debug_mode'],
            'VERBOSE_LOGGING': ['development', 'verbose_logging'],
            'MODEL_CACHE_DIR': ['llm', 'cache_dir'],
        }
        
        for env_key, config_path in env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value is not None:
                self._set_nested_value(config_path, self._convert_env_value(env_value))
                
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass
            
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
            
        # Return as string
        return value
        
    def _set_nested_value(self, path: list, value: Any):
        """Set a nested configuration value."""
        current = self.config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'llm.temperature')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        current = self.config
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
            
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Configuration section dictionary
        """
        return self.config.get(section, {})
        
    def set(self, key: str, value: Any):
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'llm.temperature')
            value: Value to set
        """
        keys = key.split('.')
        current = self.config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
            
        current[keys[-1]] = value
        
    def validate(self) -> bool:
        """
        Validate the loaded configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        required_sections = ['llm', 'agent', 'sources', 'logging']
        
        for section in required_sections:
            if section not in self.config:
                print(f"Missing required configuration section: {section}")
                return False
                
        # Validate specific required values
        if not self.get('llm.model_name'):
            print("Missing required configuration: llm.model_name")
            return False
            
        return True
        
    def create_directories(self):
        """Create necessary directories based on configuration."""
        directories = [
            self.get('logging.directory', 'logs'),
            self.get('cache.directory', 'cache'),
            self.get('llm.cache_dir', 'models'),
            'data'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    def save_config(self, path: Optional[str] = None):
        """
        Save current configuration to file.
        
        Args:
            path: Path to save configuration (default: original path)
        """
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
            
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"ConfigLoader({self.config_path})"
        
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"ConfigLoader(path={self.config_path}, sections={list(self.config.keys())})"


# Global configuration instance
config = ConfigLoader()

# Convenience functions for common operations
def get_config(key: str, default: Any = None) -> Any:
    """Get a configuration value."""
    return config.get(key, default)

def get_section(section: str) -> Dict[str, Any]:
    """Get a configuration section."""
    return config.get_section(section)

def set_config(key: str, value: Any):
    """Set a configuration value."""
    config.set(key, value)

def validate_config() -> bool:
    """Validate the current configuration."""
    return config.validate()

def create_directories():
    """Create necessary directories."""
    config.create_directories()