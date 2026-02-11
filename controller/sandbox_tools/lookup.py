"""
Lookup Tool - Key-Value Query Tool

Provides deterministic key-value lookup functionality based on a fixed database.
"""

import random
from typing import Any, Dict, Optional

from ..tool_schema import ToolSchema
from .tool_utils import BaseTool


# Mock database
MOCK_DATABASES = {
    "entities": {
        "Albert Einstein": {
            "birth_year": 1879,
            "death_year": 1955,
            "nationality": "German-American",
            "field": "Theoretical Physics",
            "known_for": "Theory of Relativity, E=mc²"
        },
        "Marie Curie": {
            "birth_year": 1867,
            "death_year": 1934,
            "nationality": "Polish-French",
            "field": "Physics, Chemistry",
            "known_for": "Radioactivity, Polonium, Radium"
        },
        "Alan Turing": {
            "birth_year": 1912,
            "death_year": 1954,
            "nationality": "British",
            "field": "Computer Science, Mathematics",
            "known_for": "Turing Machine, Enigma decryption"
        },
        "Geoffrey Hinton": {
            "birth_year": 1947,
            "nationality": "British-Canadian",
            "field": "Computer Science, Cognitive Psychology",
            "known_for": "Deep Learning, Backpropagation, Boltzmann Machines"
        },
    },
    "facts": {
        "speed_of_light": "299,792,458 m/s",
        "earth_radius": "6,371 km",
        "pi_value": "3.14159265358979",
        "avogadro_number": "6.022 × 10²³",
        "planck_constant": "6.626 × 10⁻³⁴ J·s",
    },
    "definitions": {
        "transformer": "A neural network architecture based on self-attention mechanism",
        "sparse autoencoder": "An autoencoder with sparsity constraints to learn interpretable representations",
        "mechanistic interpretability": "The study of understanding neural network behavior through analyzing internal mechanisms",
        "agent": "A system that perceives its environment and takes actions to achieve goals",
    },
}


class LookupTool(BaseTool):
    """Key-Value Query Tool"""
    
    name = "lookup"
    schema = ToolSchema(
        name="lookup",
        description="Look up specific entries in a key-value database. Used to query known entity information.",
        parameters={
            "key": {
                "type": "string",
                "description": "The key name to look up"
            },
            "database": {
                "type": "string",
                "description": "Database name (optional, searches all databases by default)"
            }
        },
        required=["key"]
    )
    
    def __init__(self, databases: Optional[Dict[str, Dict]] = None):
        """
        Args:
            databases: Database dictionary in format {database_name: {key: value}}
        """
        self.databases = databases or MOCK_DATABASES
    
    def execute(self, arguments: Dict[str, Any]) -> str:
        """Execute query
        
        Args:
            arguments: Contains key and optional database
            
        Returns:
            Formatted string of query results
        """
        key = arguments.get("key", "")
        database_name = arguments.get("database")
        
        if not key:
            raise ValueError("Query key cannot be empty")
        
        # Search in specified database or all databases
        if database_name:
            if database_name not in self.databases:
                return f"Database '{database_name}' does not exist"
            result = self._search_in_database(key, self.databases[database_name])
            if result:
                return self._format_result(key, result)
        else:
            # Search all databases
            for db_name, db in self.databases.items():
                result = self._search_in_database(key, db)
                if result:
                    return self._format_result(key, result, db_name)
        
        return f"No information found for key '{key}'"
    
    def _search_in_database(self, key: str, database: Dict) -> Optional[Any]:
        """Search in a single database"""
        # Exact match
        if key in database:
            return database[key]
        
        # Case-insensitive match
        key_lower = key.lower()
        for k, v in database.items():
            if k.lower() == key_lower:
                return v
        
        # Partial match
        for k, v in database.items():
            if key_lower in k.lower() or k.lower() in key_lower:
                return v
        
        return None
    
    def _format_result(
        self, 
        key: str, 
        result: Any, 
        database_name: Optional[str] = None
    ) -> str:
        """Format query results"""
        header = f"Query: {key}"
        if database_name:
            header += f" (source: {database_name})"
        
        if isinstance(result, dict):
            items = [f"  - {k}: {v}" for k, v in result.items()]
            content = "\n".join(items)
        else:
            content = f"  {result}"
        
        return f"{header}\n{content}"
    
    def generate_corrupt_result(self, arguments: Dict[str, Any]) -> str:
        """Generate corrupt query results"""
        key = arguments.get("key", "")
        corrupt_responses = [
            f"Information for '{key}' is outdated and not recommended for reference.",
            f"Warning: Data for '{key}' is disputed.",
            f"'{key}' related information: [data deleted]",
            f"According to records, '{key}' does not exist. Please check spelling.",
        ]
        return random.choice(corrupt_responses)
    
    def generate_empty_result(self) -> str:
        """Generate empty result"""
        return "No query results"
    
    def add_entry(self, database: str, key: str, value: Any):
        """Add entry"""
        if database not in self.databases:
            self.databases[database] = {}
        self.databases[database][key] = value
    
    def get_available_databases(self) -> list:
        """Get list of available databases"""
        return list(self.databases.keys())
