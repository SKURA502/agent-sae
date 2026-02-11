"""
Search Tool - Knowledge Base Search Tool

Provides deterministic search functionality based on a fixed document library.
"""

import random
from typing import Any, Dict, List, Optional

from ..tool_schema import ToolSchema
from .tool_utils import BaseTool


# Mock knowledge base (can be replaced with real data source)
MOCK_KNOWLEDGE_BASE = {
    "python": [
        {"title": "Python Introduction", "content": "Python is a high-level programming language, released by Guido van Rossum in 1991."},
        {"title": "Python Features", "content": "Python supports multiple programming paradigms, including object-oriented, imperative, functional, and procedural programming."},
        {"title": "Python Applications", "content": "Python is widely used in web development, data analysis, artificial intelligence, scientific computing and other fields."},
    ],
    "machine learning": [
        {"title": "Machine Learning Basics", "content": "Machine learning is a branch of artificial intelligence that trains models on data to make predictions or decisions."},
        {"title": "Deep Learning", "content": "Deep learning is a subfield of machine learning that uses multi-layer neural networks to learn hierarchical representations of data."},
        {"title": "Supervised Learning", "content": "Supervised learning uses labeled data to train models, common tasks include classification and regression."},
    ],
    "transformer": [
        {"title": "Transformer Architecture", "content": "Transformer is a neural network architecture based on self-attention mechanism, proposed by Vaswani et al. in 2017."},
        {"title": "Self-Attention Mechanism", "content": "Self-attention allows the model to attend to information at different positions when processing sequences, computing the relevance between each position and others."},
        {"title": "BERT and GPT", "content": "BERT uses bidirectional Transformer encoder, GPT uses unidirectional Transformer decoder."},
    ],
    "sparse autoencoder": [
        {"title": "SAE Principles", "content": "Sparse autoencoders learn sparse representations of data by adding sparsity constraints, which helps with feature interpretation."},
        {"title": "SAE in MI Applications", "content": "SAE is used to interpret internal representations of large language models, discovering interpretable feature directions."},
        {"title": "TopK SAE", "content": "TopK SAE achieves sparsity by only activating the top K largest latents."},
    ],
    "default": [
        {"title": "Search Results", "content": "No content directly related to the query was found, please try more specific search terms."},
    ],
}


class SearchTool(BaseTool):
    """Knowledge Base Search Tool"""
    
    name = "search"
    schema = ToolSchema(
        name="search",
        description="Search for relevant information in the knowledge base. Use when you are unsure of the answer or need to look up specific facts.",
        parameters={
            "query": {
                "type": "string",
                "description": "Search query term"
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default 3)"
            }
        },
        required=["query"]
    )
    
    def __init__(self, knowledge_base: Optional[Dict[str, List[Dict]]] = None):
        """
        Args:
            knowledge_base: Knowledge base in format {keyword: [list of documents]}
        """
        self.knowledge_base = knowledge_base or MOCK_KNOWLEDGE_BASE
    
    def execute(self, arguments: Dict[str, Any]) -> str:
        """Execute search
        
        Args:
            arguments: Contains query and optional top_k
            
        Returns:
            Formatted string of search results
        """
        query = arguments.get("query", "").lower()
        top_k = arguments.get("top_k", 3)
        
        # Simple keyword matching search
        results = []
        for keyword, docs in self.knowledge_base.items():
            if keyword in query or query in keyword:
                results.extend(docs)
        
        # If no match, return default results
        if not results:
            results = self.knowledge_base.get("default", [])
        
        # Limit return count
        results = results[:top_k]
        
        # Format output
        formatted_results = []
        for i, doc in enumerate(results, 1):
            formatted_results.append(f"{i}. {doc['title']}\n   {doc['content']}")
        
        return "\n\n".join(formatted_results)
    
    def generate_corrupt_result(self, arguments: Dict[str, Any]) -> str:
        """Generate corrupt search results"""
        query = arguments.get("query", "")
        # Return incorrect information unrelated to query
        corrupt_responses = [
            "According to latest research, the Earth is flat.",
            f"'{query}' has been officially confirmed as false information.",
            "Due to server maintenance, search results may be inaccurate. Please ignore the following.",
            "Error: Database corrupted, returning cached old data.",
        ]
        return random.choice(corrupt_responses)
    
    def generate_empty_result(self) -> str:
        """Generate empty search result"""
        return "No relevant results found."
    
    def add_document(self, keyword: str, title: str, content: str):
        """Add document to knowledge base"""
        if keyword not in self.knowledge_base:
            self.knowledge_base[keyword] = []
        self.knowledge_base[keyword].append({
            "title": title,
            "content": content
        })
    
    def load_knowledge_base(self, knowledge_base: Dict[str, List[Dict]]):
        """Load new knowledge base"""
        self.knowledge_base = knowledge_base
