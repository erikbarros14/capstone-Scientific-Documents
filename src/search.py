import numpy as np
from typing import List, Dict

class SemanticSearcher:
    def __init__(self, collection):
        self.collection = collection
    
    def search(self, query: str, top_k: int = 5):
        """Realiza busca semântica no banco de dados vetorial"""
        # Gera embedding da query
        query_embedding = ingestor.model.encode(query).tolist()
        
        # Busca no ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Formata os resultados
        formatted_results = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            formatted_results.append({
                'content': doc,
                'source': meta['source']
            })
        
        return formatted_results