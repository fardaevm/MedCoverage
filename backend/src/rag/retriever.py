from src.rag_interface.base_retriever import BaseRetriever
from src.rag_interface.base_datastore import BaseDatastore
import os
import cohere


class Retriever(BaseRetriever):
    def __init__(self, datastore = BaseDatastore):
        self.datastore = datastore
        api_key = os.getenv("COHERE_API_KEY")
        self.co = None
        if api_key:
            self.co = cohere.ClientV2(api_key=api_key)

    def search(self, query: str, top_k: int = 10) -> list[str]:
        search_results = self.datastore.search(query, top_k=top_k * 3)
        reranked_results = self._rerank(query, search_results, top_k=top_k)
        return reranked_results
    
    def _rerank(self, query: str, search_results: list[str], top_k: int=10) -> list[str]:
        if self.co is None:
            return search_results[:top_k]
        
        response = self.co.rerank(
            model="rerank-v3.5",
            query = query,
            documents = search_results,
            top_n = top_k,
        )

        indices = [r.index for r in response.results]
        return [search_results[i] for i in indices]