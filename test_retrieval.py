import sys
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).parent))

from src.rag.vector_engine import VectorRAGEngine

def main():
    print("Initializing VectorRAG Engine...")
    engine = VectorRAGEngine()
    
    query = "What did Mamata say about Modi?"
    print(f"\nQuerying: {query}")
    
    # Test Retrieval
    results = engine.retrieve(query)
    
    if not results:
        print("No results found. (Ingestion might still be running or empty DB)")
        return

    print(f"Found {len(results)} results after reranking.")
    
    for i, res in enumerate(results[:3]):
        print(f"\nResult {i+1}:")
        print(f"Score: {res['score']}")
        print(f"Text: {res['document'][:100]}...")
        print(f"Metadata: {res['metadata']}")

    # Test Context Expansion
    print("\nTesting Context Expansion...")
    context = engine.expand_context(results)
    print(f"Expanded Context Length: {len(context)} chars")
    print(f"Context Preview:\n{context[:500]}...")

if __name__ == "__main__":
    main()
