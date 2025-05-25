# semantic_search.py

import sys
from sentence_transformers import SentenceTransformer, util

# Sample corpus of documents or FAQs
documents = [
    "How do I reset my password?",
    "Where can I find the user manual?",
    "How to change my email address?",
    "Troubleshooting login issues",
    "Best practices for data privacy settings",
    "How can I contact support?",
]

# Load your fine-tuned model
model = SentenceTransformer('fine_tuned_model')  # <-- fine-tuned model

# Encode the documents
doc_embeddings = model.encode(documents, convert_to_tensor=True)

def semantic_search(query: str, top_k: int = 5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, doc_embeddings, top_k=top_k)[0]
    
    print(f"\nTop {top_k} results for your query: \"{query}\"")
    for i, hit in enumerate(hits):
        doc = documents[hit['corpus_id']]
        print(f"{i+1}. {doc} (score: {hit['score']:.4f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python semantic_search.py \"your search query here\"")
    else:
        user_query = " ".join(sys.argv[1:])
        semantic_search(user_query)
