"""
Vector Databases Fundamentals - Beginner's Guide
Everything you need to know about vector databases for AI/ML interviews
Covers concepts, use cases, benefits, and simple implementations
"""

print("=" * 80)
print("VECTOR DATABASES FUNDAMENTALS - INTERVIEW PREPARATION")
print("=" * 80)

# ============================================================================
# SECTION 1: WHAT ARE VECTOR DATABASES?
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 1: UNDERSTANDING VECTOR DATABASES")
print("=" * 60)

# Question 1: What is a Vector Database?
print("\n1. What is a Vector Database and why do we need it?")
print("-" * 53)
print("""
DEFINITION:
A vector database is a specialized database designed to store, index, and query 
high-dimensional vectors (embeddings) efficiently.

WHAT ARE VECTORS/EMBEDDINGS?
• Numerical representations of data (text, images, audio)
• Convert unstructured data into structured numerical format
• Each dimension captures some semantic meaning
• Example: "cat" → [0.2, -0.1, 0.8, 0.3, ...] (512 dimensions)

WHY TRADITIONAL DATABASES CAN'T HANDLE THIS?
• Traditional databases work with exact matches
• Vectors need similarity search (finding "similar" vectors)
• High-dimensional data requires specialized indexing
• Millions of dimensions make queries extremely slow

REAL-WORLD ANALOGY:
Think of it like a library:
- Traditional DB: Find books with exact title "Python Programming"
- Vector DB: Find books "similar to" Python Programming (includes books about coding, programming languages, software development)

KEY BENEFITS:
✅ Semantic search (meaning-based, not keyword-based)
✅ Recommendation systems
✅ AI-powered applications (ChatGPT, image search)
✅ Fraud detection and anomaly detection
✅ Content similarity and clustering
""")

def demonstrate_vector_concept():
    """Simple demonstration of vector concepts"""
    print("Vector Concept Demo:")
    
    # Simulate simple word embeddings (in reality, these are much more complex)
    word_vectors = {
        "cat": [0.8, 0.2, 0.1, 0.9],      # pet, furry, small, animal
        "dog": [0.9, 0.3, 0.2, 0.8],      # pet, furry, medium, animal  
        "lion": [0.1, 0.2, 0.9, 0.9],     # wild, furry, big, animal
        "car": [0.1, 0.1, 0.8, 0.2],      # object, metal, big, vehicle
        "bicycle": [0.1, 0.1, 0.3, 0.2]   # object, metal, small, vehicle
    }
    
    print("\nWord Embeddings (simplified 4D vectors):")
    print("Dimensions: [pet_score, furry_score, size_score, animal_score]")
    for word, vector in word_vectors.items():
        print(f"{word:8}: {vector}")
    
    # Calculate similarity using cosine similarity
    import math
    
    def cosine_similarity(vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        return dot_product / (magnitude1 * magnitude2)
    
    print("\nSimilarity Scores (1.0 = identical, 0.0 = completely different):")
    target_word = "cat"
    target_vector = word_vectors[target_word]
    
    similarities = []
    for word, vector in word_vectors.items():
        if word != target_word:
            similarity = cosine_similarity(target_vector, vector)
            similarities.append((word, similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nWords most similar to '{target_word}':")
    for word, similarity in similarities:
        print(f"{word:8}: {similarity:.3f}")
    
    print("\nObservation: 'dog' is most similar to 'cat' (both are pets, furry, animals)")
    print("This is how vector databases enable semantic search!")

demonstrate_vector_concept()

# Question 2: Types of Vector Databases
print("\n2. What are the different types of Vector Databases?")
print("-" * 54)
print("""
VECTOR DATABASE CATEGORIES:

1. PURPOSE-BUILT VECTOR DATABASES:
   • Pinecone: Fully managed cloud service
   • Weaviate: Open-source with GraphQL API
   • Qdrant: Rust-based, high performance
   • Chroma: Simple, developer-friendly

2. TRADITIONAL DATABASES WITH VECTOR SUPPORT:
   • PostgreSQL + pgvector: SQL database with vector extension
   • Elasticsearch: Search engine with vector capabilities
   • Redis: In-memory database with vector search
   • MongoDB Atlas: Document database with vector search

3. AI/ML PLATFORMS WITH VECTOR CAPABILITIES:
   • Faiss (Facebook): Library for similarity search
   • Annoy (Spotify): Approximate nearest neighbors
   • ScaNN (Google): Efficient vector similarity search

CHOOSING THE RIGHT ONE:
Factor                  | Purpose-built | Traditional+Vector | Libraries
------------------------|---------------|-------------------|----------
Ease of use           | ⭐⭐⭐⭐⭐        | ⭐⭐⭐             | ⭐⭐
Performance            | ⭐⭐⭐⭐⭐        | ⭐⭐⭐⭐           | ⭐⭐⭐⭐⭐
Cost                   | ⭐⭐            | ⭐⭐⭐⭐           | ⭐⭐⭐⭐⭐
Existing infrastructure| ⭐⭐            | ⭐⭐⭐⭐⭐         | ⭐⭐⭐
SQL support            | ⭐             | ⭐⭐⭐⭐⭐         | ⭐
""")

# Question 3: Vector Database Use Cases
print("\n3. What are the main use cases for Vector Databases?")
print("-" * 56)
print("""
MAJOR USE CASES:

1. SEMANTIC SEARCH:
   • Search by meaning, not just keywords
   • "Find articles about machine learning" (matches AI, neural networks, etc.)
   • E-commerce: "Find similar products"

2. RECOMMENDATION SYSTEMS:
   • Netflix: "Movies similar to what you watched"
   • Spotify: "Songs you might like"
   • E-commerce: "Customers who bought this also bought..."

3. RAG (Retrieval-Augmented Generation):
   • ChatGPT-like systems with custom knowledge
   • Company chatbots with internal documents
   • Question-answering systems

4. IMAGE AND VIDEO SEARCH:
   • Google Image search: "Find similar images"
   • Pinterest: Visual search
   • Security: Face recognition, object detection

5. FRAUD DETECTION:
   • Banking: Detect unusual transaction patterns
   • Insurance: Identify suspicious claims
   • E-commerce: Detect fake reviews

6. PERSONALIZATION:
   • News feed algorithms
   • Content recommendation
   • Targeted advertising

INTERVIEW TIP:
Always give concrete examples when discussing use cases!
Example: "At Netflix, when you watch a sci-fi movie, the vector database 
finds movies with similar embeddings (genre, director, actors, themes) 
to recommend next."
""")

def demonstrate_use_case_example():
    """Demonstrate a simple recommendation system"""
    print("Simple Recommendation System Demo:")
    
    # Simulate user preferences and movie features
    # Dimensions: [action, comedy, drama, sci-fi, romance]
    movies = {
        "Avengers": [0.9, 0.1, 0.2, 0.7, 0.1],
        "The Hangover": [0.2, 0.9, 0.1, 0.0, 0.3],
        "Titanic": [0.3, 0.1, 0.9, 0.0, 0.9],
        "Star Wars": [0.8, 0.2, 0.3, 0.9, 0.2],
        "The Notebook": [0.1, 0.2, 0.8, 0.0, 0.9],
        "Deadpool": [0.7, 0.8, 0.2, 0.3, 0.2]
    }
    
    # User watched and liked "Avengers"
    user_preference = movies["Avengers"]
    
    print("\nMovie Features (Action, Comedy, Drama, Sci-fi, Romance):")
    for movie, features in movies.items():
        print(f"{movie:12}: {features}")
    
    print(f"\nUser liked: Avengers {user_preference}")
    print("Finding similar movies...")
    
    import math
    
    def cosine_similarity(vec1, vec2):
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        return dot_product / (magnitude1 * magnitude2)
    
    recommendations = []
    for movie, features in movies.items():
        if movie != "Avengers":  # Don't recommend the same movie
            similarity = cosine_similarity(user_preference, features)
            recommendations.append((movie, similarity))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    print("\nRecommendations (most similar first):")
    for movie, similarity in recommendations:
        print(f"{movie:12}: {similarity:.3f}")
    
    print("\nResult: Star Wars is most similar (both action + sci-fi)")
    print("This is how Netflix/Spotify recommendations work!")

demonstrate_use_case_example()

# ============================================================================
# SECTION 2: SIMILARITY SEARCH FUNDAMENTALS
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 2: SIMILARITY SEARCH FUNDAMENTALS")
print("=" * 60)

# Question 4: How does Similarity Search work?
print("\n4. How does Similarity Search work in Vector Databases?")
print("-" * 60)
print("""
SIMILARITY SEARCH PROCESS:

1. CONVERT DATA TO VECTORS:
   • Text → Embeddings (using models like BERT, OpenAI)
   • Images → Feature vectors (using CNN models)
   • Audio → Spectral features

2. CALCULATE DISTANCE/SIMILARITY:
   • Cosine Similarity: Measures angle between vectors
   • Euclidean Distance: Straight-line distance
   • Dot Product: Simple multiplication and sum

3. FIND NEAREST NEIGHBORS:
   • k-NN: Find k most similar vectors
   • Range Search: Find all vectors within distance threshold
   • Approximate Search: Trade accuracy for speed

DISTANCE METRICS EXPLAINED:

COSINE SIMILARITY:
• Best for: Text, recommendations
• Range: -1 to 1 (1 = identical, 0 = orthogonal, -1 = opposite)
• Formula: cos(θ) = (A·B) / (|A|×|B|)

EUCLIDEAN DISTANCE:
• Best for: Images, coordinates
• Range: 0 to ∞ (0 = identical, larger = more different)
• Formula: √[(x₁-x₂)² + (y₁-y₂)² + ...]

DOT PRODUCT:
• Best for: When magnitude matters
• Range: -∞ to ∞
• Formula: A·B = a₁×b₁ + a₂×b₂ + ...

INTERVIEW TIP:
Know when to use which metric!
- Text similarity → Cosine
- Image similarity → Euclidean
- Recommendation → Cosine or Dot Product
""")

def demonstrate_distance_metrics():
    """Demonstrate different distance metrics"""
    print("Distance Metrics Comparison:")
    
    import math
    
    # Sample vectors representing documents
    doc_vectors = {
        "Python Programming": [0.8, 0.9, 0.1, 0.2],
        "Java Programming": [0.7, 0.8, 0.2, 0.1], 
        "Machine Learning": [0.3, 0.4, 0.9, 0.8],
        "Cooking Recipes": [0.1, 0.1, 0.2, 0.1]
    }
    
    query = "Programming Tutorial"
    query_vector = [0.9, 0.9, 0.1, 0.1]
    
    print(f"\nQuery: '{query}' → {query_vector}")
    print("\nDocument vectors:")
    for doc, vector in doc_vectors.items():
        print(f"{doc:18}: {vector}")
    
    # Calculate different distance metrics
    def cosine_similarity(v1, v2):
        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude1 = math.sqrt(sum(a * a for a in v1))
        magnitude2 = math.sqrt(sum(a * a for a in v2))
        return dot_product / (magnitude1 * magnitude2)
    
    def euclidean_distance(v1, v2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
    
    def dot_product(v1, v2):
        return sum(a * b for a, b in zip(v1, v2))
    
    print("\nSimilarity/Distance Calculations:")
    print("Document            Cosine    Euclidean  Dot Product")
    print("-" * 55)
    
    for doc, vector in doc_vectors.items():
        cos_sim = cosine_similarity(query_vector, vector)
        euc_dist = euclidean_distance(query_vector, vector)
        dot_prod = dot_product(query_vector, vector)
        
        print(f"{doc:18} {cos_sim:8.3f} {euc_dist:10.3f} {dot_prod:11.3f}")
    
    print("\nInterpretation:")
    print("• Cosine: Higher = more similar")
    print("• Euclidean: Lower = more similar") 
    print("• Dot Product: Higher = more similar")
    print("\nAll metrics correctly identify programming documents as most similar!")

demonstrate_distance_metrics()

# Question 5: Indexing Algorithms
print("\n5. What are the main indexing algorithms for vector search?")
print("-" * 61)
print("""
INDEXING ALGORITHMS:

1. FLAT INDEX (BRUTE FORCE):
   • Compares query against every vector
   • 100% accurate but slow
   • Good for: Small datasets (<10K vectors)
   • Time complexity: O(n×d) where n=vectors, d=dimensions

2. IVF (INVERTED FILE):
   • Divides space into clusters
   • Only searches relevant clusters
   • Trade-off between speed and accuracy
   • Good for: Medium datasets (10K-1M vectors)

3. HNSW (HIERARCHICAL NAVIGABLE SMALL WORLD):
   • Multi-layer graph structure
   • Very fast and accurate
   • Most popular for production
   • Good for: Large datasets (1M+ vectors)

4. LSH (LOCALITY SENSITIVE HASHING):
   • Hash similar vectors to same buckets
   • Fast but less accurate
   • Good for: Approximate search

5. PRODUCT QUANTIZATION (PQ):
   • Compresses vectors to save memory
   • Trades memory for some accuracy
   • Good for: Large-scale systems

PERFORMANCE COMPARISON:
Algorithm | Speed | Accuracy | Memory | Best for
----------|-------|----------|--------|----------
Flat      | ⭐     | ⭐⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐ | Small data
IVF       | ⭐⭐⭐   | ⭐⭐⭐⭐    | ⭐⭐⭐⭐  | Medium data
HNSW      | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐    | ⭐⭐⭐   | Large data
LSH       | ⭐⭐⭐⭐  | ⭐⭐⭐     | ⭐⭐⭐⭐  | Speed priority
PQ        | ⭐⭐⭐   | ⭐⭐⭐     | ⭐⭐⭐⭐⭐ | Memory priority
""")

# Question 6: Evaluation Metrics
print("\n6. How do you evaluate vector database performance?")
print("-" * 53)
print("""
EVALUATION METRICS:

1. ACCURACY METRICS:
   • Recall@k: % of relevant results in top-k results
   • Precision@k: % of returned results that are relevant
   • mAP (Mean Average Precision): Overall ranking quality

2. PERFORMANCE METRICS:
   • QPS (Queries Per Second): Throughput
   • Latency: Response time (p50, p95, p99)
   • Memory Usage: RAM consumption
   • Index Build Time: How long to create the index

3. TRADE-OFF METRICS:
   • Speed vs Accuracy: Faster often means less accurate
   • Memory vs Accuracy: Less memory often means less accurate
   • Index Size: Storage requirements

EXAMPLE EVALUATION:
Dataset: 1M vectors, 768 dimensions
Query: Find top-10 similar vectors

Algorithm | Recall@10 | QPS   | Memory | Latency
----------|-----------|-------|--------|--------
Flat      | 100%      | 50    | 3GB    | 20ms
IVF       | 95%       | 500   | 3.5GB  | 2ms
HNSW      | 98%       | 1000  | 4GB    | 1ms

INTERVIEW TIP:
Always mention the trade-offs! There's no perfect solution - 
it depends on your specific requirements (accuracy vs speed vs memory).
""")

def demonstrate_simple_indexing():
    """Demonstrate simple indexing concept"""
    print("Simple Indexing Concept Demo:")
    
    # Simulate a simple IVF-like index
    print("\nIVF (Inverted File) Indexing Example:")
    
    # Step 1: Create clusters of similar vectors
    clusters = {
        "Technology": [
            ("Python Programming", [0.9, 0.8, 0.1, 0.1]),
            ("Java Programming", [0.8, 0.9, 0.1, 0.2]),
            ("Machine Learning", [0.7, 0.6, 0.8, 0.9])
        ],
        "Food": [
            ("Pizza Recipe", [0.1, 0.1, 0.9, 0.8]),
            ("Pasta Recipe", [0.1, 0.2, 0.8, 0.9]),
            ("Salad Recipe", [0.2, 0.1, 0.7, 0.8])
        ],
        "Sports": [
            ("Football Rules", [0.1, 0.2, 0.1, 0.9]),
            ("Basketball Tips", [0.2, 0.1, 0.2, 0.8]),
            ("Tennis Guide", [0.1, 0.1, 0.3, 0.7])
        ]
    }
    
    query = "Programming Language"
    query_vector = [0.9, 0.9, 0.1, 0.1]
    
    print(f"Query: '{query}' → {query_vector}")
    
    # Step 2: Find the most relevant cluster
    def cluster_similarity(query_vec, cluster_docs):
        # Calculate average similarity to cluster
        total_similarity = 0
        for _, doc_vec in cluster_docs:
            dot_product = sum(a * b for a, b in zip(query_vec, doc_vec))
            total_similarity += dot_product
        return total_similarity / len(cluster_docs)
    
    print("\nCluster Relevance:")
    cluster_scores = {}
    for cluster_name, docs in clusters.items():
        score = cluster_similarity(query_vector, docs)
        cluster_scores[cluster_name] = score
        print(f"{cluster_name:10}: {score:.3f}")
    
    # Step 3: Search only in the most relevant cluster
    best_cluster = max(cluster_scores, key=cluster_scores.get)
    print(f"\nSearching only in '{best_cluster}' cluster (most relevant)")
    
    best_docs = clusters[best_cluster]
    similarities = []
    for doc_name, doc_vec in best_docs:
        similarity = sum(a * b for a, b in zip(query_vector, doc_vec))
        similarities.append((doc_name, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print("Results from selected cluster:")
    for doc, sim in similarities:
        print(f"{doc:18}: {sim:.3f}")
    
    print("\nBenefit: Instead of searching 9 documents, we only searched 3!")
    print("This is how indexing speeds up vector search.")

demonstrate_simple_indexing()

print("\n" + "=" * 80)
print("END OF VECTOR DATABASE FUNDAMENTALS")
print("Next: Learn about pgvector implementation in PostgreSQL...")
print("=" * 80)
