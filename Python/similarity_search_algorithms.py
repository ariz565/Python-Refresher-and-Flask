"""
Similarity Search Algorithms - Complete Guide for Beginners
Learn the algorithms behind vector similarity search
Covers k-NN, approximate methods, distance metrics, and evaluation
"""

print("=" * 80)
print("SIMILARITY SEARCH ALGORITHMS - INTERVIEW PREPARATION")
print("=" * 80)

# ============================================================================
# SECTION 1: SIMILARITY SEARCH FUNDAMENTALS
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 1: SIMILARITY SEARCH FUNDAMENTALS")
print("=" * 60)

# Question 1: What is Similarity Search?
print("\n1. What is similarity search and why is it important?")
print("-" * 54)
print("""
SIMILARITY SEARCH DEFINITION:
Similarity search is the process of finding objects (vectors) that are "similar" 
to a given query object in a high-dimensional space.

KEY CONCEPTS:
• Query Vector: The input vector we want to find similar items for
• Database Vectors: The collection of vectors to search through
• Distance Metric: How we measure "similarity" between vectors
• k-NN: Find k most similar vectors (k Nearest Neighbors)
• Range Search: Find all vectors within a distance threshold

WHY IS IT IMPORTANT?
✅ Foundation of recommendation systems
✅ Enables semantic search (meaning-based)
✅ Powers AI applications (RAG, ChatGPT)
✅ Essential for machine learning
✅ Enables content discovery and matching

REAL-WORLD APPLICATIONS:
🔍 Search Engines: "Find similar web pages"
🎵 Music Apps: "Songs you might like"
🛍️ E-commerce: "Customers who bought this also bought"
📱 Social Media: "People you may know"
🏥 Healthcare: "Similar patient cases"
🔒 Security: "Detect fraudulent transactions"

CHALLENGES:
❌ Curse of dimensionality (high-dimensional spaces are sparse)
❌ Computational complexity (brute force is O(n))
❌ Memory requirements (storing millions of vectors)
❌ Accuracy vs speed trade-offs
❌ Choosing appropriate distance metrics

INTERVIEW TIP:
Always mention the "curse of dimensionality" - as dimensions increase,
all points become roughly equidistant, making similarity less meaningful.
""")

def demonstrate_similarity_search_concept():
    """Demonstrate basic similarity search concepts"""
    print("Similarity Search Concept Demo:")
    
    import math
    import random
    
    # Simple 2D example for visualization
    points_2d = {
        "restaurants": [
            ("Pizza Place", [2, 3]),
            ("Burger Joint", [3, 2]),
            ("Sushi Bar", [8, 9]),
            ("Coffee Shop", [1, 2]),
            ("Fancy Restaurant", [9, 8])
        ]
    }
    
    query_point = ("New Restaurant", [2.5, 2.5])
    
    print(f"Query: {query_point[0]} at location {query_point[1]}")
    print("\nRestaurant locations:")
    for name, location in points_2d["restaurants"]:
        print(f"{name:15}: {location}")
    
    # Calculate distances
    def euclidean_distance(p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    
    distances = []
    for name, location in points_2d["restaurants"]:
        dist = euclidean_distance(query_point[1], location)
        distances.append((name, location, dist))
    
    # Sort by distance (nearest first)
    distances.sort(key=lambda x: x[2])
    
    print(f"\nNearest neighbors to {query_point[0]}:")
    for i, (name, location, dist) in enumerate(distances, 1):
        print(f"{i}. {name:15} - Distance: {dist:.2f}")
    
    print("\nObservation: Coffee Shop and Pizza Place are most similar")
    print("This is how similarity search works - find closest points!")
    
    # High-dimensional example
    print("\n" + "="*50)
    print("HIGH-DIMENSIONAL SIMILARITY SEARCH")
    print("="*50)
    
    # Simulate high-dimensional vectors (e.g., text embeddings)
    dimension = 100
    num_documents = 1000
    
    # Generate random "document vectors"
    random.seed(42)
    documents = []
    for i in range(num_documents):
        vector = [random.gauss(0, 1) for _ in range(dimension)]
        documents.append((f"Document_{i}", vector))
    
    # Query vector
    query_vector = [random.gauss(0, 1) for _ in range(dimension)]
    
    print(f"Dataset: {num_documents} documents in {dimension}D space")
    
    # Find most similar documents
    similarities = []
    for doc_name, doc_vector in documents:
        # Use cosine similarity
        dot_product = sum(a * b for a, b in zip(query_vector, doc_vector))
        mag_query = math.sqrt(sum(a * a for a in query_vector))
        mag_doc = math.sqrt(sum(a * a for a in doc_vector))
        
        if mag_query > 0 and mag_doc > 0:
            similarity = dot_product / (mag_query * mag_doc)
            similarities.append((doc_name, similarity))
    
    # Get top 5 most similar
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 most similar documents:")
    for i, (doc_name, similarity) in enumerate(similarities[:5], 1):
        print(f"{i}. {doc_name}: similarity = {similarity:.4f}")
    
    print(f"\nSearched through {num_documents} documents")
    print("In real applications, this could be millions of vectors!")

demonstrate_similarity_search_concept()

# Question 2: Distance Metrics Deep Dive
print("\n2. What are the different distance metrics for similarity search?")
print("-" * 67)
print("""
DISTANCE METRICS COMPARISON:

1. EUCLIDEAN DISTANCE (L2):
   • Formula: √[(x₁-y₁)² + (x₂-y₂)² + ... + (xₙ-yₙ)²]
   • Range: [0, ∞) (0 = identical, larger = more different)
   • Best for: Continuous features, image embeddings
   • Properties: Metric space (triangle inequality holds)

2. COSINE DISTANCE:
   • Formula: 1 - (A·B)/(|A|×|B|)
   • Range: [0, 2] (0 = identical direction, 2 = opposite)
   • Best for: Text embeddings, high-dimensional sparse data
   • Properties: Ignores magnitude, focuses on direction

3. MANHATTAN DISTANCE (L1):
   • Formula: |x₁-y₁| + |x₂-y₂| + ... + |xₙ-yₙ|
   • Range: [0, ∞)
   • Best for: Discrete features, outlier-robust scenarios
   • Properties: Less sensitive to outliers than Euclidean

4. DOT PRODUCT (INNER PRODUCT):
   • Formula: x₁×y₁ + x₂×y₂ + ... + xₙ×yₙ
   • Range: (-∞, ∞) (higher = more similar)
   • Best for: Normalized vectors, neural network outputs
   • Properties: Fast to compute, used in neural networks

5. HAMMING DISTANCE:
   • Formula: Number of differing bits/positions
   • Range: [0, n] where n is vector length
   • Best for: Binary vectors, categorical data
   • Properties: Discrete, used in information theory

6. MINKOWSKI DISTANCE:
   • Formula: (Σ|xᵢ-yᵢ|ᵖ)^(1/p)
   • Special cases: p=1 (Manhattan), p=2 (Euclidean)
   • Range: [0, ∞)
   • Properties: Generalizes L1 and L2 distances

CHOOSING THE RIGHT METRIC:
Data Type           | Recommended Metric    | Reason
--------------------|----------------------|---------------------------
Text embeddings     | Cosine              | Direction matters, not magnitude
Image features      | Euclidean           | Continuous values
Binary vectors      | Hamming             | Bit-wise comparison
Normalized vectors  | Dot Product         | Fast and effective
Categorical data    | Manhattan           | Robust to outliers
High-dimensional    | Cosine              | Curse of dimensionality

INTERVIEW QUESTIONS:
Q: "Why use cosine similarity for text?"
A: "Text embeddings encode semantic meaning in direction, not magnitude.
   Two documents about the same topic will point in similar directions
   regardless of length."

Q: "When would you use Manhattan distance?"
A: "When you have outliers or want equal weight for all dimensions.
   It's less sensitive to extreme values than Euclidean distance."
""")

def demonstrate_distance_metrics():
    """Demonstrate different distance metrics with examples"""
    print("Distance Metrics Comparison Demo:")
    
    import math
    import numpy as np
    
    # Example vectors representing different documents
    vectors = {
        "AI Research Paper": [0.8, 0.9, 0.2, 0.1, 0.7],
        "ML Tutorial": [0.7, 0.8, 0.3, 0.2, 0.6], 
        "Cooking Recipe": [0.1, 0.2, 0.9, 0.8, 0.1],
        "Sports Article": [0.2, 0.1, 0.1, 0.9, 0.8],
        "AI News": [0.9, 0.7, 0.1, 0.1, 0.8]
    }
    
    query = "Machine Learning Guide"
    query_vector = [0.8, 0.8, 0.2, 0.1, 0.7]
    
    print(f"Query: '{query}' → {query_vector}")
    print("\nDocument vectors:")
    for doc, vec in vectors.items():
        print(f"{doc:18}: {vec}")
    
    # Implement distance metrics
    def euclidean_distance(v1, v2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
    
    def cosine_distance(v1, v2):
        dot_product = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(a * a for a in v2))
        if mag1 == 0 or mag2 == 0:
            return 1  # Maximum distance
        return 1 - (dot_product / (mag1 * mag2))
    
    def manhattan_distance(v1, v2):
        return sum(abs(a - b) for a, b in zip(v1, v2))
    
    def dot_product_similarity(v1, v2):
        return sum(a * b for a, b in zip(v1, v2))
    
    def hamming_distance(v1, v2):
        # Convert to binary (>0.5 = 1, <=0.5 = 0)
        b1 = [1 if x > 0.5 else 0 for x in v1]
        b2 = [1 if x > 0.5 else 0 for x in v2]
        return sum(a != b for a, b in zip(b1, b2))
    
    # Calculate all distances
    metrics = {
        "Euclidean": euclidean_distance,
        "Cosine": cosine_distance,
        "Manhattan": manhattan_distance,
        "Dot Product": lambda v1, v2: -dot_product_similarity(v1, v2),  # Negative for sorting
        "Hamming": hamming_distance
    }
    
    print(f"\nDistance/Similarity calculations for '{query}':")
    print("="*70)
    
    for metric_name, metric_func in metrics.items():
        print(f"\n{metric_name} Distance/Similarity:")
        print("-" * 40)
        
        results = []
        for doc, vec in vectors.items():
            distance = metric_func(query_vector, vec)
            results.append((doc, distance))
        
        # Sort by distance (ascending for distances, descending for similarities)
        if metric_name == "Dot Product":
            results.sort(key=lambda x: x[1])  # Ascending (we negated it)
            results = [(doc, -dist) for doc, dist in results]  # Convert back
            print("(Higher values = more similar)")
        else:
            results.sort(key=lambda x: x[1])
            print("(Lower values = more similar)")
        
        for i, (doc, value) in enumerate(results, 1):
            print(f"{i}. {doc:18}: {value:.4f}")
    
    # Normalized vs non-normalized comparison
    print("\n" + "="*70)
    print("NORMALIZED vs NON-NORMALIZED VECTORS")
    print("="*70)
    
    # Original vectors
    vec1 = [3, 4]  # Length = 5
    vec2 = [6, 8]  # Length = 10, same direction as vec1
    vec3 = [1, 1]  # Length = √2, different direction
    
    # Normalized versions
    def normalize(v):
        mag = math.sqrt(sum(x*x for x in v))
        return [x/mag for x in v] if mag > 0 else v
    
    vec1_norm = normalize(vec1)
    vec2_norm = normalize(vec2)
    vec3_norm = normalize(vec3)
    
    print("Original vectors:")
    print(f"Vec1: {vec1} (magnitude: {math.sqrt(sum(x*x for x in vec1)):.2f})")
    print(f"Vec2: {vec2} (magnitude: {math.sqrt(sum(x*x for x in vec2)):.2f})")
    print(f"Vec3: {vec3} (magnitude: {math.sqrt(sum(x*x for x in vec3)):.2f})")
    
    print("\nNormalized vectors:")
    print(f"Vec1: {[f'{x:.3f}' for x in vec1_norm]}")
    print(f"Vec2: {[f'{x:.3f}' for x in vec2_norm]}")
    print(f"Vec3: {[f'{x:.3f}' for x in vec3_norm]}")
    
    print("\nDistance comparisons:")
    print(f"Euclidean(Vec1, Vec2): {euclidean_distance(vec1, vec2):.3f}")
    print(f"Euclidean(Vec1, Vec3): {euclidean_distance(vec1, vec3):.3f}")
    print(f"Cosine(Vec1, Vec2): {cosine_distance(vec1, vec2):.3f}")
    print(f"Cosine(Vec1, Vec3): {cosine_distance(vec1, vec3):.3f}")
    
    print("\nKey Insight:")
    print("• Euclidean: Vec3 appears closer to Vec1 than Vec2")
    print("• Cosine: Vec2 is closer to Vec1 (same direction)")
    print("• This shows why cosine is preferred for direction-based similarity")

demonstrate_distance_metrics()

# ============================================================================
# SECTION 2: EXACT SIMILARITY SEARCH ALGORITHMS
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 2: EXACT SIMILARITY SEARCH ALGORITHMS")
print("=" * 60)

# Question 3: Brute Force k-NN
print("\n3. How does brute force k-NN search work?")
print("-" * 45)
print("""
BRUTE FORCE k-NN ALGORITHM:
The simplest and most accurate method for finding k nearest neighbors.

ALGORITHM:
1. Calculate distance from query to every vector in database
2. Sort all distances in ascending order
3. Return the k smallest distances and their indices

PSEUDOCODE:
function bruteForceKNN(query, database, k):
    distances = []
    for i, vector in enumerate(database):
        dist = distance(query, vector)
        distances.append((dist, i))
    
    distances.sort()  // Sort by distance
    return distances[:k]  // Return k nearest

TIME COMPLEXITY:
• Distance calculations: O(n × d) where n=vectors, d=dimensions
• Sorting: O(n log n)
• Total: O(n × d + n log n) ≈ O(n × d) for large d

SPACE COMPLEXITY:
• O(n) for storing distances

ADVANTAGES:
✅ 100% accurate (finds true nearest neighbors)
✅ Simple to implement and understand
✅ Works with any distance metric
✅ No preprocessing required
✅ Supports dynamic updates

DISADVANTAGES:
❌ Very slow for large datasets
❌ Linear time complexity
❌ No memory efficiency
❌ Not practical for real-time applications

WHEN TO USE:
✅ Small datasets (<10K vectors)
✅ Ground truth generation
✅ Prototyping and testing
✅ When 100% accuracy is required
✅ Batch processing scenarios
""")

def demonstrate_brute_force_knn():
    """Demonstrate brute force k-NN implementation"""
    print("Brute Force k-NN Implementation Demo:")
    
    import time
    import heapq
    import random
    import math
    
    class BruteForceKNN:
        def __init__(self, distance_metric='euclidean'):
            self.vectors = []
            self.metadata = []
            self.distance_metric = distance_metric
        
        def add_vector(self, vector, metadata=None):
            """Add a vector to the database"""
            self.vectors.append(vector)
            self.metadata.append(metadata)
        
        def _calculate_distance(self, v1, v2):
            """Calculate distance between two vectors"""
            if self.distance_metric == 'euclidean':
                return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
            elif self.distance_metric == 'cosine':
                dot_product = sum(a * b for a, b in zip(v1, v2))
                mag1 = math.sqrt(sum(a * a for a in v1))
                mag2 = math.sqrt(sum(a * a for a in v2))
                if mag1 == 0 or mag2 == 0:
                    return 1.0
                return 1 - (dot_product / (mag1 * mag2))
            elif self.distance_metric == 'manhattan':
                return sum(abs(a - b) for a, b in zip(v1, v2))
            else:
                raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        def search(self, query, k=5):
            """Find k nearest neighbors using brute force"""
            distances = []
            
            # Calculate distance to every vector
            for i, vector in enumerate(self.vectors):
                dist = self._calculate_distance(query, vector)
                distances.append((dist, i))
            
            # Sort by distance and return top k
            distances.sort()
            
            results = []
            for dist, idx in distances[:k]:
                results.append({
                    'index': idx,
                    'distance': dist,
                    'vector': self.vectors[idx],
                    'metadata': self.metadata[idx]
                })
            
            return results
        
        def search_optimized(self, query, k=5):
            """Optimized version using heap (for large k)"""
            # Use max heap to keep only k smallest distances
            heap = []
            
            for i, vector in enumerate(self.vectors):
                dist = self._calculate_distance(query, vector)
                
                if len(heap) < k:
                    heapq.heappush(heap, (-dist, i))  # Negative for max heap
                elif dist < -heap[0][0]:  # If smaller than largest in heap
                    heapq.heapreplace(heap, (-dist, i))
            
            # Extract results and sort
            results = []
            while heap:
                neg_dist, idx = heapq.heappop(heap)
                results.append({
                    'index': idx,
                    'distance': -neg_dist,
                    'vector': self.vectors[idx],
                    'metadata': self.metadata[idx]
                })
            
            # Reverse to get ascending order
            results.reverse()
            return results
    
    # Demo with sample data
    print("\\n1. Creating sample dataset:")
    
    # Generate sample vectors (simulate document embeddings)
    random.seed(42)
    dimension = 50
    num_vectors = 1000
    
    knn = BruteForceKNN(distance_metric='cosine')
    
    print(f"Generating {num_vectors} vectors of dimension {dimension}...")
    
    for i in range(num_vectors):
        # Generate random vector
        vector = [random.gauss(0, 1) for _ in range(dimension)]
        
        # Create metadata
        metadata = {
            'id': i,
            'category': random.choice(['tech', 'sports', 'cooking', 'travel']),
            'title': f"Document_{i}"
        }
        
        knn.add_vector(vector, metadata)
    
    print(f"Created database with {len(knn.vectors)} vectors")
    
    # Test search
    print("\\n2. Testing search:")
    query = [random.gauss(0, 1) for _ in range(dimension)]
    k = 5
    
    # Benchmark search
    start_time = time.time()
    results = knn.search(query, k)
    search_time = time.time() - start_time
    
    print(f"Search completed in {search_time*1000:.2f}ms")
    print(f"\\nTop {k} results:")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['metadata']['title']} "
              f"({result['metadata']['category']}) - "
              f"Distance: {result['distance']:.4f}")
    
    # Compare with optimized version
    print("\\n3. Performance comparison:")
    
    start_time = time.time()
    results_basic = knn.search(query, k)
    time_basic = time.time() - start_time
    
    start_time = time.time()
    results_optimized = knn.search_optimized(query, k)
    time_optimized = time.time() - start_time
    
    print(f"Basic search: {time_basic*1000:.2f}ms")
    print(f"Optimized search: {time_optimized*1000:.2f}ms")
    print(f"Speedup: {time_basic/time_optimized:.1f}x")
    
    # Verify results are the same
    distances_basic = [r['distance'] for r in results_basic]
    distances_opt = [r['distance'] for r in results_optimized]
    
    print(f"Results match: {distances_basic == distances_opt}")
    
    # Scale test
    print("\\n4. Scalability test:")
    scales = [100, 500, 1000, 2000]
    
    for scale in scales:
        if scale <= len(knn.vectors):
            # Create subset
            subset_knn = BruteForceKNN(distance_metric='cosine')
            for i in range(scale):
                subset_knn.add_vector(knn.vectors[i], knn.metadata[i])
            
            # Benchmark
            start_time = time.time()
            results = subset_knn.search(query, 5)
            search_time = time.time() - start_time
            
            print(f"Scale {scale:4d}: {search_time*1000:6.1f}ms "
                  f"({search_time*1000/scale:.3f}ms per vector)")
    
    print("\\nObservation: Search time grows linearly with dataset size")
    print("This is why we need approximate methods for large datasets!")

demonstrate_brute_force_knn()

# ============================================================================
# SECTION 3: APPROXIMATE SIMILARITY SEARCH
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 3: APPROXIMATE SIMILARITY SEARCH")
print("=" * 60)

# Question 4: Locality Sensitive Hashing (LSH)
print("\n4. How does Locality Sensitive Hashing (LSH) work?")
print("-" * 54)
print("""
LOCALITY SENSITIVE HASHING (LSH):
LSH is a technique that hashes similar vectors to the same buckets,
enabling fast approximate similarity search.

KEY IDEA:
• Hash similar vectors to the same hash bucket
• Search only within the same bucket(s)
• Trade accuracy for speed

LSH PROPERTIES:
For a hash function h to be LSH:
• If vectors are similar → high probability of same hash
• If vectors are different → low probability of same hash

COMMON LSH FAMILIES:

1. RANDOM PROJECTIONS (for cosine similarity):
   • Project vectors onto random hyperplanes
   • Hash based on which side of plane vector falls
   • Multiple projections create hash signature

2. MIN-HASH (for Jaccard similarity):
   • Used for set similarity (binary vectors)
   • Hash based on minimum element in permuted sets

3. P-STABLE DISTRIBUTIONS (for Lp distances):
   • Use random projections with stable distributions
   • Works for Euclidean and Manhattan distances

ALGORITHM (Random Projections):
1. Generate k random hyperplanes
2. For each vector:
   - Project onto each hyperplane
   - Create binary signature based on sign
3. Hash vectors with same signature to same bucket
4. Search only within query's bucket

PARAMETERS:
• k: Number of hash functions (more = better accuracy)
• L: Number of hash tables (more = better recall)

ADVANTAGES:
✅ Sub-linear query time
✅ Works well for high-dimensional data
✅ Memory efficient
✅ Suitable for streaming data

DISADVANTAGES:
❌ Approximate results only
❌ Parameter tuning required
❌ May miss some neighbors
❌ Performance depends on data distribution

TIME COMPLEXITY:
• Preprocessing: O(n × d × k × L)
• Query: O(d × k × L + bucket_size)
• Space: O(n × k × L)
""")

def demonstrate_lsh():
    """Demonstrate Locality Sensitive Hashing implementation"""
    print("Locality Sensitive Hashing (LSH) Demo:")
    
    import random
    import math
    import time
    from collections import defaultdict
    
    class RandomProjectionLSH:
        def __init__(self, dimension, num_hash_functions=10, num_tables=5):
            self.dimension = dimension
            self.num_hash_functions = num_hash_functions
            self.num_tables = num_tables
            
            # Generate random hyperplanes for each table
            self.hash_tables = []
            self.hyperplanes = []
            
            random.seed(42)  # For reproducibility
            
            for table_idx in range(num_tables):
                # Generate random hyperplanes for this table
                planes = []
                for _ in range(num_hash_functions):
                    # Random unit vector (hyperplane normal)
                    plane = [random.gauss(0, 1) for _ in range(dimension)]
                    # Normalize
                    magnitude = math.sqrt(sum(x*x for x in plane))
                    if magnitude > 0:
                        plane = [x/magnitude for x in plane]
                    planes.append(plane)
                
                self.hyperplanes.append(planes)
                self.hash_tables.append(defaultdict(list))
            
            self.vectors = []
            self.metadata = []
        
        def _hash_vector(self, vector, table_idx):
            """Hash a vector using the hyperplanes for given table"""
            hash_value = []
            planes = self.hyperplanes[table_idx]
            
            for plane in planes:
                # Dot product with hyperplane
                dot_product = sum(v * p for v, p in zip(vector, plane))
                # Binary hash based on sign
                hash_value.append(1 if dot_product >= 0 else 0)
            
            # Convert binary list to string for hashing
            return ''.join(map(str, hash_value))
        
        def add_vector(self, vector, metadata=None):
            """Add vector to all hash tables"""
            vector_idx = len(self.vectors)
            self.vectors.append(vector)
            self.metadata.append(metadata)
            
            # Add to each hash table
            for table_idx in range(self.num_tables):
                hash_value = self._hash_vector(vector, table_idx)
                self.hash_tables[table_idx][hash_value].append(vector_idx)
        
        def search(self, query, k=5):
            """Search for similar vectors using LSH"""
            candidates = set()
            
            # Get candidates from all hash tables
            for table_idx in range(self.num_tables):
                hash_value = self._hash_vector(query, table_idx)
                bucket_vectors = self.hash_tables[table_idx].get(hash_value, [])
                candidates.update(bucket_vectors)
            
            # Calculate actual distances for candidates
            distances = []
            for idx in candidates:
                vector = self.vectors[idx]
                # Cosine distance
                dot_product = sum(a * b for a, b in zip(query, vector))
                mag_query = math.sqrt(sum(a * a for a in query))
                mag_vector = math.sqrt(sum(a * a for a in vector))
                
                if mag_query > 0 and mag_vector > 0:
                    cosine_sim = dot_product / (mag_query * mag_vector)
                    distance = 1 - cosine_sim
                else:
                    distance = 1.0
                
                distances.append((distance, idx))
            
            # Sort and return top k
            distances.sort()
            
            results = []
            for dist, idx in distances[:k]:
                results.append({
                    'index': idx,
                    'distance': dist,
                    'metadata': self.metadata[idx]
                })
            
            return results, len(candidates)
    
    # Demo with sample data
    print("\\n1. Creating LSH index:")
    
    dimension = 100
    num_vectors = 5000
    
    # Create LSH index
    lsh = RandomProjectionLSH(
        dimension=dimension,
        num_hash_functions=15,  # More hash functions = better accuracy
        num_tables=10          # More tables = better recall
    )
    
    # Create brute force baseline for comparison
    from collections import namedtuple
    BruteForce = namedtuple('BruteForce', ['vectors', 'metadata'])
    brute_force = BruteForce(vectors=[], metadata=[])
    
    print(f"Generating {num_vectors} random vectors...")
    
    # Generate and add vectors
    random.seed(123)
    for i in range(num_vectors):
        # Generate random vector
        vector = [random.gauss(0, 1) for _ in range(dimension)]
        
        # Normalize for cosine similarity
        magnitude = math.sqrt(sum(x*x for x in vector))
        if magnitude > 0:
            vector = [x/magnitude for x in vector]
        
        metadata = {'id': i, 'title': f'Vector_{i}'}
        
        lsh.add_vector(vector, metadata)
        brute_force.vectors.append(vector)
        brute_force.metadata.append(metadata)
    
    print(f"LSH index created with {len(lsh.vectors)} vectors")
    
    # Test search
    print("\\n2. Comparing LSH vs Brute Force:")
    
    query = [random.gauss(0, 1) for _ in range(dimension)]
    magnitude = math.sqrt(sum(x*x for x in query))
    if magnitude > 0:
        query = [x/magnitude for x in query]
    
    k = 10
    
    # LSH search
    start_time = time.time()
    lsh_results, candidates_checked = lsh.search(query, k)
    lsh_time = time.time() - start_time
    
    # Brute force search (for comparison)
    start_time = time.time()
    bf_distances = []
    for i, vector in enumerate(brute_force.vectors):
        dot_product = sum(a * b for a, b in zip(query, vector))
        distance = 1 - dot_product  # Cosine distance
        bf_distances.append((distance, i))
    
    bf_distances.sort()
    bf_results = bf_distances[:k]
    bf_time = time.time() - start_time
    
    print(f"LSH search time: {lsh_time*1000:.2f}ms")
    print(f"Brute force time: {bf_time*1000:.2f}ms")
    print(f"Speedup: {bf_time/lsh_time:.1f}x")
    print(f"Candidates checked: {candidates_checked}/{num_vectors} "
          f"({candidates_checked/num_vectors*100:.1f}%)")
    
    # Calculate recall (how many true neighbors were found)
    bf_top_indices = set(idx for _, idx in bf_results)
    lsh_top_indices = set(result['index'] for result in lsh_results)
    
    recall = len(bf_top_indices & lsh_top_indices) / len(bf_top_indices)
    print(f"Recall@{k}: {recall:.3f} ({recall*100:.1f}%)")
    
    # Show results comparison
    print(f"\\n3. Results comparison (top 5):")
    print("LSH Results:")
    for i, result in enumerate(lsh_results[:5], 1):
        print(f"{i}. Vector_{result['index']} - Distance: {result['distance']:.4f}")
    
    print("\\nBrute Force Results:")
    for i, (dist, idx) in enumerate(bf_results[:5], 1):
        print(f"{i}. Vector_{idx} - Distance: {dist:.4f}")
    
    # Parameter sensitivity analysis
    print("\\n4. Parameter sensitivity:")
    
    # Test different numbers of hash functions
    hash_function_counts = [5, 10, 15, 20]
    print("\\nHash functions vs Accuracy/Speed:")
    print("Hash Funcs | Time(ms) | Recall | Candidates")
    print("-" * 45)
    
    for num_hashes in hash_function_counts:
        test_lsh = RandomProjectionLSH(
            dimension=dimension,
            num_hash_functions=num_hashes,
            num_tables=5
        )
        
        # Add subset of vectors for faster testing
        for i in range(min(1000, num_vectors)):
            test_lsh.add_vector(brute_force.vectors[i], brute_force.metadata[i])
        
        # Test search
        start_time = time.time()
        test_results, test_candidates = test_lsh.search(query, k)
        test_time = time.time() - start_time
        
        # Calculate recall against brute force subset
        bf_subset = bf_distances[:min(1000, num_vectors)]
        bf_subset_top = set(idx for _, idx in bf_subset[:k] if idx < 1000)
        test_top = set(result['index'] for result in test_results)
        
        test_recall = len(bf_subset_top & test_top) / len(bf_subset_top) if bf_subset_top else 0
        
        print(f"{num_hashes:9d} | {test_time*1000:7.1f} | {test_recall:6.3f} | {test_candidates:10d}")
    
    print("\\nObservations:")
    print("• More hash functions → Better accuracy, slower search")
    print("• LSH trades accuracy for speed")
    print("• Effectiveness depends on data distribution")
    print("• Good for high-dimensional, sparse data")

demonstrate_lsh()

print("\n" + "=" * 80)
print("END OF SIMILARITY SEARCH ALGORITHMS")
print("Next: Learn about FastAPI integration for building vector search APIs...")
print("=" * 80)
