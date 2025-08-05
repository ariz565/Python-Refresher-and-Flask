"""
FAISS (Facebook AI Similarity Search) - Complete Beginner's Guide
Learn high-performance vector similarity search with FAISS
Covers installation, basic usage, advanced features, and optimization
"""

print("=" * 80)
print("FAISS (FACEBOOK AI SIMILARITY SEARCH) - INTERVIEW PREPARATION")
print("=" * 80)

# ============================================================================
# SECTION 1: WHAT IS FAISS?
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 1: UNDERSTANDING FAISS")
print("=" * 60)

# Question 1: What is FAISS?
print("\n1. What is FAISS and why is it important for vector search?")
print("-" * 61)
print("""
WHAT IS FAISS?
FAISS (Facebook AI Similarity Search) is a library for efficient similarity 
search and clustering of dense vectors developed by Meta AI (Facebook).

KEY FEATURES:
✅ Extremely fast similarity search
✅ Memory efficient implementations
✅ GPU acceleration support
✅ Multiple indexing algorithms
✅ Billions of vectors support
✅ Production-ready and battle-tested

WHY CHOOSE FAISS?
• Speed: Optimized for high-performance search
• Scale: Handles billions of vectors
• Flexibility: Multiple index types for different use cases
• GPU Support: Leverage GPU acceleration
• Memory Efficiency: Compressed indexes to save RAM
• Production Ready: Used by Meta, Spotify, and many others

FAISS vs OTHER SOLUTIONS:
Feature              | FAISS    | pgvector | Pinecone | Weaviate
---------------------|----------|----------|----------|----------
Speed               | ⭐⭐⭐⭐⭐    | ⭐⭐⭐      | ⭐⭐⭐⭐     | ⭐⭐⭐⭐
Scale (vectors)     | Billions | Millions | Millions | Millions  
GPU Support         | Yes      | No       | Yes      | No
Memory Efficiency   | ⭐⭐⭐⭐⭐    | ⭐⭐⭐      | ⭐⭐⭐⭐     | ⭐⭐⭐
Ease of Use        | ⭐⭐       | ⭐⭐⭐⭐⭐    | ⭐⭐⭐⭐⭐   | ⭐⭐⭐⭐
SQL Support        | No       | Yes      | No       | Limited
Cloud Managed      | No       | No       | Yes      | Yes

WHEN TO USE FAISS:
✅ Need maximum performance
✅ Large datasets (>1M vectors)
✅ Memory constraints
✅ GPU acceleration available
✅ Custom algorithms needed
✅ Research and experimentation

WHEN NOT TO USE FAISS:
❌ Need SQL queries
❌ Want managed cloud service
❌ Simple small-scale use case
❌ Need ACID transactions
❌ Want easy deployment
""")

# Question 2: FAISS Installation and Setup
print("\n2. How do you install and set up FAISS?")
print("-" * 42)
print("""
INSTALLATION OPTIONS:

1. CPU VERSION (BASIC):
   pip install faiss-cpu

2. GPU VERSION (FOR NVIDIA GPUS):
   pip install faiss-gpu

3. CONDA INSTALLATION:
   conda install -c pytorch faiss-cpu
   conda install -c pytorch faiss-gpu

4. FROM SOURCE (ADVANCED):
   git clone https://github.com/facebookresearch/faiss.git
   cd faiss
   cmake -B build .
   make -C build -j

DEPENDENCIES:
   pip install numpy          # Core arrays
   pip install matplotlib     # Visualization
   pip install scikit-learn   # ML utilities

VERIFY INSTALLATION:
   import faiss
   print(f"FAISS version: {faiss.__version__}")
   print(f"CPU support: {faiss.get_num_gpus() >= 0}")
   print(f"GPU support: {faiss.get_num_gpus()}")

BASIC IMPORTS:
   import faiss
   import numpy as np
   import time
""")

def demonstrate_faiss_setup():
    """Demonstrate FAISS setup and basic functionality"""
    print("FAISS Setup Demo:")
    print("""
    import faiss
    import numpy as np
    
    # Check FAISS installation
    print(f"FAISS version: {faiss.__version__}")
    
    # Check GPU availability
    gpu_count = faiss.get_num_gpus()
    print(f"Available GPUs: {gpu_count}")
    
    if gpu_count > 0:
        print("GPU acceleration available!")
        # List GPU devices
        for i in range(gpu_count):
            print(f"GPU {i}: Available")
    else:
        print("Using CPU version")
    
    # Test basic functionality
    dimension = 128
    nb = 1000  # number of vectors
    
    # Generate random vectors
    vectors = np.random.random((nb, dimension)).astype('float32')
    print(f"Generated {nb} vectors of dimension {dimension}")
    
    # Create simple index
    index = faiss.IndexFlatL2(dimension)
    print(f"Created index: {index}")
    
    # Add vectors to index
    index.add(vectors)
    print(f"Index contains {index.ntotal} vectors")
    
    # Test search
    k = 5  # number of nearest neighbors
    query = np.random.random((1, dimension)).astype('float32')
    
    distances, indices = index.search(query, k)
    print(f"Search results: {indices[0]}")
    print(f"Distances: {distances[0]}")
    """)

demonstrate_faiss_setup()

# ============================================================================
# SECTION 2: BASIC FAISS OPERATIONS
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 2: BASIC FAISS OPERATIONS")
print("=" * 60)

# Question 3: Basic Index Types
print("\n3. What are the basic FAISS index types?")
print("-" * 44)
print("""
FUNDAMENTAL INDEX TYPES:

1. INDEXFLAT (EXACT SEARCH):
   • IndexFlatL2: L2 (Euclidean) distance
   • IndexFlatIP: Inner Product (dot product)
   • 100% accurate but slower for large datasets
   • Good for: Small datasets, ground truth

2. INDEXIVFFLAT (APPROXIMATE SEARCH):
   • Inverted File with Flat quantizer
   • Faster than Flat but approximate
   • Good for: Medium datasets (10K-1M vectors)

3. INDEXIVFPQ (COMPRESSED SEARCH):
   • Inverted File with Product Quantization
   • Memory efficient, very fast
   • Good for: Large datasets (1M+ vectors)

4. INDEXHNSW (GRAPH-BASED):
   • Hierarchical Navigable Small World
   • High accuracy and speed
   • Good for: When accuracy is important

INDEX SELECTION GUIDE:
Dataset Size | Accuracy Priority | Memory Priority | Recommended Index
-------------|-------------------|-----------------|------------------
< 10K        | High             | Any             | IndexFlatL2
10K - 100K   | High             | Any             | IndexHNSW
100K - 1M    | High             | Low             | IndexIVFFlat
100K - 1M    | Medium           | High            | IndexIVFPQ
> 1M         | Medium           | High            | IndexIVFPQ
> 1M         | High             | Low             | IndexIVFFlat + GPU

DISTANCE METRICS:
• L2 (Euclidean): Most common, good for general use
• Inner Product: For normalized vectors, faster
• Cosine: Use IP with normalized vectors
""")

def demonstrate_basic_indexes():
    """Demonstrate basic FAISS index types"""
    print("Basic FAISS Index Types Demo:")
    print("""
    import faiss
    import numpy as np
    import time
    
    # Prepare test data
    dimension = 128
    nb = 10000  # database vectors
    nq = 100    # query vectors
    
    # Generate random data
    np.random.seed(1234)
    database = np.random.random((nb, dimension)).astype('float32')
    queries = np.random.random((nq, dimension)).astype('float32')
    
    print(f"Database: {nb} vectors of {dimension} dimensions")
    print(f"Queries: {nq} vectors\\n")
    
    # 1. IndexFlatL2 (Exact search)
    print("1. IndexFlatL2 (Exact Search)")
    index_flat = faiss.IndexFlatL2(dimension)
    
    # Add vectors
    index_flat.add(database)
    print(f"   Added {index_flat.ntotal} vectors")
    
    # Search
    k = 5
    start_time = time.time()
    distances, indices = index_flat.search(queries, k)
    search_time = time.time() - start_time
    
    print(f"   Search time: {search_time:.3f} seconds")
    print(f"   Average distance: {distances.mean():.3f}")
    
    # 2. IndexIVFFlat (Approximate search)
    print("\\n2. IndexIVFFlat (Approximate Search)")
    nlist = 100  # number of clusters
    quantizer = faiss.IndexFlatL2(dimension)
    index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    
    # Train the index (clustering)
    print("   Training index...")
    index_ivf.train(database)
    
    # Add vectors
    index_ivf.add(database)
    print(f"   Added {index_ivf.ntotal} vectors")
    
    # Set search parameters
    index_ivf.nprobe = 10  # search 10 clusters
    
    # Search
    start_time = time.time()
    distances, indices = index_ivf.search(queries, k)
    search_time = time.time() - start_time
    
    print(f"   Search time: {search_time:.3f} seconds")
    print(f"   Average distance: {distances.mean():.3f}")
    
    # 3. IndexIVFPQ (Compressed search)
    print("\\n3. IndexIVFPQ (Compressed Search)")
    nlist = 100
    m = 8      # number of subquantizers
    bits = 8   # bits per subquantizer
    
    quantizer = faiss.IndexFlatL2(dimension)
    index_pq = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, bits)
    
    # Train the index
    print("   Training index...")
    index_pq.train(database)
    
    # Add vectors
    index_pq.add(database)
    print(f"   Added {index_pq.ntotal} vectors")
    
    # Set search parameters
    index_pq.nprobe = 10
    
    # Search
    start_time = time.time()
    distances, indices = index_pq.search(queries, k)
    search_time = time.time() - start_time
    
    print(f"   Search time: {search_time:.3f} seconds")
    print(f"   Average distance: {distances.mean():.3f}")
    
    # Memory usage comparison
    print("\\nMemory Usage Comparison:")
    print(f"IndexFlatL2:  {nb * dimension * 4 / 1024 / 1024:.1f} MB")
    print(f"IndexIVFFlat: ~{nb * dimension * 4 / 1024 / 1024:.1f} MB")
    print(f"IndexIVFPQ:   ~{nb * m * bits / 8 / 1024 / 1024:.1f} MB")
    """)

demonstrate_basic_indexes()

# Question 4: Working with Real Data
print("\n4. How do you work with real vector data in FAISS?")
print("-" * 50)
print("""
REAL DATA WORKFLOW:

1. PREPARE YOUR VECTORS:
   • Convert to numpy arrays (float32)
   • Normalize if using Inner Product
   • Ensure consistent dimensions

2. CHOOSE APPROPRIATE INDEX:
   • Consider data size, accuracy, memory requirements
   • Start with simple index, then optimize

3. TRAIN INDEX (IF NEEDED):
   • Some indexes need training (IVF, PQ)
   • Use representative sample of your data

4. ADD VECTORS:
   • Batch additions for better performance
   • Can add incrementally after initial build

5. OPTIMIZE SEARCH PARAMETERS:
   • Tune nprobe, ef_search based on accuracy needs
   • Balance speed vs accuracy

WORKING WITH TEXT EMBEDDINGS:
• Common dimensions: 384, 768, 1536
• Usually use L2 or Cosine distance
• Normalize vectors for cosine similarity

WORKING WITH IMAGE EMBEDDINGS:
• Common dimensions: 512, 2048
• Usually use L2 distance
• May need preprocessing/normalization
""")

def demonstrate_real_data_workflow():
    """Demonstrate working with realistic vector data"""
    print("Real Data Workflow Demo:")
    print("""
    import faiss
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.datasets import fetch_20newsgroups
    
    # 1. Load real text data
    print("Loading text data...")
    newsgroups = fetch_20newsgroups(subset='train', categories=['sci.space', 'comp.graphics'])
    documents = newsgroups.data[:1000]  # Use subset for demo
    
    # 2. Convert text to vectors using TF-IDF
    print("Converting text to vectors...")
    vectorizer = TfidfVectorizer(max_features=512, stop_words='english')
    text_vectors = vectorizer.fit_transform(documents)
    
    # Convert to dense numpy array (FAISS needs dense arrays)
    vectors = text_vectors.toarray().astype('float32')
    dimension = vectors.shape[1]
    nb = vectors.shape[0]
    
    print(f"Created {nb} vectors of dimension {dimension}")
    
    # 3. Choose and configure index
    # For text similarity, we'll use L2 distance
    nlist = min(100, nb // 10)  # Adjust for dataset size
    
    if nb < 1000:
        # Small dataset - use exact search
        print("Using IndexFlatL2 for small dataset")
        index = faiss.IndexFlatL2(dimension)
    else:
        # Larger dataset - use approximate search
        print(f"Using IndexIVFFlat with {nlist} clusters")
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    
    # 4. Train index if needed
    if hasattr(index, 'train'):
        print("Training index...")
        index.train(vectors)
    
    # 5. Add vectors to index
    print("Adding vectors to index...")
    index.add(vectors)
    print(f"Index now contains {index.ntotal} vectors")
    
    # 6. Prepare search
    if hasattr(index, 'nprobe'):
        index.nprobe = min(10, nlist)  # Search 10 clusters
    
    # 7. Example search
    query_text = "space exploration mars mission"
    query_vector = vectorizer.transform([query_text]).toarray().astype('float32')
    
    k = 5
    distances, indices = index.search(query_vector, k)
    
    print(f"\\nSearch results for: '{query_text}'")
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        doc_preview = documents[idx][:100] + "..."
        print(f"{i+1}. Distance: {dist:.3f}")
        print(f"   Document: {doc_preview}\\n")
    
    # 8. Save/Load index
    print("Saving index...")
    faiss.write_index(index, "text_similarity.index")
    
    # Load index
    loaded_index = faiss.read_index("text_similarity.index")
    print(f"Loaded index contains {loaded_index.ntotal} vectors")
    
    # 9. Batch search for efficiency
    print("\\nBatch search example:")
    batch_queries = vectors[:3]  # Use first 3 documents as queries
    
    start_time = time.time()
    distances, indices = index.search(batch_queries, k)
    search_time = time.time() - start_time
    
    print(f"Batch search of {len(batch_queries)} queries took {search_time:.3f}s")
    print(f"Average time per query: {search_time/len(batch_queries):.3f}s")
    """)

demonstrate_real_data_workflow()

# ============================================================================
# SECTION 3: ADVANCED FAISS FEATURES
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 3: ADVANCED FAISS FEATURES")
print("=" * 60)

# Question 5: Index Factory and String Descriptions
print("\n5. How do you use FAISS Index Factory for complex indexes?")
print("-" * 62)
print("""
INDEX FACTORY:
FAISS provides a factory system to create complex indexes using string descriptions.

BASIC SYNTAX:
   index = faiss.index_factory(dimension, description, metric)

COMMON DESCRIPTIONS:
   "Flat"           - Exact search (IndexFlatL2)
   "IVF100,Flat"    - IVF with 100 clusters
   "IVF100,PQ8"     - IVF with Product Quantization
   "HNSW32"         - HNSW with 32 connections
   "IVF100,SQ8"     - IVF with Scalar Quantization

ADVANCED DESCRIPTIONS:
   "OPQ16,IVF100,PQ16"     - Optimized PQ with rotation
   "IVF4096,PQ64x4fs,RFlat" - Complex hierarchical index
   "PCA80,IVF100,PQ8"      - PCA preprocessing + IVF + PQ

METRICS:
   METRIC_L2          - Euclidean distance (default)
   METRIC_INNER_PRODUCT - Inner product (for normalized vectors)
   METRIC_L1          - Manhattan distance

PREPROCESSING:
   "PCA64"       - Reduce to 64 dimensions with PCA
   "OPQ16"       - Optimized Product Quantization rotation
   "L2norm"      - L2 normalize vectors

EXAMPLES:
   # Simple IVF index
   index = faiss.index_factory(128, "IVF100,Flat")
   
   # Compressed index with PCA
   index = faiss.index_factory(768, "PCA128,IVF100,PQ16")
   
   # HNSW index for high accuracy
   index = faiss.index_factory(512, "HNSW64")
   
   # Inner product metric
   index = faiss.index_factory(128, "IVF100,Flat", faiss.METRIC_INNER_PRODUCT)
""")

def demonstrate_index_factory():
    """Demonstrate FAISS Index Factory usage"""
    print("Index Factory Demo:")
    print("""
    import faiss
    import numpy as np
    
    # Prepare test data
    dimension = 128
    nb = 10000
    vectors = np.random.random((nb, dimension)).astype('float32')
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(vectors)
    
    print(f"Dataset: {nb} vectors of dimension {dimension}\\n")
    
    # 1. Simple indexes
    indexes = {
        "Exact Search": faiss.index_factory(dimension, "Flat"),
        "IVF Approximate": faiss.index_factory(dimension, "IVF100,Flat"),
        "Compressed IVF": faiss.index_factory(dimension, "IVF100,PQ16"),
        "HNSW": faiss.index_factory(dimension, "HNSW32")
    }
    
    # 2. Advanced indexes with preprocessing
    advanced_indexes = {
        "PCA + IVF": faiss.index_factory(dimension, "PCA64,IVF50,Flat"),
        "OPQ + IVF + PQ": faiss.index_factory(dimension, "OPQ16,IVF100,PQ16"),
        "Normalized IP": faiss.index_factory(
            dimension, "IVF100,Flat", faiss.METRIC_INNER_PRODUCT
        )
    }
    
    # 3. Train and test each index
    all_indexes = {**indexes, **advanced_indexes}
    
    for name, index in all_indexes.items():
        print(f"Testing {name}:")
        
        # Train if needed
        if not index.is_trained:
            print(f"  Training {name}...")
            index.train(vectors)
        
        # Add vectors
        index.add(vectors)
        print(f"  Added {index.ntotal} vectors")
        
        # Test search
        query = vectors[:1]  # Use first vector as query
        k = 5
        
        start_time = time.time()
        distances, indices = index.search(query, k)
        search_time = time.time() - start_time
        
        print(f"  Search time: {search_time*1000:.2f}ms")
        print(f"  First result distance: {distances[0][0]:.6f}\\n")
    
    # 4. Memory usage comparison
    print("Memory Usage (approximate):")
    memory_estimates = {
        "Flat": nb * dimension * 4,
        "IVF100,Flat": nb * dimension * 4 * 1.1,  # Small overhead
        "IVF100,PQ16": nb * 16,  # Much smaller
        "HNSW32": nb * dimension * 4 * 1.5,  # Graph overhead
        "PCA64,IVF50,Flat": nb * 64 * 4 * 1.1,  # Reduced dimension
    }
    
    for desc, memory in memory_estimates.items():
        print(f"  {desc:15}: {memory/1024/1024:.1f} MB")
    """)

demonstrate_index_factory()

# Question 6: GPU Acceleration
print("\n6. How do you use GPU acceleration with FAISS?")
print("-" * 49)
print("""
GPU ACCELERATION:
FAISS can leverage GPUs for faster search, especially with large datasets.

REQUIREMENTS:
• NVIDIA GPU with CUDA support
• faiss-gpu package installed
• Sufficient GPU memory

GPU INDEX TYPES:
• GPU versions of most CPU indexes
• Automatic memory management
• Multi-GPU support available

BASIC GPU USAGE:
   import faiss
   
   # Check GPU availability
   ngpus = faiss.get_num_gpus()
   print(f"Available GPUs: {ngpus}")
   
   # Create CPU index
   cpu_index = faiss.IndexFlatL2(dimension)
   
   # Move to GPU
   gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index)
   
   # Or create GPU index directly
   res = faiss.StandardGpuResources()
   gpu_index = faiss.GpuIndexFlatL2(res, dimension)

MULTI-GPU USAGE:
   # Use multiple GPUs
   cpu_index = faiss.IndexFlatL2(dimension)
   gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

GPU MEMORY MANAGEMENT:
   # Configure GPU resources
   res = faiss.StandardGpuResources()
   res.setTempMemory(1024 * 1024 * 1024)  # 1GB temp memory
   
WHEN TO USE GPU:
✅ Large datasets (>100K vectors)
✅ High query volume
✅ GPU memory sufficient for index
✅ Batch processing

PERFORMANCE CONSIDERATIONS:
• GPU excels at batch searches
• Memory transfer overhead for small queries
• Index building may be slower on GPU
• Consider GPU memory limitations
""")

def demonstrate_gpu_usage():
    """Demonstrate GPU usage (conceptual)"""
    print("GPU Usage Demo (Conceptual):")
    print("""
    import faiss
    import numpy as np
    import time
    
    # Check GPU availability
    ngpus = faiss.get_num_gpus()
    print(f"Available GPUs: {ngpus}")
    
    if ngpus == 0:
        print("No GPUs available, using CPU only")
        return
    
    # Prepare large dataset
    dimension = 512
    nb = 100000  # 100K vectors
    nq = 1000    # 1K queries
    
    vectors = np.random.random((nb, dimension)).astype('float32')
    queries = np.random.random((nq, dimension)).astype('float32')
    
    print(f"Dataset: {nb} vectors, {nq} queries, {dimension} dimensions")
    
    # 1. CPU Index
    print("\\n1. CPU Index Performance:")
    cpu_index = faiss.IndexFlatL2(dimension)
    cpu_index.add(vectors)
    
    start_time = time.time()
    cpu_distances, cpu_indices = cpu_index.search(queries, 10)
    cpu_time = time.time() - start_time
    print(f"   CPU search time: {cpu_time:.3f} seconds")
    
    # 2. Single GPU Index
    print("\\n2. Single GPU Index Performance:")
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    
    start_time = time.time()
    gpu_distances, gpu_indices = gpu_index.search(queries, 10)
    gpu_time = time.time() - start_time
    print(f"   GPU search time: {gpu_time:.3f} seconds")
    print(f"   Speedup: {cpu_time/gpu_time:.1f}x")
    
    # 3. Multi-GPU Index (if available)
    if ngpus > 1:
        print(f"\\n3. Multi-GPU Index ({ngpus} GPUs):")
        multi_gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
        
        start_time = time.time()
        multi_distances, multi_indices = multi_gpu_index.search(queries, 10)
        multi_time = time.time() - start_time
        print(f"   Multi-GPU search time: {multi_time:.3f} seconds")
        print(f"   Speedup vs CPU: {cpu_time/multi_time:.1f}x")
    
    # 4. GPU Memory Usage
    print("\\n4. GPU Memory Considerations:")
    vector_memory = nb * dimension * 4  # 4 bytes per float32
    print(f"   Vector data size: {vector_memory/1024/1024/1024:.2f} GB")
    
    # GPU resources configuration
    res = faiss.StandardGpuResources()
    res.setTempMemory(512 * 1024 * 1024)  # 512MB temp memory
    
    # Create GPU index with custom resources
    gpu_config = faiss.GpuIndexFlatConfig()
    gpu_config.device = 0  # GPU 0
    gpu_config.useFloat16 = True  # Use half precision to save memory
    
    custom_gpu_index = faiss.GpuIndexFlatL2(res, dimension, gpu_config)
    
    print("   Created GPU index with custom configuration")
    print("   - Using half precision (float16)")
    print("   - Custom memory limits")
    
    # 5. Batch vs Individual Queries
    print("\\n5. Batch vs Individual Query Performance:")
    
    # Individual queries (slow)
    start_time = time.time()
    for i in range(100):  # Test with 100 individual queries
        query = queries[i:i+1]
        distances, indices = gpu_index.search(query, 10)
    individual_time = time.time() - start_time
    
    # Batch queries (fast)
    start_time = time.time()
    batch_queries = queries[:100]
    distances, indices = gpu_index.search(batch_queries, 10)
    batch_time = time.time() - start_time
    
    print(f"   100 individual queries: {individual_time:.3f} seconds")
    print(f"   1 batch of 100 queries: {batch_time:.3f} seconds")
    print(f"   Batch speedup: {individual_time/batch_time:.1f}x")
    """)

demonstrate_gpu_usage()

# ============================================================================
# SECTION 4: PRODUCTION OPTIMIZATION
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 4: PRODUCTION OPTIMIZATION")
print("=" * 60)

# Question 7: Performance Tuning
print("\n7. How do you optimize FAISS performance for production?")
print("-" * 58)
print("""
PERFORMANCE OPTIMIZATION STRATEGIES:

1. INDEX SELECTION:
   • Start simple, then optimize based on requirements
   • Measure actual performance with your data
   • Consider memory vs accuracy tradeoffs

2. PARAMETER TUNING:
   IVF Indexes:
   • nlist: sqrt(N) for N vectors (rule of thumb)
   • nprobe: Start with nlist/10, increase for better accuracy
   
   HNSW Indexes:
   • M: 16-64 (higher = better accuracy, more memory)
   • ef_construction: 200-800 during build
   • ef_search: 16-512 during search
   
   PQ Indexes:
   • m: dimension/4 to dimension/8
   • bits: 8 is most common

3. DATA PREPROCESSING:
   • Normalize vectors for cosine similarity
   • Use PCA to reduce dimensions if possible
   • Remove outliers that might affect clustering

4. MEMORY OPTIMIZATION:
   • Use compressed indexes (PQ, SQ) for large datasets
   • Consider float16 on GPU
   • Monitor memory usage during index building

5. SEARCH OPTIMIZATION:
   • Use batch searches when possible
   • Cache frequently accessed indexes
   • Precompute common queries

PERFORMANCE MONITORING:
   • Track search latency percentiles (p50, p95, p99)
   • Monitor memory usage
   • Measure index build time
   • Track accuracy metrics

COMMON PERFORMANCE ISSUES:
❌ Using wrong index type for dataset size
❌ Not tuning search parameters
❌ Individual queries instead of batching
❌ Not normalizing vectors for cosine similarity
❌ Insufficient training data for clustering
""")

def demonstrate_performance_optimization():
    """Demonstrate performance optimization techniques"""
    print("Performance Optimization Demo:")
    print("""
    import faiss
    import numpy as np
    import time
    from sklearn.metrics import accuracy_score
    
    def benchmark_index(index, vectors, queries, name):
        '''Benchmark index performance'''
        print(f"\\nBenchmarking {name}:")
        
        # Training if needed
        if not index.is_trained:
            train_start = time.time()
            index.train(vectors)
            train_time = time.time() - train_start
            print(f"  Training time: {train_time:.2f}s")
        
        # Adding vectors
        add_start = time.time()
        index.add(vectors)
        add_time = time.time() - add_start
        print(f"  Add time: {add_time:.2f}s")
        
        # Search benchmark
        k = 10
        search_start = time.time()
        distances, indices = index.search(queries, k)
        search_time = time.time() - search_start
        
        print(f"  Search time: {search_time*1000:.1f}ms")
        print(f"  QPS: {len(queries)/search_time:.0f}")
        print(f"  Avg distance: {distances.mean():.4f}")
        
        return {
            'train_time': train_time if not index.is_trained else 0,
            'add_time': add_time,
            'search_time': search_time,
            'qps': len(queries)/search_time
        }
    
    # Prepare realistic dataset
    dimension = 384  # Common embedding size
    nb = 50000       # 50K vectors
    nq = 1000        # 1K queries
    
    np.random.seed(42)
    vectors = np.random.random((nb, dimension)).astype('float32')
    queries = np.random.random((nq, dimension)).astype('float32')
    
    # Normalize for cosine similarity
    faiss.normalize_L2(vectors)
    faiss.normalize_L2(queries)
    
    print(f"Dataset: {nb} vectors, {nq} queries, {dimension}D")
    
    # Test different index configurations
    configs = [
        ("Flat (Exact)", "Flat"),
        ("IVF100 (Default)", "IVF100,Flat"),
        ("IVF500 (More clusters)", "IVF500,Flat"), 
        ("IVF100,PQ32 (Compressed)", "IVF100,PQ32"),
        ("HNSW32 (Graph)", "HNSW32"),
        ("PCA128,IVF100 (Reduced)", "PCA128,IVF100,Flat")
    ]
    
    results = {}
    ground_truth = None
    
    for name, description in configs:
        # Create index
        index = faiss.index_factory(dimension, description, faiss.METRIC_INNER_PRODUCT)
        
        # Configure search parameters
        if hasattr(index, 'nprobe'):
            index.nprobe = 20  # Search more clusters for better accuracy
        if hasattr(index, 'hnsw'):
            index.hnsw.ef_search = 64  # Higher ef_search for better accuracy
        
        # Benchmark
        result = benchmark_index(index, vectors.copy(), queries, name)
        results[name] = result
        
        # Get ground truth from exact search
        if "Flat" in name:
            _, ground_truth_indices = index.search(queries, 10)
            ground_truth = ground_truth_indices
    
    # Performance comparison
    print("\\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"{'Index':<20} {'QPS':<8} {'Search(ms)':<12} {'Memory':<10}")
    print("-"*60)
    
    for name, result in results.items():
        memory_factor = {
            "Flat (Exact)": 1.0,
            "IVF100 (Default)": 1.1,
            "IVF500 (More clusters)": 1.2,
            "IVF100,PQ32 (Compressed)": 0.25,
            "HNSW32 (Graph)": 1.5,
            "PCA128,IVF100 (Reduced)": 0.7
        }
        
        memory = f"{memory_factor.get(name, 1.0):.1f}x"
        print(f"{name:<20} {result['qps']:<8.0f} {result['search_time']*1000:<12.1f} {memory:<10}")
    
    # Parameter tuning example
    print("\\n" + "="*60)
    print("PARAMETER TUNING EXAMPLE")
    print("="*60)
    
    # Test different nprobe values for IVF
    index = faiss.index_factory(dimension, "IVF100,Flat", faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    
    print("\\nIVF nprobe tuning (accuracy vs speed):")
    print(f"{'nprobe':<8} {'QPS':<8} {'Search(ms)':<12} {'Accuracy':<10}")
    print("-"*40)
    
    for nprobe in [1, 5, 10, 20, 50]:
        index.nprobe = nprobe
        
        start_time = time.time()
        distances, indices = index.search(queries, 10)
        search_time = time.time() - start_time
        
        # Calculate accuracy (if we have ground truth)
        if ground_truth is not None:
            # Simplified accuracy: how many of top results match exact search
            accuracy = np.mean([
                len(set(pred[:5]) & set(true[:5])) / 5.0
                for pred, true in zip(indices, ground_truth)
            ])
        else:
            accuracy = 0.0
        
        qps = len(queries) / search_time
        print(f"{nprobe:<8} {qps:<8.0f} {search_time*1000:<12.1f} {accuracy:<10.3f}")
    """)

demonstrate_performance_optimization()

print("\n" + "=" * 80)
print("END OF FAISS GUIDE")
print("Next: Learn about similarity search algorithms and FastAPI integration...")
print("=" * 80)
