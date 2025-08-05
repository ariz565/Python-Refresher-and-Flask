"""
pgvector with PostgreSQL - Complete Guide for Beginners
Learn how to use PostgreSQL with pgvector extension for vector operations
Covers installation, setup, operations, and real-world examples
"""

print("=" * 80)
print("PGVECTOR WITH POSTGRESQL - INTERVIEW PREPARATION")
print("=" * 80)

# ============================================================================
# SECTION 1: WHAT IS PGVECTOR?
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 1: UNDERSTANDING PGVECTOR")
print("=" * 60)

# Question 1: What is pgvector?
print("\n1. What is pgvector and why use it with PostgreSQL?")
print("-" * 56)
print("""
WHAT IS PGVECTOR?
pgvector is a PostgreSQL extension that adds vector similarity search capabilities 
to your existing PostgreSQL database.

KEY BENEFITS:
✅ Use familiar SQL syntax for vector operations
✅ ACID transactions (Atomicity, Consistency, Isolation, Durability)
✅ Existing PostgreSQL knowledge applies
✅ No need to learn new database system
✅ Combine relational data with vector data
✅ Mature ecosystem (backup, monitoring, scaling)

WHY CHOOSE PGVECTOR?
• You already use PostgreSQL
• You need ACID transactions for vectors
• You want to combine metadata with vectors
• You prefer SQL over specialized APIs
• You need proven reliability

COMPARISON:
Feature              | pgvector | Specialized Vector DB
---------------------|----------|---------------------
Learning curve       | Low      | Medium-High
SQL support          | Full     | Limited/None
ACID transactions    | Yes      | Varies
Ecosystem            | Mature   | Growing
Vector performance   | Good     | Excellent
Metadata queries     | Excellent| Good

WHEN TO USE PGVECTOR:
✅ Small to medium datasets (<10M vectors)
✅ Need complex metadata filtering
✅ Existing PostgreSQL infrastructure
✅ ACID compliance required
✅ Team knows SQL well

WHEN NOT TO USE:
❌ Very large datasets (>100M vectors)
❌ Ultra-high performance requirements
❌ Simple vector-only operations
❌ Distributed architecture needed
""")

# Question 2: Installation and Setup
print("\n2. How do you install and set up pgvector?")
print("-" * 45)
print("""
INSTALLATION METHODS:

1. DOCKER (EASIEST FOR DEVELOPMENT):
   docker run --name postgres-vector \\
     -e POSTGRES_PASSWORD=password \\
     -e POSTGRES_DB=vectordb \\
     -p 5432:5432 \\
     -d pgvector/pgvector:pg15

2. UBUNTU/DEBIAN:
   sudo apt install postgresql-15-pgvector

3. MAC (HOMEBREW):
   brew install pgvector

4. FROM SOURCE:
   git clone https://github.com/pgvector/pgvector.git
   cd pgvector
   make && sudo make install

ENABLE EXTENSION:
   CREATE EXTENSION vector;

VERIFY INSTALLATION:
   SELECT * FROM pg_extension WHERE extname = 'vector';

PYTHON DEPENDENCIES:
   pip install psycopg2-binary  # PostgreSQL adapter
   pip install pgvector         # Vector helpers
   pip install numpy           # For vector operations

CONNECTION EXAMPLE:
   import psycopg2
   
   conn = psycopg2.connect(
       host="localhost",
       port=5432,
       database="vectordb",
       user="postgres",
       password="password"
   )
""")

def demonstrate_connection():
    """Demonstrate database connection (conceptual)"""
    print("Database Connection Demo (Conceptual):")
    print("""
    import psycopg2
    from pgvector.psycopg2 import register_vector
    import numpy as np

    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="vectordb", 
        user="postgres",
        password="password"
    )
    
    # Register vector type
    register_vector(conn)
    
    # Create cursor
    cur = conn.cursor()
    
    # Enable extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Test connection
    cur.execute("SELECT version();")
    print("PostgreSQL version:", cur.fetchone()[0])
    
    # Close connections
    cur.close()
    conn.close()
    """)

demonstrate_connection()

# Question 3: Vector Data Types and Operations
print("\n3. What vector data types and operations does pgvector support?")
print("-" * 66)
print("""
VECTOR DATA TYPE:
The main data type is 'vector' which stores arrays of real numbers.

SYNTAX:
   vector(dimensions)
   
   Examples:
   vector(3)     -- 3-dimensional vector
   vector(768)   -- Common for text embeddings
   vector(1536)  -- OpenAI embedding size

SUPPORTED OPERATIONS:

1. DISTANCE OPERATORS:
   <->   Euclidean distance (L2)
   <#>   Negative dot product  
   <=>   Cosine distance

2. SIMILARITY FUNCTIONS:
   cosine_distance(v1, v2)
   l2_distance(v1, v2)
   inner_product(v1, v2)

3. VECTOR FUNCTIONS:
   vector_dims(v)           -- Get dimensions
   vector_norm(v)           -- Get magnitude
   
4. INDEXING:
   CREATE INDEX ON table USING ivfflat (vector_column vector_ops);
   CREATE INDEX ON table USING hnsw (vector_column vector_ops);

EXAMPLE OPERATIONS:
   -- Find nearest neighbors
   SELECT * FROM items 
   ORDER BY embedding <-> '[1,2,3]' 
   LIMIT 5;
   
   -- Cosine similarity
   SELECT * FROM items 
   ORDER BY embedding <=> '[1,2,3]' 
   LIMIT 5;
   
   -- Within distance threshold
   SELECT * FROM items 
   WHERE embedding <-> '[1,2,3]' < 0.5;
""")

# ============================================================================
# SECTION 2: BASIC OPERATIONS
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 2: BASIC PGVECTOR OPERATIONS")
print("=" * 60)

# Question 4: Creating Tables with Vectors
print("\n4. How do you create tables and insert vector data?")
print("-" * 51)
print("""
TABLE CREATION:

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    embedding vector(768),        -- 768-dimensional vector
    created_at TIMESTAMP DEFAULT NOW(),
    category VARCHAR(50)
);

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    description TEXT,
    price DECIMAL(10,2),
    image_embedding vector(512),  -- Image feature vector
    text_embedding vector(768)    -- Text embedding
);

INSERTING DATA:

-- Direct vector insertion
INSERT INTO documents (title, content, embedding) VALUES 
('Python Tutorial', 'Learn Python programming', '[0.1, 0.2, 0.3, ...]');

-- Using Python with real embeddings
import numpy as np

# Generate sample embedding (in real app, use models like OpenAI, BERT)
embedding = np.random.rand(768).tolist()

cur.execute(
    "INSERT INTO documents (title, content, embedding) VALUES (%s, %s, %s)",
    ("Machine Learning Guide", "Introduction to ML", embedding)
)

BULK INSERT:
-- For large datasets
COPY documents(title, content, embedding) 
FROM '/path/to/vectors.csv' 
WITH CSV HEADER;
""")

def demonstrate_table_creation():
    """Demonstrate table creation and data insertion"""
    print("Table Creation and Data Insertion Demo (SQL):")
    print("""
    -- 1. Create extension
    CREATE EXTENSION IF NOT EXISTS vector;
    
    -- 2. Create table for document embeddings
    CREATE TABLE documents (
        id SERIAL PRIMARY KEY,
        title VARCHAR(255) NOT NULL,
        content TEXT,
        embedding vector(384),  -- Using 384-dim for demo
        category VARCHAR(50),
        created_at TIMESTAMP DEFAULT NOW()
    );
    
    -- 3. Insert sample data with vectors
    INSERT INTO documents (title, content, embedding, category) VALUES 
    (
        'Python Basics', 
        'Introduction to Python programming language',
        '[0.1, 0.2, 0.3, 0.4, ...]',  -- 384 dimensions
        'Programming'
    ),
    (
        'Machine Learning 101',
        'Basics of machine learning algorithms',
        '[0.5, 0.6, 0.7, 0.8, ...]',  -- 384 dimensions  
        'AI'
    ),
    (
        'Cooking Pasta',
        'How to cook perfect pasta at home',
        '[0.9, 0.1, 0.2, 0.3, ...]',  -- 384 dimensions
        'Cooking'
    );
    
    -- 4. Verify data
    SELECT id, title, category, vector_dims(embedding) as dimensions 
    FROM documents;
    """)
    
    # Python code simulation
    print("\nPython Code for Real Insertion:")
    print("""
    import psycopg2
    import numpy as np
    from pgvector.psycopg2 import register_vector
    
    # Connect and register vector
    conn = psycopg2.connect("postgresql://user:pass@localhost/db")
    register_vector(conn)
    
    # Generate sample embedding (384 dimensions)
    embedding = np.random.rand(384).tolist()
    
    # Insert with embedding
    with conn.cursor() as cur:
        cur.execute(
            '''INSERT INTO documents (title, content, embedding, category) 
               VALUES (%s, %s, %s, %s)''',
            (
                "Deep Learning Guide",
                "Comprehensive guide to neural networks", 
                embedding,
                "AI"
            )
        )
    
    conn.commit()
    """)

demonstrate_table_creation()

# Question 5: Querying Vectors
print("\n5. How do you perform vector similarity searches?")
print("-" * 50)
print("""
BASIC SIMILARITY SEARCH:

1. NEAREST NEIGHBORS (k-NN):
   -- Find 5 most similar documents
   SELECT id, title, embedding <-> %s as distance
   FROM documents 
   ORDER BY embedding <-> %s
   LIMIT 5;

2. COSINE SIMILARITY:
   -- Find similar documents using cosine distance
   SELECT id, title, 1 - (embedding <=> %s) as cosine_similarity
   FROM documents 
   ORDER BY embedding <=> %s
   LIMIT 5;

3. THRESHOLD SEARCH:
   -- Find all documents within distance threshold
   SELECT id, title, embedding <-> %s as distance
   FROM documents 
   WHERE embedding <-> %s < 0.3
   ORDER BY distance;

4. COMBINED WITH METADATA:
   -- Search within specific category
   SELECT id, title, category, embedding <-> %s as distance
   FROM documents 
   WHERE category = 'Programming'
   ORDER BY embedding <-> %s
   LIMIT 5;

5. COMPLEX FILTERING:
   -- Search with multiple conditions
   SELECT d.id, d.title, d.category, d.embedding <-> %s as distance
   FROM documents d
   WHERE d.category IN ('Programming', 'AI')
     AND d.created_at > '2023-01-01'
     AND d.embedding <-> %s < 0.5
   ORDER BY distance
   LIMIT 10;

DISTANCE OPERATORS EXPLAINED:
Operator | Distance Type    | Range        | Best for
---------|------------------|--------------|----------
<->      | Euclidean (L2)   | 0 to ∞      | General use
<=>      | Cosine          | 0 to 2       | Text similarity  
<#>      | Negative Dot    | -∞ to ∞      | When magnitude matters
""")

def demonstrate_vector_queries():
    """Demonstrate vector query examples"""
    print("Vector Query Examples (Python + SQL):")
    print("""
    import psycopg2
    import numpy as np
    from pgvector.psycopg2 import register_vector
    
    def find_similar_documents(query_embedding, limit=5):
        '''Find documents similar to query embedding'''
        
        with conn.cursor() as cur:
            # Method 1: Euclidean distance
            cur.execute('''
                SELECT id, title, category, embedding <-> %s as distance
                FROM documents 
                ORDER BY embedding <-> %s
                LIMIT %s
            ''', (query_embedding, query_embedding, limit))
            
            results = cur.fetchall()
            print("Most similar documents (Euclidean):")
            for row in results:
                print(f"ID: {row[0]}, Title: {row[1]}, Distance: {row[3]:.3f}")
    
    def search_with_filters(query_embedding, category, max_distance=0.5):
        '''Search with category filter and distance threshold'''
        
        with conn.cursor() as cur:
            cur.execute('''
                SELECT id, title, category, embedding <-> %s as distance
                FROM documents 
                WHERE category = %s 
                  AND embedding <-> %s < %s
                ORDER BY distance
            ''', (query_embedding, category, query_embedding, max_distance))
            
            results = cur.fetchall()
            print(f"Similar documents in '{category}' category:")
            for row in results:
                print(f"ID: {row[0]}, Title: {row[1]}, Distance: {row[3]:.3f}")
    
    def cosine_similarity_search(query_embedding, limit=5):
        '''Find similar documents using cosine similarity'''
        
        with conn.cursor() as cur:
            cur.execute('''
                SELECT id, title, 
                       1 - (embedding <=> %s) as cosine_similarity
                FROM documents 
                ORDER BY embedding <=> %s
                LIMIT %s
            ''', (query_embedding, query_embedding, limit))
            
            results = cur.fetchall()
            print("Most similar documents (Cosine):")
            for row in results:
                print(f"ID: {row[0]}, Title: {row[1]}, Similarity: {row[2]:.3f}")
    
    # Example usage
    query_text = "machine learning algorithms"
    # In real app, convert text to embedding using model
    query_embedding = np.random.rand(384).tolist()
    
    find_similar_documents(query_embedding)
    search_with_filters(query_embedding, "AI")
    cosine_similarity_search(query_embedding)
    """)

demonstrate_vector_queries()

# ============================================================================
# SECTION 3: INDEXING AND PERFORMANCE
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 3: INDEXING AND PERFORMANCE")
print("=" * 60)

# Question 6: Vector Indexing
print("\n6. How do you create and optimize vector indexes?")
print("-" * 51)
print("""
PGVECTOR INDEX TYPES:

1. IVFFLAT INDEX:
   -- Fast approximate search
   CREATE INDEX ON documents USING ivfflat (embedding vector_l2_ops);
   CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);
   CREATE INDEX ON documents USING ivfflat (embedding vector_ip_ops);

2. HNSW INDEX (Hierarchical Navigable Small World):
   -- More accurate but slower to build
   CREATE INDEX ON documents USING hnsw (embedding vector_l2_ops);
   CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);

INDEX PARAMETERS:

IVFFLAT:
   -- Lists parameter (number of clusters)
   CREATE INDEX ON documents USING ivfflat (embedding vector_l2_ops) 
   WITH (lists = 100);
   
   -- Rule of thumb: lists = rows / 1000
   -- More lists = faster search, slower build

HNSW:
   -- m parameter (max connections per node)
   CREATE INDEX ON documents USING hnsw (embedding vector_l2_ops) 
   WITH (m = 16);
   
   -- ef_construction (quality during build)
   WITH (m = 16, ef_construction = 64);

CHOOSING INDEX TYPE:
Scenario              | Recommended Index | Parameters
----------------------|-------------------|------------
Small dataset (<10K)  | No index needed  | -
Medium dataset (10K-1M)| IVFFLAT          | lists = rows/1000
Large dataset (>1M)   | HNSW             | m=16, ef_construction=64
High accuracy needed  | HNSW             | Higher ef_construction
Fast inserts needed   | IVFFLAT          | Lower lists

PERFORMANCE TIPS:
✅ Create index after bulk data loading
✅ Use appropriate operator class (_l2_ops, _cosine_ops)
✅ Monitor index usage with EXPLAIN
✅ Tune parameters based on your data
""")

def demonstrate_indexing():
    """Demonstrate index creation and performance"""
    print("Vector Indexing Demo (SQL):")
    print("""
    -- 1. Check current performance without index
    EXPLAIN (ANALYZE, BUFFERS) 
    SELECT id, title FROM documents 
    ORDER BY embedding <-> '[0.1, 0.2, ...]' 
    LIMIT 5;
    
    -- Result: Seq Scan (slow for large tables)
    
    -- 2. Create IVFFLAT index for L2 distance
    CREATE INDEX idx_documents_embedding_l2 
    ON documents USING ivfflat (embedding vector_l2_ops) 
    WITH (lists = 100);
    
    -- 3. Create HNSW index for cosine similarity  
    CREATE INDEX idx_documents_embedding_cosine
    ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
    
    -- 4. Check performance with index
    EXPLAIN (ANALYZE, BUFFERS)
    SELECT id, title FROM documents 
    ORDER BY embedding <-> '[0.1, 0.2, ...]' 
    LIMIT 5;
    
    -- Result: Index Scan using idx_documents_embedding_l2 (much faster!)
    
    -- 5. Monitor index usage
    SELECT 
        schemaname,
        tablename, 
        indexname,
        idx_scan as index_scans,
        idx_tup_read as tuples_read
    FROM pg_stat_user_indexes 
    WHERE indexname LIKE '%embedding%';
    
    -- 6. Composite indexes (vector + metadata)
    CREATE INDEX idx_documents_category_embedding 
    ON documents (category, embedding vector_l2_ops);
    
    -- This helps with queries like:
    -- WHERE category = 'AI' ORDER BY embedding <-> '...'
    """)
    
    print("\nIndex Performance Comparison:")
    print("""
    Operation           | No Index | IVFFLAT | HNSW  | Notes
    --------------------|----------|---------|-------|--------
    1K vectors search   | 1ms      | 1ms     | 1ms   | All fast
    10K vectors search  | 10ms     | 2ms     | 1ms   | Index helps
    100K vectors search | 100ms    | 5ms     | 2ms   | Big difference
    1M vectors search   | 1000ms   | 20ms    | 5ms   | Index essential
    Index build time    | 0ms      | 30sec   | 60sec | HNSW slower build
    Memory usage        | Low      | Medium  | High  | Trade-off
    """)

demonstrate_indexing()

# Question 7: Performance Optimization
print("\n7. How do you optimize pgvector performance?")
print("-" * 47)
print("""
PERFORMANCE OPTIMIZATION STRATEGIES:

1. QUERY OPTIMIZATION:
   -- Use LIMIT to reduce results
   SELECT * FROM documents 
   ORDER BY embedding <-> %s 
   LIMIT 10;  -- Don't fetch all results
   
   -- Use distance thresholds
   WHERE embedding <-> %s < 0.5;  -- Filter early
   
   -- Combine filters efficiently
   WHERE category = 'AI' AND embedding <-> %s < 0.3;

2. INDEX TUNING:
   -- IVFFLAT: Tune 'lists' parameter
   -- Rule: lists = sqrt(total_rows)
   -- More lists = faster search, slower build
   
   -- HNSW: Tune 'm' and 'ef_construction'
   -- Higher values = better accuracy, more memory
   
   -- Set appropriate search parameters
   SET ivfflat.probes = 10;  -- Search more lists
   SET hnsw.ef_search = 40;  -- Search more neighbors

3. MEMORY CONFIGURATION:
   -- Increase shared_buffers
   shared_buffers = '256MB'  -- Or 25% of RAM
   
   -- Increase work_mem for sorting
   work_mem = '4MB'
   
   -- Increase maintenance_work_mem for index building
   maintenance_work_mem = '256MB'

4. BATCH OPERATIONS:
   -- Bulk insert data before creating index
   -- Use COPY instead of individual INSERTs
   -- Disable autocommit for bulk operations

5. CONNECTION POOLING:
   -- Use pgbouncer or similar
   -- Reduce connection overhead
   -- Better resource utilization

MONITORING QUERIES:
   -- Find slow vector queries
   SELECT query, mean_exec_time, calls 
   FROM pg_stat_statements 
   WHERE query LIKE '%<->%' 
   ORDER BY mean_exec_time DESC;
   
   -- Check index usage
   SELECT * FROM pg_stat_user_indexes 
   WHERE indexname LIKE '%embedding%';
""")

# ============================================================================
# SECTION 4: REAL-WORLD EXAMPLES
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 4: REAL-WORLD EXAMPLES")
print("=" * 60)

# Question 8: Complete Application Example
print("\n8. How do you build a complete application with pgvector?")
print("-" * 59)
print("""
COMPLETE EXAMPLE: DOCUMENT SEARCH SYSTEM

This example shows a semantic document search system using pgvector.
""")

def demonstrate_complete_application():
    """Complete application example"""
    print("Complete Document Search Application:")
    print("""
    # requirements.txt
    psycopg2-binary==2.9.7
    pgvector==0.2.3
    sentence-transformers==2.2.2
    fastapi==0.104.1
    uvicorn==0.24.0
    
    # 1. Database Setup (setup_db.py)
    import psycopg2
    from pgvector.psycopg2 import register_vector
    
    def setup_database():
        '''Initialize database and tables'''
        conn = psycopg2.connect(
            host="localhost", 
            database="document_search",
            user="postgres", 
            password="password"
        )
        register_vector(conn)
        
        with conn.cursor() as cur:
            # Enable extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create documents table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    title VARCHAR(255) NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(384),
                    category VARCHAR(50),
                    author VARCHAR(100),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            ''')
            
            # Create index for fast similarity search
            cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_documents_embedding 
                ON documents USING hnsw (embedding vector_cosine_ops);
            ''')
            
            # Create metadata index
            cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_documents_category 
                ON documents (category);
            ''')
        
        conn.commit()
        conn.close()
        print("Database setup complete!")
    
    # 2. Document Processor (processor.py)
    from sentence_transformers import SentenceTransformer
    import psycopg2
    from pgvector.psycopg2 import register_vector
    
    class DocumentProcessor:
        def __init__(self, db_config, model_name='all-MiniLM-L6-v2'):
            self.model = SentenceTransformer(model_name)
            self.db_config = db_config
        
        def get_connection(self):
            conn = psycopg2.connect(**self.db_config)
            register_vector(conn)
            return conn
        
        def add_document(self, title, content, category, author):
            '''Add document with embedding to database'''
            # Generate embedding
            text_to_embed = f"{title}. {content}"
            embedding = self.model.encode(text_to_embed).tolist()
            
            # Insert into database
            conn = self.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute('''
                        INSERT INTO documents 
                        (title, content, embedding, category, author)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id;
                    ''', (title, content, embedding, category, author))
                    
                    doc_id = cur.fetchone()[0]
                    conn.commit()
                    return doc_id
            finally:
                conn.close()
        
        def search_documents(self, query, category=None, limit=5):
            '''Search for similar documents'''
            # Generate query embedding
            query_embedding = self.model.encode(query).tolist()
            
            conn = self.get_connection()
            try:
                with conn.cursor() as cur:
                    if category:
                        # Search within category
                        cur.execute('''
                            SELECT id, title, content, category, author,
                                   1 - (embedding <=> %s) as similarity
                            FROM documents 
                            WHERE category = %s
                            ORDER BY embedding <=> %s
                            LIMIT %s;
                        ''', (query_embedding, category, query_embedding, limit))
                    else:
                        # Search all documents
                        cur.execute('''
                            SELECT id, title, content, category, author,
                                   1 - (embedding <=> %s) as similarity
                            FROM documents 
                            ORDER BY embedding <=> %s
                            LIMIT %s;
                        ''', (query_embedding, query_embedding, limit))
                    
                    results = cur.fetchall()
                    return [
                        {
                            'id': row[0],
                            'title': row[1], 
                            'content': row[2][:200] + '...',  # Truncate
                            'category': row[3],
                            'author': row[4],
                            'similarity': float(row[5])
                        }
                        for row in results
                    ]
            finally:
                conn.close()
    
    # 3. FastAPI Application (main.py)
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import Optional, List
    
    app = FastAPI(title="Document Search API")
    
    # Database configuration
    DB_CONFIG = {
        'host': 'localhost',
        'database': 'document_search', 
        'user': 'postgres',
        'password': 'password'
    }
    
    processor = DocumentProcessor(DB_CONFIG)
    
    class DocumentCreate(BaseModel):
        title: str
        content: str
        category: str
        author: str
    
    class SearchQuery(BaseModel):
        query: str
        category: Optional[str] = None
        limit: Optional[int] = 5
    
    @app.post("/documents/")
    async def create_document(doc: DocumentCreate):
        '''Add new document'''
        try:
            doc_id = processor.add_document(
                doc.title, doc.content, doc.category, doc.author
            )
            return {"id": doc_id, "message": "Document added successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/search/")
    async def search_documents(search: SearchQuery):
        '''Search for similar documents'''
        try:
            results = processor.search_documents(
                search.query, search.category, search.limit
            )
            return {"results": results}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}
    
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    
    # 4. Usage Examples (usage.py)
    import requests
    import json
    
    BASE_URL = "http://localhost:8000"
    
    # Add sample documents
    documents = [
        {
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a subset of AI that focuses on algorithms that learn from data...",
            "category": "AI",
            "author": "Dr. Smith"
        },
        {
            "title": "Python Programming Basics", 
            "content": "Python is a versatile programming language used for web development, data science...",
            "category": "Programming",
            "author": "Jane Doe"
        },
        {
            "title": "Deep Learning Neural Networks",
            "content": "Neural networks are computing systems inspired by biological neural networks...",
            "category": "AI", 
            "author": "Prof. Johnson"
        }
    ]
    
    # Add documents
    for doc in documents:
        response = requests.post(f"{BASE_URL}/documents/", json=doc)
        print(f"Added: {response.json()}")
    
    # Search examples
    searches = [
        {"query": "artificial intelligence algorithms", "limit": 3},
        {"query": "programming languages", "category": "Programming", "limit": 2},
        {"query": "neural network training", "category": "AI", "limit": 3}
    ]
    
    for search in searches:
        response = requests.post(f"{BASE_URL}/search/", json=search)
        results = response.json()["results"]
        
        print(f"\\nQuery: {search['query']}")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']} (similarity: {result['similarity']:.3f})")
    """)

demonstrate_complete_application()

print("\n" + "=" * 80)
print("END OF PGVECTOR GUIDE")
print("Next: Learn about FAISS for high-performance vector search...")
print("=" * 80)
