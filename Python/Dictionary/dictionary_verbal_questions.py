"""
DICTIONARY VERBAL/CONCEPTUAL INTERVIEW QUESTIONS
===============================================

Comprehensive collection of theoretical questions about Python dictionaries
commonly asked in technical interviews. Focus on understanding concepts,
not just memorizing answers.

Target Audience: All experience levels
Categories: Basic → Performance → Design → Advanced → Applications
"""

# =============================================================================
# QUICK REFERENCE - TOP 20 MOST ASKED QUESTIONS
# =============================================================================

TOP_QUESTIONS = """
TOP 20 MOST FREQUENTLY ASKED VERBAL QUESTIONS:
=============================================

1. What is a dictionary in Python and how does it work?
2. What is the time complexity of dictionary operations?
3. Why can't lists be used as dictionary keys?
4. What's the difference between dict.get() and dict[]?
5. How are hash collisions handled in Python dictionaries?
6. What changed in Python 3.7+ regarding dictionary ordering?
7. When would you use a dictionary vs list vs set?
8. How much memory do dictionaries use compared to lists?
9. What is a load factor and why is it important?
10. How would you implement a dictionary from scratch?
11. What are dictionary views and why are they useful?
12. Explain shallow vs deep copying for dictionaries?
13. How do you handle thread safety with dictionaries?
14. What are the advantages of defaultdict over regular dict?
15. How are dictionaries used in caching strategies?
16. What causes KeyError and how do you prevent it?
17. How do you debug performance issues with dictionaries?
18. What's the difference between dict() and {} syntax?
19. How are dictionaries used in web development?
20. Explain dictionary comprehensions vs traditional loops?
"""

print(TOP_QUESTIONS)

# =============================================================================
# STRUCTURED Q&A BY CATEGORY
# =============================================================================

def print_section(title, questions_dict):
    """Helper function to print Q&A sections"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    for i, (question, answer) in enumerate(questions_dict.items(), 1):
        print(f"\nQ{i}: {question}")
        print("-" * 50)
        print(f"A: {answer}")
        print()

# BASIC CONCEPTS
basic_concepts = {
    "What is a dictionary in Python?": """
A dictionary is a mutable, unordered collection of key-value pairs where:
- Keys must be immutable/hashable (strings, numbers, tuples)
- Values can be any data type
- Provides O(1) average access time using hash table implementation
- Also known as hash map, associative array, or map in other languages
Example: {'name': 'John', 'age': 30, 'city': 'NYC'}
""",

    "How are dictionaries implemented internally?": """
Python dictionaries use hash tables with these key features:
- Hash function converts keys to array indices
- Open addressing with random probing for collision resolution
- Compact representation (Python 3.6+) saves ~20% memory
- Dynamic resizing maintains ~66% load factor for performance
- Combined table stores keys, values, and hash values efficiently
""",

    "Why can't mutable objects be dictionary keys?": """
Dictionary keys must be hashable (immutable) because:
- Hash value must remain constant throughout object's lifetime
- If hash changes, dictionary lookup would fail
- Mutable objects like lists can be modified, changing their hash
- Python requires __hash__() method for dictionary keys
Valid keys: strings, numbers, tuples, frozensets
Invalid keys: lists, dicts, sets, custom mutable objects
""",

    "What's the difference between dict[key] and dict.get(key)?": """
Key differences in access patterns:
- dict[key]: Faster but raises KeyError if key doesn't exist
- dict.get(key): Slower but returns None (or default) if key missing
- dict.get(key, default): Returns custom default for missing keys
- Use [] when you're certain key exists (performance critical code)
- Use get() for safe access with unknown keys (defensive programming)
Performance: [] is ~10-15% faster than get()
""",

    "What are hash collisions and how does Python handle them?": """
Hash collision occurs when different keys produce the same hash value:
- Inevitable due to infinite keys mapping to finite hash table slots
- Python uses open addressing with random probing
- If slot is occupied, tries next available slot using probe sequence
- Good hash functions minimize collisions (SipHash for strings)
- Worst case: O(n) when all keys hash to same value (very rare)
- Resizing occurs when load factor exceeds ~66% to maintain performance
"""
}

print_section("BASIC CONCEPTS", basic_concepts)

# PERFORMANCE AND COMPLEXITY
performance = {
    "What is the time complexity of dictionary operations?": """
Average and worst-case complexities:
- Access (dict[key]): O(1) average, O(n) worst case
- Insert (dict[key] = value): O(1) average, O(n) worst case
- Delete (del dict[key]): O(1) average, O(n) worst case
- Search (key in dict): O(1) average, O(n) worst case
- Iteration: O(n) - must visit all elements

Worst case happens when many hash collisions occur, but this is extremely 
rare in practice due to Python's robust hash function.
""",

    "Why is dictionary access O(1) instead of O(log n)?": """
Hash table provides direct access via computed index:
1. Hash function converts key to array index: hash(key) % table_size
2. Direct array access at computed index: array[index]
3. No searching or traversal required unlike trees or lists
4. Example: dict['name'] → hash('name') → array[42] → value

Compare to binary search tree: O(log n) requires traversing tree levels
Hash table eliminates search by computing exact location
""",

    "How much memory do dictionaries use?": """
Memory overhead breakdown:
- Base dictionary object: ~240 bytes
- Per key-value pair: ~24 bytes (3 pointers × 8 bytes)
- Hash table: 1.5x to 3x number of elements (for low load factor)
- Keys and values: actual object sizes
- Total: dict is memory-efficient for sparse data but has overhead

Example: 1000-item dict ≈ 240 + (1000 × 24) + table overhead ≈ 50-75KB
Compare to list: more compact for dense sequential data
""",

    "When would dictionary operations be O(n)?": """
Worst-case scenarios (extremely rare):
- Pathological input causing all keys to hash to same bucket
- Maliciously crafted keys designed to cause collisions
- Hash function failure (broken or poorly designed)
- DoS attacks targeting hash collision vulnerabilities

Python's mitigation:
- SipHash with random seed prevents collision attacks
- High-quality hash functions minimize natural collisions
- Dynamic resizing maintains low load factor
- In practice, O(1) performance is virtually guaranteed
""",

    "What happens when a dictionary needs to resize?": """
Dynamic resizing process:
1. Triggered when load factor exceeds ~66% (2/3 full)
2. Allocate new hash table (typically 4x current size)
3. Rehash all existing keys to new positions
4. Copy all key-value pairs to new table
5. Free old table memory

Cost: O(n) operation but amortized over many insertions
Frequency: Happens at powers of 2 (8, 16, 32, 64... slots)
Performance: Maintains O(1) average access time long-term
"""
}

print_section("PERFORMANCE & COMPLEXITY", performance)

# PYTHON-SPECIFIC FEATURES
python_specific = {
    "How did dictionary ordering change in Python 3.7+?": """
Major behavioral change:
- Python < 3.7: Order was implementation detail, not guaranteed
- Python 3.7+: Insertion order preserved as language guarantee
- Made possible by compact dict implementation
- Backward compatibility: old code continues to work
- OrderedDict still useful for: move_to_end(), explicit ordering operations

Impact: Can now rely on dict iteration order in algorithms
Memory benefit: ~20% reduction vs older implementation
""",

    "What's the difference between dict() constructor and {} literal?": """
Syntax and performance differences:
- {} literal: Faster for empty dicts and simple cases (~3x faster)
- dict() constructor: More flexible but slower (function call overhead)

Use {} when: Creating empty dict or literal key-values
Use dict() when: Converting iterables, using keyword args, dynamic creation

Examples:
{} → empty dict (fast)
{'a': 1} → literal (fast)
dict(a=1, b=2) → keyword args
dict(pairs) → from iterable
""",

    "What are dictionary views and why are they useful?": """
Dynamic views of dictionary data:
- keys(), values(), items() return view objects (not lists)
- Views automatically update when dictionary changes
- Memory efficient: no copying of data
- Support set-like operations: intersection, union, difference
- Use list(dict.keys()) to get snapshot if needed

Benefits:
- Real-time reflection of dict state
- Memory efficient for large dictionaries
- Enable set operations on keys/items
Example: dict1.keys() & dict2.keys() → common keys
""",

    "When should you use defaultdict vs regular dict?": """
defaultdict advantages:
- Automatic initialization of missing keys
- Cleaner code for grouping/counting operations
- Better performance (no key existence checks)
- Reduces try/except or 'if key in dict' boilerplate

Use defaultdict when:
- Building nested structures
- Grouping items by categories
- Counting occurrences
- Accumulating values

Use regular dict when:
- You want KeyError for missing keys
- Need fine control over initialization
- Working with JSON/external data
""",

    "Explain dictionary comprehensions vs traditional loops?": """
Comprehension advantages:
- More Pythonic and readable for simple transformations
- Often faster (optimized C implementation)
- Memory efficient (no intermediate lists)
- Functional programming style

Traditional loop advantages:
- Better for complex logic with multiple conditions
- Easier debugging (can add print statements)
- More readable for beginners

Performance: Comprehensions typically 20-40% faster
Example: {k: v**2 for k, v in items} vs manual loop
Use comprehensions for simple transformations, loops for complex logic
"""
}

print_section("PYTHON-SPECIFIC FEATURES", python_specific)

# DESIGN AND IMPLEMENTATION
design_implementation = {
    "How would you implement a dictionary from scratch?": """
Core components needed:
1. Hash function: Convert keys to array indices
2. Bucket array: Store key-value pairs
3. Collision resolution: Handle hash conflicts
4. Dynamic resizing: Maintain performance as size grows
5. Key methods: get(), put(), delete(), iterate()

Basic implementation approach:
- Use modulo for index: hash(key) % table_size
- Linear probing for collisions: try next slot if occupied
- Resize when load factor > 0.75
- Store (key, value, hash) tuples for efficiency
""",

    "What is load factor and why is it important?": """
Load factor = number_of_elements / table_size

Importance:
- Measures how 'full' the hash table is
- Controls trade-off between space and time
- Too high (>0.75): More collisions, slower operations
- Too low (<0.5): Wasted memory
- Python maintains ~0.66 for optimal performance

When to resize:
- Resize up when load factor > 0.66
- Resize down when load factor < 0.25 (saves memory)
- Typical strategy: double size when growing
""",

    "Explain different collision resolution strategies?": """
Two main approaches:

1. Chaining (separate chaining):
   - Each bucket stores linked list of colliding items
   - Simple to implement
   - Good for high load factors
   - Cache-unfriendly (pointer chasing)

2. Open Addressing (used by Python):
   - Find next available slot in table
   - Linear probing: check next slot sequentially
   - Quadratic probing: check slots at quadratic intervals
   - Random probing: use secondary hash function
   - Cache-friendly, better performance
""",

    "How do you handle thread safety with dictionaries?": """
Python dicts are not thread-safe:

Synchronization options:
1. threading.Lock: Explicit locking around operations
2. threading.RLock: Reentrant lock for nested calls
3. queue.Queue: Thread-safe alternative for producer-consumer
4. multiprocessing.Manager: Cross-process shared dicts
5. concurrent.futures: Higher-level concurrency

Best practices:
- Read operations generally safe (GIL protection)
- Write operations need explicit synchronization
- Use context managers (with lock:) for safety
- Consider lock-free alternatives for high concurrency
""",

    "What are weak references and how do they relate to dictionaries?": """
Weak references don't prevent garbage collection:

weakref.WeakKeyDictionary:
- Keys can be garbage collected
- Entry automatically removed when key is deleted
- Useful for caches that shouldn't keep objects alive

weakref.WeakValueDictionary:
- Values can be garbage collected
- Entry removed when value is deleted
- Useful for observer patterns, callbacks

Use cases:
- Caches that shouldn't cause memory leaks
- Parent-child relationships without circular references
- Event handlers that should auto-remove
"""
}

print_section("DESIGN & IMPLEMENTATION", design_implementation)

# REAL-WORLD APPLICATIONS
applications = {
    "How are dictionaries used in web development?": """
Common web development uses:
1. HTTP headers: {'Content-Type': 'application/json'}
2. JSON parsing: API requests/responses
3. Session storage: user state between requests
4. Configuration: app settings, environment variables
5. Database results: ORM query results
6. Template context: data passed to HTML templates
7. Form data: POST request parameters
8. Caching: storing computed results

Framework examples:
- Django: request.POST, request.GET
- Flask: session, request.json
- FastAPI: JSON request/response bodies
""",

    "Explain caching strategies using dictionaries?": """
Dictionary-based caching patterns:

1. Simple cache: dict with size limit
2. LRU cache: OrderedDict + move_to_end()
3. TTL cache: timestamps for expiration
4. Write-through: update cache and storage simultaneously
5. Write-back: update cache first, storage later
6. Cache-aside: application manages cache

Implementation considerations:
- Memory limits: prevent unbounded growth
- Eviction policies: LRU, LFU, FIFO
- Thread safety: locks for concurrent access
- Serialization: for persistent caches
""",

    "How do dictionaries help in algorithm optimization?": """
Optimization techniques:

1. Replace O(n) searches with O(1) lookups:
   - Two Sum: store complements in dict
   - Group Anagrams: use sorted string as key

2. Memoization for dynamic programming:
   - Cache subproblem results
   - Fibonacci: memo[n] = fib(n-1) + fib(n-2)

3. Frequency counting:
   - Character frequency: Counter or manual dict
   - Top K elements: count then sort

4. Index mapping:
   - Store positions for fast access
   - Longest substring without repeating characters

5. State tracking:
   - Visited nodes in graph traversal
   - Current configuration in search problems
""",

    "How are dictionaries used in distributed systems?": """
Distributed computing applications:

1. Consistent hashing:
   - Map keys to servers using hash rings
   - Minimize data movement when servers added/removed
   - Used by: Amazon DynamoDB, Apache Cassandra

2. Distributed hash tables (DHT):
   - P2P systems like BitTorrent
   - Chord, Kademlia protocols

3. Sharding strategies:
   - Partition data across databases
   - Hash-based sharding: hash(key) % num_shards

4. Configuration management:
   - Service discovery: map service names to addresses
   - Feature flags: enable/disable functionality

5. Load balancing:
   - Session affinity: map users to servers
   - Request routing based on keys
""",

    "Describe relationship between dictionaries and database indexing?": """
Similar concepts in different contexts:

Hash indexes in databases:
- Same principle as Python dictionaries
- O(1) lookup for equality conditions
- Good for exact match queries
- Poor for range queries

B-tree indexes:
- O(log n) lookup but support range queries
- Better for sorting, range scans
- Most common database index type

Trade-offs:
- Hash: faster equality, no range support
- B-tree: slower individual lookup, range support
- Memory vs disk: hash better in-memory, B-tree for disk

Query optimization:
- Database chooses index based on query type
- WHERE id = 123 → hash index
- WHERE age BETWEEN 20 AND 30 → B-tree index
"""
}

print_section("REAL-WORLD APPLICATIONS", applications)

# DEBUGGING AND TROUBLESHOOTING
debugging = {
    "How do you debug performance issues with dictionaries?": """
Diagnostic approaches:

1. Profiling tools:
   - cProfile: function-level timing
   - timeit: micro-benchmarks
   - memory_profiler: memory usage tracking

2. Performance analysis:
   - Check for hash collision patterns
   - Measure key distribution uniformity
   - Analyze key types and hash quality
   - Monitor load factor and resize frequency

3. Optimization strategies:
   - Use simpler key types (strings vs complex objects)
   - Pre-compute frequently used keys
   - Consider alternative data structures
   - Implement custom __hash__ methods

4. Memory analysis:
   - sys.getsizeof() for size measurement
   - tracemalloc for memory allocation tracking
   - Look for memory leaks in long-running caches
""",

    "What causes KeyError and how do you prevent it?": """
KeyError occurs when accessing non-existent keys:

Prevention strategies:
1. Use get() with defaults: dict.get(key, default_value)
2. Check membership first: if key in dict: ...
3. Use try/except: try/except KeyError
4. Use setdefault(): dict.setdefault(key, default)
5. Use defaultdict: automatically creates missing keys

Best practices:
- Use get() for optional values
- Use [] when you're certain key exists
- Document assumptions about key existence
- Validate input data before dictionary access
- Use type hints to catch errors early
""",

    "How do you handle dictionary mutation during iteration?": """
RuntimeError prevention:

Problem: 'dictionary changed size during iteration'

Safe patterns:
1. Iterate over copy: for key in list(dict.keys())
2. Collect changes first: keys_to_delete = [...]; then delete
3. Use dict.items() copy: for k, v in list(dict.items())
4. Build new dict: new_dict = {k: v for k, v in old_dict.items() if condition}

Unsafe patterns:
- for key in dict: del dict[other_key]  # DON'T DO THIS
- for key in dict: dict[new_key] = value  # DON'T DO THIS

Performance: copying keys is expensive for large dicts
Alternative: use while loop with manual iteration control
""",

    "What are common memory leak patterns with dictionaries?": """
Memory leak scenarios:

1. Unbounded caches:
   - Dict grows without size limits
   - Solution: implement LRU eviction

2. Circular references:
   - Objects reference each other through dict
   - Solution: use weak references

3. Event handlers not removed:
   - Callback dicts holding references
   - Solution: explicit cleanup on unsubscribe

4. Large objects as values:
   - Keeping references to expensive objects
   - Solution: store IDs, fetch when needed

5. Global dictionaries:
   - Module-level dicts that never shrink
   - Solution: periodic cleanup, weak references

Detection tools:
- memory_profiler for growth tracking
- objgraph for reference analysis
- gc module for garbage collection debugging
""",

    "How do you optimize dictionary access in performance-critical code?": """
Optimization techniques:

1. Minimize hash computation:
   - Use simple, consistent key types
   - Cache complex keys if computed repeatedly
   - Avoid dynamic key generation in loops

2. Access pattern optimization:
   - Use [] when key existence is guaranteed
   - Pre-compute frequently accessed keys
   - Group related operations together

3. Memory optimization:
   - Use __slots__ in custom key objects
   - Consider intern() for string keys
   - Monitor memory fragmentation

4. Alternative data structures:
   - array.array for numeric data
   - bytes objects for binary keys
   - Custom hash implementations

5. Profiling and measurement:
   - Profile before optimizing
   - Use timeit for micro-benchmarks
   - Measure actual impact in production
"""
}

print_section("DEBUGGING & TROUBLESHOOTING", debugging)

# =============================================================================
# INTERVIEW PREPARATION STRATEGY
# =============================================================================

print(f"\n{'='*60}")
print("INTERVIEW PREPARATION STRATEGY")
print(f"{'='*60}")

preparation_strategy = """
HOW TO PREPARE FOR VERBAL DICTIONARY QUESTIONS:
==============================================

1. UNDERSTAND THE FUNDAMENTALS:
   □ Hash table implementation and collision resolution
   □ Time/space complexity analysis
   □ Python-specific features and changes
   □ Memory management and optimization

2. PRACTICE EXPLAINING CONCEPTS:
   □ Use simple analogies (phone book, filing cabinet)
   □ Draw diagrams when helpful
   □ Connect to real-world applications
   □ Explain trade-offs and alternatives

3. PREPARE FOR FOLLOW-UP QUESTIONS:
   - "Can you give an example?"
   - "How would you implement this?"
   - "What are the trade-offs?"
   - "When would you use something else?"

4. CONNECT THEORY TO PRACTICE:
   □ Share experiences from real projects
   □ Discuss optimization challenges
   □ Mention debugging scenarios
   □ Show production considerations

5. STUDY PROGRESSION:
   Day 1-2: Basic concepts and implementation
   Day 3-4: Performance and Python-specific features
   Day 5-6: Advanced topics and applications
   Day 7: Review and practice explanations

SAMPLE GOOD ANSWERS:
===================

Q: "Why are dictionaries faster than lists for lookups?"

Good Answer Structure:
1. Direct answer: "Because dictionaries use hash tables for O(1) access"
2. Explanation: "Hash function computes exact array index from key"
3. Comparison: "Lists need O(n) linear search to find elements"
4. Example: "Like using building directory vs checking every apartment"
5. Caveats: "Average case O(1), worst case O(n) with collisions"

COMMON MISTAKES TO AVOID:
========================
❌ Memorizing answers without understanding
❌ Being too vague or too detailed
❌ Not providing concrete examples
❌ Ignoring edge cases and limitations
❌ Not connecting to practical applications

SUCCESS INDICATORS:
==================
✅ Can explain concepts in simple terms
✅ Provides relevant examples spontaneously
✅ Discusses trade-offs and alternatives
✅ Connects theory to real-world usage
✅ Comfortable with follow-up questions
✅ Shows understanding of implementation details
"""

print(preparation_strategy)

print(f"\n{'='*60}")
print("VERBAL DICTIONARY QUESTIONS GUIDE COMPLETE!")
print("Master these concepts for confident interview performance!")
print(f"{'='*60}")

# =============================================================================
# QUICK REFERENCE SUMMARY
# =============================================================================

quick_reference = """
QUICK REFERENCE CHEAT SHEET:
===========================

KEY CONCEPTS:
• Hash table implementation with O(1) average access
• Open addressing with random probing for collisions
• Load factor ~66% maintained through dynamic resizing
• Insertion order preserved since Python 3.7+

PERFORMANCE:
• Access/Insert/Delete: O(1) average, O(n) worst case
• Memory: ~240 bytes + 24 bytes per key-value pair
• Resizing: O(n) operation but amortized

BEST PRACTICES:
• Use get() for safe access, [] for guaranteed keys
• Use defaultdict for automatic initialization
• Consider thread safety in concurrent applications
• Monitor memory usage for large caches

COMMON PITFALLS:
• KeyError with missing keys
• Mutation during iteration
• Using mutable objects as keys
• Unbounded cache growth
"""

print(quick_reference)
