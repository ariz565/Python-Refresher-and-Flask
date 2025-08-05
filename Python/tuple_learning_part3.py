# ===============================================================================
# COMPREHENSIVE PYTHON TUPLES LEARNING GUIDE - PART 3 (FINAL)
# Continuing from tuple_learning_part2.py
# ===============================================================================

"""
FINAL SECTIONS:
==============
10. Threading & Immutability Benefits
11. Best Practices & Common Pitfalls
12. Tuple vs Other Data Structures
"""

import time
import sys
from collections import namedtuple, defaultdict, Counter
import threading
import operator
from typing import Tuple, NamedTuple
import itertools
import functools
import concurrent.futures
import queue

# ===============================================================================
# 10. THREADING & IMMUTABILITY BENEFITS
# ===============================================================================

print("=" * 80)
print("10. THREADING & IMMUTABILITY BENEFITS")
print("=" * 80)

print("\n--- Thread-Safe Data Sharing ---")

# Tuples are inherently thread-safe for reading
shared_data = (1, 2, 3, 4, 5, "hello", "world")

def worker_function(worker_id, data):
    """Worker that reads from shared tuple"""
    total = sum(x for x in data if isinstance(x, int))
    strings = [x for x in data if isinstance(x, str)]
    print(f"Worker {worker_id}: sum={total}, strings={strings}")
    return total

# Multiple threads can safely read from the same tuple
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(worker_function, i, shared_data) 
        for i in range(3)
    ]
    
    results = [future.result() for future in futures]
    print(f"All workers computed: {results}")

print("\n--- Immutable Message Passing ---")

Message = namedtuple('Message', ['sender', 'recipient', 'content', 'timestamp'])

class MessageQueue:
    def __init__(self):
        self.queue = queue.Queue()
    
    def send_message(self, sender, recipient, content):
        """Send immutable message"""
        message = Message(sender, recipient, content, time.time())
        self.queue.put(message)
    
    def receive_message(self, timeout=1):
        """Receive message"""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

def producer(msg_queue, producer_id):
    """Producer thread sending messages"""
    for i in range(3):
        msg_queue.send_message(f"producer_{producer_id}", "consumer", f"Message {i}")
        time.sleep(0.1)

def consumer(msg_queue, consumer_id):
    """Consumer thread receiving messages"""
    received = []
    while len(received) < 6:  # Expecting 6 messages total
        message = msg_queue.receive_message()
        if message:
            received.append(message)
            print(f"Consumer {consumer_id} received: {message.content} from {message.sender}")
    return received

# Example usage
msg_queue = MessageQueue()

# Start producer and consumer threads
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    # Start 2 producers
    producer_futures = [
        executor.submit(producer, msg_queue, i) for i in range(2)
    ]
    
    # Start 1 consumer
    consumer_future = executor.submit(consumer, msg_queue, 1)
    
    # Wait for completion
    messages = consumer_future.result()
    print(f"Total messages received: {len(messages)}")

print("\n--- Lock-Free Data Structures ---")

class ImmutableStack:
    """Lock-free stack using immutable tuples"""
    
    def __init__(self, items=()):
        self._items = items
    
    def push(self, item):
        """Return new stack with item pushed"""
        return ImmutableStack((item,) + self._items)
    
    def pop(self):
        """Return (item, new_stack) or (None, self) if empty"""
        if not self._items:
            return None, self
        return self._items[0], ImmutableStack(self._items[1:])
    
    def peek(self):
        """Return top item without modifying stack"""
        return self._items[0] if self._items else None
    
    def is_empty(self):
        return len(self._items) == 0
    
    def size(self):
        return len(self._items)
    
    def __repr__(self):
        return f"ImmutableStack{self._items}"

# Example usage - thread-safe operations
stack = ImmutableStack()

def stack_operations(initial_stack, operation_id):
    """Perform stack operations in thread"""
    current_stack = initial_stack
    
    # Push some items
    for i in range(3):
        current_stack = current_stack.push(f"item_{operation_id}_{i}")
    
    # Pop one item
    item, current_stack = current_stack.pop()
    print(f"Operation {operation_id}: popped {item}, stack size: {current_stack.size()}")
    
    return current_stack

# Multiple threads working with immutable stacks
initial_stack = ImmutableStack((1, 2, 3))

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(stack_operations, initial_stack, i) 
        for i in range(3)
    ]
    
    final_stacks = [future.result() for future in futures]
    print(f"Original stack unchanged: {initial_stack}")
    for i, stack in enumerate(final_stacks):
        print(f"Final stack {i}: {stack}")

print("\n--- Atomic Updates with Tuples ---")

class AtomicCounter:
    """Atomic counter using tuples for state"""
    
    def __init__(self, initial_value=0):
        self._state = (initial_value, 0)  # (value, version)
        self._lock = threading.Lock()
    
    def increment(self):
        """Atomically increment counter"""
        with self._lock:
            value, version = self._state
            self._state = (value + 1, version + 1)
            return self._state[0]
    
    def get_value(self):
        """Get current value"""
        return self._state[0]
    
    def get_version(self):
        """Get current version"""
        return self._state[1]

def increment_worker(counter, worker_id, increments):
    """Worker that increments counter"""
    for _ in range(increments):
        value = counter.increment()
    print(f"Worker {worker_id}: final value seen: {value}")

# Test atomic counter
counter = AtomicCounter()

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = [
        executor.submit(increment_worker, counter, i, 100) 
        for i in range(5)
    ]
    
    # Wait for all to complete
    for future in futures:
        future.result()

print(f"Final counter value: {counter.get_value()}")
print(f"Final version: {counter.get_version()}")

# ===============================================================================
# 11. BEST PRACTICES & COMMON PITFALLS
# ===============================================================================

print("\n" + "=" * 80)
print("11. BEST PRACTICES & COMMON PITFALLS")
print("=" * 80)

print("\n--- Best Practices ---")
print("""
✅ TUPLE BEST PRACTICES:

1. Use tuples for immutable data that won't change
2. Use named tuples for better code readability
3. Leverage tuple unpacking for clean code
4. Use tuples as dictionary keys when you need composite keys
5. Prefer tuples over lists for fixed collections
6. Use tuples for function returns with multiple values
7. Take advantage of tuple's thread-safety for concurrent code
8. Use tuples for coordinates, database records, and configurations
9. Consider memory efficiency - tuples use less memory than lists
10. Use tuple's immutability for defensive programming
""")

print("\n--- Common Pitfalls ---")

print("❌ PITFALL 1: Forgetting comma in single-element tuple")
# Wrong - this is just an integer in parentheses
not_a_tuple = (42)
print(f"Not a tuple: {not_a_tuple}, type: {type(not_a_tuple)}")

# Correct - need comma for single-element tuple
single_tuple = (42,)
print(f"Single tuple: {single_tuple}, type: {type(single_tuple)}")

print("\n❌ PITFALL 2: Trying to modify tuple elements")
try:
    sample_tuple = (1, 2, 3)
    # This will raise TypeError
    sample_tuple[0] = 10
except TypeError as e:
    print(f"Error: {e}")

# Correct way - create new tuple
original = (1, 2, 3)
modified = (10,) + original[1:]
print(f"Original: {original}")
print(f"Modified: {modified}")

print("\n❌ PITFALL 3: Mutable objects in tuples")
# Tuple is immutable, but objects inside can be mutable
tuple_with_list = ([1, 2, 3], [4, 5, 6])
print(f"Before: {tuple_with_list}")

# This modifies the list inside the tuple
tuple_with_list[0].append(4)
print(f"After: {tuple_with_list}")

# Better approach - use immutable objects
tuple_with_tuples = ((1, 2, 3), (4, 5, 6))
print(f"Fully immutable: {tuple_with_tuples}")

print("\n❌ PITFALL 4: Performance anti-patterns")

# Inefficient: repeated concatenation
def build_tuple_wrong(n):
    result = ()
    for i in range(n):
        result = result + (i,)  # Creates new tuple each time
    return result

# Efficient: build list then convert
def build_tuple_right(n):
    result_list = []
    for i in range(n):
        result_list.append(i)
    return tuple(result_list)

# Performance comparison
import time

n = 1000
start = time.time()
tuple1 = build_tuple_wrong(n)
wrong_time = time.time() - start

start = time.time()
tuple2 = build_tuple_right(n)
right_time = time.time() - start

print(f"Wrong approach time: {wrong_time:.6f}s")
print(f"Right approach time: {right_time:.6f}s")
print(f"Speedup: {wrong_time/right_time:.2f}x")

print("\n❌ PITFALL 5: Confusing tuple unpacking scope")

def get_coordinates():
    return 10, 20, 30

# This works
x, y, z = get_coordinates()
print(f"Unpacked: x={x}, y={y}, z={z}")

# This fails if number of variables doesn't match
try:
    a, b = get_coordinates()  # Too few variables
except ValueError as e:
    print(f"Unpacking error: {e}")

# Correct ways to handle varying returns
# Use star expression
first, *rest = get_coordinates()
print(f"First: {first}, Rest: {rest}")

# Or slice the tuple
coords = get_coordinates()
x, y = coords[:2]  # Take only first two
print(f"First two coordinates: x={x}, y={y}")

print("\n✅ Performance Optimization Tips ---")

# Use tuple() constructor efficiently
# Efficient for small known sequences
small_tuple = tuple([1, 2, 3])

# For large sequences, avoid intermediate lists when possible
# Instead of: tuple([x for x in range(10000)])
# Use: tuple(range(10000))

# Tuple unpacking is faster than indexing
coords = (10, 20, 30)

# Slower
# x = coords[0]
# y = coords[1]
# z = coords[2]

# Faster
x, y, z = coords

print("\n✅ Memory Optimization Tips ---")

# Tuples are more memory efficient than lists
sample_data = range(1000)
as_tuple = tuple(sample_data)
as_list = list(sample_data)

print(f"Tuple memory: {sys.getsizeof(as_tuple)} bytes")
print(f"List memory: {sys.getsizeof(as_list)} bytes")
print(f"Memory savings: {sys.getsizeof(as_list) - sys.getsizeof(as_tuple)} bytes")

# Use slots with namedtuples for even better memory efficiency
class EfficientPoint(NamedTuple):
    x: float
    y: float
    z: float = 0.0

# vs regular class
class RegularPoint:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

efficient_point = EfficientPoint(1.0, 2.0, 3.0)
regular_point = RegularPoint(1.0, 2.0, 3.0)

print(f"Efficient point memory: {sys.getsizeof(efficient_point)} bytes")
print(f"Regular point memory: {sys.getsizeof(regular_point)} bytes")

# ===============================================================================
# 12. TUPLE VS OTHER DATA STRUCTURES
# ===============================================================================

print("\n" + "=" * 80)
print("12. TUPLE VS OTHER DATA STRUCTURES")
print("=" * 80)

print("\n--- Tuple vs List Comparison ---")

def compare_structures():
    size = 10000
    data = list(range(size))
    
    # Creation time
    start = time.time()
    tuple_data = tuple(data)
    tuple_creation_time = time.time() - start
    
    start = time.time()
    list_data = list(data)
    list_creation_time = time.time() - start
    
    # Access time
    start = time.time()
    for _ in range(1000):
        _ = tuple_data[size//2]
    tuple_access_time = time.time() - start
    
    start = time.time()
    for _ in range(1000):
        _ = list_data[size//2]
    list_access_time = time.time() - start
    
    # Memory usage
    tuple_memory = sys.getsizeof(tuple_data)
    list_memory = sys.getsizeof(list_data)
    
    print(f"Creation time - Tuple: {tuple_creation_time:.6f}s, List: {list_creation_time:.6f}s")
    print(f"Access time - Tuple: {tuple_access_time:.6f}s, List: {list_access_time:.6f}s")
    print(f"Memory usage - Tuple: {tuple_memory} bytes, List: {list_memory} bytes")

compare_structures()

print("\n--- When to Use Each Data Structure ---")

comparison_table = """
DATA STRUCTURE SELECTION GUIDE:
===============================

TUPLE:
✅ Use when: Immutable data, fixed collections, coordinates, database records
✅ Benefits: Memory efficient, hashable, thread-safe, faster iteration
❌ Limitations: Cannot modify, no dynamic sizing

LIST:
✅ Use when: Dynamic collections, need to modify elements, variable size
✅ Benefits: Mutable, many built-in methods, dynamic sizing
❌ Limitations: More memory, not hashable, not thread-safe

SET:
✅ Use when: Unique elements, membership testing, mathematical operations
✅ Benefits: O(1) membership testing, eliminates duplicates
❌ Limitations: Unordered, no indexing, elements must be hashable

DICT:
✅ Use when: Key-value relationships, fast lookups, mapping data
✅ Benefits: O(1) key lookup, flexible keys, ordered (Python 3.7+)
❌ Limitations: Keys must be hashable, more memory overhead
"""

print(comparison_table)

print("\n--- Practical Decision Matrix ---")

scenarios = [
    ("Storing x,y coordinates", "Tuple", "Immutable, fixed size, natural grouping"),
    ("User shopping cart items", "List", "Dynamic, items added/removed frequently"),
    ("Unique user IDs", "Set", "No duplicates, fast membership testing"),
    ("User profile data", "Dict", "Key-value mapping, flexible structure"),
    ("Database table row", "NamedTuple", "Immutable, field names, hashable"),
    ("Function with multiple returns", "Tuple", "Natural grouping, immutable result"),
    ("Configuration settings", "NamedTuple", "Immutable, readable field access"),
    ("Graph vertices", "Set", "Unique elements, mathematical operations"),
    ("Cache key", "Tuple", "Immutable, hashable, composite key"),
    ("Queue of tasks", "List", "Dynamic, ordered, FIFO operations")
]

print("SCENARIO DECISION MATRIX:")
print("-" * 70)
for scenario, choice, reason in scenarios:
    print(f"{scenario:<25} -> {choice:<12} ({reason})")

print("\n--- Performance Benchmarks ---")

def benchmark_operations():
    """Benchmark common operations across data structures"""
    
    size = 10000
    data = list(range(size))
    
    # Test data structures
    test_tuple = tuple(data)
    test_list = list(data)
    test_set = set(data)
    
    # Membership testing
    search_item = size // 2
    
    start = time.time()
    result = search_item in test_tuple
    tuple_membership_time = time.time() - start
    
    start = time.time()
    result = search_item in test_list
    list_membership_time = time.time() - start
    
    start = time.time()
    result = search_item in test_set
    set_membership_time = time.time() - start
    
    print(f"Membership testing for {size} elements:")
    print(f"  Tuple: {tuple_membership_time:.6f}s")
    print(f"  List:  {list_membership_time:.6f}s")
    print(f"  Set:   {set_membership_time:.6f}s")
    
    # Iteration
    start = time.time()
    for item in test_tuple:
        pass
    tuple_iter_time = time.time() - start
    
    start = time.time()
    for item in test_list:
        pass
    list_iter_time = time.time() - start
    
    print(f"\nIteration over {size} elements:")
    print(f"  Tuple: {tuple_iter_time:.6f}s")
    print(f"  List:  {list_iter_time:.6f}s")

benchmark_operations()

print("\n" + "=" * 80)
print("🎯 TUPLE MASTERY COMPLETE!")
print("=" * 80)

print("""
COMPREHENSIVE TUPLE LEARNING SUMMARY:
===================================

📚 Core Concepts Mastered:
- Tuple fundamentals and creation methods
- All tuple methods and operations
- Advanced unpacking patterns
- Performance characteristics and memory efficiency
- Named tuples and advanced typing
- Real-world applications and use cases
- Complex interview problems and solutions
- Advanced concepts for experienced developers
- System design patterns with tuples
- Threading benefits and immutability
- Best practices and common pitfalls
- Comparative analysis with other data structures

🚀 Key Takeaways:
1. Tuples are immutable, ordered collections perfect for fixed data
2. Named tuples provide structure and readability
3. Tuple unpacking enables elegant Python code
4. Immutability provides thread-safety and defensive programming
5. Memory efficiency makes tuples ideal for large datasets
6. Tuples are hashable and can serve as dictionary keys
7. Essential for coordinates, database records, and configurations

💡 Advanced Skills Acquired:
- Custom tuple-like classes and metaclasses
- Functional programming patterns with tuples
- Event sourcing and system design applications
- Lock-free data structures using immutability
- Performance optimization techniques
- Thread-safe programming patterns

You now have expert-level knowledge of Python tuples! 🌟
Ready for senior-level interviews and complex system design! 💪
""")
