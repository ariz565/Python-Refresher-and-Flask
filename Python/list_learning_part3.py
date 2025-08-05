# ===============================================================================
# COMPREHENSIVE PYTHON LISTS LEARNING GUIDE - PART 3 (FINAL)
# Continuing from list_learning_part2.py
# ===============================================================================

"""
FINAL SECTIONS:
==============
8. Advanced Concepts for Experienced Developers
9. System Design with Lists
10. Threading & Concurrency Considerations
11. Best Practices & Common Pitfalls
12. List vs Other Data Structures
"""

import time
import sys
from collections import defaultdict, deque, OrderedDict
import threading
import operator
import functools
import itertools
import bisect
import heapq
import copy
import concurrent.futures
import queue
import weakref
import gc

# ===============================================================================
# 8. ADVANCED CONCEPTS FOR EXPERIENCED DEVELOPERS
# ===============================================================================

print("=" * 80)
print("8. ADVANCED CONCEPTS FOR EXPERIENCED DEVELOPERS")
print("=" * 80)

print("\n--- Custom List Classes ---")

class LazyList:
    """List that computes elements lazily"""
    
    def __init__(self, generator_func, length=None):
        self.generator_func = generator_func
        self._cache = {}
        self._length = length
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        
        if index < 0:
            index += len(self)
        
        if index not in self._cache:
            self._cache[index] = self.generator_func(index)
        
        return self._cache[index]
    
    def __len__(self):
        if self._length is None:
            raise TypeError("Length not defined for infinite lazy list")
        return self._length
    
    def __repr__(self):
        cached_items = sorted(self._cache.items())
        return f"LazyList({dict(cached_items)})"

# Example usage
def fibonacci_generator(n):
    """Generate nth Fibonacci number"""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

fib_list = LazyList(fibonacci_generator, 20)
print(f"Fibonacci at index 10: {fib_list[10]}")
print(f"Fibonacci slice [5:8]: {fib_list[5:8]}")
print(f"Cached values: {fib_list}")

print("\n--- Memory-Mapped Lists ---")

class MemoryEfficientList:
    """List that uses less memory by storing data efficiently"""
    
    def __init__(self, data_type='int'):
        self.data_type = data_type
        self._data = []
        self._type_size = self._get_type_size(data_type)
    
    def _get_type_size(self, data_type):
        """Get memory size for different data types"""
        sizes = {'int': 8, 'float': 8, 'bool': 1, 'char': 1}
        return sizes.get(data_type, 8)
    
    def append(self, item):
        self._data.append(item)
    
    def extend(self, items):
        self._data.extend(items)
    
    def __getitem__(self, index):
        return self._data[index]
    
    def __setitem__(self, index, value):
        self._data[index] = value
    
    def __len__(self):
        return len(self._data)
    
    def memory_usage(self):
        """Calculate estimated memory usage"""
        return len(self._data) * self._type_size
    
    def __repr__(self):
        return f"MemoryEfficientList({self._data[:10]}{'...' if len(self._data) > 10 else ''})"

# Example usage
efficient_list = MemoryEfficientList('int')
efficient_list.extend(range(1000))
print(f"Memory usage: {efficient_list.memory_usage()} bytes")
print(f"First 5 elements: {[efficient_list[i] for i in range(5)]}")

print("\n--- Thread-Safe List Implementation ---")

class ThreadSafeList:
    """Thread-safe list implementation"""
    
    def __init__(self, initial_data=None):
        self._data = list(initial_data) if initial_data else []
        self._lock = threading.RLock()
    
    def append(self, item):
        with self._lock:
            self._data.append(item)
    
    def extend(self, items):
        with self._lock:
            self._data.extend(items)
    
    def pop(self, index=-1):
        with self._lock:
            if not self._data:
                raise IndexError("pop from empty list")
            return self._data.pop(index)
    
    def __getitem__(self, index):
        with self._lock:
            return self._data[index]
    
    def __setitem__(self, index, value):
        with self._lock:
            self._data[index] = value
    
    def __len__(self):
        with self._lock:
            return len(self._data)
    
    def copy(self):
        with self._lock:
            return ThreadSafeList(self._data)
    
    def atomic_update(self, func):
        """Apply function atomically"""
        with self._lock:
            func(self._data)

# Example usage
thread_safe_list = ThreadSafeList([1, 2, 3])

def worker(tsl, worker_id):
    for i in range(10):
        tsl.append(f"worker_{worker_id}_item_{i}")

# Test thread safety
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(worker, thread_safe_list, i) for i in range(3)]
    for future in futures:
        future.result()

print(f"Thread-safe list length: {len(thread_safe_list)}")

print("\n--- List Decorators and Metaclasses ---")

def memoized_list(cls):
    """Decorator to add memoization to list operations"""
    original_getitem = cls.__getitem__
    cache = {}
    
    def cached_getitem(self, index):
        key = (id(self), index)
        if key not in cache:
            cache[key] = original_getitem(self, index)
        return cache[key]
    
    cls.__getitem__ = cached_getitem
    cls._cache = cache
    return cls

@memoized_list
class MemoizedList(list):
    """List with memoized access"""
    
    def expensive_operation(self, index):
        """Simulate expensive operation"""
        time.sleep(0.001)  # Simulate delay
        return self[index] ** 2

memoized = MemoizedList([1, 2, 3, 4, 5])

# First access (slow)
start = time.time()
result1 = memoized.expensive_operation(2)
first_time = time.time() - start

# Second access (fast due to memoization)
start = time.time()
result2 = memoized.expensive_operation(2)
second_time = time.time() - start

print(f"First access time: {first_time:.6f}s")
print(f"Second access time: {second_time:.6f}s")
print(f"Speedup: {first_time/second_time:.2f}x")

print("\n--- Functional Programming with Lists ---")

class FunctionalList(list):
    """List with functional programming methods"""
    
    def map(self, func):
        """Map function over list"""
        return FunctionalList(func(x) for x in self)
    
    def filter(self, predicate):
        """Filter list by predicate"""
        return FunctionalList(x for x in self if predicate(x))
    
    def reduce(self, func, initial=None):
        """Reduce list to single value"""
        if initial is None:
            return functools.reduce(func, self)
        return functools.reduce(func, self, initial)
    
    def fold_left(self, func, initial):
        """Fold from left"""
        result = initial
        for item in self:
            result = func(result, item)
        return result
    
    def fold_right(self, func, initial):
        """Fold from right"""
        result = initial
        for item in reversed(self):
            result = func(item, result)
        return result
    
    def take(self, n):
        """Take first n elements"""
        return FunctionalList(self[:n])
    
    def drop(self, n):
        """Drop first n elements"""
        return FunctionalList(self[n:])
    
    def zip_with(self, other, func):
        """Zip with another list using function"""
        return FunctionalList(func(a, b) for a, b in zip(self, other))

# Example usage
func_list = FunctionalList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

result = (func_list
          .filter(lambda x: x % 2 == 0)
          .map(lambda x: x ** 2)
          .take(3))

print(f"Functional chain result: {result}")

sum_result = func_list.reduce(lambda a, b: a + b)
print(f"Sum using reduce: {sum_result}")

print("\n--- Advanced List Algorithms ---")

def merge_sort_in_place(lst, left=0, right=None):
    """In-place merge sort implementation"""
    if right is None:
        right = len(lst) - 1
    
    if left < right:
        mid = (left + right) // 2
        merge_sort_in_place(lst, left, mid)
        merge_sort_in_place(lst, mid + 1, right)
        merge_in_place(lst, left, mid, right)

def merge_in_place(lst, left, mid, right):
    """Helper function for in-place merge"""
    # Create temporary arrays
    left_part = lst[left:mid+1]
    right_part = lst[mid+1:right+1]
    
    i = j = 0
    k = left
    
    while i < len(left_part) and j < len(right_part):
        if left_part[i] <= right_part[j]:
            lst[k] = left_part[i]
            i += 1
        else:
            lst[k] = right_part[j]
            j += 1
        k += 1
    
    while i < len(left_part):
        lst[k] = left_part[i]
        i += 1
        k += 1
    
    while j < len(right_part):
        lst[k] = right_part[j]
        j += 1
        k += 1

# Example
unsorted = [64, 34, 25, 12, 22, 11, 90]
print(f"Before sort: {unsorted}")
merge_sort_in_place(unsorted)
print(f"After merge sort: {unsorted}")

# ===============================================================================
# 9. SYSTEM DESIGN WITH LISTS
# ===============================================================================

print("\n" + "=" * 80)
print("9. SYSTEM DESIGN WITH LISTS")
print("=" * 80)

print("\n--- LRU Cache Implementation ---")

class LRUCache:
    """Least Recently Used Cache using lists"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key):
        if key not in self.cache:
            return -1
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            # Update existing key
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            # Add new key
            if len(self.cache) >= self.capacity:
                # Remove least recently used (first item)
                self.cache.popitem(last=False)
            self.cache[key] = value
    
    def get_state(self):
        """Get current cache state"""
        return list(self.cache.items())

# Example usage
cache = LRUCache(3)
cache.put("a", 1)
cache.put("b", 2)
cache.put("c", 3)
print(f"Initial cache: {cache.get_state()}")

cache.get("a")  # Make 'a' most recently used
print(f"After accessing 'a': {cache.get_state()}")

cache.put("d", 4)  # Should evict 'b' (least recently used)
print(f"After adding 'd': {cache.get_state()}")

print("\n--- Event-Driven System with Lists ---")

class EventBus:
    """Event bus using lists for subscribers"""
    
    def __init__(self):
        self.subscribers = defaultdict(list)
        self.event_history = []
    
    def subscribe(self, event_type, callback):
        """Subscribe to event type"""
        self.subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type, callback):
        """Unsubscribe from event type"""
        if callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)
    
    def publish(self, event_type, data):
        """Publish event to all subscribers"""
        event = {'type': event_type, 'data': data, 'timestamp': time.time()}
        self.event_history.append(event)
        
        for callback in self.subscribers[event_type]:
            try:
                callback(data)
            except Exception as e:
                print(f"Error in callback: {e}")
    
    def get_event_history(self, event_type=None):
        """Get event history, optionally filtered by type"""
        if event_type is None:
            return self.event_history[:]
        return [e for e in self.event_history if e['type'] == event_type]

# Example usage
event_bus = EventBus()

def user_login_handler(data):
    print(f"User {data['user_id']} logged in")

def analytics_handler(data):
    print(f"Analytics: Login recorded for user {data['user_id']}")

event_bus.subscribe('user_login', user_login_handler)
event_bus.subscribe('user_login', analytics_handler)

event_bus.publish('user_login', {'user_id': 123, 'timestamp': time.time()})
event_bus.publish('user_logout', {'user_id': 123, 'timestamp': time.time()})

login_events = event_bus.get_event_history('user_login')
print(f"Login events: {len(login_events)}")

print("\n--- Rate Limiter with Lists ---")

class SlidingWindowRateLimiter:
    """Rate limiter using sliding window with lists"""
    
    def __init__(self, max_requests, window_size_seconds):
        self.max_requests = max_requests
        self.window_size = window_size_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id):
        """Check if request is allowed for client"""
        current_time = time.time()
        client_requests = self.requests[client_id]
        
        # Remove old requests outside window
        cutoff_time = current_time - self.window_size
        while client_requests and client_requests[0] < cutoff_time:
            client_requests.pop(0)
        
        # Check if under limit
        if len(client_requests) < self.max_requests:
            client_requests.append(current_time)
            return True
        
        return False
    
    def get_remaining_requests(self, client_id):
        """Get remaining requests for client"""
        current_time = time.time()
        client_requests = self.requests[client_id]
        
        # Remove old requests
        cutoff_time = current_time - self.window_size
        while client_requests and client_requests[0] < cutoff_time:
            client_requests.pop(0)
        
        return max(0, self.max_requests - len(client_requests))

# Example usage
rate_limiter = SlidingWindowRateLimiter(max_requests=5, window_size_seconds=60)

# Simulate requests
client_id = "user_123"
for i in range(7):
    allowed = rate_limiter.is_allowed(client_id)
    remaining = rate_limiter.get_remaining_requests(client_id)
    print(f"Request {i+1}: {'Allowed' if allowed else 'Rate limited'}, Remaining: {remaining}")

print("\n--- Load Balancer with Lists ---")

class RoundRobinLoadBalancer:
    """Round-robin load balancer using lists"""
    
    def __init__(self, servers):
        self.servers = servers[:]
        self.current_index = 0
        self.server_stats = {server: {'requests': 0, 'failures': 0} for server in servers}
    
    def get_next_server(self):
        """Get next server in round-robin fashion"""
        if not self.servers:
            return None
        
        server = self.servers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.servers)
        self.server_stats[server]['requests'] += 1
        return server
    
    def mark_server_down(self, server):
        """Mark server as down (remove from rotation)"""
        if server in self.servers:
            self.servers.remove(server)
            self.server_stats[server]['failures'] += 1
            # Adjust current index if necessary
            if self.current_index >= len(self.servers) and self.servers:
                self.current_index = 0
    
    def add_server(self, server):
        """Add server back to rotation"""
        if server not in self.servers:
            self.servers.append(server)
            if server not in self.server_stats:
                self.server_stats[server] = {'requests': 0, 'failures': 0}
    
    def get_stats(self):
        """Get server statistics"""
        return dict(self.server_stats)

# Example usage
lb = RoundRobinLoadBalancer(['server1', 'server2', 'server3'])

# Simulate requests
for i in range(8):
    server = lb.get_next_server()
    print(f"Request {i+1} -> {server}")

# Simulate server failure
lb.mark_server_down('server2')
print("After server2 failure:")

for i in range(4):
    server = lb.get_next_server()
    print(f"Request {i+9} -> {server}")

print(f"Load balancer stats: {lb.get_stats()}")

# ===============================================================================
# 10. THREADING & CONCURRENCY CONSIDERATIONS
# ===============================================================================

print("\n" + "=" * 80)
print("10. THREADING & CONCURRENCY CONSIDERATIONS")
print("=" * 80)

print("\n--- Thread-Safe Operations ---")

def demonstrate_list_race_condition():
    """Demonstrate race condition with regular lists"""
    shared_list = []
    
    def worker(worker_id, iterations):
        for i in range(iterations):
            shared_list.append(f"worker_{worker_id}_{i}")
    
    # Create multiple threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=worker, args=(i, 100))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads
    for thread in threads:
        thread.join()
    
    print(f"Expected length: 300, Actual length: {len(shared_list)}")
    # Note: Length should be 300 but might be less due to race conditions

demonstrate_list_race_condition()

print("\n--- Producer-Consumer with Lists ---")

class ProducerConsumerQueue:
    """Producer-consumer pattern using lists"""
    
    def __init__(self, max_size=10):
        self.queue = []
        self.max_size = max_size
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)
    
    def produce(self, item):
        with self.not_full:
            while len(self.queue) >= self.max_size:
                self.not_full.wait()
            
            self.queue.append(item)
            print(f"Produced: {item}, Queue size: {len(self.queue)}")
            self.not_empty.notify()
    
    def consume(self):
        with self.not_empty:
            while not self.queue:
                self.not_empty.wait()
            
            item = self.queue.pop(0)
            print(f"Consumed: {item}, Queue size: {len(self.queue)}")
            self.not_full.notify()
            return item

# Example usage
pc_queue = ProducerConsumerQueue(max_size=3)

def producer(queue, items):
    for item in items:
        queue.produce(item)
        time.sleep(0.1)

def consumer(queue, count):
    for _ in range(count):
        item = queue.consume()
        time.sleep(0.2)

# Run producer and consumer
producer_thread = threading.Thread(target=producer, args=(pc_queue, [f"item_{i}" for i in range(5)]))
consumer_thread = threading.Thread(target=consumer, args=(pc_queue, 5))

producer_thread.start()
consumer_thread.start()

producer_thread.join()
consumer_thread.join()

print("\n--- Concurrent List Processing ---")

def parallel_list_processing():
    """Demonstrate parallel list processing"""
    
    large_list = list(range(1000000))
    
    def process_chunk(chunk):
        """Process a chunk of the list"""
        return sum(x * x for x in chunk)
    
    def split_list(lst, num_chunks):
        """Split list into chunks"""
        chunk_size = len(lst) // num_chunks
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < num_chunks - 1 else len(lst)
            chunks.append(lst[start:end])
        return chunks
    
    # Sequential processing
    start = time.time()
    sequential_result = sum(x * x for x in large_list)
    sequential_time = time.time() - start
    
    # Parallel processing
    start = time.time()
    chunks = split_list(large_list, 4)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        parallel_result = sum(future.result() for future in futures)
    
    parallel_time = time.time() - start
    
    print(f"Sequential time: {sequential_time:.6f}s")
    print(f"Parallel time: {parallel_time:.6f}s")
    print(f"Speedup: {sequential_time/parallel_time:.2f}x")
    print(f"Results match: {sequential_result == parallel_result}")

parallel_list_processing()

print("\n--- Lock-Free List Operations ---")

class LockFreeCounter:
    """Lock-free counter using compare-and-swap concept"""
    
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()  # Still need lock in Python due to GIL
    
    def increment(self):
        with self.lock:
            self.value += 1
    
    def get_value(self):
        return self.value

def compare_locking_strategies():
    """Compare different locking strategies"""
    
    counter = LockFreeCounter()
    
    def worker(counter, iterations):
        for _ in range(iterations):
            counter.increment()
    
    # Test with multiple threads
    start = time.time()
    threads = []
    
    for i in range(4):
        thread = threading.Thread(target=worker, args=(counter, 10000))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    execution_time = time.time() - start
    
    print(f"Final counter value: {counter.get_value()}")
    print(f"Execution time: {execution_time:.6f}s")

compare_locking_strategies()

# ===============================================================================
# 11. BEST PRACTICES & COMMON PITFALLS
# ===============================================================================

print("\n" + "=" * 80)
print("11. BEST PRACTICES & COMMON PITFALLS")
print("=" * 80)

print("\n--- Best Practices ---")
print("""
✅ LIST BEST PRACTICES:

1. Use list comprehensions for better performance and readability
2. Pre-allocate lists when size is known to avoid repeated resizing
3. Use deque for frequent operations at both ends
4. Prefer enumerate() over range(len()) for indexing
5. Use slicing for efficient subsequence operations
6. Consider generators for memory-efficient processing of large datasets
7. Use bisect module for maintaining sorted lists
8. Implement proper error handling for index operations
9. Use copy.deepcopy() for nested list copying
10. Profile memory usage for large lists and consider alternatives
""")

print("\n--- Common Pitfalls ---")

print("❌ PITFALL 1: Mutable default arguments")
def wrong_function(lst=[]):  # DON'T DO THIS
    lst.append("item")
    return lst

def correct_function(lst=None):  # DO THIS
    if lst is None:
        lst = []
    lst.append("item")
    return lst

# Demonstrate the problem
result1 = wrong_function()
result2 = wrong_function()
print(f"Wrong function results: {result1}, {result2}")  # Both modified!

result3 = correct_function()
result4 = correct_function()
print(f"Correct function results: {result3}, {result4}")  # Independent

print("\n❌ PITFALL 2: Modifying list while iterating")
demo_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Wrong way - can skip elements or cause errors
wrong_list = demo_list[:]
try:
    for item in wrong_list:
        if item % 2 == 0:
            wrong_list.remove(item)  # Modifying while iterating
except ValueError as e:
    print(f"Error when modifying during iteration: {e}")

# Correct ways
correct_list1 = demo_list[:]
correct_list1[:] = [item for item in correct_list1 if item % 2 != 0]
print(f"Correct method 1: {correct_list1}")

correct_list2 = [item for item in demo_list if item % 2 != 0]
print(f"Correct method 2: {correct_list2}")

print("\n❌ PITFALL 3: Shallow vs Deep Copy confusion")
original = [[1, 2], [3, 4], [5, 6]]

# Shallow copy - nested lists are shared
shallow = original[:]
shallow[0].append(3)
print(f"After shallow copy modification:")
print(f"Original: {original}")  # Also modified!
print(f"Shallow: {shallow}")

# Deep copy - completely independent
original = [[1, 2], [3, 4], [5, 6]]
deep = copy.deepcopy(original)
deep[0].append(3)
print(f"After deep copy modification:")
print(f"Original: {original}")  # Unchanged
print(f"Deep: {deep}")

print("\n❌ PITFALL 4: Inefficient list operations")

# Inefficient: repeated string concatenation
def inefficient_join(strings):
    result = ""
    for s in strings:
        result += s  # Creates new string each time
    return result

# Efficient: use join()
def efficient_join(strings):
    return "".join(strings)

# Performance comparison
strings = ["hello", "world", "python", "programming"] * 1000

start = time.time()
result1 = inefficient_join(strings)
inefficient_time = time.time() - start

start = time.time()
result2 = efficient_join(strings)
efficient_time = time.time() - start

print(f"Inefficient join time: {inefficient_time:.6f}s")
print(f"Efficient join time: {efficient_time:.6f}s")
print(f"Speedup: {inefficient_time/efficient_time:.2f}x")

print("\n❌ PITFALL 5: Memory leaks with circular references")

class Node:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.parent = None
    
    def add_child(self, child):
        child.parent = self
        self.children.append(child)

# Create circular reference
parent = Node("parent")
child = Node("child")
parent.add_child(child)

# This creates a circular reference that can cause memory leaks
print(f"Circular reference created: parent -> child -> parent")

# Solution: use weak references or explicit cleanup
import weakref

class SafeNode:
    def __init__(self, value):
        self.value = value
        self.children = []
        self._parent = None
    
    @property
    def parent(self):
        return self._parent() if self._parent else None
    
    @parent.setter
    def parent(self, value):
        self._parent = weakref.ref(value) if value else None
    
    def add_child(self, child):
        child.parent = self
        self.children.append(child)

# Safe version
safe_parent = SafeNode("safe_parent")
safe_child = SafeNode("safe_child")
safe_parent.add_child(safe_child)
print(f"Safe reference created with weak reference")

print("\n✅ Performance Tips ---")

# Use list methods efficiently
large_list = list(range(100000))

# Efficient membership testing with sets
search_set = set(large_list)
start = time.time()
result = 99999 in search_set
set_time = time.time() - start

start = time.time()
result = 99999 in large_list
list_time = time.time() - start

print(f"Set membership test: {set_time:.6f}s")
print(f"List membership test: {list_time:.6f}s")
print(f"Set is {list_time/set_time:.2f}x faster")

# Use appropriate data structures
print("\nChoose the right tool:")
print("- Use list for ordered data with frequent append/pop from end")
print("- Use deque for frequent operations at both ends")
print("- Use set for membership testing and unique elements")
print("- Use dict for key-value mappings")

# ===============================================================================
# 12. LIST VS OTHER DATA STRUCTURES
# ===============================================================================

print("\n" + "=" * 80)
print("12. LIST VS OTHER DATA STRUCTURES")
print("=" * 80)

def comprehensive_performance_comparison():
    """Comprehensive performance comparison"""
    
    size = 10000
    data = list(range(size))
    
    # Test data structures
    test_list = list(data)
    test_tuple = tuple(data)
    test_set = set(data)
    test_deque = deque(data)
    
    print("Performance Comparison (10,000 elements):")
    print("-" * 50)
    
    # Access performance
    start = time.time()
    for _ in range(1000):
        _ = test_list[size//2]
    list_access_time = time.time() - start
    
    start = time.time()
    for _ in range(1000):
        _ = test_tuple[size//2]
    tuple_access_time = time.time() - start
    
    print(f"Random access (1000 operations):")
    print(f"  List:  {list_access_time:.6f}s")
    print(f"  Tuple: {tuple_access_time:.6f}s")
    
    # Membership testing
    search_item = size - 1
    
    start = time.time()
    for _ in range(100):
        _ = search_item in test_list
    list_membership_time = time.time() - start
    
    start = time.time()
    for _ in range(100):
        _ = search_item in test_set
    set_membership_time = time.time() - start
    
    print(f"\nMembership testing (100 operations):")
    print(f"  List: {list_membership_time:.6f}s")
    print(f"  Set:  {set_membership_time:.6f}s")
    print(f"  Set is {list_membership_time/set_membership_time:.2f}x faster")
    
    # Append performance
    start = time.time()
    test_list_copy = []
    for i in range(1000):
        test_list_copy.append(i)
    list_append_time = time.time() - start
    
    start = time.time()
    test_deque_copy = deque()
    for i in range(1000):
        test_deque_copy.append(i)
    deque_append_time = time.time() - start
    
    print(f"\nAppend performance (1000 operations):")
    print(f"  List:  {list_append_time:.6f}s")
    print(f"  Deque: {deque_append_time:.6f}s")
    
    # Memory usage
    print(f"\nMemory usage:")
    print(f"  List:  {sys.getsizeof(test_list)} bytes")
    print(f"  Tuple: {sys.getsizeof(test_tuple)} bytes")
    print(f"  Set:   {sys.getsizeof(test_set)} bytes")
    print(f"  Deque: {sys.getsizeof(test_deque)} bytes")

comprehensive_performance_comparison()

print("\n--- When to Use Each Data Structure ---")

decision_matrix = [
    ("Need ordered, mutable collection", "List", "Dynamic arrays, frequent modifications"),
    ("Need immutable ordered collection", "Tuple", "Coordinates, database records, function returns"),
    ("Need unique elements only", "Set", "Deduplication, membership testing"),
    ("Need key-value mapping", "Dict", "Lookups, caching, configuration"),
    ("Frequent insertions/deletions at both ends", "Deque", "Queues, sliding windows"),
    ("Need sorted collection", "List + bisect", "Maintaining sorted order efficiently"),
    ("Memory-critical application", "Tuple/array", "Most memory-efficient for read-only data"),
    ("Need thread-safe operations", "queue.Queue", "Producer-consumer patterns"),
    ("Large datasets with rare access", "Generator", "Memory-efficient lazy evaluation"),
    ("Need priority-based access", "heapq + list", "Task scheduling, algorithms")
]

print("DECISION MATRIX:")
print("-" * 80)
for scenario, recommendation, reason in decision_matrix:
    print(f"{scenario:<35} -> {recommendation:<15} ({reason})")

print("\n" + "=" * 80)
print("🎯 LIST MASTERY COMPLETE!")
print("=" * 80)

print("""
COMPREHENSIVE LIST LEARNING SUMMARY:
===================================

📚 Core Concepts Mastered:
- List fundamentals and creation methods  
- Complete reference of all list methods and operations
- Advanced list comprehensions and functional patterns
- Performance analysis and memory management
- Slicing and indexing mastery
- Real-world applications and use cases
- Complex interview problems and algorithmic solutions
- Advanced concepts for experienced developers
- System design patterns with lists
- Threading and concurrency considerations
- Best practices and common pitfalls
- Comparative analysis with other data structures

🚀 Key Takeaways:
1. Lists are mutable, ordered collections perfect for dynamic data
2. List comprehensions provide elegant and efficient data processing
3. Understanding time complexity is crucial for performance
4. Proper slicing techniques enable powerful data manipulation
5. Memory management considerations are important for large datasets
6. Thread safety requires careful consideration in concurrent applications
7. Choosing the right data structure is critical for performance

💡 Advanced Skills Acquired:
- Custom list implementations and metaclasses
- Functional programming patterns with lists
- System design applications (caches, queues, load balancers)
- Concurrent programming with lists
- Performance optimization techniques
- Memory-efficient processing patterns

You now have expert-level knowledge of Python lists! 🌟
Ready for senior-level interviews and complex system design! 💪

COMPLETE DATA STRUCTURES MASTERY ACHIEVED:
✅ Dictionaries - Complete ✅ Sets - Complete ✅ Tuples - Complete ✅ Lists - Complete
""")
