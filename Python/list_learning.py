# ===============================================================================
# COMPREHENSIVE PYTHON LISTS LEARNING GUIDE - ALL CONCEPTS & METHODS
# For Interview Preparation & Advanced Understanding
# ===============================================================================

"""
TABLE OF CONTENTS:
================
1. List Fundamentals & Creation
2. List Methods - Complete Reference
3. List Comprehensions & Advanced Patterns
4. Performance & Memory Management
5. Slicing & Indexing Mastery
6. Real-World Applications
7. Interview Problems & Solutions
8. Advanced Concepts for Experienced Developers
9. System Design with Lists
10. Threading & Concurrency Considerations
11. Best Practices & Common Pitfalls
12. List vs Other Data Structures
"""

import time
import sys
from collections import defaultdict, deque
import threading
import operator
import functools
import itertools
import bisect
import heapq
import copy

# ===============================================================================
# 1. LIST FUNDAMENTALS & CREATION
# ===============================================================================

print("=" * 80)
print("1. LIST FUNDAMENTALS & CREATION")
print("=" * 80)

# What is a List?
"""
A list is a mutable, ordered collection of items in Python.
- Mutable (can be modified after creation)
- Ordered (maintains insertion order)
- Allows duplicate elements
- Elements can be of different types
- Supports indexing and slicing
- Dynamic sizing (can grow/shrink)
- Not hashable (cannot be used as dict keys)
"""

print("\n--- List Creation Methods ---")

# 1. Using square brackets
empty_list = []
simple_list = [1, 2, 3, 4, 5]
mixed_list = [1, 'hello', 3.14, [1, 2], {'key': 'value'}]

print(f"Empty list: {empty_list}")
print(f"Simple list: {simple_list}")
print(f"Mixed types: {mixed_list}")

# 2. Using list() constructor
list_from_tuple = list((1, 2, 3, 4))
list_from_string = list("hello")
list_from_range = list(range(1, 6))
list_from_set = list({1, 2, 3, 4, 5})

print(f"From tuple: {list_from_tuple}")
print(f"From string: {list_from_string}")
print(f"From range: {list_from_range}")
print(f"From set: {list_from_set}")

# 3. Using list comprehension
squared_list = [x**2 for x in range(1, 6)]
filtered_list = [x for x in range(1, 11) if x % 2 == 0]

print(f"Squared list: {squared_list}")
print(f"Filtered list: {filtered_list}")

# 4. Using multiplication for repetition
repeated_list = [0] * 5
pattern_list = [1, 2] * 3

print(f"Repeated zeros: {repeated_list}")
print(f"Pattern repetition: {pattern_list}")

# 5. Nested lists (2D, 3D)
matrix_2d = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
matrix_3d = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]

print(f"2D matrix: {matrix_2d}")
print(f"3D matrix: {matrix_3d}")

# 6. List copying methods
original = [1, 2, 3, [4, 5]]

# Shallow copy methods
shallow_copy1 = original.copy()
shallow_copy2 = original[:]
shallow_copy3 = list(original)

# Deep copy
deep_copy = copy.deepcopy(original)

print(f"Original: {original}")
print(f"Shallow copy: {shallow_copy1}")

# Demonstrate shallow vs deep copy
original[3].append(6)
print(f"After modifying nested list:")
print(f"Original: {original}")
print(f"Shallow copy: {shallow_copy1}")  # Also changed!
print(f"Deep copy: {deep_copy}")  # Unchanged

print("\n--- List Characteristics ---")

# Mutability demonstration
mutable_list = [1, 2, 3]
print(f"Original: {mutable_list}, ID: {id(mutable_list)}")

mutable_list[0] = 10
print(f"After modification: {mutable_list}, ID: {id(mutable_list)}")  # Same ID

# Dynamic sizing
dynamic_list = [1, 2, 3]
print(f"Before append: {dynamic_list}, length: {len(dynamic_list)}")

dynamic_list.append(4)
print(f"After append: {dynamic_list}, length: {len(dynamic_list)}")

# Heterogeneous elements
heterogeneous = [1, "string", [1, 2], {"key": "value"}, lambda x: x**2]
print(f"Heterogeneous list: {heterogeneous}")

# ===============================================================================
# 2. LIST METHODS - COMPLETE REFERENCE
# ===============================================================================

print("\n" + "=" * 80)
print("2. LIST METHODS - COMPLETE REFERENCE")
print("=" * 80)

# Sample list for demonstrations
sample_list = [1, 2, 3, 2, 4, 2, 5]
print(f"Sample list: {sample_list}")

print("\n--- Adding Elements ---")

# append() - Add single element to end
demo_list = [1, 2, 3]
demo_list.append(4)
print(f"After append(4): {demo_list}")

demo_list.append([5, 6])  # Adds the entire list as one element
print(f"After append([5, 6]): {demo_list}")

# extend() - Add multiple elements to end
demo_list = [1, 2, 3]
demo_list.extend([4, 5, 6])
print(f"After extend([4, 5, 6]): {demo_list}")

demo_list.extend("abc")  # Extends with each character
print(f"After extend('abc'): {demo_list}")

# insert() - Insert element at specific position
demo_list = [1, 2, 3, 4, 5]
demo_list.insert(2, 'inserted')
print(f"After insert(2, 'inserted'): {demo_list}")

demo_list.insert(0, 'beginning')  # Insert at beginning
print(f"After insert(0, 'beginning'): {demo_list}")

demo_list.insert(-1, 'before_last')  # Insert before last element
print(f"After insert(-1, 'before_last'): {demo_list}")

print("\n--- Removing Elements ---")

# remove() - Remove first occurrence of value
demo_list = [1, 2, 3, 2, 4, 2, 5]
demo_list.remove(2)
print(f"After remove(2): {demo_list}")  # Only first 2 is removed

# pop() - Remove and return element at index (default: last)
demo_list = [1, 2, 3, 4, 5]
popped_last = demo_list.pop()
print(f"Popped last: {popped_last}, list: {demo_list}")

popped_index = demo_list.pop(1)
print(f"Popped index 1: {popped_index}, list: {demo_list}")

# clear() - Remove all elements
demo_list = [1, 2, 3, 4, 5]
demo_list.clear()
print(f"After clear(): {demo_list}")

# del statement - Remove by index or slice
demo_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
del demo_list[0]  # Remove first element
print(f"After del [0]: {demo_list}")

del demo_list[1:4]  # Remove slice
print(f"After del [1:4]: {demo_list}")

print("\n--- Finding and Counting ---")

# index() - Find first occurrence index
demo_list = [1, 2, 3, 2, 4, 2, 5]
first_index = demo_list.index(2)
print(f"First index of 2: {first_index}")

# index() with start and end parameters
next_index = demo_list.index(2, first_index + 1)
print(f"Next index of 2: {next_index}")

# count() - Count occurrences
count_2 = demo_list.count(2)
count_10 = demo_list.count(10)
print(f"Count of 2: {count_2}")
print(f"Count of 10: {count_10}")

print("\n--- Sorting and Reversing ---")

# sort() - Sort in place
demo_list = [3, 1, 4, 1, 5, 9, 2, 6]
print(f"Original: {demo_list}")

demo_list.sort()
print(f"After sort(): {demo_list}")

demo_list.sort(reverse=True)
print(f"After sort(reverse=True): {demo_list}")

# sort() with key function
words = ['banana', 'pie', 'Washington', 'book']
print(f"Words: {words}")

words.sort(key=len)  # Sort by length
print(f"Sorted by length: {words}")

words.sort(key=str.lower)  # Case-insensitive sort
print(f"Case-insensitive sort: {words}")

# Custom sort key
students = [('Alice', 85), ('Bob', 90), ('Charlie', 78)]
students.sort(key=lambda x: x[1])  # Sort by grade
print(f"Students by grade: {students}")

# reverse() - Reverse in place
demo_list = [1, 2, 3, 4, 5]
demo_list.reverse()
print(f"After reverse(): {demo_list}")

print("\n--- List Copying ---")

# copy() - Shallow copy
original = [1, 2, [3, 4]]
copied = original.copy()
print(f"Original: {original}")
print(f"Copied: {copied}")
print(f"Are they the same object? {original is copied}")

# Demonstrate shallow copy behavior
original[2].append(5)
print(f"After modifying nested list in original:")
print(f"Original: {original}")
print(f"Copied: {copied}")  # Nested list is shared!

print("\n--- Advanced List Operations ---")

# Using + and * operators
list1 = [1, 2, 3]
list2 = [4, 5, 6]
concatenated = list1 + list2
print(f"Concatenation: {list1} + {list2} = {concatenated}")

repeated = [1, 2] * 3
print(f"Repetition: [1, 2] * 3 = {repeated}")

# Using += and *= (in-place operations)
demo_list = [1, 2, 3]
demo_list += [4, 5]  # Same as extend()
print(f"After +=: {demo_list}")

demo_list *= 2  # Repeat in place
print(f"After *=: {demo_list}")

# ===============================================================================
# 3. LIST COMPREHENSIONS & ADVANCED PATTERNS
# ===============================================================================

print("\n" + "=" * 80)
print("3. LIST COMPREHENSIONS & ADVANCED PATTERNS")
print("=" * 80)

print("\n--- Basic List Comprehensions ---")

# Basic syntax: [expression for item in iterable]
squares = [x**2 for x in range(1, 6)]
print(f"Squares: {squares}")

# With condition: [expression for item in iterable if condition]
even_squares = [x**2 for x in range(1, 11) if x % 2 == 0]
print(f"Even squares: {even_squares}")

# String manipulation
words = ['hello', 'world', 'python', 'programming']
uppercase = [word.upper() for word in words]
print(f"Uppercase: {uppercase}")

# Length filtering
long_words = [word for word in words if len(word) > 5]
print(f"Long words: {long_words}")

print("\n--- Nested List Comprehensions ---")

# Flattening nested lists
nested = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [item for sublist in nested for item in sublist]
print(f"Flattened: {flattened}")

# Matrix operations
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Transpose matrix
transposed = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
print(f"Original matrix: {matrix}")
print(f"Transposed: {transposed}")

# Extract diagonal
diagonal = [matrix[i][i] for i in range(len(matrix))]
print(f"Diagonal: {diagonal}")

print("\n--- Advanced Comprehension Patterns ---")

# Multiple conditions
numbers = range(1, 21)
filtered = [x for x in numbers if x % 2 == 0 if x % 3 == 0]
print(f"Numbers divisible by both 2 and 3: {filtered}")

# Complex expressions
complex_expr = [x**2 if x % 2 == 0 else x**3 for x in range(1, 6)]
print(f"Square if even, cube if odd: {complex_expr}")

# Multiple iterables with zip
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
combined = [f"{name} is {age}" for name, age in zip(names, ages)]
print(f"Combined: {combined}")

# Cartesian product
colors = ['red', 'blue']
sizes = ['S', 'M', 'L']
products = [f"{color}-{size}" for color in colors for size in sizes]
print(f"Products: {products}")

print("\n--- Functional Programming with Comprehensions ---")

# Using functions in comprehensions
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [n for n in range(2, 50) if is_prime(n)]
print(f"Primes: {primes}")

# Lambda functions
squared_evens = [(lambda x: x**2)(x) for x in range(1, 11) if x % 2 == 0]
print(f"Squared evens: {squared_evens}")

# Method calls
sentence = "hello world python programming"
word_lengths = [len(word) for word in sentence.split()]
print(f"Word lengths: {word_lengths}")

print("\n--- Conditional Comprehensions ---")

# if-else in comprehension
numbers = range(-5, 6)
absolute_values = [x if x >= 0 else -x for x in numbers]
print(f"Absolute values: {absolute_values}")

# Complex conditionals
grades = [85, 92, 78, 96, 88, 73, 95]
letter_grades = [
    'A' if grade >= 90
    else 'B' if grade >= 80
    else 'C' if grade >= 70
    else 'D' if grade >= 60
    else 'F'
    for grade in grades
]
print(f"Letter grades: {letter_grades}")

print("\n--- Performance Considerations ---")

# Comprehension vs traditional loop (comprehensions are generally faster)
import time

# Traditional loop
start = time.time()
traditional = []
for i in range(10000):
    if i % 2 == 0:
        traditional.append(i**2)
traditional_time = time.time() - start

# List comprehension
start = time.time()
comprehension = [i**2 for i in range(10000) if i % 2 == 0]
comprehension_time = time.time() - start

print(f"Traditional loop time: {traditional_time:.6f}s")
print(f"Comprehension time: {comprehension_time:.6f}s")
print(f"Speedup: {traditional_time/comprehension_time:.2f}x")

# Memory considerations - generator vs list comprehension
# List comprehension (creates entire list in memory)
list_comp = [x**2 for x in range(1000000)]
print(f"List comprehension memory: {sys.getsizeof(list_comp)} bytes")

# Generator expression (lazy evaluation)
gen_expr = (x**2 for x in range(1000000))
print(f"Generator expression memory: {sys.getsizeof(gen_expr)} bytes")

# ===============================================================================
# 4. PERFORMANCE & MEMORY MANAGEMENT
# ===============================================================================

print("\n" + "=" * 80)
print("4. PERFORMANCE & MEMORY MANAGEMENT")
print("=" * 80)

print("\n--- Time Complexity Analysis ---")
print("""
List Operations Time Complexity:
===============================
- Access by index: O(1)
- Search (in operator): O(n)
- append(): O(1) amortized
- insert(0, item): O(n) - shifts all elements
- insert(i, item): O(n-i) - shifts elements after i
- remove(item): O(n) - searches then shifts
- pop(): O(1) - remove last
- pop(0): O(n) - remove first, shifts all
- pop(i): O(n-i) - shifts elements after i
- sort(): O(n log n)
- reverse(): O(n)
- count(): O(n)
- index(): O(n)
- extend(): O(k) where k is length of extension
- clear(): O(n)
- copy(): O(n)
""")

print("\n--- Performance Benchmarks ---")

def benchmark_list_operations():
    """Benchmark common list operations"""
    
    size = 100000
    test_list = list(range(size))
    
    # Append vs Insert at beginning
    print("Append vs Insert Performance:")
    
    # Append (fast)
    append_list = []
    start = time.time()
    for i in range(1000):
        append_list.append(i)
    append_time = time.time() - start
    
    # Insert at beginning (slow)
    insert_list = []
    start = time.time()
    for i in range(1000):
        insert_list.insert(0, i)
    insert_time = time.time() - start
    
    print(f"  Append 1000 items: {append_time:.6f}s")
    print(f"  Insert at beginning 1000 items: {insert_time:.6f}s")
    print(f"  Insert is {insert_time/append_time:.2f}x slower")
    
    # Pop from end vs beginning
    print("\nPop Performance:")
    
    pop_end_list = list(range(1000))
    start = time.time()
    while pop_end_list:
        pop_end_list.pop()
    pop_end_time = time.time() - start
    
    pop_beginning_list = list(range(1000))
    start = time.time()
    while pop_beginning_list:
        pop_beginning_list.pop(0)
    pop_beginning_time = time.time() - start
    
    print(f"  Pop from end 1000 items: {pop_end_time:.6f}s")
    print(f"  Pop from beginning 1000 items: {pop_beginning_time:.6f}s")
    print(f"  Pop from beginning is {pop_beginning_time/pop_end_time:.2f}x slower")

benchmark_list_operations()

print("\n--- Memory Management ---")

# List growth strategy
def analyze_list_growth():
    """Analyze how lists grow in memory"""
    
    lst = []
    previous_capacity = 0
    
    print("List Growth Pattern:")
    print("Length -> Capacity (Memory)")
    
    for i in range(20):
        lst.append(i)
        current_capacity = len(lst)
        memory = sys.getsizeof(lst)
        
        if current_capacity != previous_capacity:
            print(f"{len(lst):6d} -> {memory:8d} bytes")
            previous_capacity = current_capacity

analyze_list_growth()

print("\n--- Memory Optimization Techniques ---")

# 1. Use generators for large datasets
def memory_efficient_processing():
    """Demonstrate memory-efficient list processing"""
    
    # Memory-heavy approach
    large_list = [x**2 for x in range(1000000)]
    print(f"Large list memory: {sys.getsizeof(large_list)} bytes")
    
    # Memory-efficient approach
    def squares_generator(n):
        for i in range(n):
            yield i**2
    
    gen = squares_generator(1000000)
    print(f"Generator memory: {sys.getsizeof(gen)} bytes")
    
    # Process in chunks
    def process_in_chunks(iterable, chunk_size=1000):
        chunk = []
        for item in iterable:
            chunk.append(item)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
    
    # Example: sum squares in chunks
    total = 0
    for chunk in process_in_chunks(squares_generator(10000), 1000):
        total += sum(chunk)
    
    print(f"Sum of squares (processed in chunks): {total}")

memory_efficient_processing()

# 2. Pre-allocate lists when size is known
def preallocate_vs_append():
    """Compare pre-allocation vs append"""
    
    size = 100000
    
    # Append method
    start = time.time()
    append_list = []
    for i in range(size):
        append_list.append(i)
    append_time = time.time() - start
    
    # Pre-allocation method
    start = time.time()
    prealloc_list = [None] * size
    for i in range(size):
        prealloc_list[i] = i
    prealloc_time = time.time() - start
    
    print(f"Append method: {append_time:.6f}s")
    print(f"Pre-allocation method: {prealloc_time:.6f}s")
    print(f"Speedup: {append_time/prealloc_time:.2f}x")

preallocate_vs_append()

# 3. Using slots for list-like objects
class EfficientList:
    """Memory-efficient list-like class using __slots__"""
    __slots__ = ['_data', '_size']
    
    def __init__(self):
        self._data = []
        self._size = 0
    
    def append(self, item):
        self._data.append(item)
        self._size += 1
    
    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return self._size

# Compare memory usage
regular_list = [i for i in range(1000)]
efficient_list = EfficientList()
for i in range(1000):
    efficient_list.append(i)

print(f"Regular list memory: {sys.getsizeof(regular_list)} bytes")
print(f"Efficient list memory: {sys.getsizeof(efficient_list._data)} bytes")
