# ===============================================================================
# COMPREHENSIVE PYTHON SETS LEARNING GUIDE - ALL CONCEPTS & METHODS
# For Interview Preparation & Advanced Understanding
# ===============================================================================

"""
TABLE OF CONTENTS:
================
1. Set Fundamentals & Creation
2. Set Methods - Complete Reference
3. Set Operations & Mathematical Operations
4. Set Comprehensions & Advanced Patterns
5. Performance & Time Complexity
6. Memory Management & Optimization
7. Real-World Applications
8. Interview Problems & Solutions
9. Advanced Concepts for Experienced Developers
10. System Design with Sets
11. Threading & Concurrency
12. Best Practices & Common Pitfalls
"""

import time
import sys
from collections import defaultdict
import threading
import copy

# ===============================================================================
# 1. SET FUNDAMENTALS & CREATION
# ===============================================================================

print("=" * 80)
print("1. SET FUNDAMENTALS & CREATION")
print("=" * 80)

# What is a Set?
"""
A set is an unordered collection of unique elements in Python.
- Mutable (can be modified)
- No duplicate elements
- Elements must be hashable (immutable)
- No indexing (not subscriptable)
- Useful for membership testing and eliminating duplicates
"""

# Different ways to create sets
print("\n--- Set Creation Methods ---")

# 1. Using set() constructor
empty_set = set()
set_from_list = set([1, 2, 3, 3, 4])  # Duplicates automatically removed
set_from_string = set("hello")  # {'h', 'e', 'l', 'o'}
set_from_tuple = set((1, 2, 3, 4))

print(f"Empty set: {empty_set}")
print(f"From list: {set_from_list}")
print(f"From string: {set_from_string}")
print(f"From tuple: {set_from_tuple}")

# 2. Using set literal {}
literal_set = {1, 2, 3, 4, 5}
mixed_set = {1, 'hello', 3.14, (1, 2)}  # Mixed data types

print(f"Literal set: {literal_set}")
print(f"Mixed set: {mixed_set}")

# 3. Set comprehension
squared_set = {x**2 for x in range(1, 6)}
filtered_set = {x for x in range(1, 11) if x % 2 == 0}

print(f"Squared set: {squared_set}")
print(f"Filtered set: {filtered_set}")

# ===============================================================================
# 2. SET METHODS - COMPLETE REFERENCE
# ===============================================================================

print("\n" + "=" * 80)
print("2. SET METHODS - COMPLETE REFERENCE")
print("=" * 80)

# Creating sample sets for demonstration
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}
set3 = {1, 2, 3}

print(f"\nSample sets:")
print(f"set1: {set1}")
print(f"set2: {set2}")
print(f"set3: {set3}")

print("\n--- Adding Elements ---")

# add() - Add single element
demo_set = {1, 2, 3}
demo_set.add(4)
print(f"After add(4): {demo_set}")

# update() - Add multiple elements
demo_set.update([5, 6, 7])
print(f"After update([5, 6, 7]): {demo_set}")

demo_set.update({8, 9}, [10, 11])  # Multiple iterables
print(f"After update with multiple iterables: {demo_set}")

print("\n--- Removing Elements ---")

# remove() - Remove element (raises KeyError if not found)
demo_set = {1, 2, 3, 4, 5}
demo_set.remove(3)
print(f"After remove(3): {demo_set}")

# discard() - Remove element (no error if not found)
demo_set.discard(10)  # No error even though 10 doesn't exist
demo_set.discard(4)
print(f"After discard(4): {demo_set}")

# pop() - Remove and return arbitrary element
demo_set = {1, 2, 3, 4, 5}
popped = demo_set.pop()
print(f"Popped element: {popped}")
print(f"Set after pop(): {demo_set}")

# clear() - Remove all elements
demo_set.clear()
print(f"After clear(): {demo_set}")

print("\n--- Set Operations Methods ---")

# union() and |
union_result = set1.union(set2)
union_operator = set1 | set2
print(f"set1.union(set2): {union_result}")
print(f"set1 | set2: {union_operator}")

# intersection() and &
intersection_result = set1.intersection(set2)
intersection_operator = set1 & set2
print(f"set1.intersection(set2): {intersection_result}")
print(f"set1 & set2: {intersection_operator}")

# difference() and -
difference_result = set1.difference(set2)
difference_operator = set1 - set2
print(f"set1.difference(set2): {difference_result}")
print(f"set1 - set2: {difference_operator}")

# symmetric_difference() and ^
sym_diff_result = set1.symmetric_difference(set2)
sym_diff_operator = set1 ^ set2
print(f"set1.symmetric_difference(set2): {sym_diff_result}")
print(f"set1 ^ set2: {sym_diff_operator}")

print("\n--- In-place Set Operations ---")

# union_update() and |=
demo_set = {1, 2, 3}
demo_set.update({4, 5})  # Same as union_update for single argument
print(f"After union_update: {demo_set}")

# intersection_update() and &=
demo_set = {1, 2, 3, 4, 5}
demo_set.intersection_update({3, 4, 5, 6})
print(f"After intersection_update: {demo_set}")

# difference_update() and -=
demo_set = {1, 2, 3, 4, 5}
demo_set.difference_update({3, 4})
print(f"After difference_update: {demo_set}")

# symmetric_difference_update() and ^=
demo_set = {1, 2, 3}
demo_set.symmetric_difference_update({3, 4, 5})
print(f"After symmetric_difference_update: {demo_set}")

print("\n--- Subset and Superset Tests ---")

# issubset() and <=
print(f"set3.issubset(set1): {set3.issubset(set1)}")
print(f"set3 <= set1: {set3 <= set1}")

# issuperset() and >=
print(f"set1.issuperset(set3): {set1.issuperset(set3)}")
print(f"set1 >= set3: {set1 >= set3}")

# isdisjoint()
print(f"set1.isdisjoint(set2): {set1.isdisjoint(set2)}")
print(f"{1, 2}.isdisjoint({3, 4}): {{1, 2}}.isdisjoint({{3, 4}})")

print("\n--- Other Useful Methods ---")

# copy()
set_copy = set1.copy()
print(f"Original: {set1}")
print(f"Copy: {set_copy}")
print(f"Are they the same object? {set1 is set_copy}")

# len()
print(f"Length of set1: {len(set1)}")

# ===============================================================================
# 3. SET OPERATIONS & MATHEMATICAL OPERATIONS
# ===============================================================================

print("\n" + "=" * 80)
print("3. SET OPERATIONS & MATHEMATICAL OPERATIONS")
print("=" * 80)

# Mathematical Set Operations with examples
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}
C = {1, 2, 3}

print(f"Set A: {A}")
print(f"Set B: {B}")
print(f"Set C: {C}")

print("\n--- Union (A ∪ B) ---")
print("Elements in A or B or both")
print(f"A | B = {A | B}")

print("\n--- Intersection (A ∩ B) ---")
print("Elements in both A and B")
print(f"A & B = {A & B}")

print("\n--- Difference (A - B) ---")
print("Elements in A but not in B")
print(f"A - B = {A - B}")
print(f"B - A = {B - A}")

print("\n--- Symmetric Difference (A Δ B) ---")
print("Elements in A or B but not in both")
print(f"A ^ B = {A ^ B}")

print("\n--- Cartesian Product (Custom Implementation) ---")
def cartesian_product(set1, set2):
    """Returns cartesian product of two sets"""
    return {(a, b) for a in set1 for b in set2}

cart_product = cartesian_product({1, 2}, {'a', 'b'})
print(f"Cartesian product of {{1, 2}} and {{'a', 'b'}}: {cart_product}")

print("\n--- Power Set (Custom Implementation) ---")
def power_set(s):
    """Returns power set of a given set"""
    s = list(s)
    n = len(s)
    power_set_list = []
    
    for i in range(2**n):
        subset = set()
        for j in range(n):
            if i & (1 << j):
                subset.add(s[j])
        power_set_list.append(subset)
    
    return power_set_list

ps = power_set({1, 2, 3})
print(f"Power set of {{1, 2, 3}}: {ps}")

# ===============================================================================
# 4. SET COMPREHENSIONS & ADVANCED PATTERNS
# ===============================================================================

print("\n" + "=" * 80)
print("4. SET COMPREHENSIONS & ADVANCED PATTERNS")
print("=" * 80)

print("\n--- Basic Set Comprehensions ---")

# Basic comprehension
squares = {x**2 for x in range(1, 6)}
print(f"Squares: {squares}")

# With condition
even_squares = {x**2 for x in range(1, 11) if x % 2 == 0}
print(f"Even squares: {even_squares}")

# String manipulation
vowels = {char.lower() for char in "Hello World" if char.lower() in 'aeiou'}
print(f"Vowels: {vowels}")

print("\n--- Nested Set Comprehensions ---")

# Flatten nested structure
nested_list = [[1, 2], [3, 4], [5, 6]]
flattened = {item for sublist in nested_list for item in sublist}
print(f"Flattened: {flattened}")

# Matrix operations
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
diagonal = {matrix[i][i] for i in range(len(matrix))}
print(f"Diagonal elements: {diagonal}")

print("\n--- Advanced Patterns ---")

# Set comprehension with function calls
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = {n for n in range(2, 50) if is_prime(n)}
print(f"Primes up to 50: {primes}")

# Multiple conditions
multiples = {x for x in range(1, 101) if x % 3 == 0 and x % 5 == 0}
print(f"Multiples of both 3 and 5: {multiples}")

# Set comprehension with zip
names = ['Alice', 'Bob', 'Charlie']
scores = [85, 92, 78]
high_scorers = {name for name, score in zip(names, scores) if score > 80}
print(f"High scorers: {high_scorers}")

# ===============================================================================
# 5. PERFORMANCE & TIME COMPLEXITY
# ===============================================================================

print("\n" + "=" * 80)
print("5. PERFORMANCE & TIME COMPLEXITY")
print("=" * 80)

print("\n--- Time Complexity Analysis ---")
print("""
Set Operations Time Complexity:
- add(): O(1) average, O(n) worst case
- remove(), discard(): O(1) average, O(n) worst case
- pop(): O(1)
- clear(): O(n)
- copy(): O(n)
- len(): O(1)
- in operator: O(1) average, O(n) worst case

Set Operations:
- union: O(len(s1) + len(s2))
- intersection: O(min(len(s1), len(s2)))
- difference: O(len(s1))
- symmetric_difference: O(len(s1) + len(s2))
""")

print("\n--- Performance Comparison ---")

# Membership testing comparison
import time

# Large datasets
large_list = list(range(100000))
large_set = set(large_list)
search_item = 99999

# List membership test
start_time = time.time()
result1 = search_item in large_list
list_time = time.time() - start_time

# Set membership test
start_time = time.time()
result2 = search_item in large_set
set_time = time.time() - start_time

print(f"List membership test: {list_time:.6f} seconds")
print(f"Set membership test: {set_time:.6f} seconds")
print(f"Set is {list_time/set_time:.2f}x faster for membership testing")

print("\n--- Memory Usage ---")

# Memory comparison
import sys

sample_list = [1, 2, 3, 4, 5] * 1000
sample_set = set(sample_list)

print(f"List memory usage: {sys.getsizeof(sample_list)} bytes")
print(f"Set memory usage: {sys.getsizeof(sample_set)} bytes")
print(f"Set eliminates duplicates: {len(sample_list)} -> {len(sample_set)} elements")

# ===============================================================================
# 6. MEMORY MANAGEMENT & OPTIMIZATION
# ===============================================================================

print("\n" + "=" * 80)
print("6. MEMORY MANAGEMENT & OPTIMIZATION")
print("=" * 80)

print("\n--- Set Memory Optimization ---")

# Frozenset for immutable sets
immutable_set = frozenset([1, 2, 3, 4, 5])
print(f"Frozenset: {immutable_set}")
print(f"Can be used as dict key: {type(immutable_set)}")

# Example: frozenset as dictionary key
set_dict = {frozenset([1, 2]): 'value1', frozenset([3, 4]): 'value2'}
print(f"Dict with frozenset keys: {set_dict}")

print("\n--- Memory-Efficient Patterns ---")

# Using sets for deduplication
def remove_duplicates_efficient(items):
    """Memory-efficient duplicate removal"""
    return list(set(items))

# Generator expression with set
def unique_squares(n):
    """Generate unique squares up to n"""
    return {x*x for x in range(n) if x*x not in {y*y for y in range(x)}}

print(f"Unique squares: {unique_squares(10)}")

# Set intersection for large datasets
def find_common_elements_efficient(list1, list2):
    """Efficient way to find common elements"""
    return set(list1) & set(list2)

print("\n--- Weak References and Sets ---")

import weakref

class SetManager:
    def __init__(self):
        self._sets = weakref.WeakSet()
    
    def add_set(self, s):
        self._sets.add(s)
    
    def get_active_sets(self):
        return list(self._sets)

# Example usage
manager = SetManager()
temp_set = {1, 2, 3}
manager.add_set(temp_set)
print(f"Active sets: {len(manager.get_active_sets())}")

# ===============================================================================
# 7. REAL-WORLD APPLICATIONS
# ===============================================================================

print("\n" + "=" * 80)
print("7. REAL-WORLD APPLICATIONS")
print("=" * 80)

print("\n--- Application 1: Data Deduplication ---")

class DataDeduplicator:
    def __init__(self):
        self.seen_records = set()
    
    def add_record(self, record):
        """Add record if not duplicate"""
        record_hash = hash(tuple(sorted(record.items())))
        if record_hash not in self.seen_records:
            self.seen_records.add(record_hash)
            return True
        return False
    
    def get_unique_count(self):
        return len(self.seen_records)

# Example usage
dedup = DataDeduplicator()
records = [
    {'name': 'John', 'age': 30},
    {'age': 30, 'name': 'John'},  # Duplicate (same content)
    {'name': 'Jane', 'age': 25}
]

for record in records:
    is_unique = dedup.add_record(record)
    print(f"Record {record}: {'Unique' if is_unique else 'Duplicate'}")

print(f"Total unique records: {dedup.get_unique_count()}")

print("\n--- Application 2: Permission System ---")

class PermissionManager:
    def __init__(self):
        self.user_permissions = defaultdict(set)
        self.role_permissions = {
            'admin': {'read', 'write', 'delete', 'execute'},
            'user': {'read', 'write'},
            'guest': {'read'}
        }
    
    def assign_role(self, user, role):
        if role in self.role_permissions:
            self.user_permissions[user].update(self.role_permissions[role])
    
    def grant_permission(self, user, permission):
        self.user_permissions[user].add(permission)
    
    def revoke_permission(self, user, permission):
        self.user_permissions[user].discard(permission)
    
    def has_permission(self, user, permission):
        return permission in self.user_permissions[user]
    
    def get_user_permissions(self, user):
        return self.user_permissions[user].copy()

# Example usage
perm_manager = PermissionManager()
perm_manager.assign_role('alice', 'admin')
perm_manager.assign_role('bob', 'user')

print(f"Alice permissions: {perm_manager.get_user_permissions('alice')}")
print(f"Bob has delete permission: {perm_manager.has_permission('bob', 'delete')}")

print("\n--- Application 3: Graph Algorithms ---")

class Graph:
    def __init__(self):
        self.vertices = set()
        self.edges = defaultdict(set)
    
    def add_vertex(self, vertex):
        self.vertices.add(vertex)
    
    def add_edge(self, u, v):
        self.vertices.add(u)
        self.vertices.add(v)
        self.edges[u].add(v)
        self.edges[v].add(u)  # Undirected graph
    
    def get_neighbors(self, vertex):
        return self.edges[vertex]
    
    def bfs(self, start):
        """Breadth-first search using sets"""
        visited = set()
        queue = [start]
        result = []
        
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                queue.extend(self.edges[vertex] - visited)
        
        return result

# Example usage
graph = Graph()
edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D'), ('D', 'E')]
for u, v in edges:
    graph.add_edge(u, v)

print(f"BFS from A: {graph.bfs('A')}")

print("\n--- Application 4: Text Analysis ---")

class TextAnalyzer:
    def __init__(self):
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def extract_unique_words(self, text):
        """Extract unique words excluding stop words"""
        words = set(text.lower().split())
        return words - self.stop_words
    
    def find_common_words(self, text1, text2):
        """Find common words between two texts"""
        words1 = self.extract_unique_words(text1)
        words2 = self.extract_unique_words(text2)
        return words1 & words2
    
    def find_unique_to_text(self, text1, text2):
        """Find words unique to first text"""
        words1 = self.extract_unique_words(text1)
        words2 = self.extract_unique_words(text2)
        return words1 - words2

# Example usage
analyzer = TextAnalyzer()
text1 = "The quick brown fox jumps over the lazy dog"
text2 = "A quick brown fox runs through the forest"

common = analyzer.find_common_words(text1, text2)
unique = analyzer.find_unique_to_text(text1, text2)

print(f"Common words: {common}")
print(f"Words unique to text1: {unique}")

# ===============================================================================
# 8. INTERVIEW PROBLEMS & SOLUTIONS
# ===============================================================================

print("\n" + "=" * 80)
print("8. INTERVIEW PROBLEMS & SOLUTIONS")
print("=" * 80)

print("\n--- Problem 1: Find Intersection of Multiple Lists ---")

def find_intersection(*lists):
    """Find intersection of multiple lists using sets"""
    if not lists:
        return []
    
    result = set(lists[0])
    for lst in lists[1:]:
        result &= set(lst)
    
    return list(result)

# Example
lists = [[1, 2, 3, 4], [3, 4, 5, 6], [4, 5, 6, 7], [4, 6, 8, 9]]
intersection = find_intersection(*lists)
print(f"Intersection of {lists}: {intersection}")

print("\n--- Problem 2: Check if Arrays are Disjoint ---")

def are_disjoint(arr1, arr2):
    """Check if two arrays have no common elements"""
    return set(arr1).isdisjoint(set(arr2))

# Example
arr1 = [1, 2, 3, 4]
arr2 = [5, 6, 7, 8]
arr3 = [3, 4, 5, 6]

print(f"arr1 and arr2 disjoint: {are_disjoint(arr1, arr2)}")
print(f"arr1 and arr3 disjoint: {are_disjoint(arr1, arr3)}")

print("\n--- Problem 3: Find Missing Numbers ---")

def find_missing_numbers(arr, start, end):
    """Find missing numbers in a range"""
    present = set(arr)
    expected = set(range(start, end + 1))
    return sorted(expected - present)

# Example
numbers = [1, 3, 5, 7, 9]
missing = find_missing_numbers(numbers, 1, 10)
print(f"Missing numbers in range 1-10: {missing}")

print("\n--- Problem 4: Group Anagrams ---")

def group_anagrams(words):
    """Group anagrams together using sets"""
    anagram_groups = defaultdict(list)
    
    for word in words:
        # Use sorted characters as key
        key = ''.join(sorted(word.lower()))
        anagram_groups[key].append(word)
    
    return [group for group in anagram_groups.values() if len(group) > 1]

# Example
words = ['eat', 'tea', 'tan', 'ate', 'nat', 'bat']
anagrams = group_anagrams(words)
print(f"Anagram groups: {anagrams}")

print("\n--- Problem 5: Sudoku Validator ---")

def is_valid_sudoku(board):
    """Validate sudoku board using sets"""
    def is_valid_unit(unit):
        unit = [i for i in unit if i != '.']
        return len(set(unit)) == len(unit)
    
    # Check rows
    for row in board:
        if not is_valid_unit(row):
            return False
    
    # Check columns
    for col in zip(*board):
        if not is_valid_unit(col):
            return False
    
    # Check 3x3 boxes
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            box = [board[x][y] for x in range(i, i+3) for y in range(j, j+3)]
            if not is_valid_unit(box):
                return False
    
    return True

# Example (simplified 3x3 for demonstration)
mini_board = [
    ['5', '3', '.'],
    ['6', '.', '.'],
    ['.', '9', '8']
]

# Convert to 9x9 for proper testing
sudoku_board = [
    ['5','3','.','.','7','.','.','.','.'],
    ['6','.','.','1','9','5','.','.','.'],
    ['.','9','8','.','.','.','.','6','.'],
    ['8','.','.','.','6','.','.','.','3'],
    ['4','.','.','8','.','3','.','.','1'],
    ['7','.','.','.','2','.','.','.','6'],
    ['.','6','.','.','.','.','2','8','.'],
    ['.','.','.','4','1','9','.','.','5'],
    ['.','.','.','.','8','.','.','7','9']
]

print(f"Sudoku valid: {is_valid_sudoku(sudoku_board)}")

# ===============================================================================
# 9. ADVANCED CONCEPTS FOR EXPERIENCED DEVELOPERS
# ===============================================================================

print("\n" + "=" * 80)
print("9. ADVANCED CONCEPTS FOR EXPERIENCED DEVELOPERS")
print("=" * 80)

print("\n--- Custom Hashable Objects ---")

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        return isinstance(other, Point) and self.x == other.x and self.y == other.y
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"

# Using custom objects in sets
points = {Point(1, 2), Point(3, 4), Point(1, 2)}  # Duplicate will be removed
print(f"Unique points: {points}")

print("\n--- Set-based Design Patterns ---")

# Observer Pattern with Sets
class Subject:
    def __init__(self):
        self._observers = set()
    
    def attach(self, observer):
        self._observers.add(observer)
    
    def detach(self, observer):
        self._observers.discard(observer)
    
    def notify(self, message):
        for observer in self._observers:
            observer.update(message)

class Observer:
    def __init__(self, name):
        self.name = name
    
    def update(self, message):
        print(f"{self.name} received: {message}")
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return isinstance(other, Observer) and self.name == other.name

# Example usage
subject = Subject()
obs1 = Observer("Observer1")
obs2 = Observer("Observer2")

subject.attach(obs1)
subject.attach(obs2)
subject.notify("Hello Observers!")

print("\n--- Bloom Filter Implementation ---")

import hashlib

class BloomFilter:
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = set()  # Using set to simulate bit array
    
    def _hash(self, item, seed):
        return int(hashlib.md5(f"{item}{seed}".encode()).hexdigest(), 16) % self.size
    
    def add(self, item):
        for i in range(self.hash_count):
            self.bit_array.add(self._hash(item, i))
    
    def might_contain(self, item):
        for i in range(self.hash_count):
            if self._hash(item, i) not in self.bit_array:
                return False
        return True

# Example usage
bloom = BloomFilter(1000, 3)
bloom.add("hello")
bloom.add("world")

print(f"'hello' might be in filter: {bloom.might_contain('hello')}")
print(f"'python' might be in filter: {bloom.might_contain('python')}")

print("\n--- Set-based LRU Cache ---")

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.recent = set()
        self.order = []
    
    def get(self, key):
        if key in self.cache:
            self._update_order(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if len(self.cache) >= self.capacity and key not in self.cache:
            # Remove least recently used
            lru_key = self.order.pop(0)
            del self.cache[lru_key]
            self.recent.discard(lru_key)
        
        self.cache[key] = value
        self._update_order(key)
    
    def _update_order(self, key):
        if key in self.recent:
            self.order.remove(key)
        else:
            self.recent.add(key)
        self.order.append(key)

# Example usage
lru = LRUCache(3)
lru.put("a", 1)
lru.put("b", 2)
lru.put("c", 3)
print(f"Get 'a': {lru.get('a')}")
lru.put("d", 4)  # Should evict 'b'
print(f"Get 'b': {lru.get('b')}")  # Should return None

# ===============================================================================
# 10. SYSTEM DESIGN WITH SETS
# ===============================================================================

print("\n" + "=" * 80)
print("10. SYSTEM DESIGN WITH SETS")
print("=" * 80)

print("\n--- Distributed Rate Limiter ---")

import time
from collections import defaultdict

class DistributedRateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window
        self.user_requests = defaultdict(set)
    
    def is_allowed(self, user_id):
        current_time = time.time()
        user_request_times = self.user_requests[user_id]
        
        # Remove old requests outside time window
        cutoff_time = current_time - self.time_window
        self.user_requests[user_id] = {t for t in user_request_times if t > cutoff_time}
        
        # Check if under limit
        if len(self.user_requests[user_id]) < self.max_requests:
            self.user_requests[user_id].add(current_time)
            return True
        
        return False

# Example usage
rate_limiter = DistributedRateLimiter(max_requests=5, time_window=60)  # 5 requests per minute

for i in range(7):
    allowed = rate_limiter.is_allowed("user123")
    print(f"Request {i+1}: {'Allowed' if allowed else 'Rate limited'}")

print("\n--- Content Recommendation System ---")

class RecommendationEngine:
    def __init__(self):
        self.user_preferences = defaultdict(set)
        self.item_features = defaultdict(set)
        self.user_interactions = defaultdict(set)
    
    def add_user_preference(self, user_id, preferences):
        self.user_preferences[user_id].update(preferences)
    
    def add_item_features(self, item_id, features):
        self.item_features[item_id].update(features)
    
    def record_interaction(self, user_id, item_id):
        self.user_interactions[user_id].add(item_id)
    
    def recommend_items(self, user_id, num_recommendations=5):
        user_prefs = self.user_preferences[user_id]
        interacted_items = self.user_interactions[user_id]
        
        scores = {}
        for item_id, features in self.item_features.items():
            if item_id not in interacted_items:
                # Calculate similarity score
                common_features = user_prefs & features
                scores[item_id] = len(common_features)
        
        # Return top recommendations
        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:num_recommendations]

# Example usage
recommender = RecommendationEngine()
recommender.add_user_preference("user1", {"action", "sci-fi", "thriller"})
recommender.add_item_features("movie1", {"action", "adventure"})
recommender.add_item_features("movie2", {"sci-fi", "thriller", "drama"})
recommender.add_item_features("movie3", {"comedy", "romance"})

recommendations = recommender.recommend_items("user1")
print(f"Recommendations for user1: {recommendations}")

print("\n--- Social Network Graph ---")

class SocialNetwork:
    def __init__(self):
        self.followers = defaultdict(set)
        self.following = defaultdict(set)
    
    def follow(self, follower, followee):
        self.followers[followee].add(follower)
        self.following[follower].add(followee)
    
    def unfollow(self, follower, followee):
        self.followers[followee].discard(follower)
        self.following[follower].discard(followee)
    
    def get_mutual_followers(self, user1, user2):
        return self.followers[user1] & self.followers[user2]
    
    def suggest_friends(self, user_id):
        # Suggest people followed by people you follow
        suggestions = set()
        for followee in self.following[user_id]:
            suggestions.update(self.following[followee])
        
        # Remove yourself and people you already follow
        suggestions.discard(user_id)
        suggestions -= self.following[user_id]
        
        return suggestions

# Example usage
social_net = SocialNetwork()
social_net.follow("alice", "bob")
social_net.follow("alice", "charlie")
social_net.follow("bob", "david")
social_net.follow("charlie", "david")

suggestions = social_net.suggest_friends("alice")
print(f"Friend suggestions for alice: {suggestions}")

# ===============================================================================
# 11. THREADING & CONCURRENCY
# ===============================================================================

print("\n" + "=" * 80)
print("11. THREADING & CONCURRENCY")
print("=" * 80)

print("\n--- Thread-Safe Set Operations ---")

import threading
import time
import random

class ThreadSafeSet:
    def __init__(self):
        self._set = set()
        self._lock = threading.RLock()
    
    def add(self, item):
        with self._lock:
            self._set.add(item)
    
    def remove(self, item):
        with self._lock:
            self._set.discard(item)
    
    def contains(self, item):
        with self._lock:
            return item in self._set
    
    def size(self):
        with self._lock:
            return len(self._set)
    
    def snapshot(self):
        with self._lock:
            return self._set.copy()

# Example usage with threading
thread_safe_set = ThreadSafeSet()

def worker(worker_id):
    for i in range(10):
        thread_safe_set.add(f"worker_{worker_id}_item_{i}")
        time.sleep(0.01)

# Create and start threads
threads = []
for i in range(3):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join()

print(f"Final set size: {thread_safe_set.size()}")
print(f"Sample items: {list(thread_safe_set.snapshot())[:5]}")

print("\n--- Producer-Consumer with Sets ---")

class ProducerConsumer:
    def __init__(self):
        self.pending_tasks = set()
        self.completed_tasks = set()
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
    
    def produce(self, task):
        with self.condition:
            self.pending_tasks.add(task)
            self.condition.notify()
    
    def consume(self):
        with self.condition:
            while not self.pending_tasks:
                self.condition.wait()
            task = self.pending_tasks.pop()
            return task
    
    def mark_completed(self, task):
        with self.lock:
            self.completed_tasks.add(task)
    
    def get_status(self):
        with self.lock:
            return {
                'pending': len(self.pending_tasks),
                'completed': len(self.completed_tasks)
            }

# Example usage
pc = ProducerConsumer()

def producer():
    for i in range(5):
        task = f"task_{i}"
        pc.produce(task)
        print(f"Produced: {task}")
        time.sleep(0.1)

def consumer():
    for _ in range(5):
        task = pc.consume()
        print(f"Consumed: {task}")
        # Simulate work
        time.sleep(0.2)
        pc.mark_completed(task)

# Run producer and consumer
prod_thread = threading.Thread(target=producer)
cons_thread = threading.Thread(target=consumer)

prod_thread.start()
cons_thread.start()

prod_thread.join()
cons_thread.join()

print(f"Final status: {pc.get_status()}")

# ===============================================================================
# 12. BEST PRACTICES & COMMON PITFALLS
# ===============================================================================

print("\n" + "=" * 80)
print("12. BEST PRACTICES & COMMON PITFALLS")
print("=" * 80)

print("\n--- Best Practices ---")
print("""
1. Use sets for membership testing when you need O(1) lookup
2. Use frozenset for immutable sets that can be dictionary keys
3. Prefer set operations over loops for better performance
4. Use set comprehensions for readable and efficient code
5. Consider memory usage when dealing with large sets
6. Use sets for deduplication instead of manual loops
7. Leverage mathematical set operations for clean logic
8. Use sets in algorithms that benefit from uniqueness
""")

print("\n--- Common Pitfalls ---")

print("❌ PITFALL 1: Trying to create empty set with {}")
# Wrong way
# empty_set = {}  # This creates a dict, not a set!
# Correct way
empty_set = set()
print(f"Correct empty set: {empty_set}, type: {type(empty_set)}")

print("\n❌ PITFALL 2: Adding mutable objects to sets")
try:
    # This will raise TypeError
    bad_set = {[1, 2, 3]}  # Lists are not hashable
except TypeError as e:
    print(f"Error: {e}")

# Correct way - use tuples or frozensets
good_set = {(1, 2, 3), frozenset([4, 5, 6])}
print(f"Set with immutable objects: {good_set}")

print("\n❌ PITFALL 3: Assuming sets maintain order")
# Sets are unordered (before Python 3.7) or insertion-ordered (Python 3.7+)
# Don't rely on order for logic
sample_set = {3, 1, 4, 1, 5, 9, 2, 6}
print(f"Set order may not be what you expect: {sample_set}")

print("\n❌ PITFALL 4: Modifying set during iteration")
demo_set = {1, 2, 3, 4, 5}
# Wrong way - can cause RuntimeError
# for item in demo_set:
#     if item % 2 == 0:
#         demo_set.remove(item)  # Don't modify during iteration

# Correct way
demo_set = {1, 2, 3, 4, 5}
to_remove = {item for item in demo_set if item % 2 == 0}
demo_set -= to_remove
print(f"After removing even numbers: {demo_set}")

print("\n❌ PITFALL 5: Not handling KeyError with remove()")
demo_set = {1, 2, 3}
# Wrong way - can raise KeyError
# demo_set.remove(4)  # Raises KeyError if 4 not in set

# Correct ways
demo_set.discard(4)  # No error if item doesn't exist
print(f"After discard(4): {demo_set}")

# Or use conditional removal
if 2 in demo_set:
    demo_set.remove(2)
print(f"After conditional remove(2): {demo_set}")

print("\n✅ Performance Tips ---")

# Use set operations instead of loops
list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]

# Inefficient way
common_slow = []
for item in list1:
    if item in list2:
        common_slow.append(item)

# Efficient way
common_fast = list(set(list1) & set(list2))
print(f"Common elements (efficient): {common_fast}")

# Use membership testing efficiently
large_collection = set(range(10000))
# Fast: item in large_collection
# Slow: item in list(large_collection)

print("\n✅ Memory Optimization Tips ---")

# Use frozenset for read-only sets
immutable_data = frozenset([1, 2, 3, 4, 5])
# Can be used as dict keys, more memory efficient for large datasets

# Use set comprehensions instead of set(generator)
# More memory efficient
efficient_set = {x**2 for x in range(1000) if x % 2 == 0}
# Less efficient
# less_efficient = set(x**2 for x in range(1000) if x % 2 == 0)

print(f"Efficient set generation complete: {len(efficient_set)} elements")

print("\n" + "=" * 80)
print("🎯 SET MASTERY SUMMARY")
print("=" * 80)
print("""
Key Takeaways:
1. Sets provide O(1) average-case membership testing
2. Use for deduplication, uniqueness, and mathematical operations
3. Thread-safety requires explicit locking mechanisms
4. Sets are unordered collections of unique, hashable elements
5. Powerful for algorithm optimization and data processing
6. Essential for system design patterns and real-world applications

Master these concepts and you'll excel in Python set-related interviews! 🚀
""")
