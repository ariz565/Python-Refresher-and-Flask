# ===============================================================================
# COMPREHENSIVE PYTHON TUPLES LEARNING GUIDE - ALL CONCEPTS & METHODS
# For Interview Preparation & Advanced Understanding
# ===============================================================================

"""
TABLE OF CONTENTS:
================
1. Tuple Fundamentals & Creation
2. Tuple Methods & Operations - Complete Reference
3. Tuple Unpacking & Advanced Patterns
4. Performance & Memory Efficiency
5. Named Tuples & Advanced Tuple Types
6. Real-World Applications
7. Interview Problems & Solutions
8. Advanced Concepts for Experienced Developers
9. System Design with Tuples
10. Threading & Immutability Benefits
11. Best Practices & Common Pitfalls
12. Tuple vs Other Data Structures
"""

import time
import sys
from collections import namedtuple, defaultdict
import threading
import operator
from typing import Tuple, NamedTuple
import itertools

# ===============================================================================
# 1. TUPLE FUNDAMENTALS & CREATION
# ===============================================================================

print("=" * 80)
print("1. TUPLE FUNDAMENTALS & CREATION")
print("=" * 80)

# What is a Tuple?
"""
A tuple is an ordered collection of items in Python.
- Immutable (cannot be modified after creation)
- Ordered (maintains insertion order)
- Allows duplicate elements
- Elements can be of different types
- Supports indexing and slicing
- Hashable (can be used as dictionary keys)
"""

print("\n--- Tuple Creation Methods ---")

# 1. Using parentheses
empty_tuple = ()
single_item_tuple = (42,)  # Note the comma!
multi_item_tuple = (1, 2, 3, 4, 5)
mixed_tuple = (1, 'hello', 3.14, [1, 2, 3])

print(f"Empty tuple: {empty_tuple}")
print(f"Single item: {single_item_tuple}")
print(f"Multi item: {multi_item_tuple}")
print(f"Mixed types: {mixed_tuple}")

# 2. Using tuple() constructor
tuple_from_list = tuple([1, 2, 3, 4])
tuple_from_string = tuple("hello")
tuple_from_range = tuple(range(5))

print(f"From list: {tuple_from_list}")
print(f"From string: {tuple_from_string}")
print(f"From range: {tuple_from_range}")

# 3. Without parentheses (tuple packing)
packed_tuple = 1, 2, 3, 4, 5
coordinates = 10, 20
name_age = "Alice", 25

print(f"Packed tuple: {packed_tuple}")
print(f"Coordinates: {coordinates}")
print(f"Name and age: {name_age}")

# 4. Nested tuples
nested_tuple = ((1, 2), (3, 4), (5, 6))
matrix_tuple = ((1, 2, 3), (4, 5, 6), (7, 8, 9))

print(f"Nested tuple: {nested_tuple}")
print(f"Matrix tuple: {matrix_tuple}")

print("\n--- Important Tuple Characteristics ---")

# Immutability demonstration
original = (1, 2, 3)
print(f"Original tuple: {original}")
print(f"ID of original: {id(original)}")

# This creates a new tuple, doesn't modify the original
modified = original + (4, 5)
print(f"After concatenation: {modified}")
print(f"ID of modified: {id(modified)}")
print(f"Original unchanged: {original}")

# Hashable property
tuple_as_key = {(1, 2): 'value1', (3, 4): 'value2'}
print(f"Tuple as dict key: {tuple_as_key}")

# ===============================================================================
# 2. TUPLE METHODS & OPERATIONS - COMPLETE REFERENCE
# ===============================================================================

print("\n" + "=" * 80)
print("2. TUPLE METHODS & OPERATIONS - COMPLETE REFERENCE")
print("=" * 80)

# Sample tuple for demonstrations
sample_tuple = (1, 2, 3, 2, 4, 2, 5)
print(f"Sample tuple: {sample_tuple}")

print("\n--- Built-in Tuple Methods ---")

# count() - Count occurrences of an element
count_2 = sample_tuple.count(2)
count_6 = sample_tuple.count(6)  # Not in tuple
print(f"Count of 2: {count_2}")
print(f"Count of 6: {count_6}")

# index() - Find first occurrence index
index_2 = sample_tuple.index(2)
try:
    index_6 = sample_tuple.index(6)
except ValueError as e:
    print(f"Index of 6: Error - {e}")

print(f"First index of 2: {index_2}")

# index() with start and end parameters
index_2_after_3 = sample_tuple.index(2, 3)  # Start searching from index 3
print(f"Index of 2 after position 3: {index_2_after_3}")

print("\n--- Indexing and Slicing ---")

# Positive indexing
print(f"First element: {sample_tuple[0]}")
print(f"Last element: {sample_tuple[-1]}")
print(f"Second last: {sample_tuple[-2]}")

# Slicing
print(f"First 3 elements: {sample_tuple[:3]}")
print(f"Last 3 elements: {sample_tuple[-3:]}")
print(f"Every 2nd element: {sample_tuple[::2]}")
print(f"Reverse tuple: {sample_tuple[::-1]}")

# Advanced slicing
print(f"Elements 1 to 5: {sample_tuple[1:5]}")
print(f"Elements 1 to 5, step 2: {sample_tuple[1:5:2]}")

print("\n--- Tuple Operations ---")

# Concatenation
tuple1 = (1, 2, 3)
tuple2 = (4, 5, 6)
concatenated = tuple1 + tuple2
print(f"Concatenation: {tuple1} + {tuple2} = {concatenated}")

# Repetition
repeated = (1, 2) * 3
print(f"Repetition: (1, 2) * 3 = {repeated}")

# Membership testing
print(f"2 in sample_tuple: {2 in sample_tuple}")
print(f"10 in sample_tuple: {10 in sample_tuple}")

# Length
print(f"Length of sample_tuple: {len(sample_tuple)}")

# Comparison
tuple_a = (1, 2, 3)
tuple_b = (1, 2, 4)
tuple_c = (1, 2, 3)

print(f"{tuple_a} == {tuple_c}: {tuple_a == tuple_c}")
print(f"{tuple_a} < {tuple_b}: {tuple_a < tuple_b}")
print(f"{tuple_a} > {tuple_b}: {tuple_a > tuple_b}")

print("\n--- Built-in Functions with Tuples ---")

numbers = (3, 1, 4, 1, 5, 9, 2, 6)
print(f"Numbers tuple: {numbers}")

# Mathematical functions
print(f"min(): {min(numbers)}")
print(f"max(): {max(numbers)}")
print(f"sum(): {sum(numbers)}")

# sorted() - returns a list
sorted_list = sorted(numbers)
print(f"sorted(): {sorted_list} (returns list)")

# any() and all()
bool_tuple = (True, True, False)
print(f"any({bool_tuple}): {any(bool_tuple)}")
print(f"all({bool_tuple}): {all(bool_tuple)}")

# enumerate()
for i, value in enumerate(('a', 'b', 'c')):
    print(f"enumerate: index {i}, value '{value}'")

print("\n--- Tuple Conversion ---")

# Convert to other types
tuple_data = (1, 2, 3, 4, 5)
to_list = list(tuple_data)
to_set = set(tuple_data)
to_string = str(tuple_data)

print(f"Original tuple: {tuple_data}")
print(f"To list: {to_list}")
print(f"To set: {to_set}")
print(f"To string: {to_string}")

# ===============================================================================
# 3. TUPLE UNPACKING & ADVANCED PATTERNS
# ===============================================================================

print("\n" + "=" * 80)
print("3. TUPLE UNPACKING & ADVANCED PATTERNS")
print("=" * 80)

print("\n--- Basic Tuple Unpacking ---")

# Simple unpacking
coordinates = (10, 20)
x, y = coordinates
print(f"Coordinates {coordinates} unpacked to x={x}, y={y}")

# Multiple values
person_data = ("Alice", 25, "Engineer")
name, age, job = person_data
print(f"Person: name={name}, age={age}, job={job}")

print("\n--- Advanced Unpacking Patterns ---")

# Star expressions (Python 3+)
numbers = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

# First, middle, last
first, *middle, last = numbers
print(f"First: {first}, Middle: {middle}, Last: {last}")

# First few and rest
first_three, *rest = numbers[:3], numbers[3:]
print(f"First three: {first_three}, Rest: {rest}")

# Skip elements
first, _, third, *_ = numbers
print(f"First: {first}, Third: {third}")

print("\n--- Unpacking in Function Calls ---")

def calculate_distance(x1, y1, x2, y2):
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

point1 = (0, 0)
point2 = (3, 4)

# Unpacking tuples as function arguments
distance = calculate_distance(*point1, *point2)
print(f"Distance between {point1} and {point2}: {distance}")

print("\n--- Unpacking in Loops ---")

# List of tuples
people = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]

for name, age in people:
    print(f"{name} is {age} years old")

# Enumerate with unpacking
items = ("apple", "banana", "cherry")
for index, item in enumerate(items):
    print(f"Item {index}: {item}")

print("\n--- Swapping Variables ---")

# Classic tuple swapping
a, b = 10, 20
print(f"Before swap: a={a}, b={b}")
a, b = b, a
print(f"After swap: a={a}, b={b}")

# Multiple variable rotation
x, y, z = 1, 2, 3
print(f"Before rotation: x={x}, y={y}, z={z}")
x, y, z = z, x, y
print(f"After rotation: x={x}, y={y}, z={z}")

print("\n--- Function Return Unpacking ---")

def get_name_age():
    return "John", 28

def get_statistics(numbers):
    return min(numbers), max(numbers), sum(numbers), len(numbers)

# Unpack function returns
name, age = get_name_age()
print(f"Function returned: name={name}, age={age}")

min_val, max_val, total, count = get_statistics((1, 2, 3, 4, 5))
print(f"Statistics: min={min_val}, max={max_val}, sum={total}, count={count}")

print("\n--- Nested Tuple Unpacking ---")

nested_data = (("Alice", 25), ("Bob", 30))
(name1, age1), (name2, age2) = nested_data
print(f"Person 1: {name1}, {age1}")
print(f"Person 2: {name2}, {age2}")

# Complex nested structure
matrix = ((1, 2, 3), (4, 5, 6))
(a, b, c), (d, e, f) = matrix
print(f"Matrix unpacked: a={a}, b={b}, c={c}, d={d}, e={e}, f={f}")

# ===============================================================================
# 4. PERFORMANCE & MEMORY EFFICIENCY
# ===============================================================================

print("\n" + "=" * 80)
print("4. PERFORMANCE & MEMORY EFFICIENCY")
print("=" * 80)

print("\n--- Memory Efficiency Comparison ---")

# Memory usage comparison
import sys

# Same data in different structures
data = list(range(1000))
tuple_data = tuple(data)
list_data = list(data)
set_data = set(data)

print(f"Data size: 1000 elements")
print(f"Tuple memory: {sys.getsizeof(tuple_data)} bytes")
print(f"List memory: {sys.getsizeof(list_data)} bytes")
print(f"Set memory: {sys.getsizeof(set_data)} bytes")

print(f"\nTuple is {sys.getsizeof(list_data) - sys.getsizeof(tuple_data)} bytes smaller than list")

print("\n--- Performance Comparison ---")

# Access time comparison
import time

large_tuple = tuple(range(100000))
large_list = list(range(100000))

# Indexing performance
start_time = time.time()
for _ in range(10000):
    _ = large_tuple[50000]
tuple_access_time = time.time() - start_time

start_time = time.time()
for _ in range(10000):
    _ = large_list[50000]
list_access_time = time.time() - start_time

print(f"Tuple access time: {tuple_access_time:.6f} seconds")
print(f"List access time: {list_access_time:.6f} seconds")

# Iteration performance
start_time = time.time()
for item in large_tuple:
    pass
tuple_iter_time = time.time() - start_time

start_time = time.time()
for item in large_list:
    pass
list_iter_time = time.time() - start_time

print(f"Tuple iteration time: {tuple_iter_time:.6f} seconds")
print(f"List iteration time: {list_iter_time:.6f} seconds")

print("\n--- Memory Optimization Techniques ---")

# Using tuples for immutable data
class ImmutablePoint:
    def __init__(self, x, y):
        self._coords = (x, y)  # Store as tuple
    
    @property
    def x(self):
        return self._coords[0]
    
    @property
    def y(self):
        return self._coords[1]
    
    def __repr__(self):
        return f"ImmutablePoint({self.x}, {self.y})"

point = ImmutablePoint(10, 20)
print(f"Immutable point: {point}")

# Tuple interning for small tuples
small_tuple1 = (1, 2)
small_tuple2 = (1, 2)
print(f"Small tuple interning: {small_tuple1 is small_tuple2}")

# Large tuples are not interned
large_tuple1 = tuple(range(1000))
large_tuple2 = tuple(range(1000))
print(f"Large tuple interning: {large_tuple1 is large_tuple2}")

print("\n--- Time Complexity Analysis ---")
print("""
Tuple Operations Time Complexity:
- Access by index: O(1)
- Search (in operator): O(n)
- count(): O(n)
- index(): O(n)
- Slicing: O(k) where k is slice length
- Concatenation: O(n + m)
- Repetition: O(n * k)
- len(): O(1)

Space Complexity:
- Storage: O(n) where n is number of elements
- Slicing creates new tuple: O(k) additional space
- Concatenation: O(n + m) additional space
""")

# ===============================================================================
# 5. NAMED TUPLES & ADVANCED TUPLE TYPES
# ===============================================================================

print("\n" + "=" * 80)
print("5. NAMED TUPLES & ADVANCED TUPLE TYPES")
print("=" * 80)

print("\n--- Basic Named Tuples ---")

# Creating named tuple
Person = namedtuple('Person', ['name', 'age', 'city'])

# Creating instances
person1 = Person('Alice', 30, 'New York')
person2 = Person(name='Bob', age=25, city='London')

print(f"Person 1: {person1}")
print(f"Person 2: {person2}")

# Accessing fields
print(f"Person 1 name: {person1.name}")
print(f"Person 1 age: {person1.age}")
print(f"Person 1 city: {person1.city}")

# Still works with indexing
print(f"Person 1 by index: {person1[0]}, {person1[1]}, {person1[2]}")

print("\n--- Named Tuple Methods ---")

# _replace() - create new instance with some fields changed
person1_older = person1._replace(age=31)
print(f"Original: {person1}")
print(f"Aged: {person1_older}")

# _asdict() - convert to dictionary
person_dict = person1._asdict()
print(f"As dictionary: {person_dict}")

# _fields - get field names
print(f"Field names: {Person._fields}")

# _make() - create from iterable
person_data = ['Charlie', 35, 'Paris']
person3 = Person._make(person_data)
print(f"Made from list: {person3}")

print("\n--- Advanced Named Tuple Features ---")

# Named tuple with defaults (Python 3.7+)
try:
    Employee = namedtuple('Employee', ['name', 'position', 'salary', 'department'], 
                         defaults=['Unknown', 0, 'General'])
    
    # Create with some defaults
    emp1 = Employee('John', 'Developer')
    emp2 = Employee('Jane', 'Manager', 75000)
    emp3 = Employee('Bob', 'Analyst', 60000, 'Finance')
    
    print(f"Employee with defaults: {emp1}")
    print(f"Employee partial: {emp2}")
    print(f"Employee full: {emp3}")
except TypeError:
    # Fallback for older Python versions
    print("Named tuple defaults require Python 3.7+")

print("\n--- Typing with Named Tuples ---")

# Using typing.NamedTuple (Python 3.6+)
class Coordinate(NamedTuple):
    x: float
    y: float
    z: float = 0.0  # Default value
    
    def distance_from_origin(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2)**0.5

coord1 = Coordinate(3.0, 4.0)
coord2 = Coordinate(1.0, 2.0, 3.0)

print(f"2D coordinate: {coord1}")
print(f"3D coordinate: {coord2}")
print(f"Distance from origin: {coord1.distance_from_origin():.2f}")

print("\n--- Named Tuples in Data Processing ---")

# Example: Processing CSV-like data
Student = namedtuple('Student', ['id', 'name', 'grade', 'subject'])

students_data = [
    (1, 'Alice', 85, 'Math'),
    (2, 'Bob', 92, 'Science'),
    (3, 'Charlie', 78, 'Math'),
    (4, 'Diana', 95, 'Science')
]

students = [Student(*data) for data in students_data]

# Analysis using named tuples
math_students = [s for s in students if s.subject == 'Math']
high_performers = [s for s in students if s.grade >= 90]

print(f"Math students: {[s.name for s in math_students]}")
print(f"High performers: {[s.name for s in high_performers]}")

# Average grade calculation
avg_grade = sum(s.grade for s in students) / len(students)
print(f"Average grade: {avg_grade:.2f}")
