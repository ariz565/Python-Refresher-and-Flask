"""
PYTHON DICTIONARIES - COMPREHENSIVE INTERVIEW GUIDE
==================================================

This guide covers ALL dictionary concepts with interview questions and patterns.
Use this to master dictionary-based problem solving for technical interviews.

Author: Interview Preparation Guide
Date: August 2025
"""

# =============================================================================
# PART 1: DICTIONARY BASICS AND FUNDAMENTALS
# =============================================================================

print("=" * 60)
print("PART 1: DICTIONARY BASICS AND FUNDAMENTALS")
print("=" * 60)

# 1. Dictionary Creation Methods
print("\n1. DICTIONARY CREATION METHODS:")
print("-" * 40)

# Method 1: Literal syntax (most common)
dict1 = {"name": "John", "age": 25, "city": "NYC"}
print(f"Literal: {dict1}")

# Method 2: dict() constructor
dict2 = dict(name="Alice", age=30, city="LA")
print(f"Constructor: {dict2}")

# Method 3: From list of tuples
dict3 = dict([("a", 1), ("b", 2), ("c", 3)])
print(f"From tuples: {dict3}")

# Method 4: From two lists using zip
keys = ["x", "y", "z"]
values = [1, 2, 3]
dict4 = dict(zip(keys, values))
print(f"From zip: {dict4}")

# Method 5: Dictionary comprehension
dict5 = {x: x**2 for x in range(5)}
print(f"Comprehension: {dict5}")

# Method 6: fromkeys() method
dict6 = dict.fromkeys(["x", "y", "z"], 0)
print(f"fromkeys(): {dict6}")

# Method 7: Empty dictionary
empty_dict = {}
empty_dict2 = dict()
print(f"Empty dict: {empty_dict}, {empty_dict2}")

# 2. Key Properties and Rules
print("\n2. KEY PROPERTIES AND RULES:")
print("-" * 40)

# Valid key types (immutable objects only)
valid_keys = {
    "string": "String keys are most common",
    42: "Integer keys",
    3.14: "Float keys",
    (1, 2): "Tuple keys (if contents are immutable)",
    True: "Boolean keys",
    None: "None as key",
    frozenset([1, 2]): "Frozenset keys"
}
print(f"Valid keys example: {valid_keys}")

# Key rules demonstration
print("\nKey Rules:")
print("✓ Keys must be immutable (hashable)")
print("✓ Keys must be unique (duplicates overwrite)")
print("✓ Keys are case-sensitive for strings")
print("✓ Key order is preserved (Python 3.7+)")

# Invalid key examples (commented to avoid errors)
print("\nInvalid key types:")
print("✗ Lists: [1, 2] - mutable")
print("✗ Dictionaries: {'a': 1} - mutable") 
print("✗ Sets: {1, 2} - mutable")

# 3. Data Storage and Retrieval
print("\n3. DATA STORAGE AND RETRIEVAL:")
print("-" * 40)

# Creating a sample dictionary for demonstrations
student = {
    "name": "Alice Johnson",
    "age": 22,
    "grades": [85, 92, 78, 94],
    "subjects": ["Math", "Physics", "Chemistry"],
    "address": {
        "street": "123 Main St",
        "city": "Boston",
        "zip": "02101"
    },
    "graduated": False
}

print(f"Student data: {student}")

# 4. Accessing Dictionary Values
print("\n4. ACCESSING DICTIONARY VALUES:")
print("-" * 40)

# Method 1: Square bracket notation (raises KeyError if key doesn't exist)
print(f"Name: {student['name']}")
print(f"Age: {student['age']}")

# Method 2: get() method (returns None or default if key doesn't exist)
print(f"Name with get(): {student.get('name')}")
print(f"Phone with get(): {student.get('phone')}")  # Returns None
print(f"Phone with default: {student.get('phone', 'Not provided')}")

# Method 3: Accessing nested values
print(f"City: {student['address']['city']}")
print(f"First grade: {student['grades'][0]}")

# Safe nested access using get()
print(f"Safe nested access: {student.get('address', {}).get('city', 'Unknown')}")

# 5. Adding and Updating Data
print("\n5. ADDING AND UPDATING DATA:")
print("-" * 40)

# Adding new key-value pairs
student["email"] = "alice@email.com"
student["phone"] = "555-1234"
print(f"After adding email and phone: {len(student)} keys")

# Updating existing values
student["age"] = 23
student["graduated"] = True
print(f"Updated age: {student['age']}, Graduated: {student['graduated']}")

# Bulk updates using update()
additional_info = {
    "gpa": 3.8,
    "major": "Computer Science",
    "year": "Senior"
}
student.update(additional_info)
print(f"After bulk update: {len(student)} keys total")

# Using setdefault() - adds key only if it doesn't exist
student.setdefault("scholarship", "Merit Scholar")
student.setdefault("age", 25)  # Won't change existing age
print(f"Scholarship: {student['scholarship']}, Age still: {student['age']}")

# 6. Dictionary Methods (Complete Reference)
print("\n6. COMPLETE DICTIONARY METHODS REFERENCE:")
print("-" * 40)

# Create a sample dictionary for method demonstrations
sample = {"a": 1, "b": 2, "c": 3, "d": 4}
print(f"Original dict: {sample}")

# 6.1 Access Methods
print("\nACCESS METHODS:")
print(f"sample.get('a'): {sample.get('a')}")
print(f"sample.get('z', 'default'): {sample.get('z', 'default')}")
print(f"sample.setdefault('e', 5): {sample.setdefault('e', 5)}")
print(f"After setdefault: {sample}")

# 6.2 View Methods (return dictionary views)
print("\nVIEW METHODS:")
keys_view = sample.keys()
values_view = sample.values()
items_view = sample.items()
print(f"Keys: {list(keys_view)}")
print(f"Values: {list(values_view)}")
print(f"Items: {list(items_view)}")

# 6.3 Modification Methods
print("\nMODIFICATION METHODS:")
sample_copy = sample.copy()

# pop() - remove and return value
popped_value = sample_copy.pop('e')
print(f"Popped 'e': {popped_value}, Dict: {sample_copy}")

# pop() with default
popped_default = sample_copy.pop('z', 'not found')
print(f"Popped 'z' with default: {popped_default}")

# popitem() - remove and return last inserted item (Python 3.7+)
last_item = sample_copy.popitem()
print(f"Popped last item: {last_item}, Dict: {sample_copy}")

# clear() - remove all items
sample_copy.clear()
print(f"After clear(): {sample_copy}")

# 6.4 Dictionary Information Methods
print("\nINFORMATION METHODS:")
print(f"Length: len(sample) = {len(sample)}")
print(f"Is empty: len(sample) == 0 = {len(sample) == 0}")
print(f"Key exists: 'a' in sample = {'a' in sample}")
print(f"Key doesn't exist: 'z' in sample = {'z' in sample}")

# 7. Dictionary Sorting
print("\n7. DICTIONARY SORTING:")
print("-" * 40)

# Sample data for sorting demonstrations
grades = {"Alice": 85, "Bob": 92, "Charlie": 78, "Diana": 96, "Eve": 88}
print(f"Original grades: {grades}")

# Sort by keys (alphabetical)
sorted_by_keys = dict(sorted(grades.items()))
print(f"Sorted by keys: {sorted_by_keys}")

# Sort by keys (reverse alphabetical)
sorted_by_keys_desc = dict(sorted(grades.items(), reverse=True))
print(f"Sorted by keys (desc): {sorted_by_keys_desc}")

# Sort by values (ascending)
sorted_by_values = dict(sorted(grades.items(), key=lambda x: x[1]))
print(f"Sorted by values (asc): {sorted_by_values}")

# Sort by values (descending)
sorted_by_values_desc = dict(sorted(grades.items(), key=lambda x: x[1], reverse=True))
print(f"Sorted by values (desc): {sorted_by_values_desc}")

# Get top N entries
top_3 = dict(sorted(grades.items(), key=lambda x: x[1], reverse=True)[:3])
print(f"Top 3 students: {top_3}")

# Sort by string length (for string values)
names = {"a": "Alice", "b": "Bob", "c": "Christopher", "d": "Di"}
sorted_by_length = dict(sorted(names.items(), key=lambda x: len(x[1])))
print(f"Sorted by name length: {sorted_by_length}")

# 8. Dictionary Copying and Merging
print("\n8. DICTIONARY COPYING AND MERGING:")
print("-" * 40)

original = {"a": 1, "b": 2, "c": 3}

# Shallow copy methods
copy1 = original.copy()
copy2 = dict(original)
copy3 = {**original}  # Dictionary unpacking (Python 3.5+)
import copy as copy_module
copy4 = copy_module.copy(original)

print(f"Original: {original}")
print(f"Shallow copies are equal: {copy1 == copy2 == copy3 == copy4}")

# Deep copy (for nested dictionaries)
nested_dict = {"a": {"x": 1}, "b": {"y": 2}}
shallow_copy = nested_dict.copy()
deep_copy = copy_module.deepcopy(nested_dict)

# Modify nested value
nested_dict["a"]["x"] = 999
print(f"Original after modification: {nested_dict}")
print(f"Shallow copy affected: {shallow_copy}")  # Also changed!
print(f"Deep copy unaffected: {deep_copy}")     # Unchanged

# Dictionary merging techniques
dict_a = {"a": 1, "b": 2}
dict_b = {"c": 3, "d": 4}
dict_c = {"b": 20, "e": 5}  # Note: 'b' conflicts with dict_a

# Method 1: update() - modifies original
merged1 = dict_a.copy()
merged1.update(dict_b)
print(f"Merge with update(): {merged1}")

# Method 2: Dictionary unpacking (Python 3.5+)
merged2 = {**dict_a, **dict_b, **dict_c}
print(f"Merge with unpacking: {merged2}")  # Later values overwrite

# Method 3: dict() constructor
merged3 = dict(dict_a, **dict_b)
print(f"Merge with dict(): {merged3}")

# Method 4: Union operator (Python 3.9+)
# merged4 = dict_a | dict_b | dict_c
# print(f"Merge with | operator: {merged4}")

# 9. Dictionary Iteration Patterns
print("\n9. DICTIONARY ITERATION PATTERNS:")
print("-" * 40)

demo_dict = {"apple": 5, "banana": 3, "cherry": 8, "date": 2}

# Iterate over keys (default behavior)
print("Iterating over keys:")
for key in demo_dict:
    print(f"  {key}: {demo_dict[key]}")

# Iterate over keys explicitly
print("\nIterating over keys explicitly:")
for key in demo_dict.keys():
    print(f"  Key: {key}")

# Iterate over values
print("\nIterating over values:")
for value in demo_dict.values():
    print(f"  Value: {value}")

# Iterate over key-value pairs
print("\nIterating over items:")
for key, value in demo_dict.items():
    print(f"  {key} -> {value}")

# Iterate with enumeration
print("\nIterating with enumerate:")
for i, (key, value) in enumerate(demo_dict.items()):
    print(f"  {i}: {key} = {value}")

# Conditional iteration
print("\nConditional iteration (values > 4):")
for key, value in demo_dict.items():
    if value > 4:
        print(f"  {key}: {value}")

# 10. Dictionary Comprehensions (Advanced)
print("\n10. ADVANCED DICTIONARY COMPREHENSIONS:")
print("-" * 40)

# Basic comprehension
squares = {x: x**2 for x in range(6)}
print(f"Squares: {squares}")

# Conditional comprehension
even_squares = {x: x**2 for x in range(10) if x % 2 == 0}
print(f"Even squares: {even_squares}")

# Transform existing dictionary
original_prices = {"apple": 1.00, "banana": 0.50, "cherry": 2.00}
discounted = {item: price * 0.8 for item, price in original_prices.items()}
print(f"20% discount: {discounted}")

# Filter and transform
expensive_items = {item: price for item, price in original_prices.items() if price > 0.75}
print(f"Expensive items: {expensive_items}")

# Swap keys and values
swapped = {value: key for key, value in demo_dict.items()}
print(f"Swapped dict: {swapped}")

# Multiple conditions
filtered_dict = {k: v for k, v in demo_dict.items() if len(k) > 4 and v > 3}
print(f"Long names with value > 3: {filtered_dict}")

# 11. Dictionary Performance and Memory
print("\n11. DICTIONARY PERFORMANCE NOTES:")
print("-" * 40)
print("TIME COMPLEXITIES:")
print("• Access (dict[key]): O(1) average, O(n) worst case")
print("• Insert (dict[key] = value): O(1) average, O(n) worst case") 
print("• Delete (del dict[key]): O(1) average, O(n) worst case")
print("• Search (key in dict): O(1) average, O(n) worst case")
print("• Update (dict.update()): O(k) where k is number of items")
print("• Copy (dict.copy()): O(n)")

print("\nSPACE COMPLEXITY:")
print("• Dictionary overhead: ~240 bytes + 8 bytes per key-value pair")
print("• Memory efficient for sparse data")
print("• Hash table with 2/3 load factor for performance")

print("\nBEST PRACTICES:")
print("• Use get() for safe access instead of []")
print("• Use 'in' to check key existence before access")
print("• Use defaultdict for automatic initialization")
print("• Prefer dict comprehensions over loops for creation")
print("• Use setdefault() for conditional initialization")

# =============================================================================
# PART 2: DICTIONARY OPERATIONS AND METHODS
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: DICTIONARY OPERATIONS")
print("=" * 60)

sample_dict = {"a": 1, "b": 2, "c": 3, "d": 4}

# Accessing elements
print("\n1. ACCESSING ELEMENTS:")
print("-" * 40)
print(f"dict['a']: {sample_dict['a']}")
print(f"dict.get('a'): {sample_dict.get('a')}")
print(f"dict.get('z', 'default'): {sample_dict.get('z', 'default')}")

# Adding/Updating elements
print("\n2. ADDING/UPDATING:")
print("-" * 40)
sample_dict["e"] = 5  # Add new key
sample_dict["a"] = 10  # Update existing key
sample_dict.update({"f": 6, "g": 7})  # Bulk update
print(f"After updates: {sample_dict}")

# Removing elements
print("\n3. REMOVING ELEMENTS:")
print("-" * 40)
removed_pop = sample_dict.pop("g")  # Remove and return value
print(f"Removed with pop(): {removed_pop}")

removed_popitem = sample_dict.popitem()  # Remove last item (3.7+)
print(f"Removed with popitem(): {removed_popitem}")

del sample_dict["f"]  # Delete specific key
print(f"After removals: {sample_dict}")

# Dictionary views
print("\n4. DICTIONARY VIEWS:")
print("-" * 40)
print(f"Keys: {list(sample_dict.keys())}")
print(f"Values: {list(sample_dict.values())}")
print(f"Items: {list(sample_dict.items())}")

# =============================================================================
# PART 3: COMMON INTERVIEW PATTERNS
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: INTERVIEW PATTERNS & WHEN TO USE DICTIONARIES")
print("=" * 60)

print("\nWHEN TO USE DICTIONARIES:")
print("-" * 40)
print("1. FREQUENCY COUNTING - Count occurrences of elements")
print("2. MAPPING/LOOKUP - Quick O(1) key-value lookups")
print("3. GROUPING - Group elements by some criteria")
print("4. CACHING/MEMOIZATION - Store computed results")
print("5. INDEX MAPPING - Map elements to their positions")
print("6. TWO SUM PROBLEMS - Store complements for O(n) solutions")
print("7. GRAPH REPRESENTATIONS - Adjacency lists")
print("8. STATE TRACKING - Track visited nodes, current states")

# =============================================================================
# PART 4: INTERVIEW QUESTIONS WITH SOLUTIONS
# =============================================================================

print("\n" + "=" * 60)
print("PART 4: INTERVIEW QUESTIONS WITH SOLUTIONS")
print("=" * 60)

# Question 1: Character Frequency Counter
print("\n1. CHARACTER FREQUENCY COUNTER:")
print("-" * 40)
def char_frequency(s):
    """Count frequency of each character in string"""
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    return freq

# Alternative using defaultdict
from collections import defaultdict
def char_frequency_defaultdict(s):
    freq = defaultdict(int)
    for char in s:
        freq[char] += 1
    return dict(freq)

# Alternative using Counter
from collections import Counter
def char_frequency_counter(s):
    return dict(Counter(s))

test_string = "hello world"
print(f"String: '{test_string}'")
print(f"Frequency: {char_frequency(test_string)}")

# Question 2: Two Sum Problem
print("\n2. TWO SUM PROBLEM:")
print("-" * 40)
def two_sum(nums, target):
    """Find indices of two numbers that add up to target"""
    seen = {}  # {value: index}
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    
    return []

nums = [2, 7, 11, 15]
target = 9
print(f"Array: {nums}, Target: {target}")
print(f"Indices: {two_sum(nums, target)}")

# Question 3: Group Anagrams
print("\n3. GROUP ANAGRAMS:")
print("-" * 40)
def group_anagrams(strs):
    """Group strings that are anagrams of each other"""
    groups = {}
    
    for s in strs:
        # Sort characters to create key
        key = ''.join(sorted(s))
        if key not in groups:
            groups[key] = []
        groups[key].append(s)
    
    return list(groups.values())

words = ["eat", "tea", "tan", "ate", "nat", "bat"]
print(f"Words: {words}")
print(f"Grouped: {group_anagrams(words)}")

# Question 4: First Non-Repeating Character
print("\n4. FIRST NON-REPEATING CHARACTER:")
print("-" * 40)
def first_non_repeating(s):
    """Find first character that appears only once"""
    freq = {}
    
    # Count frequencies
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    
    # Find first non-repeating
    for char in s:
        if freq[char] == 1:
            return char
    
    return None

test_str = "leetcode"
print(f"String: '{test_str}'")
print(f"First non-repeating: '{first_non_repeating(test_str)}'")

# Question 5: Subarray Sum Equals K
print("\n5. SUBARRAY SUM EQUALS K:")
print("-" * 40)
def subarray_sum_k(nums, k):
    """Count number of subarrays with sum equal to k"""
    count = 0
    cumsum = 0
    sum_freq = {0: 1}  # {cumulative_sum: frequency}
    
    for num in nums:
        cumsum += num
        
        # If (cumsum - k) exists, we found valid subarrays
        if cumsum - k in sum_freq:
            count += sum_freq[cumsum - k]
        
        # Update frequency of current cumsum
        sum_freq[cumsum] = sum_freq.get(cumsum, 0) + 1
    
    return count

arr = [1, 1, 1]
k_val = 2
print(f"Array: {arr}, K: {k_val}")
print(f"Subarrays with sum {k_val}: {subarray_sum_k(arr, k_val)}")

# =============================================================================
# PART 5: ADVANCED DICTIONARY CONCEPTS
# =============================================================================

print("\n" + "=" * 60)
print("PART 5: ADVANCED CONCEPTS")
print("=" * 60)

# Dictionary Comprehensions
print("\n1. DICTIONARY COMPREHENSIONS:")
print("-" * 40)

# Basic comprehension
squares = {x: x**2 for x in range(6)}
print(f"Squares: {squares}")

# Conditional comprehension
even_squares = {x: x**2 for x in range(10) if x % 2 == 0}
print(f"Even squares: {even_squares}")

# Transform existing dict
original = {"a": 1, "b": 2, "c": 3}
doubled = {k: v*2 for k, v in original.items()}
print(f"Doubled values: {doubled}")

# Nested Dictionaries
print("\n2. NESTED DICTIONARIES:")
print("-" * 40)
nested_dict = {
    "user1": {"name": "Alice", "age": 25, "scores": [85, 90, 78]},
    "user2": {"name": "Bob", "age": 30, "scores": [92, 88, 95]}
}

# Accessing nested data
print(f"User1 name: {nested_dict['user1']['name']}")
print(f"User2 first score: {nested_dict['user2']['scores'][0]}")

# Safe nested access
def safe_get(d, keys, default=None):
    """Safely get nested dictionary value"""
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d

print(f"Safe access: {safe_get(nested_dict, ['user1', 'name'])}")
print(f"Safe access (missing): {safe_get(nested_dict, ['user3', 'name'], 'Not found')}")

# =============================================================================
# PART 6: COLLECTIONS MODULE DICTIONARIES
# =============================================================================

print("\n" + "=" * 60)
print("PART 6: COLLECTIONS MODULE")
print("=" * 60)

# defaultdict
print("\n1. DEFAULTDICT:")
print("-" * 40)
from collections import defaultdict

# Group words by first letter
words = ["apple", "banana", "apricot", "cherry", "avocado"]
groups = defaultdict(list)

for word in words:
    groups[word[0]].append(word)

print(f"Grouped by first letter: {dict(groups)}")

# Counter
print("\n2. COUNTER:")
print("-" * 40)
from collections import Counter

text = "hello world"
char_count = Counter(text)
print(f"Character counts: {char_count}")
print(f"Most common 3: {char_count.most_common(3)}")

# OrderedDict (less relevant in Python 3.7+)
print("\n3. ORDEREDDICT:")
print("-" * 40)
from collections import OrderedDict

ordered = OrderedDict([("first", 1), ("second", 2), ("third", 3)])
print(f"Ordered dict: {ordered}")

# =============================================================================
# PART 7: COMMON INTERVIEW QUESTIONS
# =============================================================================

print("\n" + "=" * 60)
print("PART 7: MORE INTERVIEW QUESTIONS")
print("=" * 60)

# Question 6: Valid Anagram
print("\n6. VALID ANAGRAM:")
print("-" * 40)
def is_anagram(s, t):
    """Check if two strings are anagrams"""
    if len(s) != len(t):
        return False
    
    char_count = {}
    
    # Count characters in first string
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    
    # Subtract characters from second string
    for char in t:
        if char not in char_count:
            return False
        char_count[char] -= 1
        if char_count[char] == 0:
            del char_count[char]
    
    return len(char_count) == 0

print(f"'listen' and 'silent': {is_anagram('listen', 'silent')}")
print(f"'hello' and 'world': {is_anagram('hello', 'world')}")

# Question 7: Top K Frequent Elements
print("\n7. TOP K FREQUENT ELEMENTS:")
print("-" * 40)
def top_k_frequent(nums, k):
    """Find k most frequent elements"""
    from collections import Counter
    import heapq
    
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)

nums = [1, 1, 1, 2, 2, 3]
k = 2
print(f"Array: {nums}, K: {k}")
print(f"Top {k} frequent: {top_k_frequent(nums, k)}")

# Question 8: Longest Substring Without Repeating Characters
print("\n8. LONGEST SUBSTRING WITHOUT REPEATING:")
print("-" * 40)
def length_longest_substring(s):
    """Find length of longest substring without repeating characters"""
    char_index = {}
    max_length = 0
    start = 0
    
    for end, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        
        char_index[char] = end
        max_length = max(max_length, end - start + 1)
    
    return max_length

test_string = "abcabcbb"
print(f"String: '{test_string}'")
print(f"Longest substring length: {length_longest_substring(test_string)}")

# =============================================================================
# PART 8: PERFORMANCE AND BEST PRACTICES
# =============================================================================

print("\n" + "=" * 60)
print("PART 8: PERFORMANCE & BEST PRACTICES")
print("=" * 60)

print("\nTIME COMPLEXITIES:")
print("-" * 40)
print("• Access/Search: O(1) average, O(n) worst case")
print("• Insert/Update: O(1) average, O(n) worst case")
print("• Delete: O(1) average, O(n) worst case")
print("• Space: O(n)")

print("\nBEST PRACTICES:")
print("-" * 40)
print("1. Use get() method for safe access")
print("2. Use defaultdict for grouping operations")
print("3. Use Counter for frequency counting")
print("4. Use dict comprehensions for transformations")
print("5. Check 'in' before accessing to avoid KeyError")
print("6. Use setdefault() for conditional initialization")

# Examples of best practices
print("\nBEST PRACTICE EXAMPLES:")
print("-" * 40)

# Bad: KeyError prone
def bad_grouping(items):
    groups = {}
    for item in items:
        key = item % 3
        if key not in groups:  # Extra check needed
            groups[key] = []
        groups[key].append(item)
    return groups

# Good: Using defaultdict
def good_grouping(items):
    groups = defaultdict(list)
    for item in items:
        key = item % 3
        groups[key].append(item)  # No checking needed
    return dict(groups)

items = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(f"Items: {items}")
print(f"Grouped by remainder: {good_grouping(items)}")

# =============================================================================
# PART 9: COMMON MISTAKES AND GOTCHAS
# =============================================================================

print("\n" + "=" * 60)
print("PART 9: COMMON MISTAKES & GOTCHAS")
print("=" * 60)

print("\nCOMMON MISTAKES:")
print("-" * 40)

# Mistake 1: Mutable default arguments
print("1. MUTABLE DEFAULT ARGUMENTS:")
def bad_function(key, d={}):  # DON'T DO THIS
    d[key] = "value"
    return d

def good_function(key, d=None):  # DO THIS
    if d is None:
        d = {}
    d[key] = "value"
    return d

# Mistake 2: Modifying dict while iterating
print("\n2. MODIFYING DICT WHILE ITERATING:")
sample = {"a": 1, "b": 2, "c": 3}

# Bad way (can cause RuntimeError)
# for key in sample:
#     if sample[key] > 1:
#         del sample[key]  # DON'T DO THIS

# Good way
to_delete = [key for key, value in sample.items() if value > 1]
for key in to_delete:
    del sample[key]

print(f"After filtering: {sample}")

# Mistake 3: Assuming order in older Python versions
print("\n3. DICTIONARY ORDERING:")
print("• Python 3.7+: Dictionaries maintain insertion order")
print("• Python < 3.7: Order not guaranteed")
print("• Use OrderedDict for explicit ordering in older versions")

# =============================================================================
# PART 10: INTERVIEW PREPARATION CHECKLIST
# =============================================================================

print("\n" + "=" * 60)
print("PART 10: INTERVIEW PREPARATION CHECKLIST")
print("=" * 60)

checklist = """
DICTIONARY INTERVIEW CHECKLIST:
================================

✓ BASIC OPERATIONS:
  - Creation methods (literal, constructor, comprehension)
  - Access ([], get(), setdefault())
  - Update (assignment, update(), merge)
  - Delete (del, pop(), popitem(), clear())

✓ COMMON PATTERNS:
  - Frequency counting
  - Two-pointer/Two-sum problems
  - Grouping and categorization
  - Caching and memoization
  - Graph representations (adjacency lists)

✓ IMPORTANT METHODS:
  - get(key, default)
  - setdefault(key, default)
  - pop(key, default)
  - popitem()
  - keys(), values(), items()
  - update()

✓ COLLECTIONS MODULE:
  - Counter (frequency counting)
  - defaultdict (automatic initialization)
  - OrderedDict (explicit ordering)

✓ PERFORMANCE:
  - O(1) average time complexity
  - Hash collision handling
  - Memory efficiency considerations

✓ COMMON INTERVIEW PROBLEMS:
  - Two Sum variants
  - Anagram problems
  - Frequency analysis
  - Substring problems
  - Graph traversal with state tracking

PRACTICE PROBLEMS:
==================
BEGINNER (0-1 years):
1. Group Anagrams
2. Top K Frequent Elements
3. Subarray Sum Equals K
4. Longest Substring Without Repeating Characters
5. Valid Anagram
6. First Non-Repeating Character
7. Word Pattern
8. Isomorphic Strings
9. Contains Duplicate II
10. Minimum Window Substring

INTERMEDIATE-ADVANCED (2-3 years):
11. LRU Cache Implementation
12. Design Word Dictionary (Trie + Wildcard)
13. Design Twitter Timeline
14. Alien Dictionary (Topological Sort)
15. Design In-Memory File System
16. Design Hit Counter
17. Autocomplete System with Ranking
18. Rate Limiter (Token Bucket)
19. Consistent Hashing Implementation
20. Design Chat System
21. Design URL Shortener
22. Design Search Autocomplete
23. Design Distributed Cache
24. Design Time-based Key-Value Store
25. Design Logger System

ADVANCED PATTERNS TO MASTER:
============================
• Sliding Window + HashMap
• Prefix Sum + HashMap
• Two Pointers + HashMap
• Backtracking + Memoization
• Graph Algorithms + State Tracking
• Design Patterns (Factory, Observer)
• Distributed Systems Concepts
• Concurrency with Thread-Safe Operations
• Memory-Efficient Data Structures
• Time-Series Data Handling

SYSTEM DESIGN COMPONENTS:
=========================
• Caching Strategies (LRU, LFU, TTL)
• Load Balancing (Consistent Hashing)
• Rate Limiting (Token Bucket, Sliding Window)
• Data Partitioning (Sharding)
• Replication and Consistency
• Message Queues and Event Systems
• Search and Indexing Systems
• Real-time Data Processing
• Monitoring and Logging
• API Design and Versioning
"""

print(checklist)

# =============================================================================
# PART 11: ADVANCED INTERVIEW QUESTIONS (2-3 YEARS EXPERIENCE)
# =============================================================================

print("\n" + "=" * 60)
print("PART 11: ADVANCED INTERVIEW QUESTIONS (EXPERIENCED PROFESSIONALS)")
print("=" * 60)

# Question 9: LRU Cache Implementation
print("\n9. LRU CACHE IMPLEMENTATION:")
print("-" * 40)
class LRUCache:
    """Least Recently Used Cache with O(1) operations"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # key -> node
        
        # Create dummy head and tail nodes
        self.head = self.Node(0, 0)
        self.tail = self.Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head
    
    class Node:
        def __init__(self, key, val):
            self.key = key
            self.val = val
            self.prev = None
            self.next = None
    
    def get(self, key):
        if key in self.cache:
            node = self.cache[key]
            # Move to front (most recently used)
            self._remove(node)
            self._add_to_front(node)
            return node.val
        return -1
    
    def put(self, key, value):
        if key in self.cache:
            # Update existing
            node = self.cache[key]
            node.val = value
            self._remove(node)
            self._add_to_front(node)
        else:
            # Add new
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]
            
            new_node = self.Node(key, value)
            self.cache[key] = new_node
            self._add_to_front(new_node)
    
    def _remove(self, node):
        """Remove node from doubly linked list"""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _add_to_front(self, node):
        """Add node right after head"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

# Test LRU Cache
lru = LRUCache(2)
lru.put(1, 1)
lru.put(2, 2)
print(f"Get 1: {lru.get(1)}")  # 1
lru.put(3, 3)  # evicts key 2
print(f"Get 2: {lru.get(2)}")  # -1 (not found)
print(f"Get 3: {lru.get(3)}")  # 3

# Question 10: Design Word Dictionary (Trie + Dict)
print("\n10. WORD DICTIONARY WITH WILDCARD SEARCH:")
print("-" * 40)
class WordDictionary:
    """Dictionary that supports wildcard '.' character"""
    
    def __init__(self):
        self.trie = {}
    
    def addWord(self, word):
        node = self.trie
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['#'] = True  # End of word marker
    
    def search(self, word):
        return self._dfs(word, 0, self.trie)
    
    def _dfs(self, word, index, node):
        if index == len(word):
            return '#' in node
        
        char = word[index]
        if char == '.':
            # Wildcard: try all possible characters
            for child_char in node:
                if child_char != '#' and self._dfs(word, index + 1, node[child_char]):
                    return True
            return False
        else:
            if char not in node:
                return False
            return self._dfs(word, index + 1, node[char])

# Test Word Dictionary
wd = WordDictionary()
wd.addWord("bad")
wd.addWord("dad")
wd.addWord("mad")
print(f"Search 'pad': {wd.search('pad')}")  # False
print(f"Search 'bad': {wd.search('bad')}")  # True
print(f"Search '.ad': {wd.search('.ad')}")  # True
print(f"Search 'b..': {wd.search('b..')}")  # True

# Question 11: Design Twitter (Advanced System Design)
print("\n11. DESIGN TWITTER (SIMPLIFIED):")
print("-" * 40)
from collections import defaultdict
import heapq

class Twitter:
    """Simplified Twitter with follow/unfollow and news feed"""
    
    def __init__(self):
        self.tweets = defaultdict(list)  # user_id -> [(timestamp, tweet_id)]
        self.following = defaultdict(set)  # user_id -> set of followed users
        self.timestamp = 0
    
    def postTweet(self, userId, tweetId):
        self.tweets[userId].append((self.timestamp, tweetId))
        self.timestamp += 1
    
    def getNewsFeed(self, userId):
        """Get 10 most recent tweets from user and people they follow"""
        # Get all relevant users (self + following)
        relevant_users = self.following[userId] | {userId}
        
        # Collect all tweets with timestamps
        all_tweets = []
        for user in relevant_users:
            for timestamp, tweet_id in self.tweets[user]:
                all_tweets.append((timestamp, tweet_id))
        
        # Sort by timestamp (most recent first) and return top 10
        all_tweets.sort(reverse=True)
        return [tweet_id for _, tweet_id in all_tweets[:10]]
    
    def follow(self, followerId, followeeId):
        if followerId != followeeId:
            self.following[followerId].add(followeeId)
    
    def unfollow(self, followerId, followeeId):
        self.following[followerId].discard(followeeId)

# Test Twitter
twitter = Twitter()
twitter.postTweet(1, 5)
print(f"User 1 news feed: {twitter.getNewsFeed(1)}")  # [5]
twitter.follow(1, 2)
twitter.postTweet(2, 6)
print(f"User 1 news feed: {twitter.getNewsFeed(1)}")  # [6, 5]

# Question 12: Alien Dictionary (Topological Sort)
print("\n12. ALIEN DICTIONARY (TOPOLOGICAL SORT):")
print("-" * 40)
def alien_order(words):
    """Find order of characters in alien language"""
    from collections import defaultdict, deque
    
    # Build graph
    graph = defaultdict(set)
    in_degree = defaultdict(int)
    
    # Initialize all characters
    for word in words:
        for char in word:
            in_degree[char] = 0
    
    # Build edges
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        min_len = min(len(word1), len(word2))
        
        # Check for invalid case
        if len(word1) > len(word2) and word1[:min_len] == word2[:min_len]:
            return ""  # Invalid
        
        # Find first different character
        for j in range(min_len):
            if word1[j] != word2[j]:
                if word2[j] not in graph[word1[j]]:
                    graph[word1[j]].add(word2[j])
                    in_degree[word2[j]] += 1
                break
    
    # Topological sort using Kahn's algorithm
    queue = deque([char for char in in_degree if in_degree[char] == 0])
    result = []
    
    while queue:
        char = queue.popleft()
        result.append(char)
        
        for neighbor in graph[char]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return ''.join(result) if len(result) == len(in_degree) else ""

words = ["wrt", "wrf", "er", "ett", "rftt"]
print(f"Words: {words}")
print(f"Alien order: '{alien_order(words)}'")

# Question 13: Design In-Memory File System
print("\n13. DESIGN IN-MEMORY FILE SYSTEM:")
print("-" * 40)
class FileSystem:
    """In-memory file system with directories and files"""
    
    def __init__(self):
        self.root = {}
    
    def ls(self, path):
        """List contents of directory or return file name"""
        node = self._navigate(path)
        if isinstance(node, str):  # It's a file
            return [path.split('/')[-1]]
        else:  # It's a directory
            return sorted(node.keys())
    
    def mkdir(self, path):
        """Create directory"""
        self._navigate(path, create=True)
    
    def addContentToFile(self, filePath, content):
        """Add content to file (create if doesn't exist)"""
        parts = filePath.split('/')
        dir_path = '/'.join(parts[:-1]) if len(parts) > 1 else '/'
        filename = parts[-1]
        
        dir_node = self._navigate(dir_path, create=True)
        if filename not in dir_node:
            dir_node[filename] = ""
        dir_node[filename] += content
    
    def readContentFromFile(self, filePath):
        """Read content from file"""
        parts = filePath.split('/')
        dir_path = '/'.join(parts[:-1]) if len(parts) > 1 else '/'
        filename = parts[-1]
        
        dir_node = self._navigate(dir_path)
        return dir_node[filename]
    
    def _navigate(self, path, create=False):
        """Navigate to path, optionally creating directories"""
        if path == '/':
            return self.root
        
        parts = [p for p in path.split('/') if p]
        node = self.root
        
        for part in parts:
            if part not in node:
                if create:
                    node[part] = {}
                else:
                    raise KeyError(f"Path not found: {path}")
            node = node[part]
        
        return node

# Test File System
fs = FileSystem()
print(f"ls /: {fs.ls('/')}")  # []
fs.mkdir("/a/b/c")
fs.addContentToFile("/a/b/c/d", "hello")
print(f"ls /: {fs.ls('/')}")  # ['a']
print(f"ls /a/b/c: {fs.ls('/a/b/c')}")  # ['d']
print(f"Read file: '{fs.readContentFromFile('/a/b/c/d')}'")  # 'hello'

# Question 14: Design Hit Counter
print("\n14. DESIGN HIT COUNTER:")
print("-" * 40)
from collections import deque

class HitCounter:
    """Count hits in last 5 minutes (300 seconds)"""
    
    def __init__(self):
        self.hits = deque()  # Store (timestamp, count) pairs
    
    def hit(self, timestamp):
        """Record a hit at given timestamp"""
        if self.hits and self.hits[-1][0] == timestamp:
            # Same timestamp, increment count
            self.hits[-1] = (timestamp, self.hits[-1][1] + 1)
        else:
            # New timestamp
            self.hits.append((timestamp, 1))
        
        # Clean old hits
        self._cleanup(timestamp)
    
    def getHits(self, timestamp):
        """Get number of hits in last 300 seconds"""
        self._cleanup(timestamp)
        return sum(count for _, count in self.hits)
    
    def _cleanup(self, timestamp):
        """Remove hits older than 300 seconds"""
        while self.hits and self.hits[0][0] <= timestamp - 300:
            self.hits.popleft()

# Test Hit Counter
hc = HitCounter()
hc.hit(1)
hc.hit(2)
hc.hit(3)
print(f"Hits at timestamp 4: {hc.getHits(4)}")  # 3
hc.hit(300)
print(f"Hits at timestamp 300: {hc.getHits(300)}")  # 4
print(f"Hits at timestamp 301: {hc.getHits(301)}")  # 3 (hit at 1 expired)

# Question 15: Implement Autocomplete System
print("\n15. AUTOCOMPLETE SYSTEM:")
print("-" * 40)
class AutocompleteSystem:
    """Autocomplete system with hot ranking"""
    
    def __init__(self, sentences, times):
        self.trie = {}
        self.hot = {}  # sentence -> frequency
        self.current_input = ""
        
        # Build initial trie and hot ranking
        for sentence, time in zip(sentences, times):
            self.hot[sentence] = time
            self._add_to_trie(sentence)
    
    def input(self, c):
        if c == '#':
            # End of input, save and reset
            if self.current_input:
                self.hot[self.current_input] = self.hot.get(self.current_input, 0) + 1
                self._add_to_trie(self.current_input)
            self.current_input = ""
            return []
        else:
            # Continue building input
            self.current_input += c
            return self._get_suggestions()
    
    def _add_to_trie(self, sentence):
        """Add sentence to trie"""
        node = self.trie
        for char in sentence:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['#'] = True
    
    def _get_suggestions(self):
        """Get top 3 suggestions for current input"""
        # Find all sentences with current prefix
        candidates = []
        for sentence in self.hot:
            if sentence.startswith(self.current_input):
                candidates.append((self.hot[sentence], sentence))
        
        # Sort by frequency (desc) then lexicographically (asc)
        candidates.sort(key=lambda x: (-x[0], x[1]))
        return [sentence for _, sentence in candidates[:3]]

# Test Autocomplete
sentences = ["i love you", "island", "iroman", "i love leetcode"]
times = [5, 3, 2, 2]
ac = AutocompleteSystem(sentences, times)

print(f"Input 'i': {ac.input('i')}")
print(f"Input ' ': {ac.input(' ')}")
print(f"Input 'a': {ac.input('a')}")
print(f"Input '#': {ac.input('#')}")  # Save "i a"

# =============================================================================
# PART 12: SYSTEM DESIGN PATTERNS WITH DICTIONARIES
# =============================================================================

print("\n" + "=" * 60)
print("PART 12: SYSTEM DESIGN PATTERNS")
print("=" * 60)

# Rate Limiter using Token Bucket
print("\n16. RATE LIMITER (TOKEN BUCKET):")
print("-" * 40)
import time

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()
        self.user_buckets = {}  # user_id -> (tokens, last_refill)
    
    def is_allowed(self, user_id):
        """Check if user is allowed to make request"""
        current_time = time.time()
        
        if user_id not in self.user_buckets:
            self.user_buckets[user_id] = [self.capacity, current_time]
        
        tokens, last_refill = self.user_buckets[user_id]
        
        # Refill tokens
        time_passed = current_time - last_refill
        tokens_to_add = time_passed * self.refill_rate
        tokens = min(self.capacity, tokens + tokens_to_add)
        
        # Check if request allowed
        if tokens >= 1:
            tokens -= 1
            self.user_buckets[user_id] = [tokens, current_time]
            return True
        else:
            self.user_buckets[user_id] = [tokens, current_time]
            return False

# Consistent Hashing for Load Balancing
print("\n17. CONSISTENT HASHING:")
print("-" * 40)
import hashlib

class ConsistentHash:
    """Consistent hashing for distributed systems"""
    
    def __init__(self, replicas=150):
        self.replicas = replicas
        self.ring = {}  # hash -> server
        self.sorted_hashes = []
        self.servers = set()
    
    def _hash(self, key):
        """Hash function"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_server(self, server):
        """Add server to the ring"""
        self.servers.add(server)
        for i in range(self.replicas):
            key = f"{server}:{i}"
            hash_val = self._hash(key)
            self.ring[hash_val] = server
            self.sorted_hashes.append(hash_val)
        self.sorted_hashes.sort()
    
    def remove_server(self, server):
        """Remove server from the ring"""
        self.servers.discard(server)
        for i in range(self.replicas):
            key = f"{server}:{i}"
            hash_val = self._hash(key)
            del self.ring[hash_val]
            self.sorted_hashes.remove(hash_val)
    
    def get_server(self, key):
        """Get server responsible for key"""
        if not self.ring:
            return None
        
        hash_val = self._hash(key)
        
        # Find first server clockwise
        for server_hash in self.sorted_hashes:
            if hash_val <= server_hash:
                return self.ring[server_hash]
        
        # Wrap around to first server
        return self.ring[self.sorted_hashes[0]]

# Test Consistent Hashing
ch = ConsistentHash()
ch.add_server("server1")
ch.add_server("server2")
ch.add_server("server3")

keys = ["user1", "user2", "user3", "user4", "user5"]
for key in keys:
    server = ch.get_server(key)
    print(f"Key '{key}' -> {server}")

print("\n" + "=" * 60)
print("ADVANCED DICTIONARY CONCEPTS COMPLETE!")
print("These problems test deep understanding of data structures,")
print("system design, and algorithmic thinking!")
print("=" * 60)