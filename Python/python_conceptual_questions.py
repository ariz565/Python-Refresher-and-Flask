# ===============================================================================
# PYTHON LIST COMPREHENSIONS, LAMBDA FUNCTIONS & CONCEPTUAL QUESTIONS
# Complete Guide for Advanced Python Interviews
# ===============================================================================

"""
COMPREHENSIVE CONCEPTUAL QUESTIONS COVERAGE:
===========================================
1. List Comprehensions (Basic to Advanced)
2. Lambda Functions and Functional Programming
3. OOP Conceptual Questions
4. What is OOP? vs Other Paradigms
5. Classes and Objects Deep Dive
6. Python Keywords and Built-ins
7. Memory Management Concepts
8. Tricky Python Concepts
9. Interview Brain Teasers
10. Advanced Python Features
"""

print("=" * 100)
print("PYTHON LIST COMPREHENSIONS, LAMBDA FUNCTIONS & CONCEPTUAL QUESTIONS")
print("=" * 100)

# ===============================================================================
# 1. LIST COMPREHENSIONS (BASIC TO ADVANCED)
# ===============================================================================

print("\n" + "=" * 80)
print("1. LIST COMPREHENSIONS")
print("=" * 80)

print("""
Q: What are list comprehensions? Why use them?

A: List comprehensions provide a concise way to create lists based on existing sequences.

Syntax: [expression for item in iterable if condition]

Benefits:
- More readable and concise
- Faster than traditional loops
- Memory efficient
- Pythonic way of writing
""")

# BASIC LIST COMPREHENSIONS
print("1. BASIC LIST COMPREHENSIONS:")

# Traditional way
squares_traditional = []
for x in range(10):
    squares_traditional.append(x**2)

# List comprehension way
squares_lc = [x**2 for x in range(10)]

print(f"Traditional: {squares_traditional}")
print(f"List Comp:   {squares_lc}")

# CONDITIONAL LIST COMPREHENSIONS
print("\n2. CONDITIONAL LIST COMPREHENSIONS:")

numbers = range(20)

# Even numbers only
evens = [x for x in numbers if x % 2 == 0]
print(f"Even numbers: {evens}")

# Conditional expression (ternary operator)
positive_negative = [x if x > 0 else -x for x in [-5, -2, 0, 3, 7]]
print(f"Absolute values: {positive_negative}")

# NESTED LIST COMPREHENSIONS
print("\n3. NESTED LIST COMPREHENSIONS:")

# Matrix transpose
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
transposed = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
print(f"Original matrix: {matrix}")
print(f"Transposed: {transposed}")

# Flattening nested lists
nested_list = [[1, 2], [3, 4], [5, 6]]
flattened = [item for sublist in nested_list for item in sublist]
print(f"Nested: {nested_list}")
print(f"Flattened: {flattened}")

# ADVANCED LIST COMPREHENSIONS
print("\n4. ADVANCED LIST COMPREHENSIONS:")

# Multiple conditions
data = range(1, 101)
special_numbers = [x for x in data if x % 3 == 0 if x % 5 == 0]
print(f"Numbers divisible by both 3 and 5: {special_numbers[:5]}...")

# Working with strings
words = ["hello", "world", "python", "programming"]
capitalized = [word.upper() for word in words if len(word) > 5]
print(f"Long words capitalized: {capitalized}")

# Dictionary from list comprehension
word_lengths = {word: len(word) for word in words}
print(f"Word lengths: {word_lengths}")

# Set comprehension
unique_lengths = {len(word) for word in words}
print(f"Unique word lengths: {unique_lengths}")

# GENERATOR EXPRESSIONS (Memory Efficient)
print("\n5. GENERATOR EXPRESSIONS:")

# Generator expression (memory efficient for large datasets)
squares_gen = (x**2 for x in range(1000000))
print(f"Generator: {squares_gen}")
print(f"First 5 squares: {[next(squares_gen) for _ in range(5)]}")

# Memory comparison demonstration
import sys
list_comp = [x for x in range(1000)]
gen_expr = (x for x in range(1000))
print(f"List comprehension size: {sys.getsizeof(list_comp)} bytes")
print(f"Generator expression size: {sys.getsizeof(gen_expr)} bytes")

# PRACTICAL EXAMPLES
print("\n6. PRACTICAL LIST COMPREHENSION EXAMPLES:")

# File processing simulation
file_lines = ["  hello world  ", "PYTHON programming", "  Data Science  ", "machine learning"]
cleaned_lines = [line.strip().title() for line in file_lines if line.strip()]
print(f"Cleaned lines: {cleaned_lines}")

# Data filtering and transformation
student_scores = [
    {"name": "Alice", "score": 85},
    {"name": "Bob", "score": 78},
    {"name": "Charlie", "score": 92},
    {"name": "Diana", "score": 69}
]

high_scorers = [student["name"] for student in student_scores if student["score"] >= 80]
print(f"High scorers: {high_scorers}")

# Complex transformations
coords = [(1, 2), (3, 4), (5, 6)]
distances = [((x**2 + y**2)**0.5) for x, y in coords]
print(f"Distances from origin: {[round(d, 2) for d in distances]}")

# ===============================================================================
# 2. LAMBDA FUNCTIONS AND FUNCTIONAL PROGRAMMING
# ===============================================================================

print("\n" + "=" * 80)
print("2. LAMBDA FUNCTIONS AND FUNCTIONAL PROGRAMMING")
print("=" * 80)

print("""
Q: What are lambda functions? When to use them?

A: Lambda functions are anonymous functions defined with the 'lambda' keyword.

Syntax: lambda arguments: expression

Use cases:
- Short, simple functions
- Function arguments (map, filter, sort)
- Event-driven programming
- Functional programming paradigms

Limitations:
- Single expression only
- No statements (print, return, etc.)
- Less readable for complex logic
""")

# BASIC LAMBDA FUNCTIONS
print("1. BASIC LAMBDA FUNCTIONS:")

# Simple lambda
square = lambda x: x**2
print(f"Square of 5: {square(5)}")

# Lambda with multiple arguments
add = lambda x, y: x + y
print(f"Add 3 and 7: {add(3, 7)}")

# Lambda with default arguments
greet = lambda name, greeting="Hello": f"{greeting}, {name}!"
print(f"Greeting: {greet('Alice')}")
print(f"Custom greeting: {greet('Bob', 'Hi')}")

# LAMBDA WITH BUILT-IN FUNCTIONS
print("\n2. LAMBDA WITH MAP, FILTER, REDUCE:")

from functools import reduce

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# MAP - apply function to each element
squares = list(map(lambda x: x**2, numbers))
print(f"Squares: {squares}")

# FILTER - filter elements based on condition
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(f"Even numbers: {evens}")

# REDUCE - reduce sequence to single value
sum_all = reduce(lambda x, y: x + y, numbers)
print(f"Sum of all numbers: {sum_all}")

# Product of all numbers
product = reduce(lambda x, y: x * y, numbers)
print(f"Product of all numbers: {product}")

# LAMBDA FOR SORTING
print("\n3. LAMBDA FOR SORTING:")

# Sort by different criteria
students = [
    ("Alice", 85, 20),
    ("Bob", 78, 22),
    ("Charlie", 92, 19),
    ("Diana", 69, 21)
]

# Sort by grade (second element)
by_grade = sorted(students, key=lambda student: student[1])
print(f"Sorted by grade: {by_grade}")

# Sort by age (third element), descending
by_age_desc = sorted(students, key=lambda student: student[2], reverse=True)
print(f"Sorted by age (desc): {by_age_desc}")

# Sort by name length
by_name_length = sorted(students, key=lambda student: len(student[0]))
print(f"Sorted by name length: {by_name_length}")

# ADVANCED LAMBDA EXAMPLES
print("\n4. ADVANCED LAMBDA EXAMPLES:")

# Lambda returning lambda (closure)
def multiplier(n):
    return lambda x: x * n

double = multiplier(2)
triple = multiplier(3)
print(f"Double 5: {double(5)}")
print(f"Triple 5: {triple(5)}")

# Lambda with conditional expressions
max_lambda = lambda x, y: x if x > y else y
print(f"Max of 10 and 7: {max_lambda(10, 7)}")

# Lambda for data processing
data = [
    {"name": "product1", "price": 100, "quantity": 5},
    {"name": "product2", "price": 200, "quantity": 3},
    {"name": "product3", "price": 50, "quantity": 10}
]

# Calculate total value for each product
with_total = list(map(lambda item: {**item, "total": item["price"] * item["quantity"]}, data))
print(f"With totals: {with_total}")

# Find expensive products
expensive = list(filter(lambda item: item["price"] > 75, data))
print(f"Expensive products: {expensive}")

# FUNCTIONAL PROGRAMMING CONCEPTS
print("\n5. FUNCTIONAL PROGRAMMING CONCEPTS:")

# Higher-order functions
def apply_operation(numbers, operation):
    return [operation(x) for x in numbers]

nums = [1, 2, 3, 4, 5]
results = apply_operation(nums, lambda x: x**3)
print(f"Cubes: {results}")

# Function composition
def compose(f, g):
    return lambda x: f(g(x))

add_one = lambda x: x + 1
square_it = lambda x: x**2
add_then_square = compose(square_it, add_one)

print(f"Add 1 then square 5: {add_then_square(5)}")  # (5+1)^2 = 36

# Partial functions
from functools import partial

def power(base, exponent):
    return base ** exponent

# Create specialized functions
square_func = partial(power, exponent=2)
cube_func = partial(power, exponent=3)

print(f"Square of 4: {square_func(4)}")
print(f"Cube of 4: {cube_func(4)}")

# ===============================================================================
# 3. OOP CONCEPTUAL QUESTIONS
# ===============================================================================

print("\n" + "=" * 80)
print("3. OOP CONCEPTUAL QUESTIONS")
print("=" * 80)

print("""
Q: What is Object-Oriented Programming? How does it differ from other paradigms?

A: OOP is a programming paradigm based on objects containing data and code.

KEY CONCEPTS:
1. Class: Blueprint for creating objects
2. Object: Instance of a class
3. Encapsulation: Bundling data and methods
4. Inheritance: Creating new classes from existing ones
5. Polymorphism: Same interface, different implementations
6. Abstraction: Hiding complex implementation details

PARADIGM COMPARISON:
- Procedural: Functions and procedures
- Functional: Mathematical functions, immutability
- Object-Oriented: Objects and classes
""")

# CLASSES AND OBJECTS DEEP DIVE
print("CLASSES AND OBJECTS DEEP DIVE:")

class BankAccount:
    """Comprehensive example of OOP concepts"""
    
    # Class variables (shared by all instances)
    bank_name = "Python Bank"
    interest_rate = 0.02
    total_accounts = 0
    
    def __init__(self, account_holder, initial_balance=0):
        # Instance variables (unique to each instance)
        self.account_holder = account_holder
        self.balance = initial_balance
        self.transaction_history = []
        
        # Increment class variable
        BankAccount.total_accounts += 1
        self.account_number = f"ACC{BankAccount.total_accounts:04d}"
        
        self._log_transaction("Account created", initial_balance)
    
    def deposit(self, amount):
        """Instance method"""
        if amount > 0:
            self.balance += amount
            self._log_transaction("Deposit", amount)
            return True
        return False
    
    def withdraw(self, amount):
        """Instance method with validation"""
        if 0 < amount <= self.balance:
            self.balance -= amount
            self._log_transaction("Withdrawal", -amount)
            return True
        return False
    
    def _log_transaction(self, transaction_type, amount):
        """Private method (convention)"""
        from datetime import datetime
        self.transaction_history.append({
            "type": transaction_type,
            "amount": amount,
            "timestamp": datetime.now(),
            "balance": self.balance
        })
    
    @classmethod
    def get_bank_info(cls):
        """Class method - operates on class, not instance"""
        return f"Bank: {cls.bank_name}, Total Accounts: {cls.total_accounts}"
    
    @staticmethod
    def validate_account_number(account_number):
        """Static method - utility function"""
        return account_number.startswith("ACC") and len(account_number) == 7
    
    def __str__(self):
        """String representation for users"""
        return f"Account {self.account_number}: {self.account_holder} - ${self.balance:.2f}"
    
    def __repr__(self):
        """String representation for developers"""
        return f"BankAccount('{self.account_holder}', {self.balance})"
    
    def __eq__(self, other):
        """Equality comparison"""
        if isinstance(other, BankAccount):
            return self.account_number == other.account_number
        return False

# Demonstration
print("\nBank Account Example:")
acc1 = BankAccount("Alice", 1000)
acc2 = BankAccount("Bob", 500)

print(f"Account 1: {acc1}")
print(f"Account 2: {acc2}")

acc1.deposit(200)
acc2.withdraw(100)

print(f"After transactions:")
print(f"Account 1: {acc1}")
print(f"Account 2: {acc2}")

print(f"Bank info: {BankAccount.get_bank_info()}")
print(f"Valid account number: {BankAccount.validate_account_number('ACC0001')}")

# ===============================================================================
# 4. PYTHON KEYWORDS AND BUILT-INS
# ===============================================================================

print("\n" + "=" * 80)
print("4. PYTHON KEYWORDS AND BUILT-INS")
print("=" * 80)

print("""
Q: Explain important Python keywords and their usage:
""")

# PYTHON KEYWORDS DEMONSTRATION
import keyword

print(f"Total Python keywords: {len(keyword.kwlist)}")
print(f"Keywords: {keyword.kwlist}")

print("\nKey Python Keywords with Examples:")

# and, or, not
result_and = True and False
result_or = True or False
result_not = not True
print(f"Logical operators: and={result_and}, or={result_or}, not={result_not}")

# is vs ==
a = [1, 2, 3]
b = [1, 2, 3]
c = a
print(f"== vs is: a==b: {a==b}, a is b: {a is b}, a is c: {a is c}")

# in operator
print(f"'python' in 'python programming': {'python' in 'python programming'}")
print(f"3 in [1,2,3,4]: {3 in [1,2,3,4]}")

# pass, break, continue
print("\nControl flow keywords:")
for i in range(5):
    if i == 2:
        continue  # Skip iteration
    if i == 4:
        break     # Exit loop
    if i == 1:
        pass      # Do nothing
    print(f"Loop iteration: {i}")

# try, except, finally, else
print("\nException handling:")
try:
    result = 10 / 2
except ZeroDivisionError:
    print("Cannot divide by zero")
else:
    print(f"Division successful: {result}")
finally:
    print("Finally block always executes")

# with statement (context managers)
print("\nContext manager:")
with open(__file__, 'r') as f:
    first_line = f.readline()
    print(f"First line of this file: {first_line[:50]}...")

# yield (generators)
def fibonacci_generator(n):
    """Generator function using yield"""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

print(f"\nFibonacci using generator: {list(fibonacci_generator(10))}")

# global and nonlocal
global_var = "I'm global"

def outer_function():
    local_var = "I'm local to outer"
    
    def inner_function():
        nonlocal local_var
        global global_var
        local_var = "Modified by inner"
        global_var = "Modified globally"
    
    inner_function()
    return local_var

result = outer_function()
print(f"After function calls: local={result}, global={global_var}")

# ===============================================================================
# 5. TRICKY PYTHON CONCEPTS
# ===============================================================================

print("\n" + "=" * 80)
print("5. TRICKY PYTHON CONCEPTS")
print("=" * 80)

print("""
Q: Explain these tricky Python concepts:
""")

# MUTABLE DEFAULT ARGUMENTS (Common Pitfall)
print("1. MUTABLE DEFAULT ARGUMENTS:")

def bad_function(data=[]):  # DANGEROUS!
    data.append(1)
    return data

def good_function(data=None):  # SAFE!
    if data is None:
        data = []
    data.append(1)
    return data

print(f"Bad function call 1: {bad_function()}")
print(f"Bad function call 2: {bad_function()}")  # Accumulates!
print(f"Good function call 1: {good_function()}")
print(f"Good function call 2: {good_function()}")

# LATE BINDING CLOSURES
print("\n2. LATE BINDING CLOSURES:")

# Problem
functions_bad = []
for i in range(3):
    functions_bad.append(lambda: i)  # i is captured by reference

print("Bad closures:", [f() for f in functions_bad])  # All return 2!

# Solution 1: Default argument
functions_good1 = []
for i in range(3):
    functions_good1.append(lambda x=i: x)  # Capture by value

print("Good closures 1:", [f() for f in functions_good1])

# Solution 2: Using functools.partial
from functools import partial

def return_value(x):
    return x

functions_good2 = [partial(return_value, i) for i in range(3)]
print("Good closures 2:", [f() for f in functions_good2])

# CHAINED COMPARISONS
print("\n3. CHAINED COMPARISONS:")

x = 5
result = 1 < x < 10
print(f"1 < {x} < 10: {result}")

# Tricky case
result_tricky = False == False in [False]
print(f"False == False in [False]: {result_tricky}")  # True!
# This is equivalent to: (False == False) and (False in [False])

# BOOLEAN EVALUATION
print("\n4. BOOLEAN EVALUATION:")

# Truthiness of different types
values = [0, 1, "", "hello", [], [1], {}, {"a": 1}, None]
for val in values:
    print(f"bool({repr(val)}): {bool(val)}")

# Short-circuit evaluation
def expensive_function():
    print("Expensive function called!")
    return True

result = False and expensive_function()  # expensive_function not called
print(f"Short-circuit result: {result}")

# OPERATOR PRECEDENCE
print("\n5. OPERATOR PRECEDENCE:")

# Tricky expression
result = 2 + 3 * 4 ** 2
print(f"2 + 3 * 4 ** 2 = {result}")  # 2 + 3 * 16 = 50

# Boolean operators
result = True or False and False
print(f"True or False and False = {result}")  # True (and has higher precedence)

# ===============================================================================
# 6. MEMORY MANAGEMENT CONCEPTS
# ===============================================================================

print("\n" + "=" * 80)
print("6. MEMORY MANAGEMENT CONCEPTS")
print("=" * 80)

print("""
Q: How does Python manage memory? Explain reference counting and garbage collection.
""")

# REFERENCE COUNTING
print("1. REFERENCE COUNTING:")

import sys

a = [1, 2, 3]
print(f"Reference count of a: {sys.getrefcount(a)}")

b = a  # Another reference
print(f"Reference count after b = a: {sys.getrefcount(a)}")

del b  # Remove reference
print(f"Reference count after del b: {sys.getrefcount(a)}")

# OBJECT IDENTITY
print("\n2. OBJECT IDENTITY:")

x = 256
y = 256
print(f"x is y (256): {x is y}")  # True (integer caching)

x = 257
y = 257
print(f"x is y (257): {x is y}")  # May be False

x = "hello"
y = "hello"
print(f"x is y ('hello'): {x is y}")  # True (string interning)

# WEAK REFERENCES
print("\n3. WEAK REFERENCES:")

import weakref

class MyClass:
    def __init__(self, name):
        self.name = name
    
    def __del__(self):
        print(f"{self.name} deleted")

obj = MyClass("Object1")
weak_ref = weakref.ref(obj)

print(f"Weak reference alive: {weak_ref() is not None}")
del obj
print(f"Weak reference alive after del: {weak_ref() is not None}")

# GARBAGE COLLECTION
print("\n4. GARBAGE COLLECTION:")

import gc

# Circular reference example
class Node:
    def __init__(self, name):
        self.name = name
        self.ref = None

node1 = Node("Node1")
node2 = Node("Node2")
node1.ref = node2
node2.ref = node1  # Circular reference

# Check garbage collector stats
print(f"Garbage collection stats: {gc.get_stats()}")

# Force garbage collection
collected = gc.collect()
print(f"Objects collected by GC: {collected}")

# ===============================================================================
# 7. INTERVIEW BRAIN TEASERS
# ===============================================================================

print("\n" + "=" * 80)
print("7. INTERVIEW BRAIN TEASERS")
print("=" * 80)

print("""
Q: Solve these Python brain teasers:
""")

# TEASER 1: List multiplication
print("1. List multiplication:")
list1 = [[0] * 3] * 3
list2 = [[0] * 3 for _ in range(3)]

list1[0][0] = 1
list2[0][0] = 1

print(f"list1 after list1[0][0] = 1: {list1}")  # All rows affected!
print(f"list2 after list2[0][0] = 1: {list2}")  # Only first row affected

# TEASER 2: Function arguments
print("\n2. Function arguments:")

def modify_list(lst):
    lst.append(4)
    lst = [1, 2, 3]  # This doesn't affect the original list
    lst.append(5)

original = [1, 2, 3]
modify_list(original)
print(f"Original list after function: {original}")  # [1, 2, 3, 4]

# TEASER 3: Class variables vs instance variables
print("\n3. Class vs instance variables:")

class Counter:
    count = 0
    
    def __init__(self):
        Counter.count += 1
        self.instance_count = Counter.count

c1 = Counter()
c2 = Counter()
c3 = Counter()

print(f"Class count: {Counter.count}")  # 3
print(f"Instance counts: {c1.instance_count}, {c2.instance_count}, {c3.instance_count}")

# TEASER 4: Generator expressions vs list comprehensions
print("\n4. Generator vs list comprehension:")

def side_effect():
    print("Side effect!")
    return 1

# List comprehension evaluates immediately
list_comp = [side_effect() for _ in range(3)]
print("List comprehension created")

# Generator expression evaluates lazily
gen_expr = (side_effect() for _ in range(3))
print("Generator expression created")
print("Consuming generator:")
list(gen_expr)

print("\n" + "=" * 80)
print("CONCEPTUAL QUESTIONS COMPLETE!")
print("=" * 80)
print("""
🎯 COMPREHENSIVE COVERAGE ACHIEVED:
✅ List comprehensions (basic to advanced)
✅ Lambda functions and functional programming
✅ OOP conceptual deep dive
✅ Python keywords and built-ins
✅ Tricky Python concepts and pitfalls
✅ Memory management concepts
✅ Interview brain teasers

📝 KEY INTERVIEW POINTS COVERED:
- List comprehensions vs generator expressions
- Lambda functions with map, filter, reduce
- Functional programming concepts
- OOP paradigm comparison
- Python keyword usage
- Memory management and garbage collection
- Common Python pitfalls and gotchas
- Brain teasers and trick questions

🚀 READY FOR ADVANCED PYTHON INTERVIEWS!
This covers all the conceptual questions, tricky scenarios,
and advanced Python features that experienced developers
need to master for technical interviews.
""")
