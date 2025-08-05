"""
Complete Python Fundamentals Interview Questions
Covering Basic to Intermediate Level Python Concepts
Based on comprehensive Python interview preparation material
"""

print("=" * 80)
print("COMPLETE PYTHON FUNDAMENTALS INTERVIEW QUESTIONS")
print("=" * 80)

# ============================================================================
# SECTION 1: PYTHON BASICS AND INTRODUCTION
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 1: PYTHON BASICS AND INTRODUCTION")
print("=" * 50)

# Question 1: What is Python?
print("\n1. What is Python?")
print("-" * 30)
print("""
Python is a high-level, interpreted, general-purpose programming language created by 
Guido van Rossum and released in 1991. Key features include:

• Simple and readable syntax that looks like English
• Interpreted language (executes line by line)
• Cross-platform compatibility (Windows, Mac, Linux)
• Free and open-source
• Supports multiple programming paradigms (OOP, functional)
• Extensive standard library and third-party packages
• Used in: Web development, Data Science, AI/ML, Automation, etc.

Why Python is popular:
- Easy to learn and use
- Versatile and powerful
- Large community support
- Rich ecosystem of libraries
""")

# Question 2: Key Features of Python
print("\n2. What are the key features of Python?")
print("-" * 40)
features = {
    "Easy to Learn": "Simple syntax similar to English",
    "Interpreted": "No compilation step, executes line by line",
    "Cross-platform": "Runs on Windows, Mac, Linux",
    "Object-Oriented": "Supports classes and objects",
    "Functional Programming": "Supports functional programming concepts",
    "Dynamic Typing": "No need to declare variable types",
    "Extensive Libraries": "Rich standard library + third-party packages",
    "Free and Open Source": "No licensing costs",
    "Memory Management": "Automatic garbage collection"
}

for feature, description in features.items():
    print(f"• {feature}: {description}")

# Question 3: Python Variables
print("\n3. What are variables in Python?")
print("-" * 35)
print("""
Variables are containers that store data values. In Python:
• No need to declare variable types explicitly
• Python automatically determines the type
• Variable names should be descriptive
• Follow naming conventions (snake_case)
""")

# Examples of variables
name = "Alice"
age = 25
height = 5.6
is_student = True

print("Examples:")
print(f"name = '{name}' (type: {type(name).__name__})")
print(f"age = {age} (type: {type(age).__name__})")
print(f"height = {height} (type: {type(height).__name__})")
print(f"is_student = {is_student} (type: {type(is_student).__name__})")

# Question 4: Python Data Types
print("\n4. What are Python's built-in data types?")
print("-" * 45)
print("""
Python has several built-in data types categorized as:

NUMERIC TYPES:
• int: Integer numbers (1, 2, -5)
• float: Decimal numbers (3.14, -2.5)
• complex: Complex numbers (2+3j)
• bool: Boolean values (True, False)

SEQUENCE TYPES:
• str: Text/String data ("hello")
• list: Ordered, mutable collection [1, 2, 3]
• tuple: Ordered, immutable collection (1, 2, 3)
• range: Sequence of numbers range(10)

MAPPING TYPE:
• dict: Key-value pairs {"name": "Alice"}

SET TYPES:
• set: Unordered, unique elements {1, 2, 3}
• frozenset: Immutable set

BINARY TYPES:
• bytes, bytearray, memoryview
""")

# Question 5: Mutable vs Immutable
print("\n5. What is the difference between mutable and immutable types?")
print("-" * 65)
print("""
MUTABLE TYPES (can be changed after creation):
• list, dict, set, bytearray
• Can modify, add, or remove elements
• Changes affect the original object

IMMUTABLE TYPES (cannot be changed after creation):
• int, float, str, tuple, frozenset, bool
• Any operation creates a new object
• Original object remains unchanged
""")

# Examples
print("Examples:")
# Mutable example
my_list = [1, 2, 3]
print(f"Original list: {my_list}")
my_list.append(4)
print(f"After append: {my_list}")

# Immutable example
my_string = "hello"
print(f"Original string: '{my_string}'")
new_string = my_string + " world"
print(f"Original unchanged: '{my_string}'")
print(f"New string: '{new_string}'")

# ============================================================================
# SECTION 2: PYTHON OPERATORS AND CONTROL FLOW
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 2: OPERATORS AND CONTROL FLOW")
print("=" * 50)

# Question 6: Python Operators
print("\n6. What are the different types of operators in Python?")
print("-" * 55)
print("""
ARITHMETIC OPERATORS:
+ (addition), - (subtraction), * (multiplication)
/ (division), // (floor division), % (modulo), ** (exponent)

COMPARISON OPERATORS:
== (equal), != (not equal), < (less than), > (greater than)
<= (less than or equal), >= (greater than or equal)

LOGICAL OPERATORS:
and, or, not

ASSIGNMENT OPERATORS:
=, +=, -=, *=, /=, //=, %=, **=

MEMBERSHIP OPERATORS:
in, not in

IDENTITY OPERATORS:
is, is not

BITWISE OPERATORS:
& (AND), | (OR), ^ (XOR), ~ (NOT), << (left shift), >> (right shift)
""")

# Examples
print("Examples:")
a, b = 10, 3
print(f"a = {a}, b = {b}")
print(f"a + b = {a + b}")
print(f"a // b = {a // b}")
print(f"a % b = {a % b}")
print(f"a ** b = {a ** b}")
print(f"a == b: {a == b}")
print(f"a > b: {a > b}")

# Question 7: Conditional Statements
print("\n7. What are conditional statements in Python?")
print("-" * 47)
print("""
Conditional statements execute code based on certain conditions:

• if: Execute code if condition is True
• elif: Check additional conditions (else if)
• else: Execute code if all conditions are False
""")

# Example
age = 18
if age >= 18:
    status = "Adult"
elif age >= 13:
    status = "Teenager"
else:
    status = "Child"

print(f"Example: Age {age} → Status: {status}")

# Question 8: Loops in Python
print("\n8. What are loops in Python?")
print("-" * 32)
print("""
Loops are used to repeat code execution:

FOR LOOP:
• Used when you know the number of iterations
• Iterates over sequences (lists, strings, ranges)
• Syntax: for item in sequence:

WHILE LOOP:
• Used when condition-based repetition is needed
• Continues while condition is True
• Syntax: while condition:
""")

# Examples
print("For loop example:")
for i in range(3):
    print(f"  Iteration {i}")

print("While loop example:")
count = 0
while count < 3:
    print(f"  Count: {count}")
    count += 1

# Question 9: Break, Continue, Pass
print("\n9. What are break, continue, and pass statements?")
print("-" * 53)
print("""
BREAK:
• Exits the loop completely
• Control moves to the next statement after the loop

CONTINUE:
• Skips the current iteration
• Moves to the next iteration of the loop

PASS:
• Does nothing (null operation)
• Used as placeholder where code is syntactically required
""")

# Examples
print("Examples:")
print("Break example:")
for i in range(5):
    if i == 3:
        break
    print(f"  {i}")

print("Continue example:")
for i in range(5):
    if i == 2:
        continue
    print(f"  {i}")

# ============================================================================
# SECTION 3: FUNCTIONS AND MODULES
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 3: FUNCTIONS AND MODULES")
print("=" * 50)

# Question 10: Functions in Python
print("\n10. How do you define and use functions in Python?")
print("-" * 50)
print("""
Functions are reusable blocks of code that perform specific tasks.

SYNTAX:
def function_name(parameters):
    \"\"\"docstring\"\"\"
    # function body
    return value  # optional

BENEFITS:
• Code reusability
• Better organization
• Easier debugging and maintenance
• Modularity
""")

# Example function
def greet(name, greeting="Hello"):
    """Function to greet a person"""
    return f"{greeting}, {name}!"

print("Example:")
print(f"greet('Alice'): {greet('Alice')}")
print(f"greet('Bob', 'Hi'): {greet('Bob', 'Hi')}")

# Question 11: Function Parameters
print("\n11. What are parameters and arguments in Python functions?")
print("-" * 59)
print("""
PARAMETERS: Variables in function definition
ARGUMENTS: Actual values passed when calling function

TYPES OF PARAMETERS:
• Positional parameters: def func(a, b)
• Default parameters: def func(a, b=10)
• Keyword arguments: func(a=5, b=10)
• Variable-length arguments: *args, **kwargs
""")

def example_function(pos_arg, default_arg=10, *args, **kwargs):
    """Example function showing different parameter types"""
    print(f"Positional: {pos_arg}")
    print(f"Default: {default_arg}")
    print(f"*args: {args}")
    print(f"**kwargs: {kwargs}")

print("Example:")
example_function(1, 2, 3, 4, 5, name="Alice", age=25)

# Question 12: Return Statement
print("\n12. What is the return statement in Python?")
print("-" * 45)
print("""
The return statement:
• Sends a value back to the function caller
• Ends function execution immediately
• If no return statement, function returns None
• Can return multiple values (as tuple)
""")

def calculate_operations(a, b):
    """Function returning multiple values"""
    return a + b, a - b, a * b, a / b

result = calculate_operations(10, 3)
print(f"Example: calculate_operations(10, 3) = {result}")

# Question 13: Lambda Functions
print("\n13. What are lambda functions in Python?")
print("-" * 42)
print("""
Lambda functions are small, anonymous functions:
• Can take any number of arguments
• Can only contain one expression
• Return the result of the expression
• Often used with map(), filter(), sort()

SYNTAX: lambda arguments: expression
""")

# Examples
square = lambda x: x ** 2
add = lambda x, y: x + y

print("Examples:")
print(f"square = lambda x: x ** 2")
print(f"square(5) = {square(5)}")
print(f"add = lambda x, y: x + y")
print(f"add(3, 4) = {add(3, 4)}")

# Using with built-in functions
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print(f"map(lambda x: x**2, {numbers}) = {squared}")

# Question 14: Modules and Packages
print("\n14. What are modules and packages in Python?")
print("-" * 46)
print("""
MODULE:
• A single Python file (.py) containing code
• Can contain functions, classes, variables
• Imported using 'import' statement
• Promotes code reusability and organization

PACKAGE:
• A directory containing multiple modules
• Must contain __init__.py file
• Organized hierarchically using dot notation
• Helps avoid naming conflicts

IMPORTING:
• import module_name
• from module_name import function_name
• import module_name as alias
• from module_name import *
""")

import math
from datetime import datetime

print("Examples:")
print(f"import math; math.sqrt(16) = {math.sqrt(16)}")
print(f"from datetime import datetime; datetime.now() = {datetime.now()}")

# ============================================================================
# SECTION 4: DATA STRUCTURES - LISTS AND TUPLES
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 4: DATA STRUCTURES - LISTS AND TUPLES")
print("=" * 50)

# Question 15: Lists in Python
print("\n15. What are lists in Python and how do you use them?")
print("-" * 52)
print("""
Lists are ordered, mutable collections that can store different data types:

CHARACTERISTICS:
• Ordered: Elements have a defined order
• Mutable: Can change, add, remove elements
• Allow duplicates
• Can store mixed data types
• Zero-indexed

COMMON OPERATIONS:
• Access: list[index]
• Modify: list[index] = value
• Add: append(), insert(), extend()
• Remove: remove(), pop(), del
• Other: sort(), reverse(), len(), in
""")

# Examples
fruits = ["apple", "banana", "cherry"]
print(f"Original list: {fruits}")

# Access elements
print(f"First element: {fruits[0]}")
print(f"Last element: {fruits[-1]}")

# Modify elements
fruits[1] = "orange"
print(f"After modification: {fruits}")

# Add elements
fruits.append("grape")
fruits.insert(0, "mango")
print(f"After adding: {fruits}")

# Remove elements
removed = fruits.pop()
print(f"Removed '{removed}': {fruits}")

# Question 16: List Comprehensions
print("\n16. What are list comprehensions in Python?")
print("-" * 43)
print("""
List comprehensions provide a concise way to create lists:

SYNTAX: [expression for item in iterable if condition]

BENEFITS:
• More readable and concise than loops
• Often faster than equivalent for loops
• Pythonic way of creating lists
• Can include conditions for filtering
""")

# Examples
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Basic list comprehension
squares = [x**2 for x in numbers]
print(f"Squares: {squares}")

# With condition
even_squares = [x**2 for x in numbers if x % 2 == 0]
print(f"Even squares: {even_squares}")

# Nested comprehension
matrix = [[i*j for j in range(1, 4)] for i in range(1, 4)]
print(f"Matrix: {matrix}")

# Question 17: Tuples in Python
print("\n17. What are tuples and how do they differ from lists?")
print("-" * 54)
print("""
Tuples are ordered, immutable collections:

CHARACTERISTICS:
• Ordered: Elements have a defined order
• Immutable: Cannot change after creation
• Allow duplicates
• Can store mixed data types
• Zero-indexed

DIFFERENCES FROM LISTS:
╔════════════╦══════════╦════════════╗
║  Feature   ║   List   ║   Tuple    ║
╠════════════╬══════════╬════════════╣
║ Mutability ║ Mutable  ║ Immutable  ║
║ Syntax     ║ [1,2,3]  ║ (1,2,3)    ║
║ Speed      ║ Slower   ║ Faster     ║
║ Memory     ║ More     ║ Less       ║
║ Use Case   ║ Dynamic  ║ Fixed data ║
╚════════════╩══════════╩════════════╝
""")

# Examples
coordinates = (10, 20)
rgb_color = (255, 0, 128)
mixed_tuple = ("Alice", 25, True)

print(f"Coordinates: {coordinates}")
print(f"RGB Color: {rgb_color}")
print(f"Mixed tuple: {mixed_tuple}")

# Tuple unpacking
x, y = coordinates
print(f"Unpacked: x={x}, y={y}")

# Question 18: Slicing in Python
print("\n18. What is slicing in Python?")
print("-" * 34)
print("""
Slicing extracts a portion of sequences (strings, lists, tuples):

SYNTAX: sequence[start:stop:step]
• start: Starting index (inclusive)
• stop: Ending index (exclusive)
• step: Step size (default 1)

FEATURES:
• Negative indices count from the end
• Missing values use defaults
• Can reverse sequences with negative step
""")

# Examples
text = "Python Programming"
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(f"Text: '{text}'")
print(f"text[0:6]: '{text[0:6]}'")
print(f"text[7:]: '{text[7:]}'")
print(f"text[:6]: '{text[:6]}'")
print(f"text[::2]: '{text[::2]}'")
print(f"text[::-1]: '{text[::-1]}'")

print(f"\nNumbers: {numbers}")
print(f"numbers[2:8]: {numbers[2:8]}")
print(f"numbers[::2]: {numbers[::2]}")
print(f"numbers[::-1]: {numbers[::-1]}")

# ============================================================================
# SECTION 5: DICTIONARIES AND SETS
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 5: DICTIONARIES AND SETS")
print("=" * 50)

# Question 19: Dictionaries in Python
print("\n19. What are dictionaries and how are they used?")
print("-" * 48)
print("""
Dictionaries store data in key-value pairs:

CHARACTERISTICS:
• Unordered (Python 3.7+ maintains insertion order)
• Mutable: Can add, modify, remove items
• Keys must be immutable and unique
• Values can be any data type
• Fast lookup by key

COMMON OPERATIONS:
• Access: dict[key] or dict.get(key)
• Add/Modify: dict[key] = value
• Remove: del dict[key], dict.pop(key)
• Methods: keys(), values(), items()
""")

# Examples
person = {
    "name": "Alice",
    "age": 30,
    "city": "New York",
    "skills": ["Python", "JavaScript", "SQL"]
}

print(f"Person: {person}")
print(f"Name: {person['name']}")
print(f"Age: {person.get('age', 'Unknown')}")

# Modify dictionary
person["age"] = 31
person["email"] = "alice@example.com"
print(f"Updated: {person}")

# Dictionary methods
print(f"Keys: {list(person.keys())}")
print(f"Values: {list(person.values())}")

# Question 20: Dictionary Comprehensions
print("\n20. What are dictionary comprehensions?")
print("-" * 39)
print("""
Dictionary comprehensions create dictionaries concisely:

SYNTAX: {key_expression: value_expression for item in iterable if condition}

BENEFITS:
• More readable than loops
• Efficient way to create dictionaries
• Can include filtering conditions
""")

# Examples
numbers = [1, 2, 3, 4, 5]

# Basic dictionary comprehension
squares_dict = {x: x**2 for x in numbers}
print(f"Squares: {squares_dict}")

# With condition
even_squares = {x: x**2 for x in numbers if x % 2 == 0}
print(f"Even squares: {even_squares}")

# From existing data
names = ["Alice", "Bob", "Charlie"]
name_lengths = {name: len(name) for name in names}
print(f"Name lengths: {name_lengths}")

# Question 21: Sets in Python
print("\n21. What are sets and what are they used for?")
print("-" * 44)
print("""
Sets are unordered collections of unique elements:

CHARACTERISTICS:
• Unordered: No defined order
• Mutable: Can add/remove elements
• No duplicates: Automatically removes duplicates
• Elements must be immutable

OPERATIONS:
• Add: add()
• Remove: remove(), discard(), pop()
• Set operations: union, intersection, difference
• Membership: in, not in

USE CASES:
• Remove duplicates
• Fast membership testing
• Mathematical set operations
""")

# Examples
numbers_set = {1, 2, 3, 4, 5}
colors = {"red", "green", "blue"}

print(f"Numbers set: {numbers_set}")
print(f"Colors set: {colors}")

# Set operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

print(f"Set1: {set1}")
print(f"Set2: {set2}")
print(f"Union: {set1 | set2}")
print(f"Intersection: {set1 & set2}")
print(f"Difference: {set1 - set2}")

# Remove duplicates
numbers_with_duplicates = [1, 2, 2, 3, 3, 3, 4, 5, 5]
unique_numbers = list(set(numbers_with_duplicates))
print(f"Remove duplicates: {numbers_with_duplicates} → {unique_numbers}")

print("\n" + "=" * 80)
print("END OF PYTHON FUNDAMENTALS SECTION")
print("Continue with advanced topics in the next file...")
print("=" * 80)
