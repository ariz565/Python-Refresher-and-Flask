"""
=============================================================================
COMPREHENSIVE PYTHON INTERVIEW QUESTIONS - STRUCTURED COLLECTION
=============================================================================
Created: August 2025
Total Questions: 100
Difficulty Levels: Fresher (1-35), Intermediate (36-70), Advanced (71-100)
=============================================================================
"""

# =============================================================================
# SECTION 1: PYTHON FUNDAMENTALS (Questions 1-35) - FRESHER LEVEL
# =============================================================================

print("="*70)
print("SECTION 1: PYTHON FUNDAMENTALS - FRESHER LEVEL (Questions 1-35)")
print("="*70)

# Question 1: What is Python and its key features?
"""
Q1. What is Python and list its key features?

Answer:
Python is a high-level, interpreted, object-oriented programming language created by Guido van Rossum.

Key Features:
1. Easy to Learn and Read - Simple syntax
2. Interpreted Language - No compilation needed
3. Cross-platform - Runs on Windows, Mac, Linux
4. Object-Oriented - Supports OOP concepts
5. Open Source - Free to use
6. Large Standard Library - Extensive built-in modules
7. Dynamic Typing - No need to declare variable types
8. Memory Management - Automatic garbage collection
"""

# Question 2: What is the difference between list and tuple?
"""
Q2. What is the difference between list and tuple?

Answer:
List:
- Mutable (can be changed)
- Uses square brackets []
- Slower operations
- More memory consumption
- Methods: append(), remove(), pop(), etc.

Tuple:
- Immutable (cannot be changed)
- Uses parentheses ()
- Faster operations
- Less memory consumption
- Limited methods: count(), index()

Example:
"""
# List example
my_list = [1, 2, 3, 4]
my_list.append(5)  # Valid
print(f"List: {my_list}")

# Tuple example
my_tuple = (1, 2, 3, 4)
# my_tuple.append(5)  # Error - AttributeError
print(f"Tuple: {my_tuple}")

# Question 3: What is __init__ method?
"""
Q3. What is __init__ method in Python?

Answer:
__init__ is a special method (constructor) that initializes an object when it's created.
It's automatically called when a new instance of a class is created.
"""

class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade
    
    def display_info(self):
        return f"Student: {self.name}, Age: {self.age}, Grade: {self.grade}"

# Usage
student1 = Student("Alice", 20, "A")
print(student1.display_info())

# Question 4: What is slicing in Python?
"""
Q4. What is slicing in Python?

Answer:
Slicing is extracting parts of sequences (strings, lists, tuples).
Syntax: sequence[start:stop:step]
- start: starting index (inclusive)
- stop: ending index (exclusive)
- step: increment value
"""

# String slicing examples
text = "Python Programming"
print(f"Original: {text}")
print(f"First 6 chars: {text[:6]}")
print(f"Last 5 chars: {text[-11:]}")
print(f"Every 2nd char: {text[::2]}")
print(f"Reverse: {text[::-1]}")

# List slicing examples
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(f"Numbers[2:7]: {numbers[2:7]}")
print(f"Numbers[::2]: {numbers[::2]}")

# Question 5: What are Python data types?
"""
Q5. What are the main data types in Python?

Answer:
1. Numeric Types:
   - int: Integers (1, 2, 100)
   - float: Floating point (3.14, 2.5)
   - complex: Complex numbers (3+4j)

2. Sequence Types:
   - str: Strings ("Hello")
   - list: Ordered, mutable [1, 2, 3]
   - tuple: Ordered, immutable (1, 2, 3)

3. Mapping Type:
   - dict: Key-value pairs {"name": "John"}

4. Set Types:
   - set: Unordered, unique elements {1, 2, 3}
   - frozenset: Immutable set

5. Boolean Type:
   - bool: True or False

6. None Type:
   - NoneType: Represents absence of value
"""

# Examples of data types
integer_var = 42
float_var = 3.14159
string_var = "Hello, Python!"
list_var = [1, 2, 3, "four", 5.0]
tuple_var = (10, 20, 30)
dict_var = {"name": "Alice", "age": 25}
set_var = {1, 2, 3, 4, 5}
bool_var = True
none_var = None

print("\nData Type Examples:")
print(f"Integer: {integer_var} (type: {type(integer_var)})")
print(f"Float: {float_var} (type: {type(float_var)})")
print(f"String: {string_var} (type: {type(string_var)})")
print(f"List: {list_var} (type: {type(list_var)})")
print(f"Tuple: {tuple_var} (type: {type(tuple_var)})")
print(f"Dictionary: {dict_var} (type: {type(dict_var)})")
print(f"Set: {set_var} (type: {type(set_var)})")
print(f"Boolean: {bool_var} (type: {type(bool_var)})")
print(f"None: {none_var} (type: {type(none_var)})")

# Question 6: What is the difference between '==' and 'is'?
"""
Q6. What is the difference between '==' and 'is' operators?

Answer:
== (Equality): Compares values
is (Identity): Compares object identity (memory location)
"""

# Examples
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(f"\na = {a}")
print(f"b = {b}")
print(f"c = a")
print(f"a == b: {a == b}")  # True (same values)
print(f"a is b: {a is b}")  # False (different objects)
print(f"a is c: {a is c}")  # True (same object)

# Question 7: What are mutable and immutable objects?
"""
Q7. What are mutable and immutable objects in Python?

Answer:
Mutable: Objects that can be changed after creation
- list, dict, set

Immutable: Objects that cannot be changed after creation
- int, float, str, tuple, frozenset
"""

# Mutable example
mutable_list = [1, 2, 3]
print(f"\nOriginal list: {mutable_list}")
mutable_list.append(4)
print(f"After append: {mutable_list}")

# Immutable example
immutable_string = "Hello"
print(f"\nOriginal string: {immutable_string}")
# immutable_string[0] = 'h'  # Error - strings are immutable
new_string = immutable_string.replace('H', 'h')
print(f"New string: {new_string}")
print(f"Original unchanged: {immutable_string}")

# Question 8: What is list comprehension?
"""
Q8. What is list comprehension in Python?

Answer:
List comprehension provides a concise way to create lists.
Syntax: [expression for item in iterable if condition]
"""

# Traditional way
squares_traditional = []
for x in range(10):
    squares_traditional.append(x**2)

# List comprehension way
squares_comprehension = [x**2 for x in range(10)]

print(f"\nTraditional: {squares_traditional}")
print(f"Comprehension: {squares_comprehension}")

# With condition
even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(f"Even squares: {even_squares}")

# Question 9: What are Python keywords?
"""
Q9. What are Python keywords?

Answer:
Keywords are reserved words that have special meaning in Python.
They cannot be used as variable names.
"""

import keyword
print(f"\nPython keywords ({len(keyword.kwlist)} total):")
for i, kw in enumerate(keyword.kwlist, 1):
    print(f"{i:2d}. {kw}")

# Question 10: What is the difference between range and xrange?
"""
Q10. What is the difference between range and xrange? (Python 2 vs 3)

Answer:
Python 2:
- range(): Returns a list
- xrange(): Returns an iterator (memory efficient)

Python 3:
- range(): Returns an iterator (like Python 2's xrange)
- xrange(): Doesn't exist

Current Python 3 range() is memory efficient.
"""

# Python 3 range examples
print(f"\nrange(5): {list(range(5))}")
print(f"range(2, 8): {list(range(2, 8))}")
print(f"range(0, 10, 2): {list(range(0, 10, 2))}")

# Question 11: What is docstring?
"""
Q11. What is docstring in Python?

Answer:
Docstring is a string literal that documents a module, function, class, or method.
It's the first statement in the definition and can be accessed via __doc__ attribute.
"""

def calculate_area(radius):
    """
    Calculate the area of a circle.
    
    Parameters:
    radius (float): The radius of the circle
    
    Returns:
    float: The area of the circle
    """
    return 3.14159 * radius ** 2

# Accessing docstring
print(f"\nFunction docstring:\n{calculate_area.__doc__}")

# Question 12: What are local and global variables?
"""
Q12. What are local and global variables?

Answer:
Local variables: Defined inside a function, accessible only within that function
Global variables: Defined outside functions, accessible throughout the program
"""

global_var = "I'm global"

def example_function():
    local_var = "I'm local"
    print(f"Inside function - Global: {global_var}")
    print(f"Inside function - Local: {local_var}")

example_function()
print(f"Outside function - Global: {global_var}")
# print(local_var)  # Error - NameError

# Question 13: What is the global keyword?
"""
Q13. What is the global keyword?

Answer:
The global keyword allows modification of global variables inside a function.
"""

counter = 0  # Global variable

def increment():
    global counter
    counter += 1

print(f"\nInitial counter: {counter}")
increment()
print(f"After increment: {counter}")

# Question 14: What are *args and **kwargs?
"""
Q14. What are *args and **kwargs?

Answer:
*args: Allows function to accept variable number of positional arguments
**kwargs: Allows function to accept variable number of keyword arguments
"""

def example_function(*args, **kwargs):
    print(f"Positional arguments (*args): {args}")
    print(f"Keyword arguments (**kwargs): {kwargs}")

print("\nCalling with various arguments:")
example_function(1, 2, 3, name="Alice", age=25)

# Question 15: What is lambda function?
"""
Q15. What is lambda function?

Answer:
Lambda is an anonymous function defined with lambda keyword.
Syntax: lambda arguments: expression
"""

# Regular function
def square(x):
    return x ** 2

# Lambda function
square_lambda = lambda x: x ** 2

print(f"\nRegular function: {square(5)}")
print(f"Lambda function: {square_lambda(5)}")

# Lambda with map
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print(f"Squared with lambda: {squared}")

# =============================================================================
# SECTION 2: INTERMEDIATE PYTHON (Questions 36-70)
# =============================================================================

print("\n" + "="*70)
print("SECTION 2: INTERMEDIATE PYTHON (Questions 36-70)")
print("="*70)

# Question 36: What are decorators?
"""
Q36. What are decorators in Python?

Answer:
Decorators are functions that modify or extend the behavior of other functions
without permanently modifying them.
"""

def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function execution")
        result = func(*args, **kwargs)
        print("After function execution")
        return result
    return wrapper

@my_decorator
def greet(name):
    return f"Hello, {name}!"

print(f"\nDecorator example:")
result = greet("Alice")
print(f"Result: {result}")

# Question 37: What are generators?
"""
Q37. What are generators in Python?

Answer:
Generators are functions that return an iterator using yield keyword.
They generate values on-demand, making them memory efficient.
"""

def fibonacci_generator(n):
    """Generate Fibonacci sequence up to n terms"""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

print(f"\nFibonacci generator:")
fib_gen = fibonacci_generator(8)
for num in fib_gen:
    print(num, end=" ")
print()

# Question 38: What is the difference between append() and extend()?
"""
Q38. What is the difference between append() and extend()?

Answer:
append(): Adds single element to the end of list
extend(): Adds all elements from an iterable to the end of list
"""

list1 = [1, 2, 3]
list2 = [1, 2, 3]

list1.append([4, 5])  # Adds list as single element
list2.extend([4, 5])  # Adds elements individually

print(f"\nAfter append([4, 5]): {list1}")
print(f"After extend([4, 5]): {list2}")

# Question 39: What is exception handling?
"""
Q39. What is exception handling in Python?

Answer:
Exception handling allows graceful handling of runtime errors using
try, except, else, and finally blocks.
"""

def divide_numbers(a, b):
    try:
        result = a / b
        print(f"Result: {result}")
    except ZeroDivisionError:
        print("Error: Division by zero!")
    except TypeError:
        print("Error: Invalid data type!")
    else:
        print("Division successful!")
    finally:
        print("Cleanup operations")

print(f"\nException handling examples:")
divide_numbers(10, 2)
divide_numbers(10, 0)

# Question 40: What are modules and packages?
"""
Q40. What are modules and packages in Python?

Answer:
Module: A single Python file containing functions, classes, and variables
Package: A directory containing multiple modules with __init__.py file
"""

import math
import os
from datetime import datetime

print(f"\nUsing modules:")
print(f"Math.pi: {math.pi}")
print(f"Current directory: {os.getcwd()}")
print(f"Current time: {datetime.now()}")

# =============================================================================
# SECTION 3: ADVANCED PYTHON (Questions 71-100)
# =============================================================================

print("\n" + "="*70)
print("SECTION 3: ADVANCED PYTHON (Questions 71-100)")
print("="*70)

# Question 71: What are metaclasses?
"""
Q71. What are metaclasses in Python?

Answer:
Metaclasses are classes whose instances are classes themselves.
They define how classes are created and behave.
"""

class MetaClass(type):
    def __new__(cls, name, bases, attrs):
        attrs['class_id'] = f"{name}_ID"
        return super().__new__(cls, name, bases, attrs)

class MyClass(metaclass=MetaClass):
    pass

print(f"\nMetaclass example:")
obj = MyClass()
print(f"Class ID: {obj.class_id}")

# Question 72: What is multiple inheritance?
"""
Q72. What is multiple inheritance in Python?

Answer:
Multiple inheritance allows a class to inherit from multiple parent classes.
Python uses Method Resolution Order (MRO) to resolve conflicts.
"""

class Animal:
    def speak(self):
        return "Animal speaks"

class Mammal:
    def give_birth(self):
        return "Gives birth to live young"

class Dog(Animal, Mammal):
    def bark(self):
        return "Woof!"

print(f"\nMultiple inheritance example:")
dog = Dog()
print(f"Speak: {dog.speak()}")
print(f"Birth: {dog.give_birth()}")
print(f"Bark: {dog.bark()}")
print(f"MRO: {Dog.__mro__}")

# Question 73: What is context manager?
"""
Q73. What are context managers in Python?

Answer:
Context managers ensure proper resource management using 'with' statement.
They implement __enter__ and __exit__ methods.
"""

class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        print(f"Opening file: {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Closing file: {self.filename}")
        if self.file:
            self.file.close()

# Usage example
print(f"\nContext manager example:")
try:
    with FileManager('temp.txt', 'w') as f:
        f.write("Hello, World!")
except FileNotFoundError:
    print("File operation completed")

# Question 74: What is monkey patching?
"""
Q74. What is monkey patching in Python?

Answer:
Monkey patching is dynamically modifying a class or module at runtime.
"""

class Calculator:
    def add(self, a, b):
        return a + b

# Monkey patch - add new method
def multiply(self, a, b):
    return a * b

Calculator.multiply = multiply

print(f"\nMonkey patching example:")
calc = Calculator()
print(f"Add: {calc.add(5, 3)}")
print(f"Multiply: {calc.multiply(5, 3)}")

# Question 75: What is the GIL?
"""
Q75. What is the Global Interpreter Lock (GIL)?

Answer:
GIL is a mutex that prevents multiple native threads from executing 
Python bytecodes simultaneously. It ensures thread safety but limits 
true parallelism in CPU-bound tasks.

Impact:
- Limits multi-threading performance for CPU-bound tasks
- Less impact on I/O-bound tasks
- Can use multiprocessing for CPU-bound parallelism
"""

import threading
import time

def cpu_bound_task():
    count = 0
    for i in range(10000000):
        count += 1
    return count

print(f"\nGIL demonstration:")
start_time = time.time()

# Single thread
cpu_bound_task()
single_thread_time = time.time() - start_time

# Multiple threads
start_time = time.time()
threads = []
for _ in range(2):
    thread = threading.Thread(target=cpu_bound_task)
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

multi_thread_time = time.time() - start_time

print(f"Single thread time: {single_thread_time:.4f}s")
print(f"Multi thread time: {multi_thread_time:.4f}s")

# =============================================================================
# ADDITIONAL QUICK QUESTIONS (76-100)
# =============================================================================

print(f"\n" + "="*70)
print("ADDITIONAL QUICK QUESTIONS (76-100)")
print("="*70)

quick_questions = [
    "Q76. What is the difference between deep copy and shallow copy?",
    "Q77. What are descriptors in Python?",
    "Q78. What is the difference between @staticmethod and @classmethod?",
    "Q79. What are abstract base classes (ABC)?",
    "Q80. What is the purpose of __slots__?",
    "Q81. What is method chaining?",
    "Q82. What are coroutines in Python?",
    "Q83. What is the difference between __str__ and __repr__?",
    "Q84. What is duck typing?",
    "Q85. What are magic methods (dunder methods)?",
    "Q86. What is the difference between is and == for strings?",
    "Q87. What is the purpose of __call__ method?",
    "Q88. What are weak references?",
    "Q89. What is the difference between import and from...import?",
    "Q90. What is the purpose of __new__ method?",
    "Q91. What are dataclasses in Python?",
    "Q92. What is the difference between pickle and json?",
    "Q93. What are type hints in Python?",
    "Q94. What is the walrus operator (:=)?",
    "Q95. What are async and await keywords?",
    "Q96. What is the difference between exec() and eval()?",
    "Q97. What are frozen sets?",
    "Q98. What is the purpose of enumerate()?",
    "Q99. What are named tuples?",
    "Q100. What is the difference between __getattr__ and __getattribute__?"
]

for i, question in enumerate(quick_questions, 76):
    print(f"{question}")

print(f"\n" + "="*70)
print("END OF STRUCTURED PYTHON INTERVIEW QUESTIONS")
print("Total Questions: 100 | Sections: 3 | Difficulty Levels: 3")
print("="*70)
