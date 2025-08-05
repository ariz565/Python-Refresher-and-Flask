"""
Complete Python Advanced Concepts Interview Questions
Covering decorators, generators, iterators, context managers, and advanced Python features
Based on comprehensive Python interview preparation material
"""

print("=" * 80)
print("COMPLETE PYTHON ADVANCED CONCEPTS INTERVIEW QUESTIONS")
print("=" * 80)

# ============================================================================
# SECTION 1: DECORATORS AND CLOSURES
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 1: DECORATORS AND CLOSURES")
print("=" * 50)

# Question 1: Decorators in Python
print("\n1. What are decorators in Python?")
print("-" * 35)
print("""
Decorators are a design pattern that allows you to modify or extend the behavior
of functions or classes without permanently modifying their code.

CHARACTERISTICS:
• Functions that take another function as argument
• Return a modified version of the function
• Use @ syntax for clean application
• Can be stacked (multiple decorators)
• Enable aspect-oriented programming

COMMON USE CASES:
• Logging and debugging
• Authentication and authorization
• Timing and performance monitoring
• Caching and memoization
• Input validation
""")

# Basic decorator example
def timing_decorator(func):
    """Decorator to measure function execution time"""
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def logging_decorator(func):
    """Decorator to log function calls"""
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned: {result}")
        return result
    return wrapper

# Applying decorators
@timing_decorator
@logging_decorator
def calculate_square(n):
    """Calculate square of a number"""
    import time
    time.sleep(0.1)  # Simulate some work
    return n ** 2

print("Decorator Example:")
result = calculate_square(5)

# Question 2: Decorator with Parameters
print("\n2. How do you create decorators with parameters?")
print("-" * 49)
print("""
Decorators with parameters require an additional layer of functions:

STRUCTURE:
def decorator_with_params(param1, param2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Use param1, param2 here
            return func(*args, **kwargs)
        return wrapper
    return decorator

This creates a decorator factory that returns the actual decorator.
""")

def retry_decorator(max_attempts=3, delay=1):
    """Decorator to retry function execution on failure"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(delay)
        return wrapper
    return decorator

def cache_decorator(max_size=100):
    """Decorator to cache function results"""
    def decorator(func):
        cache = {}
        def wrapper(*args, **kwargs):
            # Create a key from arguments
            key = str(args) + str(sorted(kwargs.items()))
            if key in cache:
                print(f"Cache hit for {func.__name__}")
                return cache[key]
            
            result = func(*args, **kwargs)
            if len(cache) < max_size:
                cache[key] = result
                print(f"Cached result for {func.__name__}")
            return result
        return wrapper
    return decorator

@retry_decorator(max_attempts=2, delay=0.5)
@cache_decorator(max_size=50)
def fibonacci(n):
    """Calculate fibonacci number (inefficient for demo)"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print("Decorator with Parameters Example:")
print(f"fibonacci(10) = {fibonacci(10)}")
print(f"fibonacci(10) = {fibonacci(10)}")  # Should hit cache

# Question 3: Class-based Decorators
print("\n3. How do you create class-based decorators?")
print("-" * 44)
print("""
Class-based decorators use the __call__ method to make the class callable:

ADVANTAGES:
• Can maintain state between calls
• More complex logic and configuration
• Can implement multiple decorator behaviors
• Object-oriented approach
""")

class CountCalls:
    """Class-based decorator to count function calls"""
    def __init__(self, func):
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"{self.func.__name__} has been called {self.count} times")
        return self.func(*args, **kwargs)
    
    def reset_count(self):
        self.count = 0

class RateLimiter:
    """Class-based decorator for rate limiting"""
    def __init__(self, max_calls=5, time_window=60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            import time
            current_time = time.time()
            
            # Remove old calls outside time window
            self.calls = [call_time for call_time in self.calls 
                         if current_time - call_time < self.time_window]
            
            if len(self.calls) >= self.max_calls:
                raise Exception(f"Rate limit exceeded. Max {self.max_calls} calls per {self.time_window} seconds")
            
            self.calls.append(current_time)
            return func(*args, **kwargs)
        return wrapper

@CountCalls
def greet(name):
    return f"Hello, {name}!"

@RateLimiter(max_calls=3, time_window=10)
def api_call():
    return "API response"

print("Class-based Decorators Example:")
print(greet("Alice"))
print(greet("Bob"))
print(greet("Charlie"))

# Question 4: Closures
print("\n4. What are closures in Python?")
print("-" * 34)
print("""
A closure is a function that retains access to variables from its enclosing scope
even after the outer function has finished executing.

CHARACTERISTICS:
• Inner function references variables from outer function
• Outer function returns the inner function
• Inner function "closes over" the outer variables
• Variables persist in memory

USE CASES:
• Creating specialized functions
• Implementing decorators
• Factory functions
• Maintaining state without classes
""")

def create_multiplier(factor):
    """Closure factory function"""
    def multiplier(number):
        return number * factor  # 'factor' is from enclosing scope
    return multiplier

def create_accumulator(initial_value=0):
    """Closure maintaining state"""
    total = initial_value
    
    def accumulate(value):
        nonlocal total  # Allows modification of enclosing variable
        total += value
        return total
    
    def get_total():
        return total
    
    def reset():
        nonlocal total
        total = initial_value
    
    # Return multiple functions as a tuple
    accumulate.get_total = get_total
    accumulate.reset = reset
    return accumulate

print("Closures Example:")
# Create specialized multiplier functions
double = create_multiplier(2)
triple = create_multiplier(3)

print(f"double(5) = {double(5)}")
print(f"triple(4) = {triple(4)}")

# Create accumulator with state
acc = create_accumulator(10)
print(f"Initial total: {acc.get_total()}")
print(f"Add 5: {acc(5)}")
print(f"Add 3: {acc(3)}")
print(f"Current total: {acc.get_total()}")

# ============================================================================
# SECTION 2: GENERATORS AND ITERATORS
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 2: GENERATORS AND ITERATORS")
print("=" * 50)

# Question 5: Iterators in Python
print("\n5. What are iterators in Python?")
print("-" * 35)
print("""
An iterator is an object that implements the iterator protocol:
• __iter__(): Returns the iterator object itself
• __next__(): Returns the next item from the iterator

CHARACTERISTICS:
• Remembers its state during iteration
• Raises StopIteration when no more items
• Memory efficient (one item at a time)
• Can be used with for loops, next(), etc.

BUILT-IN ITERABLES:
• list, tuple, string, dict, set
• range, enumerate, zip, map, filter
""")

class NumberIterator:
    """Custom iterator that generates numbers up to a limit"""
    def __init__(self, limit):
        self.limit = limit
        self.current = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < self.limit:
            current = self.current
            self.current += 1
            return current
        else:
            raise StopIteration

class FibonacciIterator:
    """Iterator for Fibonacci sequence"""
    def __init__(self, max_count):
        self.max_count = max_count
        self.count = 0
        self.a, self.b = 0, 1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.count < self.max_count:
            if self.count == 0:
                self.count += 1
                return self.a
            elif self.count == 1:
                self.count += 1
                return self.b
            else:
                self.a, self.b = self.b, self.a + self.b
                self.count += 1
                return self.b
        else:
            raise StopIteration

print("Iterator Examples:")
# Number iterator
numbers = NumberIterator(5)
print("Number iterator:", list(numbers))

# Fibonacci iterator
fib = FibonacciIterator(8)
print("Fibonacci iterator:", list(fib))

# Using next() manually
manual_iter = NumberIterator(3)
print(f"Manual iteration: {next(manual_iter)}, {next(manual_iter)}, {next(manual_iter)}")

# Question 6: Generators
print("\n6. What are generators in Python?")
print("-" * 34)
print("""
Generators are a special type of iterator that use the 'yield' keyword
instead of 'return' to produce a sequence of values.

ADVANTAGES:
• Memory efficient (lazy evaluation)
• Simpler syntax than iterators
• Automatic __iter__ and __next__ implementation
• Can be infinite sequences
• Resume execution from where they left off

TYPES:
• Generator functions (using yield)
• Generator expressions (like list comprehensions)
""")

def count_up_to(limit):
    """Generator function counting up to limit"""
    count = 1
    while count <= limit:
        print(f"Generating {count}")
        yield count
        count += 1

def fibonacci_generator(max_count):
    """Generator for Fibonacci sequence"""
    a, b = 0, 1
    count = 0
    while count < max_count:
        yield a
        a, b = b, a + b
        count += 1

def infinite_fibonacci():
    """Infinite Fibonacci generator"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

def read_large_file(filename):
    """Generator for reading large files line by line"""
    try:
        with open(filename, 'r') as file:
            for line in file:
                yield line.strip()
    except FileNotFoundError:
        yield "File not found"

print("Generator Examples:")
# Generator function
print("Count up to 3:")
for num in count_up_to(3):
    print(f"  Got: {num}")

# Fibonacci generator
print("Fibonacci generator:")
fib_gen = fibonacci_generator(6)
print(f"  Fibonacci: {list(fib_gen)}")

# Generator expression
print("Generator expression:")
squares_gen = (x**2 for x in range(5))
print(f"  Squares: {list(squares_gen)}")

# Infinite generator (limited for demo)
print("Infinite Fibonacci (first 5):")
inf_fib = infinite_fibonacci()
for i, fib_num in enumerate(inf_fib):
    if i >= 5:
        break
    print(f"  {fib_num}")

# Question 7: Generator vs Iterator vs List
print("\n7. What's the difference between generators, iterators, and lists?")
print("-" * 69)
print("""
MEMORY USAGE:
• List: Stores all items in memory at once
• Iterator: Stores state, generates one item at a time
• Generator: Similar to iterator but simpler syntax

PERFORMANCE:
• List: Fast access to any element, high memory usage
• Iterator: Memory efficient, only forward iteration
• Generator: Memory efficient, resumable execution

SYNTAX:
• List: [1, 2, 3, 4, 5]
• Iterator: Custom class with __iter__ and __next__
• Generator: Function with yield or (expression)
""")

import sys

# Memory comparison
def create_list(n):
    return [x for x in range(n)]

def create_generator(n):
    return (x for x in range(n))

def create_iterator(n):
    return NumberIterator(n)

print("Memory Usage Comparison (for n=1000):")
list_obj = create_list(1000)
gen_obj = create_generator(1000)
iter_obj = create_iterator(1000)

print(f"List size: {sys.getsizeof(list_obj)} bytes")
print(f"Generator size: {sys.getsizeof(gen_obj)} bytes")
print(f"Iterator size: {sys.getsizeof(iter_obj)} bytes")

# Question 8: Yield vs Return
print("\n8. What's the difference between yield and return?")
print("-" * 50)
print("""
RETURN:
• Terminates function execution
• Returns a single value
• Function cannot resume from where it left off
• Memory is freed after return

YIELD:
• Pauses function execution
• Returns a generator object
• Function can resume from yield point
• Maintains local state between calls
""")

def function_with_return():
    """Function using return"""
    print("Before return")
    return "Returned value"
    print("After return")  # This never executes

def function_with_yield():
    """Function using yield"""
    print("Before first yield")
    yield "First yield"
    print("Between yields")
    yield "Second yield"
    print("After second yield")

print("Return vs Yield Example:")
# Function with return
print("Function with return:")
result = function_with_return()
print(f"Result: {result}")

# Function with yield
print("\nFunction with yield:")
gen = function_with_yield()
print("Generator created")
print(f"First call: {next(gen)}")
print(f"Second call: {next(gen)}")

# ============================================================================
# SECTION 3: CONTEXT MANAGERS AND WITH STATEMENT
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 3: CONTEXT MANAGERS AND WITH STATEMENT")
print("=" * 50)

# Question 9: Context Managers
print("\n9. What are context managers in Python?")
print("-" * 41)
print("""
Context managers are objects that define the runtime context for executing
a block of code using the 'with' statement.

PROTOCOL:
• __enter__(): Setup code, returns resource
• __exit__(): Cleanup code, handles exceptions

BENEFITS:
• Automatic resource management
• Exception-safe cleanup
• Cleaner, more readable code
• Prevents resource leaks

COMMON USE CASES:
• File handling
• Database connections
• Thread locks
• Network connections
""")

class FileManager:
    """Custom file context manager"""
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        print(f"Opening file: {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_value, traceback):
        print(f"Closing file: {self.filename}")
        if self.file:
            self.file.close()
        if exc_type:
            print(f"Exception occurred: {exc_type.__name__}: {exc_value}")
        return False  # Don't suppress exceptions

class DatabaseConnection:
    """Mock database context manager"""
    def __init__(self, database_name):
        self.database_name = database_name
        self.connection = None
    
    def __enter__(self):
        print(f"Connecting to database: {self.database_name}")
        self.connection = f"Connection to {self.database_name}"
        return self.connection
    
    def __exit__(self, exc_type, exc_value, traceback):
        print(f"Disconnecting from database: {self.database_name}")
        self.connection = None
        return False

print("Context Manager Examples:")
# File manager example
try:
    with FileManager("temp_file.txt", "w") as f:
        f.write("Hello, Context Manager!")
        print("File operations completed")
except Exception as e:
    print(f"Error: {e}")

# Database connection example
with DatabaseConnection("mydb") as conn:
    print(f"Using connection: {conn}")
    print("Performing database operations...")

# Question 10: contextlib Module
print("\n10. How do you use the contextlib module?")
print("-" * 42)
print("""
The contextlib module provides utilities for working with context managers:

KEY FUNCTIONS:
• @contextmanager: Decorator to create context managers from generators
• closing(): Automatically call close() on objects
• suppress(): Suppress specified exceptions
• ExitStack(): Manage multiple context managers

ADVANTAGES:
• Simpler syntax than implementing __enter__ and __exit__
• Less boilerplate code
• Generator-based approach
""")

from contextlib import contextmanager, closing, suppress, ExitStack
import io

@contextmanager
def timer_context(name):
    """Context manager to time code execution"""
    import time
    print(f"Starting timer: {name}")
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"Timer {name}: {end_time - start_time:.4f} seconds")

@contextmanager
def temporary_file(content):
    """Context manager for temporary file"""
    import tempfile
    import os
    
    # Setup
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    temp_file.write(content)
    temp_file.close()
    
    try:
        yield temp_file.name
    finally:
        # Cleanup
        os.unlink(temp_file.name)
        print(f"Temporary file deleted: {temp_file.name}")

print("contextlib Examples:")
# Timer context manager
with timer_context("Example operation"):
    import time
    time.sleep(0.1)
    print("Doing some work...")

# Suppress exceptions
print("Suppress example:")
with suppress(ValueError, TypeError):
    int("not a number")  # This would normally raise ValueError
    print("This won't execute")
print("Continued execution after suppressed exception")

# Multiple context managers with ExitStack
print("ExitStack example:")
with ExitStack() as stack:
    cm1 = stack.enter_context(timer_context("Operation 1"))
    cm2 = stack.enter_context(timer_context("Operation 2"))
    print("All context managers are active")
    time.sleep(0.05)

# ============================================================================
# SECTION 4: METACLASSES AND DESCRIPTORS
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 4: METACLASSES AND DESCRIPTORS")
print("=" * 50)

# Question 11: Metaclasses
print("\n11. What are metaclasses in Python?")
print("-" * 37)
print("""
A metaclass is a class whose instances are classes themselves.
In Python, classes are objects, and metaclasses are the classes of those objects.

CONCEPTS:
• type is the default metaclass
• Classes are instances of metaclasses
• Metaclasses define how classes are created
• Control class creation and behavior

USE CASES:
• Singleton pattern
• Automatic registration systems
• API frameworks (like Django ORM)
• Code validation and modification
""")

class SingletonMeta(type):
    """Metaclass for implementing Singleton pattern"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Singleton(metaclass=SingletonMeta):
    """Singleton class using metaclass"""
    def __init__(self, value):
        self.value = value

class ValidatorMeta(type):
    """Metaclass that validates class definitions"""
    def __new__(mcs, name, bases, attrs):
        # Ensure all methods have docstrings
        for attr_name, attr_value in attrs.items():
            if callable(attr_value) and not attr_name.startswith('_'):
                if not hasattr(attr_value, '__doc__') or not attr_value.__doc__:
                    raise ValueError(f"Method {attr_name} must have a docstring")
        
        return super().__new__(mcs, name, bases, attrs)

class ValidatedClass(metaclass=ValidatorMeta):
    """Class using validator metaclass"""
    def process_data(self, data):
        """Process the input data"""
        return data.upper()

print("Metaclass Examples:")
# Singleton example
s1 = Singleton("first")
s2 = Singleton("second")
print(f"s1 is s2: {s1 is s2}")
print(f"s1.value: {s1.value}, s2.value: {s2.value}")

# Validator example
validated = ValidatedClass()
print(f"Validated class method: {validated.process_data('hello')}")

# Question 12: Descriptors
print("\n12. What are descriptors in Python?")
print("-" * 36)
print("""
Descriptors are objects that define how attribute access is handled
through special methods: __get__, __set__, and __delete__.

TYPES:
• Data descriptors: Define __get__ and __set__
• Non-data descriptors: Define only __get__
• Properties are built on descriptors

PROTOCOL:
• __get__(self, obj, objtype): Get attribute value
• __set__(self, obj, value): Set attribute value
• __delete__(self, obj): Delete attribute

USE CASES:
• Property-like behavior
• Type checking and validation
• Lazy loading
• Logging attribute access
""")

class ValidatedAttribute:
    """Descriptor for validated attributes"""
    def __init__(self, validator=None, default=None):
        self.validator = validator
        self.default = default
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name, self.default)
    
    def __set__(self, obj, value):
        if self.validator and not self.validator(value):
            raise ValueError(f"Invalid value for {self.name}: {value}")
        obj.__dict__[self.name] = value

class LoggedAttribute:
    """Descriptor that logs attribute access"""
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        print(f"Getting {self.name}: {self.value}")
        return self.value
    
    def __set__(self, obj, value):
        print(f"Setting {self.name}: {self.value} -> {value}")
        self.value = value

class Person:
    """Class using descriptors"""
    # Validator descriptors
    age = ValidatedAttribute(lambda x: isinstance(x, int) and 0 <= x <= 150, 0)
    name = ValidatedAttribute(lambda x: isinstance(x, str) and len(x) > 0, "")
    
    # Logged attribute
    score = LoggedAttribute(0)
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

print("Descriptor Examples:")
person = Person("Alice", 30)
print(f"Person: {person.name}, {person.age}")

# Using logged attribute
person.score = 85
retrieved_score = person.score

# Validation
try:
    person.age = -5  # Should raise ValueError
except ValueError as e:
    print(f"Validation error: {e}")

print("\n" + "=" * 80)
print("END OF ADVANCED CONCEPTS SECTION")
print("Continue with error handling and file operations in the next file...")
print("=" * 80)
