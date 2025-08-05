"""
Complete Python Exception Handling & File Operations Interview Questions
Covering error handling, file I/O, modules, packages, and memory management
Based on comprehensive Python interview preparation material
"""

print("=" * 80)
print("COMPLETE PYTHON EXCEPTION HANDLING & FILE OPERATIONS QUESTIONS")
print("=" * 80)

# ============================================================================
# SECTION 1: EXCEPTION HANDLING
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 1: EXCEPTION HANDLING")
print("=" * 50)

# Question 1: Exception Handling Basics
print("\n1. How does exception handling work in Python?")
print("-" * 47)
print("""
Exception handling allows you to handle errors gracefully without crashing the program:

SYNTAX:
try:
    # Code that might raise an exception
except SpecificException:
    # Handle specific exception
except (Exception1, Exception2):
    # Handle multiple exceptions
except Exception as e:
    # Handle any exception
else:
    # Executed if no exception occurs
finally:
    # Always executed (cleanup code)

KEY CONCEPTS:
• try: Contains potentially problematic code
• except: Handles specific exceptions
• else: Runs only if no exceptions occurred
• finally: Always runs (cleanup)
• raise: Manually raise exceptions
""")

def divide_numbers(a, b):
    """Function demonstrating exception handling"""
    try:
        print(f"Attempting to divide {a} by {b}")
        result = a / b
        print(f"Division successful!")
        return result
    except ZeroDivisionError:
        print("Error: Cannot divide by zero!")
        return None
    except TypeError:
        print("Error: Invalid types for division!")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    else:
        print("No exceptions occurred")
    finally:
        print("Division operation completed")

print("Basic Exception Handling Example:")
print(f"Result 1: {divide_numbers(10, 2)}")
print(f"Result 2: {divide_numbers(10, 0)}")
print(f"Result 3: {divide_numbers('10', 2)}")

# Question 2: Common Python Exceptions
print("\n2. What are the common built-in exceptions in Python?")
print("-" * 56)
print("""
COMMON EXCEPTIONS:
• SyntaxError: Invalid Python syntax
• NameError: Variable not defined
• TypeError: Wrong data type for operation
• ValueError: Right type but wrong value
• IndexError: List index out of range
• KeyError: Dictionary key doesn't exist
• AttributeError: Object has no attribute
• FileNotFoundError: File doesn't exist
• ZeroDivisionError: Division by zero
• ImportError: Module import failed
• RuntimeError: Generic runtime error
""")

def demonstrate_exceptions():
    """Demonstrate common exceptions"""
    exceptions_demo = [
        ("NameError", lambda: print(undefined_variable)),
        ("TypeError", lambda: "string" + 5),
        ("ValueError", lambda: int("not_a_number")),
        ("IndexError", lambda: [1, 2, 3][10]),
        ("KeyError", lambda: {"a": 1}["b"]),
        ("AttributeError", lambda: "string".non_existent_method()),
        ("ZeroDivisionError", lambda: 10 / 0),
    ]
    
    for exc_name, func in exceptions_demo:
        try:
            func()
        except Exception as e:
            print(f"{exc_name}: {type(e).__name__} - {e}")

print("Common Exceptions Demo:")
demonstrate_exceptions()

# Question 3: Custom Exceptions
print("\n3. How do you create custom exceptions in Python?")
print("-" * 49)
print("""
Custom exceptions are created by inheriting from built-in exception classes:

BEST PRACTICES:
• Inherit from appropriate base exception
• Provide meaningful error messages
• Include relevant context information
• Use descriptive exception names
• Group related exceptions in hierarchy
""")

class ValidationError(Exception):
    """Base class for validation errors"""
    pass

class AgeValidationError(ValidationError):
    """Exception for age validation errors"""
    def __init__(self, age, message="Invalid age"):
        self.age = age
        self.message = message
        super().__init__(f"{message}: {age}")

class EmailValidationError(ValidationError):
    """Exception for email validation errors"""
    def __init__(self, email, message="Invalid email format"):
        self.email = email
        self.message = message
        super().__init__(f"{message}: {email}")

class BankAccountError(Exception):
    """Base class for bank account errors"""
    pass

class InsufficientFundsError(BankAccountError):
    """Exception for insufficient funds"""
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        message = f"Insufficient funds: Balance ${balance}, Attempted ${amount}"
        super().__init__(message)

class User:
    """User class with validation"""
    def __init__(self, name, age, email):
        self.name = name
        self.age = self.validate_age(age)
        self.email = self.validate_email(email)
    
    def validate_age(self, age):
        if not isinstance(age, int):
            raise AgeValidationError(age, "Age must be an integer")
        if age < 0 or age > 150:
            raise AgeValidationError(age, "Age must be between 0 and 150")
        return age
    
    def validate_email(self, email):
        if not isinstance(email, str) or "@" not in email:
            raise EmailValidationError(email)
        return email

print("Custom Exceptions Example:")
try:
    user1 = User("Alice", 25, "alice@example.com")
    print(f"Created user: {user1.name}")
except ValidationError as e:
    print(f"Validation error: {e}")

try:
    user2 = User("Bob", -5, "invalid_email")
except AgeValidationError as e:
    print(f"Age error: {e}")
except EmailValidationError as e:
    print(f"Email error: {e}")

# Question 4: Exception Chaining
print("\n4. What is exception chaining in Python?")
print("-" * 43)
print("""
Exception chaining links exceptions to show the original cause:

SYNTAX:
• raise NewException from original_exception
• raise NewException() from None (suppress chain)

BENEFITS:
• Preserves original error context
• Helps in debugging
• Shows complete error chain
• Better error reporting
""")

def process_data(data):
    """Function that demonstrates exception chaining"""
    try:
        # Simulate data processing
        result = int(data) / 2
        return result
    except ValueError as e:
        # Chain the exception
        raise ValueError("Data processing failed") from e
    except ZeroDivisionError as e:
        # Different chaining
        raise RuntimeError("Calculation error occurred") from e

def handle_file_operation(filename):
    """Function demonstrating exception suppression"""
    try:
        with open(filename, 'r') as f:
            content = f.read()
        return content
    except FileNotFoundError as e:
        # Suppress the original exception chain
        raise ValueError("Configuration file is missing") from None

print("Exception Chaining Example:")
try:
    process_data("invalid")
except ValueError as e:
    print(f"Chained exception: {e}")
    print(f"Original cause: {e.__cause__}")

try:
    handle_file_operation("nonexistent.txt")
except ValueError as e:
    print(f"Suppressed chain: {e}")
    print(f"Original cause: {e.__cause__}")

# ============================================================================
# SECTION 2: FILE OPERATIONS
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 2: FILE OPERATIONS")
print("=" * 50)

# Question 5: File Handling in Python
print("\n5. How do you handle file operations in Python?")
print("-" * 48)
print("""
Python provides built-in functions for file operations:

OPENING FILES:
• open(filename, mode): Opens a file
• Modes: 'r' (read), 'w' (write), 'a' (append), 'x' (exclusive)
• Binary modes: 'rb', 'wb', 'ab'
• Text encoding: encoding='utf-8'

READING FILES:
• read(): Read entire file
• readline(): Read one line
• readlines(): Read all lines as list
• For loop: Iterate line by line

WRITING FILES:
• write(): Write string to file
• writelines(): Write list of strings

BEST PRACTICES:
• Always use 'with' statement (context manager)
• Handle exceptions appropriately
• Specify encoding explicitly
• Close files properly
""")

import os
import tempfile

def demonstrate_file_operations():
    """Demonstrate various file operations"""
    # Create a temporary file for demonstration
    temp_dir = tempfile.gettempdir()
    test_file = os.path.join(temp_dir, "test_file.txt")
    
    # Writing to file
    print("Writing to file:")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("Hello, World!\n")
        f.write("This is line 2.\n")
        f.writelines(["Line 3\n", "Line 4\n"])
    print(f"File written: {test_file}")
    
    # Reading entire file
    print("\nReading entire file:")
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()
        print(content)
    
    # Reading line by line
    print("Reading line by line:")
    with open(test_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            print(f"Line {line_num}: {line.strip()}")
    
    # Appending to file
    print("\nAppending to file:")
    with open(test_file, 'a', encoding='utf-8') as f:
        f.write("Appended line\n")
    
    # Reading with readlines()
    print("All lines as list:")
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(lines)
    
    # Cleanup
    os.remove(test_file)
    print(f"File removed: {test_file}")

demonstrate_file_operations()

# Question 6: Working with Different File Types
print("\n6. How do you work with different file types?")
print("-" * 46)
print("""
Python can handle various file types with appropriate libraries:

TEXT FILES:
• Default mode for human-readable files
• Specify encoding (UTF-8, ASCII, etc.)
• Handle line endings properly

BINARY FILES:
• Use 'b' mode for images, executables, etc.
• Work with bytes instead of strings
• No encoding/decoding

CSV FILES:
• Use csv module
• Handle different delimiters and formats
• DictReader for dictionary-based access

JSON FILES:
• Use json module
• Load/dump Python objects
• Handle serialization/deserialization
""")

import json
import csv

def demonstrate_file_types():
    """Demonstrate working with different file types"""
    temp_dir = tempfile.gettempdir()
    
    # JSON file operations
    json_file = os.path.join(temp_dir, "data.json")
    data = {
        "name": "Alice",
        "age": 30,
        "skills": ["Python", "JavaScript", "SQL"],
        "is_employed": True
    }
    
    print("JSON file operations:")
    # Write JSON
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"JSON written to: {json_file}")
    
    # Read JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
    print(f"Loaded JSON: {loaded_data}")
    
    # CSV file operations
    csv_file = os.path.join(temp_dir, "data.csv")
    csv_data = [
        {"name": "Alice", "age": 30, "city": "New York"},
        {"name": "Bob", "age": 25, "city": "Los Angeles"},
        {"name": "Charlie", "age": 35, "city": "Chicago"}
    ]
    
    print("\nCSV file operations:")
    # Write CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["name", "age", "city"])
        writer.writeheader()
        writer.writerows(csv_data)
    print(f"CSV written to: {csv_file}")
    
    # Read CSV
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        print("CSV data:")
        for row in reader:
            print(f"  {row}")
    
    # Binary file operations
    binary_file = os.path.join(temp_dir, "binary_data.bin")
    binary_data = b'\x00\x01\x02\x03\x04\x05'
    
    print("\nBinary file operations:")
    # Write binary
    with open(binary_file, 'wb') as f:
        f.write(binary_data)
    print(f"Binary data written to: {binary_file}")
    
    # Read binary
    with open(binary_file, 'rb') as f:
        loaded_binary = f.read()
    print(f"Loaded binary data: {loaded_binary}")
    
    # Cleanup
    for file_path in [json_file, csv_file, binary_file]:
        if os.path.exists(file_path):
            os.remove(file_path)

demonstrate_file_types()

# Question 7: File System Operations
print("\n7. How do you perform file system operations in Python?")
print("-" * 56)
print("""
Python's os and pathlib modules provide file system operations:

OS MODULE:
• os.path: Path manipulation
• os.listdir(): List directory contents
• os.makedirs(): Create directories
• os.remove(): Delete files
• os.rmdir(): Remove directories

PATHLIB MODULE (Modern approach):
• Path objects for cross-platform paths
• More intuitive and readable
• Object-oriented interface
• Better path manipulation

SHUTIL MODULE:
• High-level file operations
• Copy, move, delete operations
• Directory tree operations
""")

import shutil
from pathlib import Path

def demonstrate_filesystem_operations():
    """Demonstrate file system operations"""
    # Create a temporary directory for demonstration
    base_dir = Path(tempfile.gettempdir()) / "python_demo"
    
    print("File System Operations Demo:")
    
    # Create directory structure
    print("\n1. Creating directory structure:")
    (base_dir / "subdir1").mkdir(parents=True, exist_ok=True)
    (base_dir / "subdir2").mkdir(exist_ok=True)
    print(f"Created directories in: {base_dir}")
    
    # Create files
    print("\n2. Creating files:")
    file1 = base_dir / "file1.txt"
    file2 = base_dir / "subdir1" / "file2.txt"
    
    file1.write_text("Content of file 1")
    file2.write_text("Content of file 2")
    print(f"Created files: {file1.name}, {file2.name}")
    
    # List directory contents
    print("\n3. Directory contents:")
    for item in base_dir.rglob("*"):
        if item.is_file():
            print(f"File: {item.relative_to(base_dir)}")
        elif item.is_dir():
            print(f"Dir:  {item.relative_to(base_dir)}")
    
    # Path information
    print(f"\n4. Path information for {file1}:")
    print(f"   Name: {file1.name}")
    print(f"   Stem: {file1.stem}")
    print(f"   Suffix: {file1.suffix}")
    print(f"   Parent: {file1.parent}")
    print(f"   Exists: {file1.exists()}")
    print(f"   Size: {file1.stat().st_size} bytes")
    
    # Copy operations
    print("\n5. Copy operations:")
    copied_file = base_dir / "file1_copy.txt"
    shutil.copy2(file1, copied_file)
    print(f"Copied {file1.name} to {copied_file.name}")
    
    # Move operations
    print("\n6. Move operations:")
    moved_file = base_dir / "subdir2" / "moved_file.txt"
    shutil.move(str(copied_file), str(moved_file))
    print(f"Moved {copied_file.name} to {moved_file}")
    
    # Search for files
    print("\n7. Finding files:")
    txt_files = list(base_dir.glob("**/*.txt"))
    print(f"Found .txt files: {[f.name for f in txt_files]}")
    
    # Cleanup
    print("\n8. Cleanup:")
    shutil.rmtree(base_dir)
    print(f"Removed directory tree: {base_dir}")

demonstrate_filesystem_operations()

# ============================================================================
# SECTION 3: MODULES AND PACKAGES
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 3: MODULES AND PACKAGES")
print("=" * 50)

# Question 8: Python Modules
print("\n8. What are modules and how do you create them?")
print("-" * 49)
print("""
A module is a file containing Python code that can define functions, classes, and variables:

CREATING MODULES:
• Save Python code in a .py file
• Import using the filename (without .py)
• Module name becomes the namespace

IMPORTING MODULES:
• import module_name
• from module_name import function_name
• from module_name import function_name as alias
• import module_name as alias

MODULE SEARCH PATH:
• Current directory
• PYTHONPATH environment variable
• Standard library directories
• Site-packages directory

SPECIAL VARIABLES:
• __name__: Module name or '__main__'
• __file__: Module file path
• __doc__: Module docstring
""")

# Simulate creating a module (normally would be in separate file)
module_code = '''
"""
Example module demonstrating Python module concepts
"""

# Module-level variable
MODULE_VERSION = "1.0.0"

# Module-level function
def greet(name):
    """Greet a person"""
    return f"Hello, {name}!"

def calculate_area(length, width):
    """Calculate rectangle area"""
    return length * width

# Module-level class
class Calculator:
    """Simple calculator class"""
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b

# Code that runs when module is executed directly
if __name__ == "__main__":
    print("Module is being run directly")
    print(f"Version: {MODULE_VERSION}")
else:
    print(f"Module imported: {__name__}")
'''

# Write the module to a temporary file
temp_dir = tempfile.gettempdir()
module_file = os.path.join(temp_dir, "example_module.py")
with open(module_file, 'w') as f:
    f.write(module_code)

# Simulate importing (in real scenario, you'd just import)
print("Module Creation Example:")
print("Created example_module.py with functions and classes")
print("The module contains:")
print("- MODULE_VERSION variable")
print("- greet() function")
print("- calculate_area() function")  
print("- Calculator class")
print("- __name__ == '__main__' check")

# Cleanup
os.remove(module_file)

# Question 9: Python Packages
print("\n9. What are packages and how do you create them?")
print("-" * 50)
print("""
A package is a directory containing multiple modules with an __init__.py file:

PACKAGE STRUCTURE:
mypackage/
    __init__.py
    module1.py
    module2.py
    subpackage/
        __init__.py
        submodule.py

__init__.py FILE:
• Marks directory as a package
• Can be empty or contain initialization code
• Controls what gets imported with 'from package import *'
• Can import and expose submodules

IMPORTING FROM PACKAGES:
• from mypackage import module1
• from mypackage.module1 import function
• from mypackage.subpackage import submodule
• import mypackage.module1 as m1

BENEFITS:
• Organize related modules
• Avoid naming conflicts
• Create hierarchical structure
• Distribute code easily
""")

def demonstrate_package_structure():
    """Demonstrate package creation and structure"""
    base_dir = Path(tempfile.gettempdir()) / "demo_package"
    
    # Create package structure
    package_dir = base_dir / "mypackage"
    subpackage_dir = package_dir / "subpackage"
    
    package_dir.mkdir(parents=True, exist_ok=True)
    subpackage_dir.mkdir(exist_ok=True)
    
    # Create __init__.py files
    (package_dir / "__init__.py").write_text('''
"""
MyPackage - A demonstration package
"""
__version__ = "1.0.0"

# Import functions to make them available at package level
from .math_utils import add, multiply
from .string_utils import reverse_string

__all__ = ["add", "multiply", "reverse_string"]
''')
    
    (subpackage_dir / "__init__.py").write_text('''
"""
Subpackage for advanced utilities
"""
from .advanced import advanced_function
''')
    
    # Create module files
    (package_dir / "math_utils.py").write_text('''
"""Math utilities module"""

def add(a, b):
    """Add two numbers"""
    return a + b

def multiply(a, b):
    """Multiply two numbers"""
    return a * b

def divide(a, b):
    """Divide two numbers"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
''')
    
    (package_dir / "string_utils.py").write_text('''
"""String utilities module"""

def reverse_string(s):
    """Reverse a string"""
    return s[::-1]

def capitalize_words(s):
    """Capitalize each word"""
    return ' '.join(word.capitalize() for word in s.split())
''')
    
    (subpackage_dir / "advanced.py").write_text('''
"""Advanced utilities"""

def advanced_function():
    """An advanced function"""
    return "This is an advanced function"
''')
    
    print("Package Structure Created:")
    print("mypackage/")
    print("├── __init__.py")
    print("├── math_utils.py")
    print("├── string_utils.py")
    print("└── subpackage/")
    print("    ├── __init__.py")
    print("    └── advanced.py")
    
    # Cleanup
    shutil.rmtree(base_dir)
    print("\nPackage structure demonstrated (cleaned up)")

demonstrate_package_structure()

# Question 10: Module vs Package vs Library
print("\n10. What's the difference between module, package, and library?")
print("-" * 67)
print("""
MODULE:
• Single .py file containing Python code
• Contains functions, classes, variables
• Imported with 'import module_name'
• Example: math.py, random.py

PACKAGE:
• Directory containing multiple modules
• Has __init__.py file
• Organized collection of modules
• Example: numpy, requests

LIBRARY:
• Collection of packages and modules
• Provides specific functionality
• Can be third-party or built-in
• Example: NumPy library, Django library

FRAMEWORK:
• Comprehensive platform for development
• Provides structure and tools
• More than just libraries
• Example: Django, Flask, TensorFlow

HIERARCHY:
Library > Package > Module > Function/Class
""")

# ============================================================================
# SECTION 4: MEMORY MANAGEMENT
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 4: MEMORY MANAGEMENT")
print("=" * 50)

# Question 11: Python Memory Management
print("\n11. How does Python handle memory management?")
print("-" * 48)
print("""
Python handles memory management automatically through several mechanisms:

REFERENCE COUNTING:
• Each object tracks how many references point to it
• When count reaches zero, object is deleted immediately
• Simple and deterministic
• Can't handle circular references

GARBAGE COLLECTION:
• Handles circular references
• Uses mark-and-sweep algorithm
• Runs periodically or when memory pressure is high
• Can be controlled via gc module

MEMORY POOLS:
• Small objects use pre-allocated memory pools
• Reduces fragmentation and allocation overhead
• Improves performance for frequent allocations

PRIVATE HEAP:
• All Python objects stored in private heap
• Managed by Python memory manager
• Not accessible to programmer directly
""")

import gc
import sys

def demonstrate_memory_management():
    """Demonstrate memory management concepts"""
    print("Memory Management Demo:")
    
    # Reference counting
    print("\n1. Reference counting:")
    x = [1, 2, 3, 4, 5]
    print(f"Reference count for x: {sys.getrefcount(x)}")
    
    y = x  # Create another reference
    print(f"Reference count after y = x: {sys.getrefcount(x)}")
    
    del y  # Remove reference
    print(f"Reference count after del y: {sys.getrefcount(x)}")
    
    # Garbage collection
    print("\n2. Garbage collection:")
    print(f"Garbage collection enabled: {gc.isenabled()}")
    print(f"Garbage collection counts: {gc.get_count()}")
    
    # Create circular reference
    class Node:
        def __init__(self, value):
            self.value = value
            self.ref = None
    
    # Create circular reference
    node1 = Node(1)
    node2 = Node(2)
    node1.ref = node2
    node2.ref = node1
    
    print("Created circular reference")
    
    # Delete references (but circular reference remains)
    del node1, node2
    
    # Force garbage collection
    collected = gc.collect()
    print(f"Objects collected by garbage collector: {collected}")
    
    # Memory usage
    print("\n3. Memory information:")
    print(f"Reference count for integer 1: {sys.getrefcount(1)}")
    print(f"Size of empty list: {sys.getsizeof([])} bytes")
    print(f"Size of list with 5 elements: {sys.getsizeof([1, 2, 3, 4, 5])} bytes")
    print(f"Size of empty dict: {sys.getsizeof({})} bytes")
    print(f"Size of dict with 3 items: {sys.getsizeof({'a': 1, 'b': 2, 'c': 3})} bytes")

demonstrate_memory_management()

# Question 12: Memory Optimization Techniques
print("\n12. What are some memory optimization techniques in Python?")
print("-" * 62)
print("""
MEMORY OPTIMIZATION TECHNIQUES:

1. USE GENERATORS INSTEAD OF LISTS:
   • Generate values on-demand
   • Constant memory usage regardless of size

2. USE __slots__ IN CLASSES:
   • Reduces memory overhead per instance
   • Prevents dynamic attribute creation

3. OPTIMIZE DATA STRUCTURES:
   • Use tuples instead of lists when possible
   • Use sets for membership testing
   • Use appropriate data types

4. MANAGE OBJECT LIFECYCLE:
   • Delete unused objects explicitly
   • Avoid circular references
   • Use weak references when appropriate

5. USE BUILT-IN FUNCTIONS:
   • map(), filter(), zip() are memory efficient
   • Use itertools for efficient iteration
""")

class OptimizedClass:
    """Class with __slots__ for memory optimization"""
    __slots__ = ['x', 'y', 'z']
    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class RegularClass:
    """Regular class without __slots__"""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def memory_optimization_demo():
    """Demonstrate memory optimization techniques"""
    print("Memory Optimization Demo:")
    
    # Compare __slots__ vs regular class
    print("\n1. __slots__ optimization:")
    regular = RegularClass(1, 2, 3)
    optimized = OptimizedClass(1, 2, 3)
    
    print(f"Regular class size: {sys.getsizeof(regular)} bytes")
    print(f"__slots__ class size: {sys.getsizeof(optimized)} bytes")
    print(f"Regular class __dict__: {sys.getsizeof(regular.__dict__)} bytes")
    
    # Generator vs list
    print("\n2. Generator vs List:")
    
    def number_generator(n):
        for i in range(n):
            yield i
    
    # Small comparison
    n = 1000
    list_obj = list(range(n))
    gen_obj = number_generator(n)
    
    print(f"List of {n} numbers: {sys.getsizeof(list_obj)} bytes")
    print(f"Generator object: {sys.getsizeof(gen_obj)} bytes")
    
    # Data structure optimization
    print("\n3. Data structure optimization:")
    
    # List vs tuple
    data_list = [1, 2, 3, 4, 5]
    data_tuple = (1, 2, 3, 4, 5)
    
    print(f"List size: {sys.getsizeof(data_list)} bytes")
    print(f"Tuple size: {sys.getsizeof(data_tuple)} bytes")
    
    # Set for membership testing
    items_list = list(range(1000))
    items_set = set(range(1000))
    
    print(f"List size (1000 items): {sys.getsizeof(items_list)} bytes")
    print(f"Set size (1000 items): {sys.getsizeof(items_set)} bytes")

memory_optimization_demo()

print("\n" + "=" * 80)
print("END OF EXCEPTION HANDLING & FILE OPERATIONS SECTION")
print("Continue with standard library and advanced topics in the next file...")
print("=" * 80)
