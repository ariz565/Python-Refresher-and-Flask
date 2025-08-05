"""
Complete Python Standard Library & Testing Interview Questions
Covering standard library modules, testing frameworks, and best practices
Based on comprehensive Python interview preparation material
"""

print("=" * 80)
print("COMPLETE PYTHON STANDARD LIBRARY & TESTING QUESTIONS")
print("=" * 80)

# ============================================================================
# SECTION 1: STANDARD LIBRARY MODULES
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 1: STANDARD LIBRARY MODULES")
print("=" * 50)

# Question 1: Essential Standard Library Modules
print("\n1. What are the most important Python standard library modules?")
print("-" * 66)
print("""
ESSENTIAL STANDARD LIBRARY MODULES:

DATA STRUCTURES & UTILITIES:
• collections: Counter, defaultdict, deque, namedtuple
• itertools: Efficient looping tools
• functools: Higher-order functions and operations
• operator: Function equivalents of operators

DATE & TIME:
• datetime: Date and time manipulation
• time: Time-related functions
• calendar: Calendar-related functions

FILE & SYSTEM:
• os: Operating system interface
• sys: System-specific parameters
• pathlib: Object-oriented filesystem paths
• shutil: High-level file operations
• glob: Unix-style pathname pattern expansion

TEXT PROCESSING:
• re: Regular expressions
• string: String constants and classes
• textwrap: Text wrapping and filling

NETWORKING & WEB:
• urllib: URL handling modules
• http: HTTP modules
• json: JSON encoder and decoder
• email: Email handling package

CONCURRENCY:
• threading: Thread-based parallelism
• multiprocessing: Process-based parallelism
• asyncio: Asynchronous I/O
• concurrent.futures: High-level interface
""")

# Question 2: Collections Module
print("\n2. How do you use the collections module?")
print("-" * 42)
print("""
The collections module provides specialized container datatypes:

COUNTER: Count hashable objects
DEFAULTDICT: Dictionary with default values
DEQUE: Double-ended queue
NAMEDTUPLE: Tuple subclass with named fields
CHAINMAP: Combine multiple dictionaries
ORDEREDDICT: Dictionary preserving insertion order (Python 3.7+ regular dict does this)
""")

from collections import Counter, defaultdict, deque, namedtuple, ChainMap

def demonstrate_collections():
    """Demonstrate collections module usage"""
    print("Collections Module Demo:")
    
    # Counter examples
    print("\n1. Counter - counting elements:")
    text = "hello world"
    char_count = Counter(text)
    print(f"Character counts in '{text}': {char_count}")
    
    words = ["apple", "banana", "apple", "orange", "banana", "apple"]
    word_count = Counter(words)
    print(f"Word counts: {word_count}")
    print(f"Most common 2: {word_count.most_common(2)}")
    
    # defaultdict examples
    print("\n2. defaultdict - dictionary with default values:")
    dd = defaultdict(list)
    dd['fruits'].append('apple')
    dd['fruits'].append('banana')
    dd['vegetables'].append('carrot')
    print(f"defaultdict example: {dict(dd)}")
    
    # Group words by length
    words = ['cat', 'dog', 'elephant', 'tiger', 'ant']
    grouped = defaultdict(list)
    for word in words:
        grouped[len(word)].append(word)
    print(f"Words grouped by length: {dict(grouped)}")
    
    # deque examples
    print("\n3. deque - double-ended queue:")
    dq = deque([1, 2, 3])
    dq.appendleft(0)  # Add to left
    dq.append(4)      # Add to right
    print(f"After appends: {dq}")
    
    dq.popleft()      # Remove from left
    dq.pop()          # Remove from right
    print(f"After pops: {dq}")
    
    # Rotating deque
    dq.rotate(1)      # Rotate right
    print(f"After rotate(1): {dq}")
    
    # namedtuple examples
    print("\n4. namedtuple - named fields:")
    Person = namedtuple('Person', ['name', 'age', 'city'])
    john = Person('John', 30, 'New York')
    print(f"Person: {john}")
    print(f"Name: {john.name}, Age: {john.age}")
    
    # namedtuple methods
    print(f"As dict: {john._asdict()}")
    jane = john._replace(name='Jane', age=25)
    print(f"Modified: {jane}")
    
    # ChainMap examples
    print("\n5. ChainMap - combining dictionaries:")
    dict1 = {'a': 1, 'b': 2}
    dict2 = {'b': 3, 'c': 4}
    dict3 = {'c': 5, 'd': 6}
    
    cm = ChainMap(dict1, dict2, dict3)
    print(f"ChainMap: {cm}")
    print(f"Value of 'b': {cm['b']}")  # First occurrence
    print(f"All keys: {list(cm.keys())}")

demonstrate_collections()

# Question 3: itertools Module
print("\n3. How do you use itertools for efficient iteration?")
print("-" * 52)
print("""
itertools provides efficient tools for creating iterators:

INFINITE ITERATORS:
• count(): Infinite arithmetic sequence
• cycle(): Infinite repetition of iterable
• repeat(): Infinite repetition of single value

TERMINATING ITERATORS:
• chain(): Flatten iterables
• compress(): Filter based on selectors
• dropwhile(): Drop while predicate is true
• takewhile(): Take while predicate is true
• filterfalse(): Filter false values
• islice(): Slice iterator
• zip_longest(): Zip with different lengths

COMBINATORIAL ITERATORS:
• product(): Cartesian product
• permutations(): All permutations
• combinations(): All combinations
• combinations_with_replacement(): Combinations with repetition
""")

import itertools

def demonstrate_itertools():
    """Demonstrate itertools module usage"""
    print("Itertools Module Demo:")
    
    # Infinite iterators (limited output)
    print("\n1. Infinite iterators:")
    
    # count
    counter = itertools.count(10, 2)  # Start at 10, step by 2
    print(f"count(10, 2) first 5: {list(itertools.islice(counter, 5))}")
    
    # cycle
    colors = itertools.cycle(['red', 'green', 'blue'])
    print(f"cycle colors first 7: {list(itertools.islice(colors, 7))}")
    
    # repeat
    repeated = itertools.repeat('hello', 3)
    print(f"repeat 'hello' 3 times: {list(repeated)}")
    
    # Terminating iterators
    print("\n2. Terminating iterators:")
    
    # chain
    list1 = [1, 2, 3]
    list2 = [4, 5, 6]
    list3 = [7, 8, 9]
    chained = itertools.chain(list1, list2, list3)
    print(f"chain: {list(chained)}")
    
    # compress
    data = ['a', 'b', 'c', 'd', 'e']
    selectors = [1, 0, 1, 0, 1]
    compressed = itertools.compress(data, selectors)
    print(f"compress: {list(compressed)}")
    
    # dropwhile and takewhile
    numbers = [1, 3, 5, 24, 7, 11, 9, 2]
    dropped = itertools.dropwhile(lambda x: x < 10, numbers)
    print(f"dropwhile x < 10: {list(dropped)}")
    
    taken = itertools.takewhile(lambda x: x < 10, numbers)
    print(f"takewhile x < 10: {list(taken)}")
    
    # Combinatorial iterators
    print("\n3. Combinatorial iterators:")
    
    # product
    colors = ['red', 'blue']
    sizes = ['S', 'M', 'L']
    products = itertools.product(colors, sizes)
    print(f"product: {list(products)}")
    
    # permutations
    items = ['A', 'B', 'C']
    perms = itertools.permutations(items, 2)
    print(f"permutations of 2: {list(perms)}")
    
    # combinations
    combs = itertools.combinations(items, 2)
    print(f"combinations of 2: {list(combs)}")
    
    # combinations with replacement
    combs_rep = itertools.combinations_with_replacement(['A', 'B'], 2)
    print(f"combinations with replacement: {list(combs_rep)}")

demonstrate_itertools()

# Question 4: functools Module
print("\n4. How do you use functools for functional programming?")
print("-" * 56)
print("""
functools provides utilities for higher-order functions:

DECORATORS:
• lru_cache: Least-recently-used cache decorator
• wraps: Update wrapper function attributes
• singledispatch: Single-dispatch generic functions

FUNCTION TOOLS:
• partial: Partial function application
• reduce: Apply function cumulatively
• cache: Simple cache decorator (Python 3.9+)

COMPARISON TOOLS:
• cmp_to_key: Convert comparison function to key function
• total_ordering: Generate comparison methods
""")

import functools
import time

def demonstrate_functools():
    """Demonstrate functools module usage"""
    print("Functools Module Demo:")
    
    # lru_cache decorator
    print("\n1. lru_cache decorator:")
    
    @functools.lru_cache(maxsize=128)
    def fibonacci(n):
        """Fibonacci with memoization"""
        if n < 2:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    start = time.time()
    result = fibonacci(30)
    cached_time = time.time() - start
    print(f"Fibonacci(30) = {result}")
    print(f"Cache info: {fibonacci.cache_info()}")
    
    # partial function application
    print("\n2. partial function application:")
    
    def multiply(x, y, z):
        return x * y * z
    
    # Create partial function
    double = functools.partial(multiply, 2)
    triple = functools.partial(multiply, 3)
    
    print(f"double(5, 3): {double(5, 3)}")  # multiply(2, 5, 3)
    print(f"triple(4, 2): {triple(4, 2)}")  # multiply(3, 4, 2)
    
    # reduce function
    print("\n3. reduce function:")
    
    numbers = [1, 2, 3, 4, 5]
    
    # Sum using reduce
    sum_result = functools.reduce(lambda x, y: x + y, numbers)
    print(f"Sum using reduce: {sum_result}")
    
    # Find maximum using reduce
    max_result = functools.reduce(lambda x, y: x if x > y else y, numbers)
    print(f"Max using reduce: {max_result}")
    
    # Product using reduce
    product = functools.reduce(lambda x, y: x * y, numbers)
    print(f"Product using reduce: {product}")
    
    # singledispatch decorator
    print("\n4. singledispatch decorator:")
    
    @functools.singledispatch
    def process(arg):
        """Generic process function"""
        print(f"Processing {type(arg).__name__}: {arg}")
    
    @process.register
    def _(arg: int):
        print(f"Processing integer: {arg * 2}")
    
    @process.register
    def _(arg: str):
        print(f"Processing string: {arg.upper()}")
    
    @process.register
    def _(arg: list):
        print(f"Processing list of {len(arg)} items")
    
    process(42)
    process("hello")
    process([1, 2, 3, 4, 5])
    process(3.14)

demonstrate_functools()

# ============================================================================
# SECTION 2: TESTING FRAMEWORKS
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 2: TESTING FRAMEWORKS")
print("=" * 50)

# Question 5: unittest Framework
print("\n5. How do you use the unittest framework?")
print("-" * 43)
print("""
unittest is Python's built-in testing framework:

BASIC STRUCTURE:
• Inherit from unittest.TestCase
• Test methods start with 'test_'
• Use assert methods for verification
• setUp() and tearDown() for test fixtures

ASSERT METHODS:
• assertEqual(a, b): a == b
• assertNotEqual(a, b): a != b
• assertTrue(x): bool(x) is True
• assertFalse(x): bool(x) is False
• assertIs(a, b): a is b
• assertIsNone(x): x is None
• assertIn(a, b): a in b
• assertRaises(exc): Exception is raised

TEST FIXTURES:
• setUp(): Called before each test
• tearDown(): Called after each test
• setUpClass(): Called once before all tests
• tearDownClass(): Called once after all tests
""")

import unittest
from io import StringIO
import sys

class Calculator:
    """Simple calculator for testing examples"""
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
    
    def multiply(self, a, b):
        return a * b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

class TestCalculator(unittest.TestCase):
    """Test cases for Calculator class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.calc = Calculator()
    
    def test_add(self):
        """Test addition method"""
        self.assertEqual(self.calc.add(2, 3), 5)
        self.assertEqual(self.calc.add(-1, 1), 0)
        self.assertEqual(self.calc.add(0, 0), 0)
    
    def test_subtract(self):
        """Test subtraction method"""
        self.assertEqual(self.calc.subtract(5, 3), 2)
        self.assertEqual(self.calc.subtract(1, 1), 0)
        self.assertEqual(self.calc.subtract(0, 5), -5)
    
    def test_multiply(self):
        """Test multiplication method"""
        self.assertEqual(self.calc.multiply(3, 4), 12)
        self.assertEqual(self.calc.multiply(0, 5), 0)
        self.assertEqual(self.calc.multiply(-2, 3), -6)
    
    def test_divide(self):
        """Test division method"""
        self.assertEqual(self.calc.divide(10, 2), 5)
        self.assertEqual(self.calc.divide(7, 2), 3.5)
        
        # Test exception
        with self.assertRaises(ValueError):
            self.calc.divide(10, 0)
    
    def test_divide_error_message(self):
        """Test division error message"""
        with self.assertRaises(ValueError) as context:
            self.calc.divide(10, 0)
        self.assertEqual(str(context.exception), "Cannot divide by zero")

def run_unittest_example():
    """Run unittest example"""
    print("Unittest Framework Demo:")
    
    # Capture test output
    test_output = StringIO()
    runner = unittest.TextTestRunner(stream=test_output, verbosity=2)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCalculator)
    
    # Run tests
    result = runner.run(suite)
    
    # Print results
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")

run_unittest_example()

# Question 6: Testing Best Practices
print("\n6. What are testing best practices in Python?")
print("-" * 47)
print("""
TESTING BEST PRACTICES:

TEST ORGANIZATION:
• One test file per module
• Descriptive test names
• Group related tests in classes
• Use test fixtures appropriately

TEST DESIGN:
• Test one thing at a time
• Use AAA pattern (Arrange, Act, Assert)
• Make tests independent
• Include both positive and negative tests

TEST COVERAGE:
• Aim for high test coverage
• Test edge cases and error conditions
• Use code coverage tools
• Don't just chase 100% coverage

MOCKING AND FIXTURES:
• Mock external dependencies
• Use test doubles for isolation
• Create reusable test fixtures
• Clean up after tests

CONTINUOUS TESTING:
• Run tests automatically
• Test early and often
• Use CI/CD pipelines
• Keep tests fast
""")

# Question 7: Mocking and Test Doubles
print("\n7. How do you use mocking in Python tests?")
print("-" * 44)
print("""
Mocking allows you to replace parts of your system with mock objects:

UNITTEST.MOCK MODULE:
• Mock: General purpose mock object
• MagicMock: Mock with magic methods
• patch: Decorator/context manager for mocking
• patch.object: Mock specific object methods
• patch.multiple: Mock multiple things

MOCK FEATURES:
• Return values and side effects
• Track calls and arguments
• Assertions about usage
• Automatic attribute creation

WHEN TO MOCK:
• External services (APIs, databases)
• File system operations
• Time-dependent code
• Complex dependencies
""")

from unittest.mock import Mock, MagicMock, patch, mock_open

def demonstrate_mocking():
    """Demonstrate mocking concepts"""
    print("Mocking Demo:")
    
    # Basic Mock usage
    print("\n1. Basic Mock usage:")
    mock_obj = Mock()
    mock_obj.method.return_value = "mocked result"
    
    result = mock_obj.method("arg1", "arg2")
    print(f"Mock result: {result}")
    print(f"Mock called: {mock_obj.method.called}")
    print(f"Call count: {mock_obj.method.call_count}")
    print(f"Call args: {mock_obj.method.call_args}")
    
    # Mock with side effects
    print("\n2. Mock with side effects:")
    mock_func = Mock(side_effect=[1, 2, 3, ValueError("Error!")])
    
    try:
        for i in range(4):
            result = mock_func()
            print(f"Call {i+1}: {result}")
    except ValueError as e:
        print(f"Call 4: {e}")
    
    # MagicMock for magic methods
    print("\n3. MagicMock for magic methods:")
    magic_mock = MagicMock()
    magic_mock.__len__.return_value = 5
    magic_mock.__getitem__.return_value = "item"
    
    print(f"Length: {len(magic_mock)}")
    print(f"Item access: {magic_mock[0]}")
    
    # Patch decorator example
    print("\n4. Patch decorator example:")
    
    def get_current_time():
        import datetime
        return datetime.datetime.now()
    
    @patch('__main__.get_current_time')
    def test_with_patch(mock_time):
        mock_time.return_value = "2023-01-01 12:00:00"
        result = get_current_time()
        print(f"Mocked time: {result}")
    
    test_with_patch()
    
    # File operations mocking
    print("\n5. File operations mocking:")
    
    with patch('builtins.open', mock_open(read_data="file content")) as mock_file:
        with open('test.txt', 'r') as f:
            content = f.read()
        print(f"Mocked file content: {content}")
        mock_file.assert_called_once_with('test.txt', 'r')

demonstrate_mocking()

# Question 8: Property-Based Testing
print("\n8. What is property-based testing?")
print("-" * 37)
print("""
Property-based testing generates random test inputs to verify properties:

CONCEPT:
• Define properties that should always hold
• Generate random inputs automatically
• Framework finds edge cases
• Shrinks failing examples to minimal cases

HYPOTHESIS LIBRARY:
• Popular Python property-based testing library
• Generates diverse test data
• Provides statistical insights
• Integrates with unittest and pytest

BENEFITS:
• Finds edge cases humans miss
• Tests with diverse inputs
• Reduces test maintenance
• Provides confidence in correctness

EXAMPLE PROPERTIES:
• Reversing a list twice gives original
• Adding then subtracting gives original
• Serializing then deserializing preserves data
""")

# Note: hypothesis is not part of standard library
print("Property-Based Testing Example (conceptual):")
print("""
# Example using hypothesis library (not demonstrated due to dependency)

from hypothesis import given, strategies as st
import hypothesis

@given(st.lists(st.integers()))
def test_reverse_twice_is_identity(lst):
    assert list(reversed(list(reversed(lst)))) == lst

@given(st.integers(), st.integers())
def test_addition_commutative(x, y):
    assert x + y == y + x

@given(st.text())
def test_encode_decode_identity(text):
    assert text.encode('utf-8').decode('utf-8') == text
""")

# ============================================================================
# SECTION 3: PERFORMANCE TESTING AND PROFILING
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 3: PERFORMANCE TESTING AND PROFILING")
print("=" * 50)

# Question 9: Performance Measurement
print("\n9. How do you measure and profile Python code performance?")
print("-" * 62)
print("""
TIMING TOOLS:
• time.time(): Wall clock time
• time.perf_counter(): High-resolution timer
• timeit: Accurate timing of small code snippets
• datetime: Date and time operations

PROFILING TOOLS:
• cProfile: Built-in profiler
• profile: Pure Python profiler
• line_profiler: Line-by-line profiling
• memory_profiler: Memory usage profiling

PERFORMANCE ANALYSIS:
• Identify bottlenecks
• Measure memory usage
• Analyze algorithm complexity
• Compare implementations
""")

import timeit
import cProfile
import pstats
from io import StringIO

def demonstrate_performance_measurement():
    """Demonstrate performance measurement techniques"""
    print("Performance Measurement Demo:")
    
    # timeit usage
    print("\n1. Using timeit:")
    
    # Compare list comprehension vs map
    setup = "data = range(1000)"
    list_comp = "[x**2 for x in data]"
    map_func = "list(map(lambda x: x**2, data))"
    
    time_list_comp = timeit.timeit(list_comp, setup, number=1000)
    time_map = timeit.timeit(map_func, setup, number=1000)
    
    print(f"List comprehension: {time_list_comp:.6f} seconds")
    print(f"Map function: {time_map:.6f} seconds")
    print(f"Ratio: {time_list_comp/time_map:.2f}")
    
    # Different string concatenation methods
    print("\n2. String concatenation comparison:")
    
    methods = {
        "Plus operator": "result = ''; [result := result + str(i) for i in range(100)]",
        "Join method": "''.join(str(i) for i in range(100))",
        "F-string": "''.join(f'{i}' for i in range(100))"
    }
    
    for name, code in methods.items():
        time_taken = timeit.timeit(code, number=1000)
        print(f"{name}: {time_taken:.6f} seconds")
    
    # cProfile usage
    print("\n3. Using cProfile:")
    
    def slow_function():
        """Intentionally slow function for profiling"""
        total = 0
        for i in range(100000):
            total += i ** 2
        return total
    
    def another_function():
        """Another function that calls slow_function"""
        return slow_function() * 2
    
    # Profile the function
    pr = cProfile.Profile()
    pr.enable()
    
    result = another_function()
    
    pr.disable()
    
    # Get profiling results
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)  # Print top 10 functions
    
    profile_output = s.getvalue()
    lines = profile_output.split('\n')[:15]  # First 15 lines
    print('\n'.join(lines))

demonstrate_performance_measurement()

# Question 10: Memory Profiling
print("\n10. How do you profile memory usage in Python?")
print("-" * 49)
print("""
MEMORY PROFILING TECHNIQUES:

BUILT-IN TOOLS:
• sys.getsizeof(): Object size in bytes
• tracemalloc: Trace memory allocations
• gc: Garbage collector interface
• resource: System resource information

THIRD-PARTY TOOLS:
• memory_profiler: Line-by-line memory profiling
• pympler: Advanced memory analysis
• objgraph: Object reference graphs
• heapy: Heap analysis

MEMORY OPTIMIZATION:
• Identify memory leaks
• Find large objects
• Optimize data structures
• Monitor memory growth
""")

import tracemalloc
import gc

def demonstrate_memory_profiling():
    """Demonstrate memory profiling techniques"""
    print("Memory Profiling Demo:")
    
    # Basic memory measurement
    print("\n1. Basic memory measurement:")
    
    # Small objects
    small_list = [1, 2, 3, 4, 5]
    large_list = list(range(100000))
    
    print(f"Small list size: {sys.getsizeof(small_list)} bytes")
    print(f"Large list size: {sys.getsizeof(large_list)} bytes")
    
    # Dictionary vs list
    data_dict = {i: i**2 for i in range(1000)}
    data_list = [(i, i**2) for i in range(1000)]
    
    print(f"Dictionary size: {sys.getsizeof(data_dict)} bytes")
    print(f"List of tuples size: {sys.getsizeof(data_list)} bytes")
    
    # tracemalloc usage
    print("\n2. Using tracemalloc:")
    
    tracemalloc.start()
    
    # Allocate some memory
    data = []
    for i in range(10000):
        data.append([i] * 10)
    
    # Get current memory usage
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
    
    # Get top memory allocations
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print("\nTop 3 memory allocations:")
    for index, stat in enumerate(top_stats[:3], 1):
        print(f"{index}. {stat}")
    
    tracemalloc.stop()
    
    # Garbage collection info
    print("\n3. Garbage collection info:")
    print(f"GC counts: {gc.get_count()}")
    print(f"GC thresholds: {gc.get_threshold()}")
    
    # Force garbage collection
    collected = gc.collect()
    print(f"Objects collected: {collected}")

demonstrate_memory_profiling()

print("\n" + "=" * 80)
print("END OF STANDARD LIBRARY & TESTING SECTION")
print("Continue with web development and data science topics in the next file...")
print("=" * 80)
