# ===============================================================================
# PYTHON OOP PILLAR 3: POLYMORPHISM
# Real-Life Examples & Complete Mastery Guide
# ===============================================================================

"""
COMPREHENSIVE POLYMORPHISM COVERAGE:
===================================
1. Method Overloading Simulation
2. Operator Overloading (__dunder__ methods)
3. Dynamic Method Resolution
4. Duck Typing and Protocols
5. Abstract Base Classes for Polymorphism
6. Runtime Type Checking and Dispatch
7. Multiple Dispatch Patterns
8. Context-Dependent Behavior
9. Protocol-Based Programming
10. Advanced Polymorphic Patterns
"""

import operator
from typing import Protocol, Union, Any, List, Dict, Optional, Type, runtime_checkable
from functools import singledispatch, singledispatchmethod
from abc import ABC, abstractmethod
import json
import pickle
from datetime import datetime, timedelta
from enum import Enum
import copy

# ===============================================================================
# 1. METHOD OVERLOADING SIMULATION
# ===============================================================================

print("=" * 80)
print("1. METHOD OVERLOADING SIMULATION")
print("=" * 80)

print("\n--- Real-Life Example: Mathematical Calculator ---")

class Calculator:
    """Calculator demonstrating method overloading patterns"""
    
    def add(self, *args, **kwargs):
        """Polymorphic add method supporting multiple argument patterns"""
        
        # Pattern 1: Two numbers
        if len(args) == 2 and all(isinstance(x, (int, float)) for x in args):
            return self._add_numbers(args[0], args[1])
        
        # Pattern 2: List of numbers
        elif len(args) == 1 and isinstance(args[0], (list, tuple)):
            return self._add_list(args[0])
        
        # Pattern 3: Multiple numbers
        elif len(args) > 2 and all(isinstance(x, (int, float)) for x in args):
            return self._add_multiple(*args)
        
        # Pattern 4: Complex numbers
        elif len(args) == 2 and all(isinstance(x, complex) for x in args):
            return self._add_complex(args[0], args[1])
        
        # Pattern 5: Matrices (lists of lists)
        elif (len(args) == 2 and 
              all(isinstance(x, list) and all(isinstance(row, list) for row in x) for x in args)):
            return self._add_matrices(args[0], args[1])
        
        # Pattern 6: With precision specification
        elif 'precision' in kwargs:
            if len(args) == 2:
                result = self._add_numbers(args[0], args[1])
                return round(result, kwargs['precision'])
        
        # Pattern 7: String concatenation
        elif all(isinstance(x, str) for x in args):
            return self._add_strings(*args)
        
        else:
            raise TypeError(f"Unsupported argument types for add: {[type(x) for x in args]}")
    
    def _add_numbers(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Add two numbers"""
        print(f"Adding two numbers: {a} + {b}")
        return a + b
    
    def _add_list(self, numbers: List[Union[int, float]]) -> Union[int, float]:
        """Add all numbers in a list"""
        print(f"Adding list of {len(numbers)} numbers")
        return sum(numbers)
    
    def _add_multiple(self, *numbers: Union[int, float]) -> Union[int, float]:
        """Add multiple numbers"""
        print(f"Adding {len(numbers)} numbers")
        return sum(numbers)
    
    def _add_complex(self, a: complex, b: complex) -> complex:
        """Add complex numbers"""
        print(f"Adding complex numbers: {a} + {b}")
        return a + b
    
    def _add_matrices(self, matrix1: List[List[float]], matrix2: List[List[float]]) -> List[List[float]]:
        """Add two matrices"""
        print(f"Adding {len(matrix1)}x{len(matrix1[0])} matrices")
        
        if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
            raise ValueError("Matrices must have the same dimensions")
        
        result = []
        for i in range(len(matrix1)):
            row = []
            for j in range(len(matrix1[0])):
                row.append(matrix1[i][j] + matrix2[i][j])
            result.append(row)
        
        return result
    
    def _add_strings(self, *strings: str) -> str:
        """Concatenate strings"""
        print(f"Concatenating {len(strings)} strings")
        return ' '.join(strings)
    
    def multiply(self, *args, **kwargs):
        """Polymorphic multiply method"""
        
        # Pattern 1: Two numbers
        if len(args) == 2 and all(isinstance(x, (int, float)) for x in args):
            return self._multiply_numbers(args[0], args[1])
        
        # Pattern 2: Number and list (scalar multiplication)
        elif (len(args) == 2 and isinstance(args[0], (int, float)) and 
              isinstance(args[1], list)):
            return self._scalar_multiply(args[0], args[1])
        
        # Pattern 3: String repetition
        elif (len(args) == 2 and isinstance(args[0], str) and 
              isinstance(args[1], int)):
            return self._repeat_string(args[0], args[1])
        
        # Pattern 4: Matrix multiplication
        elif (len(args) == 2 and 
              all(isinstance(x, list) and all(isinstance(row, list) for row in x) for x in args)):
            return self._multiply_matrices(args[0], args[1])
        
        else:
            raise TypeError(f"Unsupported argument types for multiply: {[type(x) for x in args]}")
    
    def _multiply_numbers(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Multiply two numbers"""
        print(f"Multiplying two numbers: {a} * {b}")
        return a * b
    
    def _scalar_multiply(self, scalar: Union[int, float], vector: List[Union[int, float]]) -> List[Union[int, float]]:
        """Multiply vector by scalar"""
        print(f"Scalar multiplication: {scalar} * vector of length {len(vector)}")
        return [scalar * x for x in vector]
    
    def _repeat_string(self, string: str, count: int) -> str:
        """Repeat string"""
        print(f"Repeating string '{string}' {count} times")
        return string * count
    
    def _multiply_matrices(self, matrix1: List[List[float]], matrix2: List[List[float]]) -> List[List[float]]:
        """Matrix multiplication"""
        print(f"Multiplying {len(matrix1)}x{len(matrix1[0])} and {len(matrix2)}x{len(matrix2[0])} matrices")
        
        if len(matrix1[0]) != len(matrix2):
            raise ValueError("Number of columns in first matrix must equal number of rows in second matrix")
        
        result = []
        for i in range(len(matrix1)):
            row = []
            for j in range(len(matrix2[0])):
                sum_product = 0
                for k in range(len(matrix2)):
                    sum_product += matrix1[i][k] * matrix2[k][j]
                row.append(sum_product)
            result.append(row)
        
        return result

# Example usage - Method Overloading
print("Demonstrating method overloading simulation:")

calc = Calculator()

# Different add patterns
print(f"Two numbers: {calc.add(5, 3)}")
print(f"List of numbers: {calc.add([1, 2, 3, 4, 5])}")
print(f"Multiple numbers: {calc.add(1, 2, 3, 4)}")
print(f"Complex numbers: {calc.add(3+4j, 1+2j)}")
print(f"With precision: {calc.add(1/3, 1/3, precision=2)}")
print(f"String concatenation: {calc.add('Hello', 'World', 'Python')}")

# Matrix addition
matrix1 = [[1, 2], [3, 4]]
matrix2 = [[5, 6], [7, 8]]
print(f"Matrix addition: {calc.add(matrix1, matrix2)}")

# Different multiply patterns
print(f"Number multiplication: {calc.multiply(4, 5)}")
print(f"Scalar multiplication: {calc.multiply(3, [1, 2, 3])}")
print(f"String repetition: {calc.multiply('Hi! ', 3)}")

# ===============================================================================
# 2. OPERATOR OVERLOADING (__dunder__ methods)
# ===============================================================================

print("\n" + "=" * 80)
print("2. OPERATOR OVERLOADING (__dunder__ methods)")
print("=" * 80)

print("\n--- Real-Life Example: Money and Currency System ---")

class Money:
    """Money class with comprehensive operator overloading"""
    
    # Exchange rates (simplified for demo)
    EXCHANGE_RATES = {
        ('USD', 'EUR'): 0.85,
        ('EUR', 'USD'): 1.18,
        ('USD', 'GBP'): 0.75,
        ('GBP', 'USD'): 1.33,
        ('EUR', 'GBP'): 0.88,
        ('GBP', 'EUR'): 1.14
    }
    
    def __init__(self, amount: float, currency: str = 'USD'):
        self.amount = float(amount)
        self.currency = currency.upper()
        self._validate_currency()
    
    def _validate_currency(self):
        """Validate currency code"""
        valid_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD']
        if self.currency not in valid_currencies:
            raise ValueError(f"Unsupported currency: {self.currency}")
    
    def _convert_to_currency(self, target_currency: str) -> float:
        """Convert amount to target currency"""
        if self.currency == target_currency:
            return self.amount
        
        rate_key = (self.currency, target_currency)
        if rate_key in self.EXCHANGE_RATES:
            return self.amount * self.EXCHANGE_RATES[rate_key]
        
        # Try reverse rate
        reverse_key = (target_currency, self.currency)
        if reverse_key in self.EXCHANGE_RATES:
            return self.amount / self.EXCHANGE_RATES[reverse_key]
        
        raise ValueError(f"No exchange rate available for {self.currency} to {target_currency}")
    
    # Arithmetic operators
    def __add__(self, other):
        """Addition: money + money or money + number"""
        if isinstance(other, Money):
            # Convert other to same currency
            other_amount = other._convert_to_currency(self.currency)
            return Money(self.amount + other_amount, self.currency)
        elif isinstance(other, (int, float)):
            return Money(self.amount + other, self.currency)
        else:
            return NotImplemented
    
    def __radd__(self, other):
        """Right addition: number + money"""
        return self.__add__(other)
    
    def __sub__(self, other):
        """Subtraction: money - money or money - number"""
        if isinstance(other, Money):
            other_amount = other._convert_to_currency(self.currency)
            return Money(self.amount - other_amount, self.currency)
        elif isinstance(other, (int, float)):
            return Money(self.amount - other, self.currency)
        else:
            return NotImplemented
    
    def __rsub__(self, other):
        """Right subtraction: number - money"""
        if isinstance(other, (int, float)):
            return Money(other - self.amount, self.currency)
        else:
            return NotImplemented
    
    def __mul__(self, other):
        """Multiplication: money * number"""
        if isinstance(other, (int, float)):
            return Money(self.amount * other, self.currency)
        else:
            return NotImplemented
    
    def __rmul__(self, other):
        """Right multiplication: number * money"""
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """Division: money / number or money / money"""
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide money by zero")
            return Money(self.amount / other, self.currency)
        elif isinstance(other, Money):
            # Return ratio as float
            other_amount = other._convert_to_currency(self.currency)
            if other_amount == 0:
                raise ZeroDivisionError("Cannot divide by zero money")
            return self.amount / other_amount
        else:
            return NotImplemented
    
    def __floordiv__(self, other):
        """Floor division: money // number"""
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide money by zero")
            return Money(self.amount // other, self.currency)
        else:
            return NotImplemented
    
    def __mod__(self, other):
        """Modulo: money % number"""
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide money by zero")
            return Money(self.amount % other, self.currency)
        else:
            return NotImplemented
    
    def __pow__(self, other):
        """Power: money ** number (for compound interest)"""
        if isinstance(other, (int, float)):
            return Money(self.amount ** other, self.currency)
        else:
            return NotImplemented
    
    # Comparison operators
    def __eq__(self, other):
        """Equality: money == money"""
        if isinstance(other, Money):
            other_amount = other._convert_to_currency(self.currency)
            return abs(self.amount - other_amount) < 0.01  # Account for floating point precision
        elif isinstance(other, (int, float)):
            return abs(self.amount - other) < 0.01
        else:
            return False
    
    def __ne__(self, other):
        """Inequality: money != money"""
        return not self.__eq__(other)
    
    def __lt__(self, other):
        """Less than: money < money"""
        if isinstance(other, Money):
            other_amount = other._convert_to_currency(self.currency)
            return self.amount < other_amount
        elif isinstance(other, (int, float)):
            return self.amount < other
        else:
            return NotImplemented
    
    def __le__(self, other):
        """Less than or equal: money <= money"""
        return self.__lt__(other) or self.__eq__(other)
    
    def __gt__(self, other):
        """Greater than: money > money"""
        if isinstance(other, Money):
            other_amount = other._convert_to_currency(self.currency)
            return self.amount > other_amount
        elif isinstance(other, (int, float)):
            return self.amount > other
        else:
            return NotImplemented
    
    def __ge__(self, other):
        """Greater than or equal: money >= money"""
        return self.__gt__(other) or self.__eq__(other)
    
    # Unary operators
    def __neg__(self):
        """Negation: -money"""
        return Money(-self.amount, self.currency)
    
    def __pos__(self):
        """Positive: +money"""
        return Money(+self.amount, self.currency)
    
    def __abs__(self):
        """Absolute value: abs(money)"""
        return Money(abs(self.amount), self.currency)
    
    # Type conversion operators
    def __int__(self):
        """Convert to int: int(money)"""
        return int(self.amount)
    
    def __float__(self):
        """Convert to float: float(money)"""
        return float(self.amount)
    
    def __round__(self, ndigits=2):
        """Round: round(money, digits)"""
        return Money(round(self.amount, ndigits), self.currency)
    
    # String representation
    def __str__(self):
        """String representation"""
        return f"{self.amount:.2f} {self.currency}"
    
    def __repr__(self):
        """Developer representation"""
        return f"Money({self.amount}, '{self.currency}')"
    
    # Container operators
    def __bool__(self):
        """Boolean conversion: bool(money)"""
        return self.amount != 0
    
    def __hash__(self):
        """Hash for use in sets and dictionaries"""
        # Convert to USD for consistent hashing
        usd_amount = self._convert_to_currency('USD')
        return hash((round(usd_amount, 2), 'USD'))
    
    # Copy operators
    def __copy__(self):
        """Shallow copy"""
        return Money(self.amount, self.currency)
    
    def __deepcopy__(self, memo):
        """Deep copy"""
        return Money(copy.deepcopy(self.amount, memo), copy.deepcopy(self.currency, memo))

class BankAccount:
    """Bank account that works with Money objects"""
    
    def __init__(self, account_number: str, initial_balance: Money = None):
        self.account_number = account_number
        self.balance = initial_balance or Money(0)
        self.transactions = []
    
    def deposit(self, amount: Money):
        """Deposit money (using += operator)"""
        self.balance += amount
        self.transactions.append(f"Deposit: +{amount}")
        print(f"Deposited {amount}. New balance: {self.balance}")
    
    def withdraw(self, amount: Money):
        """Withdraw money (using -= operator)"""
        if self.balance >= amount:
            self.balance -= amount
            self.transactions.append(f"Withdrawal: -{amount}")
            print(f"Withdrew {amount}. New balance: {self.balance}")
        else:
            print(f"Insufficient funds. Balance: {self.balance}, Attempted: {amount}")
    
    def apply_interest(self, rate: float):
        """Apply interest (using ** operator for compound interest)"""
        self.balance = self.balance * (1 + rate)
        self.transactions.append(f"Interest applied: {rate*100:.2f}%")
        print(f"Interest applied. New balance: {self.balance}")

# Example usage - Operator Overloading
print("Demonstrating operator overloading with Money:")

# Create money objects
usd_100 = Money(100, 'USD')
eur_50 = Money(50, 'EUR')
gbp_25 = Money(25, 'GBP')

print(f"USD: {usd_100}")
print(f"EUR: {eur_50}")
print(f"GBP: {gbp_25}")

# Arithmetic operations
total = usd_100 + eur_50  # Automatic currency conversion
print(f"USD + EUR = {total}")

difference = usd_100 - eur_50
print(f"USD - EUR = {difference}")

doubled = usd_100 * 2
print(f"USD * 2 = {doubled}")

half = usd_100 / 2
print(f"USD / 2 = {half}")

# Comparison operations
print(f"USD > EUR? {usd_100 > eur_50}")
print(f"USD == USD? {usd_100 == Money(100, 'USD')}")

# Unary operations
print(f"-USD = {-usd_100}")
print(f"abs(-USD) = {abs(-usd_100)}")
print(f"round(USD/3, 2) = {round(usd_100/3, 2)}")

# Bank account operations
account = BankAccount("ACC001", Money(500, 'USD'))
account.deposit(Money(100, 'EUR'))  # Auto-converted
account.withdraw(Money(50, 'USD'))
account.apply_interest(0.05)  # 5% interest

# ===============================================================================
# 3. DUCK TYPING AND PROTOCOLS
# ===============================================================================

print("\n" + "=" * 80)
print("3. DUCK TYPING AND PROTOCOLS")
print("=" * 80)

print("\n--- Real-Life Example: File Processing System ---")

@runtime_checkable
class Readable(Protocol):
    """Protocol for readable objects"""
    
    def read(self, size: int = -1) -> str:
        """Read data from the object"""
        ...
    
    def readline(self) -> str:
        """Read a single line"""
        ...
    
    def close(self) -> None:
        """Close the readable object"""
        ...

@runtime_checkable
class Writable(Protocol):
    """Protocol for writable objects"""
    
    def write(self, data: str) -> int:
        """Write data to the object"""
        ...
    
    def flush(self) -> None:
        """Flush the write buffer"""
        ...
    
    def close(self) -> None:
        """Close the writable object"""
        ...

@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that can be serialized"""
    
    def serialize(self) -> str:
        """Serialize object to string"""
        ...
    
    @classmethod
    def deserialize(cls, data: str):
        """Deserialize string to object"""
        ...

# Duck typing implementations
class StringBuffer:
    """String buffer that behaves like a file"""
    
    def __init__(self, initial_data: str = ""):
        self._data = initial_data
        self._position = 0
        self._closed = False
    
    def read(self, size: int = -1) -> str:
        """Read from string buffer"""
        if self._closed:
            raise ValueError("Buffer is closed")
        
        if size == -1:
            result = self._data[self._position:]
            self._position = len(self._data)
        else:
            result = self._data[self._position:self._position + size]
            self._position += len(result)
        
        return result
    
    def readline(self) -> str:
        """Read a line from buffer"""
        if self._closed:
            raise ValueError("Buffer is closed")
        
        start = self._position
        newline_pos = self._data.find('\n', start)
        
        if newline_pos == -1:
            result = self._data[start:]
            self._position = len(self._data)
        else:
            result = self._data[start:newline_pos + 1]
            self._position = newline_pos + 1
        
        return result
    
    def write(self, data: str) -> int:
        """Write to string buffer"""
        if self._closed:
            raise ValueError("Buffer is closed")
        
        # Insert data at current position
        self._data = self._data[:self._position] + data + self._data[self._position:]
        self._position += len(data)
        return len(data)
    
    def flush(self) -> None:
        """Flush buffer (no-op for string buffer)"""
        pass
    
    def close(self) -> None:
        """Close the buffer"""
        self._closed = True
    
    def get_contents(self) -> str:
        """Get full buffer contents"""
        return self._data

class NetworkStream:
    """Simulated network stream"""
    
    def __init__(self, url: str):
        self.url = url
        self._connected = False
        self._buffer = f"Data from {url}\nLine 2\nLine 3\n"
        self._position = 0
    
    def connect(self):
        """Connect to network resource"""
        self._connected = True
        print(f"Connected to {self.url}")
    
    def read(self, size: int = -1) -> str:
        """Read from network stream"""
        if not self._connected:
            raise ConnectionError("Not connected")
        
        # Simulate network delay
        print("Reading from network...")
        
        if size == -1:
            result = self._buffer[self._position:]
            self._position = len(self._buffer)
        else:
            result = self._buffer[self._position:self._position + size]
            self._position += len(result)
        
        return result
    
    def readline(self) -> str:
        """Read line from network stream"""
        if not self._connected:
            raise ConnectionError("Not connected")
        
        start = self._position
        newline_pos = self._buffer.find('\n', start)
        
        if newline_pos == -1:
            result = self._buffer[start:]
            self._position = len(self._buffer)
        else:
            result = self._buffer[start:newline_pos + 1]
            self._position = newline_pos + 1
        
        return result
    
    def close(self) -> None:
        """Close network connection"""
        self._connected = False
        print(f"Disconnected from {self.url}")

class MemoryCache:
    """Memory cache that can be written to"""
    
    def __init__(self, max_size: int = 1024):
        self.max_size = max_size
        self._data = ""
        self._closed = False
    
    def write(self, data: str) -> int:
        """Write to cache"""
        if self._closed:
            raise ValueError("Cache is closed")
        
        if len(self._data) + len(data) > self.max_size:
            # Trim old data
            overflow = len(self._data) + len(data) - self.max_size
            self._data = self._data[overflow:]
        
        self._data += data
        print(f"Wrote {len(data)} bytes to cache")
        return len(data)
    
    def flush(self) -> None:
        """Flush cache (persist to disk in real implementation)"""
        print("Cache flushed to persistent storage")
    
    def close(self) -> None:
        """Close cache"""
        self.flush()
        self._closed = True
    
    def get_contents(self) -> str:
        """Get cache contents"""
        return self._data

# Duck typing processor
class FileProcessor:
    """Processor that works with any readable/writable object"""
    
    @staticmethod
    def copy_data(source: Readable, destination: Writable, chunk_size: int = 1024):
        """Copy data from readable to writable object using duck typing"""
        print(f"Copying data (chunk size: {chunk_size})")
        
        try:
            while True:
                chunk = source.read(chunk_size)
                if not chunk:
                    break
                
                destination.write(chunk)
                destination.flush()
            
            print("Copy completed successfully")
        
        finally:
            source.close()
            destination.close()
    
    @staticmethod
    def count_lines(source: Readable) -> int:
        """Count lines in readable object"""
        print("Counting lines...")
        
        try:
            line_count = 0
            while True:
                line = source.readline()
                if not line:
                    break
                line_count += 1
            
            print(f"Found {line_count} lines")
            return line_count
        
        finally:
            source.close()
    
    @staticmethod
    def process_with_protocol(obj: Any):
        """Process object based on supported protocols"""
        print(f"Processing object of type: {type(obj).__name__}")
        
        if isinstance(obj, Readable):
            print("✓ Object supports reading")
            data = obj.read(50)  # Read first 50 characters
            print(f"Preview: {data[:30]}...")
        
        if isinstance(obj, Writable):
            print("✓ Object supports writing")
            obj.write("Test data written by processor\n")
        
        if isinstance(obj, Serializable):
            print("✓ Object supports serialization")
            serialized = obj.serialize()
            print(f"Serialized data length: {len(serialized)}")

# Serializable implementations
class Document:
    """Document that can be serialized"""
    
    def __init__(self, title: str, content: str, author: str):
        self.title = title
        self.content = content
        self.author = author
        self.created_at = datetime.now()
    
    def serialize(self) -> str:
        """Serialize document to JSON"""
        data = {
            'title': self.title,
            'content': self.content,
            'author': self.author,
            'created_at': self.created_at.isoformat()
        }
        return json.dumps(data)
    
    @classmethod
    def deserialize(cls, data: str):
        """Deserialize JSON to document"""
        parsed = json.loads(data)
        doc = cls(parsed['title'], parsed['content'], parsed['author'])
        doc.created_at = datetime.fromisoformat(parsed['created_at'])
        return doc
    
    def __str__(self):
        return f"Document('{self.title}' by {self.author})"

# Example usage - Duck Typing and Protocols
print("Demonstrating duck typing and protocols:")

# Create different objects that implement the protocols
string_buffer = StringBuffer("Hello World\nLine 2\nLine 3\n")
network_stream = NetworkStream("https://api.example.com/data")
memory_cache = MemoryCache()

# Test protocol checking
print(f"StringBuffer is Readable: {isinstance(string_buffer, Readable)}")
print(f"StringBuffer is Writable: {isinstance(string_buffer, Writable)}")
print(f"MemoryCache is Writable: {isinstance(memory_cache, Writable)}")

# Use duck typing - objects behave like files
network_stream.connect()
line_count = FileProcessor.count_lines(network_stream)

# Copy data between different types
string_buffer2 = StringBuffer("Source data\nFrom string buffer\n")
FileProcessor.copy_data(string_buffer2, memory_cache)

# Process with protocol detection
document = Document("Test Doc", "Document content here", "John Doe")

# Test different protocol implementations
string_buffer3 = StringBuffer()
FileProcessor.process_with_protocol(string_buffer3)

memory_cache2 = MemoryCache()
FileProcessor.process_with_protocol(memory_cache2)

print("\n" + "=" * 80)
print("POLYMORPHISM PART 1 COMPLETE!")
print("=" * 80)
print("""
🎯 CONCEPTS COVERED:
✅ Method Overloading Simulation
✅ Operator Overloading (__dunder__ methods)
✅ Duck Typing and Protocols

📝 REAL-LIFE EXAMPLES:
- Mathematical Calculator (Method Overloading)
- Money and Currency System (Operator Overloading)
- File Processing System (Duck Typing & Protocols)

🚀 NEXT: Continue with polymorphism_advanced.py for Multiple Dispatch and Advanced Patterns
""")
