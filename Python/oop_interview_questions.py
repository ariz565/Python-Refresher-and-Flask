# ===============================================================================
# PYTHON OOP INTERVIEW QUESTIONS
# Complete Guide for Experienced Python Developers
# ===============================================================================

"""
COMPREHENSIVE OOP INTERVIEW COVERAGE:
====================================
1. Conceptual Questions (Theory & Understanding)
2. Practical Coding Questions
3. Design Pattern Questions
4. Debugging & Problem-Solving
5. Architecture & System Design
6. Tricky & Edge Case Questions
7. Performance & Optimization
8. Advanced OOP Concepts
9. Real-World Scenario Questions
10. Code Review & Best Practices
"""

import time
import weakref
import threading
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Union, Protocol
from datetime import datetime
import json
import copy

print("=" * 100)
print("PYTHON OOP INTERVIEW QUESTIONS FOR EXPERIENCED DEVELOPERS")
print("=" * 100)

# ===============================================================================
# 1. CONCEPTUAL QUESTIONS (THEORY & UNDERSTANDING)
# ===============================================================================

print("\n" + "=" * 80)
print("1. CONCEPTUAL QUESTIONS")
print("=" * 80)

print("""
Q1: Explain the difference between composition and inheritance with examples.
Answer:
- Inheritance: "IS-A" relationship (Car IS-A Vehicle)
- Composition: "HAS-A" relationship (Car HAS-A Engine)
- Composition is often preferred for flexibility and loose coupling
""")

# Inheritance example
class Vehicle:
    def __init__(self, brand):
        self.brand = brand
    
    def start(self):
        print(f"{self.brand} vehicle starting")

class Car(Vehicle):  # IS-A relationship
    def __init__(self, brand, model):
        super().__init__(brand)
        self.model = model

# Composition example
class Engine:
    def __init__(self, horsepower):
        self.horsepower = horsepower
    
    def start(self):
        print(f"Engine with {self.horsepower}HP starting")

class CarWithComposition:  # HAS-A relationship
    def __init__(self, brand, engine):
        self.brand = brand
        self.engine = engine  # Composition
    
    def start(self):
        print(f"{self.brand} starting")
        self.engine.start()

print("\nExample demonstration:")
car_inheritance = Car("Toyota", "Camry")
car_inheritance.start()

engine = Engine(200)
car_composition = CarWithComposition("Honda", engine)
car_composition.start()

print("""
Q2: What is the Method Resolution Order (MRO) and how does it work?
Answer:
- MRO determines the order in which methods are looked up in inheritance hierarchy
- Python uses C3 linearization algorithm
- Can be viewed using ClassName.__mro__ or ClassName.mro()
- Ensures consistent method resolution in multiple inheritance
""")

class A:
    def method(self):
        print("A.method")

class B(A):
    def method(self):
        print("B.method")
        super().method()

class C(A):
    def method(self):
        print("C.method")
        super().method()

class D(B, C):  # Multiple inheritance
    def method(self):
        print("D.method")
        super().method()

print(f"\nMRO for class D: {D.__mro__}")
d = D()
d.method()

print("""
Q3: Explain the difference between __new__ and __init__.
Answer:
- __new__: Creates and returns the instance (constructor)
- __init__: Initializes the created instance
- __new__ is called before __init__
- __new__ is a static method, __init__ is an instance method
""")

class SingletonExample:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            print("Creating new instance")
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            print("Initializing instance")
            self.initialized = True

print("\nSingleton demonstration:")
s1 = SingletonExample()
s2 = SingletonExample()
print(f"Same instance? {s1 is s2}")

# ===============================================================================
# 2. PRACTICAL CODING QUESTIONS
# ===============================================================================

print("\n" + "=" * 80)
print("2. PRACTICAL CODING QUESTIONS")
print("=" * 80)

print("""
Q4: Implement a decorator that logs method calls with timing information.
""")

def log_method_calls(func):
    """Decorator to log method calls with timing"""
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        class_name = self.__class__.__name__
        method_name = func.__name__
        
        print(f"[{datetime.now()}] Calling {class_name}.{method_name}")
        print(f"  Args: {args}, Kwargs: {kwargs}")
        
        try:
            result = func(self, *args, **kwargs)
            duration = time.time() - start_time
            print(f"  ✓ Completed in {duration:.4f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            print(f"  ✗ Failed in {duration:.4f}s: {e}")
            raise
    
    return wrapper

class DataProcessor:
    @log_method_calls
    def process_data(self, data: List[int]) -> int:
        """Process data with some computation"""
        time.sleep(0.1)  # Simulate processing
        return sum(data)
    
    @log_method_calls
    def divide_data(self, data: List[int], divisor: int) -> List[float]:
        """Divide each element by divisor"""
        if divisor == 0:
            raise ValueError("Cannot divide by zero")
        return [x / divisor for x in data]

print("Testing logged methods:")
processor = DataProcessor()
result = processor.process_data([1, 2, 3, 4, 5])
print(f"Result: {result}")

try:
    processor.divide_data([10, 20, 30], 0)
except ValueError:
    print("Caught expected error")

print("""
Q5: Implement a property that validates email addresses and caches the domain.
""")

import re

class User:
    def __init__(self, name: str):
        self.name = name
        self._email = None
        self._cached_domain = None
    
    @property
    def email(self):
        return self._email
    
    @email.setter
    def email(self, value):
        if not isinstance(value, str):
            raise TypeError("Email must be a string")
        
        # Validate email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, value):
            raise ValueError("Invalid email format")
        
        self._email = value
        # Cache the domain
        self._cached_domain = value.split('@')[1]
        print(f"Email set to {value}, domain cached: {self._cached_domain}")
    
    @property
    def domain(self):
        """Get cached domain from email"""
        return self._cached_domain
    
    def __repr__(self):
        return f"User(name='{self.name}', email='{self.email}')"

print("Testing email validation and domain caching:")
user = User("John Doe")
user.email = "john.doe@example.com"
print(f"User: {user}")
print(f"Domain: {user.domain}")

try:
    user.email = "invalid-email"
except ValueError as e:
    print(f"Validation error: {e}")

print("""
Q6: Implement a context manager class for database connections.
""")

class DatabaseConnection:
    """Context manager for database connections"""
    
    def __init__(self, host: str, database: str):
        self.host = host
        self.database = database
        self.connection = None
        self.transaction_active = False
    
    def __enter__(self):
        """Enter context - establish connection"""
        print(f"Connecting to {self.database} on {self.host}")
        self.connection = f"connection_to_{self.database}"
        print("Connection established")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - cleanup connection"""
        if exc_type is not None:
            print(f"Exception occurred: {exc_type.__name__}: {exc_val}")
            if self.transaction_active:
                print("Rolling back transaction")
                self.rollback()
        
        if self.connection:
            print("Closing database connection")
            self.connection = None
        
        # Return False to propagate exceptions
        return False
    
    def execute(self, query: str) -> str:
        """Execute a database query"""
        if not self.connection:
            raise RuntimeError("No active connection")
        
        print(f"Executing query: {query}")
        return f"Result of: {query}"
    
    def begin_transaction(self):
        """Begin database transaction"""
        self.transaction_active = True
        print("Transaction started")
    
    def commit(self):
        """Commit transaction"""
        if self.transaction_active:
            print("Transaction committed")
            self.transaction_active = False
    
    def rollback(self):
        """Rollback transaction"""
        if self.transaction_active:
            print("Transaction rolled back")
            self.transaction_active = False

print("Testing database context manager:")

# Successful operation
with DatabaseConnection("localhost", "mydb") as db:
    db.begin_transaction()
    result = db.execute("SELECT * FROM users")
    print(f"Query result: {result}")
    db.commit()

# Operation with error
try:
    with DatabaseConnection("localhost", "mydb") as db:
        db.begin_transaction()
        db.execute("SELECT * FROM users")
        raise ValueError("Simulated database error")
        db.commit()
except ValueError:
    print("Error handled gracefully")

# ===============================================================================
# 3. DESIGN PATTERN QUESTIONS
# ===============================================================================

print("\n" + "=" * 80)
print("3. DESIGN PATTERN QUESTIONS")
print("=" * 80)

print("""
Q7: Implement the Observer pattern for a stock price monitoring system.
""")

from abc import ABC, abstractmethod
from typing import List

class StockObserver(ABC):
    """Abstract observer for stock price changes"""
    
    @abstractmethod
    def update(self, stock_symbol: str, price: float, change: float):
        pass

class Stock:
    """Subject in observer pattern"""
    
    def __init__(self, symbol: str, price: float):
        self.symbol = symbol
        self._price = price
        self._observers: List[StockObserver] = []
    
    def attach(self, observer: StockObserver):
        """Attach an observer"""
        if observer not in self._observers:
            self._observers.append(observer)
            print(f"Observer attached to {self.symbol}")
    
    def detach(self, observer: StockObserver):
        """Detach an observer"""
        if observer in self._observers:
            self._observers.remove(observer)
            print(f"Observer detached from {self.symbol}")
    
    def notify(self, change: float):
        """Notify all observers of price change"""
        for observer in self._observers:
            observer.update(self.symbol, self._price, change)
    
    @property
    def price(self):
        return self._price
    
    @price.setter
    def price(self, new_price: float):
        old_price = self._price
        self._price = new_price
        change = new_price - old_price
        self.notify(change)

class EmailAlertObserver(StockObserver):
    """Observer that sends email alerts"""
    
    def __init__(self, email: str, threshold: float):
        self.email = email
        self.threshold = threshold
    
    def update(self, stock_symbol: str, price: float, change: float):
        if abs(change) >= self.threshold:
            print(f"📧 EMAIL to {self.email}: {stock_symbol} price changed by ${change:.2f} to ${price:.2f}")

class PortfolioObserver(StockObserver):
    """Observer that updates portfolio value"""
    
    def __init__(self, shares: int):
        self.shares = shares
        self.portfolio_value = 0
    
    def update(self, stock_symbol: str, price: float, change: float):
        self.portfolio_value = price * self.shares
        value_change = change * self.shares
        print(f"📈 PORTFOLIO: {stock_symbol} portfolio value: ${self.portfolio_value:.2f} (${value_change:+.2f})")

class TradingBotObserver(StockObserver):
    """Observer that executes trades based on price changes"""
    
    def __init__(self, buy_threshold: float, sell_threshold: float):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
    
    def update(self, stock_symbol: str, price: float, change: float):
        if change <= self.buy_threshold:
            print(f"🤖 BOT: BUY signal for {stock_symbol} at ${price:.2f} (change: ${change:.2f})")
        elif change >= self.sell_threshold:
            print(f"🤖 BOT: SELL signal for {stock_symbol} at ${price:.2f} (change: ${change:.2f})")

print("Testing Observer pattern:")

# Create stock
apple_stock = Stock("AAPL", 150.00)

# Create observers
email_alert = EmailAlertObserver("trader@example.com", 5.0)
portfolio = PortfolioObserver(100)  # 100 shares
trading_bot = TradingBotObserver(-3.0, 5.0)  # Buy if drops $3, sell if rises $5

# Attach observers
apple_stock.attach(email_alert)
apple_stock.attach(portfolio)
apple_stock.attach(trading_bot)

# Simulate price changes
print("\nPrice changes:")
apple_stock.price = 155.50  # +$5.50
apple_stock.price = 147.25  # -$8.25
apple_stock.price = 152.75  # +$5.50

print("""
Q8: Implement the Factory Method pattern for creating different types of loggers.
""")

class Logger(ABC):
    """Abstract logger base class"""
    
    @abstractmethod
    def log(self, level: str, message: str):
        pass
    
    @abstractmethod
    def close(self):
        pass

class FileLogger(Logger):
    """Logger that writes to files"""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.file_handle = None
    
    def log(self, level: str, message: str):
        if not self.file_handle:
            self.file_handle = f"file_handle_for_{self.filename}"
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        print(f"FILE LOG ({self.filename}): {log_entry}")
    
    def close(self):
        if self.file_handle:
            print(f"Closing file: {self.filename}")
            self.file_handle = None

class DatabaseLogger(Logger):
    """Logger that writes to database"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection = None
    
    def log(self, level: str, message: str):
        if not self.connection:
            self.connection = f"db_connection_{self.connection_string}"
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"DB LOG: INSERT INTO logs VALUES ('{timestamp}', '{level}', '{message}')")
    
    def close(self):
        if self.connection:
            print("Closing database connection")
            self.connection = None

class ConsoleLogger(Logger):
    """Logger that writes to console"""
    
    def __init__(self, color_enabled: bool = True):
        self.color_enabled = color_enabled
    
    def log(self, level: str, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if self.color_enabled:
            colors = {
                'INFO': '\033[94m',    # Blue
                'WARNING': '\033[93m', # Yellow
                'ERROR': '\033[91m',   # Red
                'DEBUG': '\033[92m'    # Green
            }
            reset = '\033[0m'
            color = colors.get(level, '')
            print(f"CONSOLE: {color}[{timestamp}] {level}: {message}{reset}")
        else:
            print(f"CONSOLE: [{timestamp}] {level}: {message}")
    
    def close(self):
        print("Console logger closed")

class LoggerFactory(ABC):
    """Abstract factory for creating loggers"""
    
    @abstractmethod
    def create_logger(self, config: Dict[str, Any]) -> Logger:
        pass

class FileLoggerFactory(LoggerFactory):
    """Factory for file loggers"""
    
    def create_logger(self, config: Dict[str, Any]) -> Logger:
        filename = config.get('filename', 'default.log')
        return FileLogger(filename)

class DatabaseLoggerFactory(LoggerFactory):
    """Factory for database loggers"""
    
    def create_logger(self, config: Dict[str, Any]) -> Logger:
        connection_string = config.get('connection_string', 'localhost:5432')
        return DatabaseLogger(connection_string)

class ConsoleLoggerFactory(LoggerFactory):
    """Factory for console loggers"""
    
    def create_logger(self, config: Dict[str, Any]) -> Logger:
        color_enabled = config.get('color_enabled', True)
        return ConsoleLogger(color_enabled)

class LoggerManager:
    """Manager that uses factory pattern to create loggers"""
    
    def __init__(self):
        self._factories = {
            'file': FileLoggerFactory(),
            'database': DatabaseLoggerFactory(),
            'console': ConsoleLoggerFactory()
        }
    
    def create_logger(self, logger_type: str, config: Dict[str, Any]) -> Logger:
        """Create logger using appropriate factory"""
        if logger_type not in self._factories:
            raise ValueError(f"Unknown logger type: {logger_type}")
        
        factory = self._factories[logger_type]
        return factory.create_logger(config)
    
    def register_factory(self, logger_type: str, factory: LoggerFactory):
        """Register new logger factory"""
        self._factories[logger_type] = factory

print("Testing Factory Method pattern:")

# Create logger manager
logger_manager = LoggerManager()

# Create different types of loggers
file_logger = logger_manager.create_logger('file', {'filename': 'app.log'})
db_logger = logger_manager.create_logger('database', {'connection_string': 'postgres://localhost'})
console_logger = logger_manager.create_logger('console', {'color_enabled': True})

# Use loggers
loggers = [file_logger, db_logger, console_logger]

for logger in loggers:
    logger.log('INFO', 'Application started')
    logger.log('WARNING', 'Low memory warning')
    logger.log('ERROR', 'Database connection failed')
    logger.close()
    print()

# ===============================================================================
# 4. DEBUGGING & PROBLEM-SOLVING QUESTIONS
# ===============================================================================

print("\n" + "=" * 80)
print("4. DEBUGGING & PROBLEM-SOLVING QUESTIONS")
print("=" * 80)

print("""
Q9: Find and fix the issues in this code (multiple bugs):
""")

# Buggy code example
class BuggyBankAccount:
    """Bank account with several bugs - find and fix them!"""
    
    def __init__(self, account_number, balance=0):
        self.account_number = account_number
        self.balance = balance
        self.transactions = []
    
    def deposit(self, amount):
        # Bug 1: No validation
        self.balance += amount
        self.transactions.append(f"Deposit: {amount}")
        return self.balance
    
    def withdraw(self, amount):
        # Bug 2: No overdraft protection
        self.balance -= amount
        self.transactions.append(f"Withdrawal: {amount}")
        return self.balance
    
    def get_balance(self):
        # Bug 3: Returning mutable reference
        return self.transactions
    
    def transfer(self, other_account, amount):
        # Bug 4: No error handling
        self.withdraw(amount)
        other_account.deposit(amount)
    
    def __eq__(self, other):
        # Bug 5: Improper equality comparison
        return self.balance == other.balance

print("Demonstrating bugs:")
try:
    buggy1 = BuggyBankAccount("ACC001", 1000)
    buggy2 = BuggyBankAccount("ACC002", 500)
    
    # This will show the bugs
    buggy1.deposit(-100)  # Bug 1: Negative deposit allowed
    print(f"Balance after negative deposit: {buggy1.balance}")
    
    buggy1.withdraw(2000)  # Bug 2: Overdraft allowed
    print(f"Balance after overdraft: {buggy1.balance}")
    
    transactions = buggy1.get_balance()  # Bug 3: Wrong return value
    print(f"Get balance returned: {transactions}")
    
    # Bug 4: No error handling in transfer
    # Bug 5: Wrong equality comparison
    print(f"Accounts equal? {buggy1 == buggy2}")
    
except Exception as e:
    print(f"Error: {e}")

print("\nFixed version:")

class FixedBankAccount:
    """Fixed bank account implementation"""
    
    def __init__(self, account_number: str, initial_balance: float = 0):
        if not isinstance(account_number, str) or not account_number:
            raise ValueError("Account number must be a non-empty string")
        
        if not isinstance(initial_balance, (int, float)) or initial_balance < 0:
            raise ValueError("Initial balance must be a non-negative number")
        
        self.account_number = account_number
        self._balance = float(initial_balance)  # Private attribute
        self._transactions = []
    
    def deposit(self, amount: float) -> float:
        """Deposit money with validation"""
        if not isinstance(amount, (int, float)):
            raise TypeError("Amount must be a number")
        
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        
        self._balance += amount
        self._transactions.append({
            'type': 'deposit',
            'amount': amount,
            'timestamp': datetime.now(),
            'balance_after': self._balance
        })
        
        return self._balance
    
    def withdraw(self, amount: float) -> float:
        """Withdraw money with overdraft protection"""
        if not isinstance(amount, (int, float)):
            raise TypeError("Amount must be a number")
        
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        
        self._balance -= amount
        self._transactions.append({
            'type': 'withdrawal',
            'amount': amount,
            'timestamp': datetime.now(),
            'balance_after': self._balance
        })
        
        return self._balance
    
    def get_balance(self) -> float:
        """Get current balance (returns copy, not reference)"""
        return self._balance
    
    def get_transactions(self) -> List[Dict]:
        """Get transaction history (returns copy)"""
        return copy.deepcopy(self._transactions)
    
    def transfer(self, other_account: 'FixedBankAccount', amount: float) -> bool:
        """Transfer money with proper error handling"""
        if not isinstance(other_account, FixedBankAccount):
            raise TypeError("Target must be a BankAccount instance")
        
        try:
            # Use a transaction-like approach
            self.withdraw(amount)
            try:
                other_account.deposit(amount)
                return True
            except Exception:
                # Rollback if deposit fails
                self.deposit(amount)
                raise
        except Exception as e:
            print(f"Transfer failed: {e}")
            return False
    
    def __eq__(self, other) -> bool:
        """Proper equality comparison"""
        if not isinstance(other, FixedBankAccount):
            return False
        return self.account_number == other.account_number
    
    def __str__(self) -> str:
        return f"BankAccount({self.account_number}, balance=${self._balance:.2f})"
    
    def __repr__(self) -> str:
        return f"FixedBankAccount('{self.account_number}', {self._balance})"

print("Testing fixed version:")
fixed1 = FixedBankAccount("ACC001", 1000)
fixed2 = FixedBankAccount("ACC002", 500)

# Test fixed functionality
fixed1.deposit(100)
print(f"After deposit: {fixed1}")

try:
    fixed1.deposit(-50)  # Should raise error
except ValueError as e:
    print(f"Deposit validation: {e}")

try:
    fixed1.withdraw(2000)  # Should raise error
except ValueError as e:
    print(f"Withdrawal validation: {e}")

# Test proper transfer
success = fixed1.transfer(fixed2, 200)
print(f"Transfer success: {success}")
print(f"Account 1: {fixed1}")
print(f"Account 2: {fixed2}")

# Test proper equality
fixed3 = FixedBankAccount("ACC001", 500)  # Same account number, different balance
print(f"Accounts 1 and 3 equal? {fixed1 == fixed3}")  # Should be True (same account number)

print("""
Q10: Identify the memory leak in this code and fix it:
""")

class MemoryLeakExample:
    """Class with memory leak issues"""
    
    # Class variable that accumulates references
    all_instances = []  # Memory leak: never cleaned up
    
    def __init__(self, name: str):
        self.name = name
        self.children = []
        self.parent = None
        
        # Memory leak: strong reference in class variable
        MemoryLeakExample.all_instances.append(self)
    
    def add_child(self, child: 'MemoryLeakExample'):
        """Create circular reference (memory leak)"""
        self.children.append(child)
        child.parent = self  # Circular reference!
    
    def __del__(self):
        print(f"Deleting {self.name}")

print("Demonstrating memory leak:")

# Create objects with circular references
parent = MemoryLeakExample("Parent")
child1 = MemoryLeakExample("Child1")
child2 = MemoryLeakExample("Child2")

parent.add_child(child1)
parent.add_child(child2)

print(f"Total instances tracked: {len(MemoryLeakExample.all_instances)}")

# Even when we delete references, objects won't be garbage collected
# due to circular references and class variable holding references
del parent, child1, child2

print("References deleted, but objects still in memory due to leaks")

# Fixed version
class MemoryLeakFixed:
    """Fixed version without memory leaks"""
    
    # Use WeakSet to avoid holding strong references
    _all_instances = weakref.WeakSet()
    
    def __init__(self, name: str):
        self.name = name
        self.children = []
        self._parent_ref = None  # Use weak reference for parent
        
        # Use weak reference in class tracking
        MemoryLeakFixed._all_instances.add(self)
    
    @property
    def parent(self):
        """Get parent using weak reference"""
        if self._parent_ref is not None:
            return self._parent_ref()
        return None
    
    @parent.setter
    def parent(self, value):
        """Set parent using weak reference"""
        if value is not None:
            self._parent_ref = weakref.ref(value)
        else:
            self._parent_ref = None
    
    def add_child(self, child: 'MemoryLeakFixed'):
        """Add child without creating circular reference"""
        self.children.append(child)
        child.parent = self  # Uses weak reference
    
    def remove_child(self, child: 'MemoryLeakFixed'):
        """Remove child and break reference"""
        if child in self.children:
            self.children.remove(child)
            child.parent = None
    
    @classmethod
    def get_instance_count(cls):
        """Get current number of live instances"""
        return len(cls._all_instances)
    
    def __del__(self):
        print(f"Properly deleting {self.name}")

print("\nTesting fixed version:")

# Create objects without memory leaks
parent_fixed = MemoryLeakFixed("Parent")
child1_fixed = MemoryLeakFixed("Child1")
child2_fixed = MemoryLeakFixed("Child2")

parent_fixed.add_child(child1_fixed)
parent_fixed.add_child(child2_fixed)

print(f"Instance count: {MemoryLeakFixed.get_instance_count()}")
print(f"Child1's parent: {child1_fixed.parent.name if child1_fixed.parent else None}")

# Clean deletion
del parent_fixed, child1_fixed, child2_fixed

# Force garbage collection to show proper cleanup
import gc
gc.collect()

print(f"Instance count after deletion: {MemoryLeakFixed.get_instance_count()}")

# ===============================================================================
# 5. TRICKY & EDGE CASE QUESTIONS
# ===============================================================================

print("\n" + "=" * 80)
print("5. TRICKY & EDGE CASE QUESTIONS")
print("=" * 80)

print("""
Q11: Explain the output of this code and why it behaves this way:
""")

class TrickyClass:
    class_var = []  # Mutable class variable - shared across all instances!
    
    def __init__(self, name):
        self.name = name
        self.instance_var = []
    
    def add_to_class_var(self, item):
        TrickyClass.class_var.append(item)
    
    def add_to_instance_var(self, item):
        self.instance_var.append(item)

print("Testing tricky class behavior:")

obj1 = TrickyClass("Object1")
obj2 = TrickyClass("Object2")

obj1.add_to_class_var("item1")
obj2.add_to_class_var("item2")

obj1.add_to_instance_var("instance1")
obj2.add_to_instance_var("instance2")

print(f"obj1.class_var: {obj1.class_var}")        # ['item1', 'item2'] - shared!
print(f"obj2.class_var: {obj2.class_var}")        # ['item1', 'item2'] - shared!
print(f"obj1.instance_var: {obj1.instance_var}")  # ['instance1'] - separate
print(f"obj2.instance_var: {obj2.instance_var}")  # ['instance2'] - separate

print("""
Explanation: 
- class_var is shared across ALL instances (dangerous with mutable objects)
- instance_var is unique to each instance
- Modifying class_var affects all instances
""")

print("""
Q12: What happens with method resolution in this complex inheritance?
""")

class A:
    def method(self):
        print("A.method")
        return "A"

class B:
    def method(self):
        print("B.method")
        return "B"

class C(A):
    def method(self):
        print("C.method")
        result = super().method()
        return f"C-{result}"

class D(B):
    def method(self):
        print("D.method")
        result = super().method()
        return f"D-{result}"

class E(C, D):  # Multiple inheritance
    def method(self):
        print("E.method")
        result = super().method()
        return f"E-{result}"

print("Complex inheritance method resolution:")
print(f"MRO for E: {[cls.__name__ for cls in E.__mro__]}")

e = E()
result = e.method()
print(f"Final result: {result}")

print("""
Explanation:
- MRO follows C3 linearization: E -> C -> D -> B -> A -> object
- super() follows the MRO, not the inheritance tree
- This is why B.method is called, not A.method (even though C inherits from A)
""")

print("""
Q13: Demonstrate the difference between shallow and deep copy with nested objects:
""")

class Address:
    def __init__(self, street, city):
        self.street = street
        self.city = city
    
    def __repr__(self):
        return f"Address('{self.street}', '{self.city}')"

class Person:
    def __init__(self, name, address):
        self.name = name
        self.address = address
        self.hobbies = []
    
    def __repr__(self):
        return f"Person('{self.name}', {self.address}, {self.hobbies})"

print("Testing shallow vs deep copy:")

# Create original object
original_address = Address("123 Main St", "New York")
original_person = Person("John", original_address)
original_person.hobbies.append("reading")

# Shallow copy
import copy
shallow_copy = copy.copy(original_person)
shallow_copy.name = "Jane"  # This won't affect original
shallow_copy.address.city = "Boston"  # This WILL affect original!
shallow_copy.hobbies.append("swimming")  # This WILL affect original!

print("After shallow copy modifications:")
print(f"Original: {original_person}")
print(f"Shallow:  {shallow_copy}")
print(f"Same address object? {original_person.address is shallow_copy.address}")

# Deep copy
deep_copy = copy.deepcopy(original_person)
deep_copy.name = "Bob"
deep_copy.address.city = "Chicago"  # This won't affect original
deep_copy.hobbies.append("cycling")  # This won't affect original

print("\nAfter deep copy modifications:")
print(f"Original: {original_person}")
print(f"Deep:     {deep_copy}")
print(f"Same address object? {original_person.address is deep_copy.address}")

print("\n" + "=" * 80)
print("OOP INTERVIEW QUESTIONS PART 1 COMPLETE!")
print("=" * 80)
print("""
🎯 TOPICS COVERED:
✅ Conceptual Questions (Theory & Understanding)
✅ Practical Coding Questions
✅ Design Pattern Questions
✅ Debugging & Problem-Solving
✅ Tricky & Edge Case Questions

📝 KEY AREAS TESTED:
- Inheritance vs Composition
- Method Resolution Order (MRO)
- Constructor patterns (__new__ vs __init__)
- Decorators and context managers
- Observer and Factory patterns
- Memory management and circular references
- Class vs instance variables
- Shallow vs deep copying

🚀 NEXT: Continue with more advanced interview questions and architecture patterns
""")
