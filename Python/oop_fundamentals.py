# ===============================================================================
# PYTHON OOP FUNDAMENTALS - CLASSES & OBJECTS
# Real-Life Examples & Complete Mastery Guide
# ===============================================================================

"""
COMPREHENSIVE OOP FUNDAMENTALS COVERAGE:
======================================
1. Class & Object Basics
2. Attributes & Methods
3. Instance vs Class Variables
4. Constructor (__init__) & Destructor (__del__)
5. Method Types (Instance, Class, Static)
6. Properties & Descriptors
7. String Representation Methods
8. Advanced Class Features
9. Real-World Applications
10. Best Practices & Common Pitfalls
"""

import time
import weakref
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging

# ===============================================================================
# 1. CLASS & OBJECT BASICS
# ===============================================================================

print("=" * 80)
print("1. CLASS & OBJECT BASICS")
print("=" * 80)

print("\n--- Real-Life Example: Car Management System ---")

class Car:
    """
    Real-life example: Represents a car in a dealership management system
    """
    
    def __init__(self, make: str, model: str, year: int, price: float):
        """Initialize a new car"""
        self.make = make
        self.model = model
        self.year = year
        self.price = price
        self.mileage = 0
        self.is_running = False
        self.fuel_level = 100  # Percentage
    
    def start_engine(self):
        """Start the car engine"""
        if not self.is_running:
            self.is_running = True
            print(f"{self.make} {self.model} engine started!")
        else:
            print(f"{self.make} {self.model} is already running!")
    
    def stop_engine(self):
        """Stop the car engine"""
        if self.is_running:
            self.is_running = False
            print(f"{self.make} {self.model} engine stopped!")
        else:
            print(f"{self.make} {self.model} is already stopped!")
    
    def drive(self, distance: float):
        """Drive the car for a given distance"""
        if not self.is_running:
            print("Cannot drive! Please start the engine first.")
            return
        
        if self.fuel_level <= 0:
            print("Cannot drive! No fuel remaining.")
            return
        
        # Calculate fuel consumption (simplified)
        fuel_consumed = distance * 0.1  # 10 miles per unit fuel
        
        if fuel_consumed > self.fuel_level:
            max_distance = self.fuel_level / 0.1
            print(f"Not enough fuel! Can only drive {max_distance:.1f} miles.")
            return
        
        self.mileage += distance
        self.fuel_level -= fuel_consumed
        print(f"Drove {distance} miles. Total mileage: {self.mileage}, Fuel: {self.fuel_level:.1f}%")
    
    def refuel(self, amount: float = None):
        """Refuel the car"""
        if amount is None:
            # Fill up completely
            fuel_added = 100 - self.fuel_level
            self.fuel_level = 100
        else:
            fuel_added = min(amount, 100 - self.fuel_level)
            self.fuel_level += fuel_added
        
        print(f"Added {fuel_added:.1f}% fuel. Current fuel level: {self.fuel_level:.1f}%")
    
    def get_info(self):
        """Get car information"""
        return {
            'make': self.make,
            'model': self.model,
            'year': self.year,
            'price': self.price,
            'mileage': self.mileage,
            'fuel_level': self.fuel_level,
            'is_running': self.is_running
        }

# Example usage
print("Creating car objects:")
car1 = Car("Toyota", "Camry", 2023, 28000)
car2 = Car("Tesla", "Model 3", 2023, 45000)

print(f"Car 1: {car1.make} {car1.model}")
print(f"Car 2: {car2.make} {car2.model}")

# Demonstrate object behavior
car1.start_engine()
car1.drive(50)
car1.refuel()
car1.stop_engine()

print(f"Car 1 info: {car1.get_info()}")

# ===============================================================================
# 2. ATTRIBUTES & METHODS
# ===============================================================================

print("\n" + "=" * 80)
print("2. ATTRIBUTES & METHODS")
print("=" * 80)

print("\n--- Real-Life Example: Bank Account System ---")

class BankAccount:
    """
    Real-life example: Bank account management system
    Demonstrates different types of attributes and methods
    """
    
    # Class variable (shared by all instances)
    bank_name = "Python National Bank"
    interest_rate = 0.02  # 2% annual interest
    
    def __init__(self, account_holder: str, initial_balance: float = 0):
        # Instance variables (unique to each instance)
        self.account_holder = account_holder
        self.account_number = self._generate_account_number()
        self.balance = initial_balance
        self.transaction_history = []
        self.created_date = datetime.now()
        self.is_active = True
        
        # Private attributes (convention: start with underscore)
        self._pin = None
        self.__security_code = self._generate_security_code()  # Name mangling
        
        self._add_transaction("Account created", initial_balance)
    
    def _generate_account_number(self):
        """Private method to generate account number"""
        import random
        return f"ACC{random.randint(100000, 999999)}"
    
    def _generate_security_code(self):
        """Private method to generate security code"""
        import random
        return random.randint(1000, 9999)
    
    def _add_transaction(self, description: str, amount: float):
        """Private method to add transaction to history"""
        transaction = {
            'date': datetime.now(),
            'description': description,
            'amount': amount,
            'balance': self.balance
        }
        self.transaction_history.append(transaction)
    
    def deposit(self, amount: float):
        """Public method to deposit money"""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        
        if not self.is_active:
            raise Exception("Account is inactive")
        
        self.balance += amount
        self._add_transaction(f"Deposit", amount)
        print(f"Deposited ${amount:.2f}. New balance: ${self.balance:.2f}")
    
    def withdraw(self, amount: float):
        """Public method to withdraw money"""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        
        if not self.is_active:
            raise Exception("Account is inactive")
        
        if amount > self.balance:
            raise Exception("Insufficient funds")
        
        self.balance -= amount
        self._add_transaction(f"Withdrawal", -amount)
        print(f"Withdrew ${amount:.2f}. New balance: ${self.balance:.2f}")
    
    def transfer(self, other_account, amount: float):
        """Transfer money to another account"""
        if not isinstance(other_account, BankAccount):
            raise TypeError("Can only transfer to BankAccount instance")
        
        # Withdraw from this account
        self.withdraw(amount)
        # Deposit to other account
        other_account.deposit(amount)
        
        print(f"Transferred ${amount:.2f} to {other_account.account_holder}")
    
    def apply_interest(self):
        """Apply annual interest to the account"""
        if self.balance > 0:
            interest = self.balance * self.interest_rate
            self.balance += interest
            self._add_transaction("Interest applied", interest)
            print(f"Interest applied: ${interest:.2f}. New balance: ${self.balance:.2f}")
    
    def set_pin(self, pin: str):
        """Set PIN for the account"""
        if len(pin) != 4 or not pin.isdigit():
            raise ValueError("PIN must be 4 digits")
        self._pin = pin
        print("PIN set successfully")
    
    def verify_pin(self, pin: str):
        """Verify PIN"""
        return self._pin == pin
    
    def get_account_summary(self):
        """Get account summary"""
        return {
            'account_holder': self.account_holder,
            'account_number': self.account_number,
            'balance': self.balance,
            'bank_name': self.bank_name,
            'created_date': self.created_date,
            'is_active': self.is_active,
            'total_transactions': len(self.transaction_history)
        }
    
    def get_transaction_history(self, limit: int = 10):
        """Get recent transaction history"""
        return self.transaction_history[-limit:]

# Example usage
print("Creating bank accounts:")
account1 = BankAccount("Alice Johnson", 1000)
account2 = BankAccount("Bob Smith", 500)

print(f"Account 1: {account1.account_holder} - {account1.account_number}")
print(f"Account 2: {account2.account_holder} - {account2.account_number}")

# Demonstrate methods
account1.deposit(500)
account1.withdraw(200)
account1.transfer(account2, 100)
account1.apply_interest()

print(f"Account 1 summary: {account1.get_account_summary()}")

# ===============================================================================
# 3. INSTANCE VS CLASS VARIABLES
# ===============================================================================

print("\n" + "=" * 80)
print("3. INSTANCE VS CLASS VARIABLES")
print("=" * 80)

print("\n--- Real-Life Example: Employee Management System ---")

class Employee:
    """
    Real-life example: Employee management system
    Demonstrates instance vs class variables
    """
    
    # Class variables (shared by all instances)
    company_name = "TechCorp Inc."
    total_employees = 0
    base_salary = 50000
    departments = ["Engineering", "Sales", "Marketing", "HR", "Finance"]
    
    def __init__(self, name: str, department: str, position: str):
        # Instance variables (unique to each instance)
        self.name = name
        self.department = department
        self.position = position
        self.employee_id = self._generate_employee_id()
        self.hire_date = datetime.now()
        self.salary = self.base_salary  # Starts with base salary
        self.performance_score = 0
        self.is_active = True
        
        # Increment total employees (class variable)
        Employee.total_employees += 1
    
    def _generate_employee_id(self):
        """Generate unique employee ID"""
        import random
        return f"EMP{Employee.total_employees + 1:04d}"
    
    def set_salary(self, new_salary: float):
        """Set employee salary"""
        if new_salary < 0:
            raise ValueError("Salary cannot be negative")
        
        old_salary = self.salary
        self.salary = new_salary
        print(f"{self.name}'s salary updated from ${old_salary:,.2f} to ${new_salary:,.2f}")
    
    def give_raise(self, percentage: float):
        """Give percentage raise"""
        if percentage < 0:
            raise ValueError("Raise percentage cannot be negative")
        
        raise_amount = self.salary * (percentage / 100)
        new_salary = self.salary + raise_amount
        self.set_salary(new_salary)
        print(f"{self.name} received a {percentage}% raise (${raise_amount:,.2f})")
    
    def update_performance(self, score: float):
        """Update performance score (0-100)"""
        if not 0 <= score <= 100:
            raise ValueError("Performance score must be between 0 and 100")
        
        self.performance_score = score
        print(f"{self.name}'s performance score updated to {score}")
        
        # Automatic raise for high performers
        if score >= 90:
            self.give_raise(10)  # 10% raise for excellent performance
        elif score >= 80:
            self.give_raise(5)   # 5% raise for good performance
    
    @classmethod
    def get_company_stats(cls):
        """Get company statistics (class method)"""
        return {
            'company_name': cls.company_name,
            'total_employees': cls.total_employees,
            'base_salary': cls.base_salary,
            'departments': cls.departments
        }
    
    @classmethod
    def set_base_salary(cls, new_base_salary: float):
        """Set new base salary for all future employees (class method)"""
        if new_base_salary < 0:
            raise ValueError("Base salary cannot be negative")
        
        old_base = cls.base_salary
        cls.base_salary = new_base_salary
        print(f"Base salary updated from ${old_base:,.2f} to ${new_base_salary:,.2f}")
    
    @staticmethod
    def calculate_annual_tax(salary: float, tax_rate: float = 0.25):
        """Calculate annual tax (static method - doesn't access class/instance data)"""
        if salary < 0 or tax_rate < 0:
            raise ValueError("Salary and tax rate must be non-negative")
        
        return salary * tax_rate
    
    def get_employee_info(self):
        """Get comprehensive employee information"""
        return {
            'name': self.name,
            'employee_id': self.employee_id,
            'department': self.department,
            'position': self.position,
            'salary': self.salary,
            'hire_date': self.hire_date,
            'performance_score': self.performance_score,
            'is_active': self.is_active,
            'company': self.company_name,
            'annual_tax': self.calculate_annual_tax(self.salary)
        }
    
    @classmethod
    def create_intern(cls, name: str, department: str):
        """Factory method to create intern with specific settings"""
        intern = cls(name, department, "Intern")
        intern.set_salary(30000)  # Lower salary for interns
        return intern
    
    def __del__(self):
        """Destructor - called when object is deleted"""
        if self.is_active:
            Employee.total_employees -= 1
            print(f"Employee {self.name} removed from system")

# Example usage
print("Creating employees:")

# Regular employees
emp1 = Employee("Alice Cooper", "Engineering", "Software Engineer")
emp2 = Employee("Bob Wilson", "Sales", "Sales Manager")
emp3 = Employee("Carol Brown", "Marketing", "Marketing Specialist")

print(f"Total employees: {Employee.total_employees}")

# Using class methods
print(f"Company stats: {Employee.get_company_stats()}")

# Create intern using factory method
intern = Employee.create_intern("David Kim", "Engineering")

# Demonstrate instance vs class variables
print(f"\nEmployee 1 company: {emp1.company_name}")
print(f"Employee 2 company: {emp2.company_name}")

# Change class variable - affects all instances
Employee.company_name = "TechCorp Global"
print(f"After company name change:")
print(f"Employee 1 company: {emp1.company_name}")
print(f"Employee 2 company: {emp2.company_name}")

# Demonstrate performance and salary updates
emp1.update_performance(95)  # Should trigger automatic raise
emp2.update_performance(85)  # Should trigger smaller raise

# Static method usage
tax_owed = Employee.calculate_annual_tax(75000, 0.28)
print(f"Annual tax for $75,000 salary: ${tax_owed:,.2f}")

print(f"Final employee info for {emp1.name}:")
print(emp1.get_employee_info())

# ===============================================================================
# 4. CONSTRUCTOR (__init__) & DESTRUCTOR (__del__)
# ===============================================================================

print("\n" + "=" * 80)
print("4. CONSTRUCTOR & DESTRUCTOR")
print("=" * 80)

print("\n--- Real-Life Example: Database Connection Manager ---")

class DatabaseConnection:
    """
    Real-life example: Database connection management
    Demonstrates constructor and destructor usage
    """
    
    # Class variable to track active connections
    active_connections = 0
    max_connections = 10
    
    def __init__(self, host: str, database: str, username: str, password: str):
        """Constructor - establish database connection"""
        
        # Check connection limit
        if DatabaseConnection.active_connections >= DatabaseConnection.max_connections:
            raise Exception(f"Maximum connections ({DatabaseConnection.max_connections}) reached")
        
        self.host = host
        self.database = database
        self.username = username
        self._password = password  # Private attribute
        self.connection_id = self._generate_connection_id()
        self.connected_at = datetime.now()
        self.is_connected = False
        self.query_count = 0
        
        # Simulate connection establishment
        self._establish_connection()
        
        # Update class variable
        DatabaseConnection.active_connections += 1
        
        print(f"Database connection established: {self.connection_id}")
        print(f"Active connections: {DatabaseConnection.active_connections}")
    
    def _generate_connection_id(self):
        """Generate unique connection ID"""
        import random
        return f"CONN_{random.randint(1000, 9999)}_{int(time.time())}"
    
    def _establish_connection(self):
        """Simulate establishing database connection"""
        # In real implementation, this would connect to actual database
        time.sleep(0.1)  # Simulate connection time
        self.is_connected = True
        print(f"Connected to {self.database} on {self.host}")
    
    def execute_query(self, query: str):
        """Execute a database query"""
        if not self.is_connected:
            raise Exception("Not connected to database")
        
        # Simulate query execution
        print(f"Executing query: {query[:50]}{'...' if len(query) > 50 else ''}")
        time.sleep(0.05)  # Simulate query time
        
        self.query_count += 1
        return f"Query executed successfully. Total queries: {self.query_count}"
    
    def close_connection(self):
        """Manually close the connection"""
        if self.is_connected:
            self.is_connected = False
            DatabaseConnection.active_connections -= 1
            print(f"Connection {self.connection_id} closed manually")
            print(f"Active connections: {DatabaseConnection.active_connections}")
    
    def get_connection_info(self):
        """Get connection information"""
        return {
            'connection_id': self.connection_id,
            'host': self.host,
            'database': self.database,
            'username': self.username,
            'connected_at': self.connected_at,
            'is_connected': self.is_connected,
            'query_count': self.query_count,
            'duration': datetime.now() - self.connected_at
        }
    
    def __del__(self):
        """Destructor - automatically close connection when object is deleted"""
        if hasattr(self, 'is_connected') and self.is_connected:
            # Only decrement if connection was successfully established
            DatabaseConnection.active_connections -= 1
            print(f"Destructor called: Connection {self.connection_id} closed automatically")
            print(f"Active connections: {DatabaseConnection.active_connections}")

# Example usage with proper resource management
print("Creating database connections:")

try:
    # Create connections
    db1 = DatabaseConnection("localhost", "users_db", "admin", "password123")
    db2 = DatabaseConnection("server1.com", "products_db", "user1", "secret456")
    
    # Use connections
    db1.execute_query("SELECT * FROM users WHERE active = 1")
    db2.execute_query("INSERT INTO products (name, price) VALUES ('Laptop', 999.99)")
    
    print(f"DB1 info: {db1.get_connection_info()}")
    
    # Manually close one connection
    db1.close_connection()
    
    # The other connection will be closed automatically by destructor
    
except Exception as e:
    print(f"Error: {e}")

print(f"Final active connections: {DatabaseConnection.active_connections}")

# ===============================================================================
# 5. METHOD TYPES (INSTANCE, CLASS, STATIC)
# ===============================================================================

print("\n" + "=" * 80)
print("5. METHOD TYPES")
print("=" * 80)

print("\n--- Real-Life Example: Shopping Cart System ---")

class ShoppingCart:
    """
    Real-life example: E-commerce shopping cart
    Demonstrates instance, class, and static methods
    """
    
    # Class variables
    tax_rate = 0.08  # 8% tax
    total_carts_created = 0
    active_carts = set()
    
    def __init__(self, customer_id: str):
        """Instance method - constructor"""
        self.customer_id = customer_id
        self.cart_id = self._generate_cart_id()
        self.items = []  # List of {'product': str, 'price': float, 'quantity': int}
        self.created_at = datetime.now()
        self.is_active = True
        
        # Update class variables
        ShoppingCart.total_carts_created += 1
        ShoppingCart.active_carts.add(self.cart_id)
        
        print(f"Shopping cart created for customer {customer_id}: {self.cart_id}")
    
    def _generate_cart_id(self):
        """Private instance method"""
        import random
        return f"CART_{self.customer_id}_{random.randint(1000, 9999)}"
    
    # INSTANCE METHODS (work with specific instance data)
    def add_item(self, product: str, price: float, quantity: int = 1):
        """Instance method - add item to this specific cart"""
        if not self.is_active:
            raise Exception("Cart is not active")
        
        if price < 0 or quantity < 1:
            raise ValueError("Price must be non-negative and quantity must be positive")
        
        # Check if item already exists in cart
        for item in self.items:
            if item['product'] == product:
                item['quantity'] += quantity
                print(f"Updated {product} quantity to {item['quantity']}")
                return
        
        # Add new item
        item = {'product': product, 'price': price, 'quantity': quantity}
        self.items.append(item)
        print(f"Added {quantity} x {product} @ ${price:.2f} to cart")
    
    def remove_item(self, product: str, quantity: int = None):
        """Instance method - remove item from this specific cart"""
        if not self.is_active:
            raise Exception("Cart is not active")
        
        for i, item in enumerate(self.items):
            if item['product'] == product:
                if quantity is None or quantity >= item['quantity']:
                    # Remove entire item
                    removed_item = self.items.pop(i)
                    print(f"Removed all {removed_item['product']} from cart")
                else:
                    # Reduce quantity
                    item['quantity'] -= quantity
                    print(f"Reduced {product} quantity by {quantity}, remaining: {item['quantity']}")
                return
        
        print(f"Product {product} not found in cart")
    
    def get_subtotal(self):
        """Instance method - calculate subtotal for this cart"""
        return sum(item['price'] * item['quantity'] for item in self.items)
    
    def get_tax_amount(self):
        """Instance method - calculate tax for this cart"""
        return self.get_subtotal() * self.tax_rate
    
    def get_total(self):
        """Instance method - calculate total for this cart"""
        return self.get_subtotal() + self.get_tax_amount()
    
    def checkout(self):
        """Instance method - process checkout for this cart"""
        if not self.items:
            raise Exception("Cannot checkout empty cart")
        
        total = self.get_total()
        print(f"Processing checkout for cart {self.cart_id}")
        print(f"Subtotal: ${self.get_subtotal():.2f}")
        print(f"Tax: ${self.get_tax_amount():.2f}")
        print(f"Total: ${total:.2f}")
        
        # Deactivate cart after checkout
        self.is_active = False
        ShoppingCart.active_carts.discard(self.cart_id)
        
        return {
            'cart_id': self.cart_id,
            'customer_id': self.customer_id,
            'items': self.items.copy(),
            'subtotal': self.get_subtotal(),
            'tax': self.get_tax_amount(),
            'total': total,
            'checkout_time': datetime.now()
        }
    
    # CLASS METHODS (work with class data, can create instances)
    @classmethod
    def get_cart_statistics(cls):
        """Class method - get statistics about all carts"""
        return {
            'total_carts_created': cls.total_carts_created,
            'active_carts_count': len(cls.active_carts),
            'active_cart_ids': list(cls.active_carts),
            'current_tax_rate': cls.tax_rate
        }
    
    @classmethod
    def set_tax_rate(cls, new_rate: float):
        """Class method - update tax rate for all carts"""
        if not 0 <= new_rate <= 1:
            raise ValueError("Tax rate must be between 0 and 1")
        
        old_rate = cls.tax_rate
        cls.tax_rate = new_rate
        print(f"Tax rate updated from {old_rate:.2%} to {new_rate:.2%}")
    
    @classmethod
    def create_guest_cart(cls):
        """Class method - factory method for guest checkout"""
        import random
        guest_id = f"GUEST_{random.randint(10000, 99999)}"
        return cls(guest_id)
    
    @classmethod
    def cleanup_inactive_carts(cls):
        """Class method - remove inactive carts from tracking"""
        # In real implementation, this might clean up old abandoned carts
        initial_count = len(cls.active_carts)
        # For demo, we'll just report current state
        print(f"Cleanup complete. Active carts: {len(cls.active_carts)}")
        return initial_count - len(cls.active_carts)
    
    # STATIC METHODS (utility functions, don't access class/instance data)
    @staticmethod
    def calculate_shipping_cost(weight: float, distance: float):
        """Static method - calculate shipping cost based on weight and distance"""
        if weight <= 0 or distance <= 0:
            raise ValueError("Weight and distance must be positive")
        
        base_cost = 5.0
        weight_cost = weight * 0.5  # $0.50 per lb
        distance_cost = distance * 0.1  # $0.10 per mile
        
        return base_cost + weight_cost + distance_cost
    
    @staticmethod
    def validate_product_code(product_code: str):
        """Static method - validate product code format"""
        import re
        # Example: Product code should be like "PROD-12345"
        pattern = r'^PROD-\d{5}$'
        return bool(re.match(pattern, product_code))
    
    @staticmethod
    def format_currency(amount: float, currency: str = "USD"):
        """Static method - format currency for display"""
        if currency == "USD":
            return f"${amount:.2f}"
        elif currency == "EUR":
            return f"€{amount:.2f}"
        elif currency == "GBP":
            return f"£{amount:.2f}"
        else:
            return f"{amount:.2f} {currency}"
    
    def get_cart_summary(self):
        """Instance method - get comprehensive cart summary"""
        return {
            'cart_id': self.cart_id,
            'customer_id': self.customer_id,
            'items': self.items.copy(),
            'item_count': len(self.items),
            'total_quantity': sum(item['quantity'] for item in self.items),
            'subtotal': self.get_subtotal(),
            'tax_amount': self.get_tax_amount(),
            'total': self.get_total(),
            'is_active': self.is_active,
            'created_at': self.created_at
        }

# Example usage demonstrating all method types
print("Creating shopping carts:")

# Create carts using constructor (instance method)
cart1 = ShoppingCart("CUST001")
cart2 = ShoppingCart("CUST002")

# Create guest cart using class method
guest_cart = ShoppingCart.create_guest_cart()

# Use instance methods
cart1.add_item("Laptop", 999.99, 1)
cart1.add_item("Mouse", 29.99, 2)
cart1.add_item("Keyboard", 79.99, 1)

cart2.add_item("Phone", 699.99, 1)
cart2.add_item("Case", 24.99, 1)

print(f"\nCart1 summary: {cart1.get_cart_summary()}")

# Use static methods
shipping_cost = ShoppingCart.calculate_shipping_cost(5.0, 100)
print(f"Shipping cost for 5 lbs, 100 miles: {ShoppingCart.format_currency(shipping_cost)}")

is_valid_code = ShoppingCart.validate_product_code("PROD-12345")
print(f"Product code PROD-12345 is valid: {is_valid_code}")

# Use class methods
print(f"\nCart statistics: {ShoppingCart.get_cart_statistics()}")

# Change tax rate (affects all carts)
ShoppingCart.set_tax_rate(0.10)  # 10% tax

print(f"Cart1 total after tax change: {ShoppingCart.format_currency(cart1.get_total())}")

# Checkout cart1
checkout_receipt = cart1.checkout()
print(f"\nCheckout receipt: {checkout_receipt}")

print(f"Final cart statistics: {ShoppingCart.get_cart_statistics()}")

print("\n" + "=" * 80)
print("OOP FUNDAMENTALS PART 1 COMPLETE!")
print("=" * 80)
print("""
🎯 CONCEPTS COVERED:
✅ Class & Object Basics with Real-Life Examples
✅ Attributes & Methods (Public, Private, Protected)
✅ Instance vs Class Variables
✅ Constructor (__init__) & Destructor (__del__)
✅ Method Types (Instance, Class, Static)

📝 REAL-LIFE EXAMPLES:
- Car Management System
- Bank Account System  
- Employee Management System
- Database Connection Manager
- Shopping Cart System

🚀 NEXT: Continue with inheritance.py for OOP Pillar 1 (Inheritance)
""")
