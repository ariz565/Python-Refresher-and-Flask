# ===============================================================================
# PYTHON OOP PILLAR 2: ENCAPSULATION
# Real-Life Examples & Complete Mastery Guide
# ===============================================================================

"""
COMPREHENSIVE ENCAPSULATION COVERAGE:
====================================
1. Data Hiding & Access Modifiers
2. Private, Protected, and Public Members
3. Property Decorators (@property)
4. Getters and Setters
5. Computed Properties
6. Property Validation and Type Checking
7. Read-Only and Write-Only Properties
8. Class-Level Properties and Descriptors
9. Advanced Property Patterns
10. Best Practices & Security Considerations
"""

import re
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union, Callable
from functools import wraps
import weakref
import threading
from enum import Enum

# ===============================================================================
# 1. DATA HIDING & ACCESS MODIFIERS
# ===============================================================================

print("=" * 80)
print("1. DATA HIDING & ACCESS MODIFIERS")
print("=" * 80)

print("\n--- Real-Life Example: Bank Account System ---")

class BankAccount:
    """Bank account demonstrating access control levels"""
    
    # Class variable (public)
    bank_name = "SecureBank"
    
    # Class variable (protected)
    _interest_rates = {
        'savings': 0.02,
        'checking': 0.001,
        'premium': 0.035
    }
    
    # Class variable (private)
    __regulatory_code = "FDIC-2024-001"
    
    def __init__(self, account_number: str, account_type: str, initial_balance: float = 0.0):
        # Public attributes
        self.account_number = account_number
        self.account_type = account_type
        self.creation_date = datetime.now()
        
        # Protected attributes (single underscore)
        self._account_holder = None
        self._credit_limit = 0.0
        self._last_transaction_date = None
        
        # Private attributes (double underscore)
        self.__balance = initial_balance
        self.__pin = None
        self.__transaction_history = []
        self.__is_frozen = False
        self.__failed_attempts = 0
    
    # Public methods
    def get_account_info(self):
        """Get basic account information (public data only)"""
        return {
            'account_number': self.account_number,
            'account_type': self.account_type,
            'bank_name': self.bank_name,
            'creation_date': self.creation_date,
            'is_active': not self.__is_frozen
        }
    
    def check_balance_public(self, pin: str):
        """Public method to check balance with PIN verification"""
        if not self._verify_pin(pin):
            return "Access denied: Invalid PIN"
        
        if self.__is_frozen:
            return "Account is frozen"
        
        return f"Balance: ${self.__balance:.2f}"
    
    # Protected methods (single underscore)
    def _verify_pin(self, entered_pin: str) -> bool:
        """Protected method for PIN verification"""
        if self.__pin is None:
            return False
        
        if entered_pin == self.__pin:
            self.__failed_attempts = 0
            return True
        else:
            self.__failed_attempts += 1
            if self.__failed_attempts >= 3:
                self.__freeze_account()
            return False
    
    def _calculate_interest(self):
        """Protected method to calculate interest"""
        if self.account_type in self._interest_rates:
            return self.__balance * self._interest_rates[self.account_type]
        return 0.0
    
    def _log_transaction(self, transaction_type: str, amount: float, description: str = ""):
        """Protected method to log transactions"""
        transaction = {
            'timestamp': datetime.now(),
            'type': transaction_type,
            'amount': amount,
            'balance_after': self.__balance,
            'description': description
        }
        self.__transaction_history.append(transaction)
        self._last_transaction_date = datetime.now()
    
    # Private methods (double underscore)
    def __freeze_account(self):
        """Private method to freeze account (internal security)"""
        self.__is_frozen = True
        self._log_transaction("SECURITY", 0, "Account frozen due to failed PIN attempts")
    
    def __validate_amount(self, amount: float) -> bool:
        """Private method to validate transaction amounts"""
        if not isinstance(amount, (int, float)):
            return False
        if amount <= 0:
            return False
        if amount > 10000:  # Daily limit
            return False
        return True
    
    def __encrypt_data(self, data: str) -> str:
        """Private method for data encryption"""
        # Simple encryption for demo (don't use in production)
        return hashlib.sha256(data.encode()).hexdigest()
    
    # Public interface methods
    def set_pin(self, new_pin: str) -> bool:
        """Set account PIN"""
        if len(new_pin) != 4 or not new_pin.isdigit():
            return False
        
        self.__pin = self.__encrypt_data(new_pin)
        return True
    
    def deposit(self, amount: float, pin: str) -> str:
        """Deposit money to account"""
        if not self._verify_pin(pin):
            return "Access denied: Invalid PIN"
        
        if self.__is_frozen:
            return "Account is frozen"
        
        if not self.__validate_amount(amount):
            return "Invalid amount"
        
        self.__balance += amount
        self._log_transaction("DEPOSIT", amount)
        return f"Deposited ${amount:.2f}. New balance: ${self.__balance:.2f}"
    
    def withdraw(self, amount: float, pin: str) -> str:
        """Withdraw money from account"""
        if not self._verify_pin(pin):
            return "Access denied: Invalid PIN"
        
        if self.__is_frozen:
            return "Account is frozen"
        
        if not self.__validate_amount(amount):
            return "Invalid amount"
        
        if amount > self.__balance + self._credit_limit:
            return "Insufficient funds"
        
        self.__balance -= amount
        self._log_transaction("WITHDRAWAL", amount)
        return f"Withdrew ${amount:.2f}. New balance: ${self.__balance:.2f}"
    
    # Demonstration of name mangling
    def demonstrate_name_mangling(self):
        """Show how name mangling works"""
        print(f"Accessing private __balance via name mangling: {self._BankAccount__balance}")
        print(f"Private __pin exists: {hasattr(self, '_BankAccount__pin')}")
        print(f"Failed attempts: {self._BankAccount__failed_attempts}")

# Example usage - Access Modifiers
print("Creating bank account with access control:")

account = BankAccount("ACC001", "savings", 1000.0)
account.set_pin("1234")

# Public access
print(f"Account info: {account.get_account_info()}")
print(f"Bank name (public): {account.bank_name}")

# Accessing protected members (possible but not recommended)
print(f"Interest rates (protected): {account._interest_rates}")
print(f"Credit limit (protected): {account._credit_limit}")

# Accessing private members (name mangled)
try:
    # This will raise AttributeError
    print(account.__balance)
except AttributeError as e:
    print(f"Cannot access __balance directly: {e}")

# Accessing via name mangling (possible but should never be done)
print(f"Balance via name mangling: {account._BankAccount__balance}")

# Proper access through public interface
print(account.check_balance_public("1234"))
print(account.deposit(500, "1234"))
print(account.withdraw(200, "1234"))

# Demonstrate name mangling
account.demonstrate_name_mangling()

# ===============================================================================
# 2. PROPERTY DECORATORS (@property)
# ===============================================================================

print("\n" + "=" * 80)
print("2. PROPERTY DECORATORS (@property)")
print("=" * 80)

print("\n--- Real-Life Example: Employee Management System ---")

class Employee:
    """Employee class demonstrating property decorators"""
    
    def __init__(self, first_name: str, last_name: str, employee_id: str, salary: float):
        self._first_name = first_name
        self._last_name = last_name
        self._employee_id = employee_id
        self._salary = salary
        self._department = None
        self._hire_date = datetime.now()
        self._performance_rating = 3.0  # Scale of 1-5
        self._is_active = True
        self._vacation_days = 20
        self._email_domain = "company.com"
    
    # Read-only property
    @property
    def full_name(self) -> str:
        """Full name as read-only computed property"""
        return f"{self._first_name} {self._last_name}"
    
    # Read-only property with computation
    @property
    def email(self) -> str:
        """Auto-generated email address"""
        first_clean = re.sub(r'[^a-zA-Z]', '', self._first_name.lower())
        last_clean = re.sub(r'[^a-zA-Z]', '', self._last_name.lower())
        return f"{first_clean}.{last_clean}@{self._email_domain}"
    
    # Read-only property with complex computation
    @property
    def years_of_service(self) -> float:
        """Calculate years of service"""
        delta = datetime.now() - self._hire_date
        return round(delta.days / 365.25, 2)
    
    # Property with getter and setter
    @property
    def salary(self) -> float:
        """Salary with validation"""
        return self._salary
    
    @salary.setter
    def salary(self, value: float):
        """Set salary with validation"""
        if not isinstance(value, (int, float)):
            raise TypeError("Salary must be a number")
        
        if value < 0:
            raise ValueError("Salary cannot be negative")
        
        if value > 1000000:
            raise ValueError("Salary exceeds maximum limit")
        
        # Log salary change
        old_salary = self._salary
        self._salary = float(value)
        print(f"Salary updated from ${old_salary:,.2f} to ${self._salary:,.2f}")
    
    # Property with getter, setter, and deleter
    @property
    def department(self) -> Optional[str]:
        """Department with full lifecycle management"""
        return self._department
    
    @department.setter
    def department(self, value: Optional[str]):
        """Set department with validation"""
        if value is not None and not isinstance(value, str):
            raise TypeError("Department must be a string")
        
        if value is not None and len(value.strip()) == 0:
            raise ValueError("Department cannot be empty")
        
        old_dept = self._department
        self._department = value.strip() if value else None
        print(f"Department changed from '{old_dept}' to '{self._department}'")
    
    @department.deleter
    def department(self):
        """Remove employee from department"""
        old_dept = self._department
        self._department = None
        print(f"Employee removed from department '{old_dept}'")
    
    # Property with complex validation
    @property
    def performance_rating(self) -> float:
        """Performance rating (1.0 - 5.0)"""
        return self._performance_rating
    
    @performance_rating.setter
    def performance_rating(self, value: float):
        """Set performance rating with validation"""
        if not isinstance(value, (int, float)):
            raise TypeError("Performance rating must be a number")
        
        if not 1.0 <= value <= 5.0:
            raise ValueError("Performance rating must be between 1.0 and 5.0")
        
        self._performance_rating = float(value)
        
        # Automatic actions based on performance
        if value >= 4.5:
            print(f"Outstanding performance! Consider for promotion.")
        elif value <= 2.0:
            print(f"Performance improvement plan recommended.")
    
    # Computed property based on other properties
    @property
    def annual_bonus(self) -> float:
        """Calculate annual bonus based on performance and salary"""
        if not self._is_active:
            return 0.0
        
        base_bonus_rate = 0.1  # 10% base
        performance_multiplier = self._performance_rating / 5.0
        service_bonus = min(self.years_of_service * 0.01, 0.05)  # Max 5%
        
        total_rate = base_bonus_rate * performance_multiplier + service_bonus
        return self._salary * total_rate
    
    # Property with caching
    @property
    def tax_bracket(self) -> str:
        """Calculate tax bracket (cached computation)"""
        if not hasattr(self, '_cached_tax_bracket'):
            # Simulate expensive computation
            if self._salary < 10000:
                bracket = "10%"
            elif self._salary < 40000:
                bracket = "12%"
            elif self._salary < 85000:
                bracket = "22%"
            elif self._salary < 160000:
                bracket = "24%"
            else:
                bracket = "32%"
            
            self._cached_tax_bracket = bracket
            print(f"Tax bracket computed and cached: {bracket}")
        
        return self._cached_tax_bracket
    
    # Clear cache when salary changes
    @salary.setter
    def salary(self, value: float):
        """Override to clear cache when salary changes"""
        # Clear cached tax bracket
        if hasattr(self, '_cached_tax_bracket'):
            delattr(self, '_cached_tax_bracket')
        
        # Call original setter logic
        if not isinstance(value, (int, float)):
            raise TypeError("Salary must be a number")
        
        if value < 0:
            raise ValueError("Salary cannot be negative")
        
        if value > 1000000:
            raise ValueError("Salary exceeds maximum limit")
        
        old_salary = self._salary
        self._salary = float(value)
        print(f"Salary updated from ${old_salary:,.2f} to ${self._salary:,.2f}")

# Example usage - Property Decorators
print("Creating employee with property decorators:")

emp = Employee("John", "Doe", "EMP001", 75000)

# Read-only properties
print(f"Full name: {emp.full_name}")
print(f"Email: {emp.email}")
print(f"Years of service: {emp.years_of_service}")

# Property with setter
print(f"Current salary: ${emp.salary:,.2f}")
emp.salary = 80000  # Triggers setter with validation

# Property with getter, setter, and deleter
emp.department = "Engineering"
print(f"Department: {emp.department}")
del emp.department  # Triggers deleter

# Property with validation
emp.performance_rating = 4.7  # Triggers outstanding performance message
print(f"Performance rating: {emp.performance_rating}")

# Computed properties
print(f"Annual bonus: ${emp.annual_bonus:,.2f}")
print(f"Tax bracket: {emp.tax_bracket}")  # First call - computes and caches
print(f"Tax bracket: {emp.tax_bracket}")  # Second call - uses cache

# Change salary to see cache clearing
emp.salary = 90000
print(f"Tax bracket after salary change: {emp.tax_bracket}")  # Recomputes

# ===============================================================================
# 3. ADVANCED PROPERTY PATTERNS
# ===============================================================================

print("\n" + "=" * 80)
print("3. ADVANCED PROPERTY PATTERNS")
print("=" * 80)

print("\n--- Real-Life Example: Smart Home IoT Device ---")

class SmartThermostat:
    """Smart thermostat with advanced property patterns"""
    
    def __init__(self, device_id: str, location: str):
        self._device_id = device_id
        self._location = location
        
        # Temperature settings
        self._current_temp = 70.0
        self._target_temp = 72.0
        self._temp_unit = "F"  # F or C
        
        # Operating modes
        self._mode = "auto"  # auto, heat, cool, off
        self._fan_speed = "auto"  # auto, low, medium, high
        
        # Scheduling and automation
        self._schedule = {}
        self._eco_mode = False
        self._learning_enabled = True
        
        # Constraints and limits
        self._min_temp = 50.0
        self._max_temp = 90.0
        self._temp_change_limit = 10.0  # Max change per hour
        
        # Monitoring
        self._last_temp_change = datetime.now()
        self._energy_usage = 0.0
        self._sensor_readings = []
        
        # Thread safety
        self._lock = threading.Lock()
    
    # Property with unit conversion
    @property
    def current_temp(self) -> float:
        """Current temperature in selected unit"""
        return self._current_temp
    
    @current_temp.setter
    def current_temp(self, value: float):
        """Set current temperature (from sensor)"""
        with self._lock:
            if self._temp_unit == "C":
                # Convert Celsius to Fahrenheit for internal storage
                fahrenheit = (value * 9/5) + 32
            else:
                fahrenheit = value
            
            self._current_temp = fahrenheit
            self._log_sensor_reading(fahrenheit)
    
    # Property with constraints and rate limiting
    @property
    def target_temp(self) -> float:
        """Target temperature with unit conversion"""
        if self._temp_unit == "C":
            return (self._target_temp - 32) * 5/9
        return self._target_temp
    
    @target_temp.setter
    def target_temp(self, value: float):
        """Set target temperature with validation and rate limiting"""
        with self._lock:
            # Convert to Fahrenheit for internal storage
            if self._temp_unit == "C":
                fahrenheit = (value * 9/5) + 32
            else:
                fahrenheit = value
            
            # Validate temperature range
            if not self._min_temp <= fahrenheit <= self._max_temp:
                raise ValueError(f"Temperature must be between {self._min_temp}°F and {self._max_temp}°F")
            
            # Check rate limiting
            time_since_last_change = datetime.now() - self._last_temp_change
            temp_change = abs(fahrenheit - self._target_temp)
            
            if time_since_last_change.total_seconds() < 3600 and temp_change > self._temp_change_limit:
                raise ValueError(f"Temperature change limited to {self._temp_change_limit}°F per hour")
            
            old_temp = self._target_temp
            self._target_temp = fahrenheit
            self._last_temp_change = datetime.now()
            
            print(f"Target temperature changed from {old_temp:.1f}°F to {fahrenheit:.1f}°F")
            self._optimize_energy_usage()
    
    # Property with enum-like validation
    @property
    def mode(self) -> str:
        """Operating mode"""
        return self._mode
    
    @mode.setter
    def mode(self, value: str):
        """Set operating mode with validation"""
        valid_modes = ["auto", "heat", "cool", "off"]
        if value not in valid_modes:
            raise ValueError(f"Mode must be one of: {valid_modes}")
        
        old_mode = self._mode
        self._mode = value
        print(f"Mode changed from '{old_mode}' to '{value}'")
        
        # Trigger mode-specific actions
        if value == "off":
            self._energy_usage = 0
        elif value == "eco":
            self._optimize_energy_usage()
    
    # Property with complex validation
    @property
    def temp_unit(self) -> str:
        """Temperature unit (F or C)"""
        return self._temp_unit
    
    @temp_unit.setter
    def temp_unit(self, value: str):
        """Change temperature unit"""
        if value not in ["F", "C"]:
            raise ValueError("Temperature unit must be 'F' or 'C'")
        
        old_unit = self._temp_unit
        self._temp_unit = value
        print(f"Temperature unit changed from {old_unit} to {value}")
    
    # Property with lazy loading and caching
    @property
    def energy_efficiency_rating(self) -> str:
        """Calculate energy efficiency rating (cached)"""
        cache_key = '_cached_efficiency_rating'
        cache_time_key = '_cached_efficiency_time'
        
        # Check if cache is valid (5 minutes)
        if (hasattr(self, cache_time_key) and 
            datetime.now() - getattr(self, cache_time_key) < timedelta(minutes=5)):
            return getattr(self, cache_key)
        
        # Calculate efficiency rating
        if self._energy_usage == 0:
            rating = "A+"
        elif self._energy_usage < 50:
            rating = "A"
        elif self._energy_usage < 100:
            rating = "B"
        elif self._energy_usage < 150:
            rating = "C"
        else:
            rating = "D"
        
        # Cache the result
        setattr(self, cache_key, rating)
        setattr(self, cache_time_key, datetime.now())
        
        print(f"Energy efficiency rating calculated and cached: {rating}")
        return rating
    
    # Property with observer pattern
    @property
    def eco_mode(self) -> bool:
        """Eco mode status"""
        return self._eco_mode
    
    @eco_mode.setter
    def eco_mode(self, value: bool):
        """Enable/disable eco mode with cascading effects"""
        if not isinstance(value, bool):
            raise TypeError("Eco mode must be boolean")
        
        old_value = self._eco_mode
        self._eco_mode = value
        
        if value and not old_value:
            # Enabling eco mode
            print("Eco mode enabled - optimizing for energy efficiency")
            self._target_temp = max(self._target_temp - 2, self._min_temp)
            self._fan_speed = "low"
        elif not value and old_value:
            # Disabling eco mode
            print("Eco mode disabled - returning to comfort settings")
            self._target_temp = min(self._target_temp + 2, self._max_temp)
            self._fan_speed = "auto"
    
    # Read-only computed property
    @property
    def status_summary(self) -> Dict[str, Any]:
        """Complete status summary"""
        return {
            'device_id': self._device_id,
            'location': self._location,
            'current_temp': self.current_temp,
            'target_temp': self.target_temp,
            'temp_unit': self._temp_unit,
            'mode': self._mode,
            'eco_mode': self._eco_mode,
            'fan_speed': self._fan_speed,
            'energy_usage': self._energy_usage,
            'efficiency_rating': self.energy_efficiency_rating,
            'last_updated': datetime.now()
        }
    
    # Helper methods
    def _log_sensor_reading(self, temp: float):
        """Log temperature sensor reading"""
        reading = {
            'timestamp': datetime.now(),
            'temperature': temp,
            'unit': 'F'
        }
        self._sensor_readings.append(reading)
        
        # Keep only last 100 readings
        if len(self._sensor_readings) > 100:
            self._sensor_readings = self._sensor_readings[-100:]
    
    def _optimize_energy_usage(self):
        """Optimize energy usage based on settings"""
        # Simulate energy calculation
        temp_diff = abs(self._target_temp - self._current_temp)
        base_usage = temp_diff * 2
        
        if self._eco_mode:
            base_usage *= 0.8
        
        if self._mode == "off":
            base_usage = 0
        
        self._energy_usage = base_usage
        
        # Clear efficiency rating cache
        if hasattr(self, '_cached_efficiency_rating'):
            delattr(self, '_cached_efficiency_rating')
            delattr(self, '_cached_efficiency_time')

# Example usage - Advanced Property Patterns
print("Creating smart thermostat with advanced properties:")

thermostat = SmartThermostat("THERM001", "Living Room")

# Basic property usage
print(f"Initial status: Current={thermostat.current_temp}°{thermostat.temp_unit}, Target={thermostat.target_temp}°{thermostat.temp_unit}")

# Property with unit conversion
thermostat.temp_unit = "C"
print(f"After unit change: Target={thermostat.target_temp:.1f}°C")

# Property with constraints
try:
    thermostat.target_temp = 35  # 35°C = 95°F, might exceed max
except ValueError as e:
    print(f"Temperature constraint: {e}")

# Property with validation
thermostat.mode = "heat"
print(f"Mode: {thermostat.mode}")

# Property with observer pattern
thermostat.eco_mode = True

# Property with caching
print(f"Efficiency rating: {thermostat.energy_efficiency_rating}")
print(f"Efficiency rating (cached): {thermostat.energy_efficiency_rating}")

# Complete status
status = thermostat.status_summary
print(f"Complete status: {len(status)} fields")

print("\n" + "=" * 80)
print("ENCAPSULATION PART 1 COMPLETE!")
print("=" * 80)
print("""
🎯 CONCEPTS COVERED:
✅ Data Hiding & Access Modifiers
✅ Private, Protected, and Public Members
✅ Property Decorators (@property)
✅ Advanced Property Patterns

📝 REAL-LIFE EXAMPLES:
- Bank Account System (Access Control)
- Employee Management System (Property Decorators)
- Smart Home IoT Device (Advanced Properties)

🚀 NEXT: Continue with encapsulation_advanced.py for Descriptors and Security
""")
