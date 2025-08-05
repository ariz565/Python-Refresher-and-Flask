# ===============================================================================
# PYTHON OOP PILLAR 1: INHERITANCE (ADVANCED CONCEPTS)
# Real-Life Examples & Complete Mastery Guide - Part 2
# ===============================================================================

"""
ADVANCED INHERITANCE COVERAGE:
==============================
5. Abstract Base Classes (ABC)
6. Mixins & Composition vs Inheritance
7. Diamond Problem Resolution
8. Advanced Inheritance Patterns
9. Real-World Design Patterns
10. Best Practices & Common Pitfalls
"""

import abc
from abc import ABC, abstractmethod, abstractproperty
from typing import Protocol, runtime_checkable, Optional, List, Dict, Any, Union
import time
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging

# ===============================================================================
# 5. ABSTRACT BASE CLASSES (ABC)
# ===============================================================================

print("=" * 80)
print("5. ABSTRACT BASE CLASSES (ABC)")
print("=" * 80)

print("\n--- Real-Life Example: Data Processing Pipeline ---")

class DataProcessor(ABC):
    """Abstract base class for data processors"""
    
    def __init__(self, name: str):
        self.name = name
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None
        self.end_time = None
    
    @abstractmethod
    def validate_data(self, data: Any) -> bool:
        """Validate input data - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def process_data(self, data: Any) -> Any:
        """Process the data - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def save_result(self, result: Any, destination: str) -> bool:
        """Save processed result - must be implemented by subclasses"""
        pass
    
    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """Return list of supported data formats"""
        pass
    
    # Concrete methods that can be used by all subclasses
    def start_processing(self):
        """Start the processing session"""
        self.start_time = datetime.now()
        self.processed_count = 0
        self.error_count = 0
        print(f"{self.name} processor started at {self.start_time}")
    
    def end_processing(self):
        """End the processing session"""
        self.end_time = datetime.now()
        duration = self.end_time - self.start_time
        print(f"{self.name} processor finished. Duration: {duration}")
        print(f"Processed: {self.processed_count}, Errors: {self.error_count}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        
        return {
            'name': self.name,
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'duration_seconds': duration,
            'supported_formats': self.supported_formats
        }
    
    def execute_pipeline(self, data_items: List[Any], destination: str):
        """Execute the complete processing pipeline"""
        self.start_processing()
        
        for item in data_items:
            try:
                # Step 1: Validate
                if not self.validate_data(item):
                    print(f"Validation failed for item: {item}")
                    self.error_count += 1
                    continue
                
                # Step 2: Process
                result = self.process_data(item)
                
                # Step 3: Save
                if self.save_result(result, destination):
                    self.processed_count += 1
                else:
                    self.error_count += 1
                    
            except Exception as e:
                print(f"Error processing item {item}: {e}")
                self.error_count += 1
        
        self.end_processing()

class CSVProcessor(DataProcessor):
    """Concrete implementation for CSV processing"""
    
    def __init__(self):
        super().__init__("CSV")
        self.delimiter = ","
        self.headers = []
    
    @property
    def supported_formats(self) -> List[str]:
        """CSV processor supports CSV and TSV formats"""
        return ["csv", "tsv", "txt"]
    
    def validate_data(self, data: str) -> bool:
        """Validate CSV data"""
        if not isinstance(data, str):
            return False
        
        # Check if data contains delimiter
        if self.delimiter not in data:
            return False
        
        # Check if all rows have same number of columns
        rows = data.strip().split('\n')
        if len(rows) < 2:  # Need at least headers and one data row
            return False
        
        first_row_cols = len(rows[0].split(self.delimiter))
        for row in rows[1:]:
            if len(row.split(self.delimiter)) != first_row_cols:
                return False
        
        return True
    
    def process_data(self, data: str) -> Dict[str, List[Any]]:
        """Process CSV data into structured format"""
        rows = data.strip().split('\n')
        headers = [h.strip() for h in rows[0].split(self.delimiter)]
        self.headers = headers
        
        processed_data = {header: [] for header in headers}
        
        for row in rows[1:]:
            values = [v.strip() for v in row.split(self.delimiter)]
            for i, header in enumerate(headers):
                # Try to convert to appropriate type
                value = values[i]
                try:
                    # Try integer first
                    value = int(value)
                except ValueError:
                    try:
                        # Try float
                        value = float(value)
                    except ValueError:
                        # Keep as string
                        pass
                
                processed_data[header].append(value)
        
        return processed_data
    
    def save_result(self, result: Dict[str, List[Any]], destination: str) -> bool:
        """Save processed CSV result"""
        try:
            # Simulate saving to file
            print(f"Saving CSV result to {destination}")
            print(f"Headers: {list(result.keys())}")
            print(f"Rows: {len(list(result.values())[0]) if result else 0}")
            return True
        except Exception as e:
            print(f"Failed to save CSV result: {e}")
            return False

class JSONProcessor(DataProcessor):
    """Concrete implementation for JSON processing"""
    
    def __init__(self):
        super().__init__("JSON")
        self.schema = {}
    
    @property
    def supported_formats(self) -> List[str]:
        """JSON processor supports JSON format"""
        return ["json", "jsonl"]
    
    def validate_data(self, data: str) -> bool:
        """Validate JSON data"""
        import json
        try:
            json.loads(data)
            return True
        except (json.JSONDecodeError, TypeError):
            return False
    
    def process_data(self, data: str) -> Dict[str, Any]:
        """Process JSON data"""
        import json
        parsed_data = json.loads(data)
        
        # Add metadata
        processed_result = {
            'data': parsed_data,
            'processed_at': datetime.now().isoformat(),
            'processor': self.name,
            'data_type': type(parsed_data).__name__
        }
        
        # If it's a list, add count
        if isinstance(parsed_data, list):
            processed_result['item_count'] = len(parsed_data)
        
        # If it's a dict, add key count
        elif isinstance(parsed_data, dict):
            processed_result['key_count'] = len(parsed_data.keys())
        
        return processed_result
    
    def save_result(self, result: Dict[str, Any], destination: str) -> bool:
        """Save processed JSON result"""
        try:
            import json
            # Simulate saving to file
            print(f"Saving JSON result to {destination}")
            print(f"Data type: {result.get('data_type', 'unknown')}")
            print(f"Item count: {result.get('item_count', 'N/A')}")
            return True
        except Exception as e:
            print(f"Failed to save JSON result: {e}")
            return False

class XMLProcessor(DataProcessor):
    """Concrete implementation for XML processing"""
    
    def __init__(self):
        super().__init__("XML")
        self.namespace = {}
    
    @property
    def supported_formats(self) -> List[str]:
        """XML processor supports XML format"""
        return ["xml", "html", "xhtml"]
    
    def validate_data(self, data: str) -> bool:
        """Validate XML data"""
        # Simple XML validation
        if not isinstance(data, str):
            return False
        
        # Check for basic XML structure
        if not (data.strip().startswith('<') and data.strip().endswith('>')):
            return False
        
        # Count opening and closing tags (simplified)
        opening_tags = data.count('<') - data.count('</')
        closing_tags = data.count('</')
        
        return opening_tags == closing_tags + 1  # +1 for self-closing tags
    
    def process_data(self, data: str) -> Dict[str, Any]:
        """Process XML data"""
        # Simplified XML processing (in real world, use xml.etree.ElementTree)
        
        # Extract tag names
        import re
        tags = re.findall(r'<(\w+)', data)
        unique_tags = list(set(tags))
        
        processed_result = {
            'raw_data': data,
            'unique_tags': unique_tags,
            'tag_count': len(tags),
            'unique_tag_count': len(unique_tags),
            'processed_at': datetime.now().isoformat(),
            'processor': self.name
        }
        
        return processed_result
    
    def save_result(self, result: Dict[str, Any], destination: str) -> bool:
        """Save processed XML result"""
        try:
            print(f"Saving XML result to {destination}")
            print(f"Unique tags: {result.get('unique_tag_count', 0)}")
            print(f"Total tags: {result.get('tag_count', 0)}")
            return True
        except Exception as e:
            print(f"Failed to save XML result: {e}")
            return False

# Example usage - Abstract Base Classes
print("Demonstrating Abstract Base Classes:")

# Cannot instantiate abstract class
try:
    # This would raise TypeError
    # processor = DataProcessor("test")
    pass
except TypeError as e:
    print(f"Cannot instantiate abstract class: {e}")

# Create concrete implementations
csv_processor = CSVProcessor()
json_processor = JSONProcessor()
xml_processor = XMLProcessor()

# Test CSV processing
csv_data = """name,age,city
John,25,New York
Jane,30,Los Angeles
Bob,35,Chicago"""

csv_processor.execute_pipeline([csv_data], "output.csv")
print(f"CSV Stats: {csv_processor.get_stats()}")

# Test JSON processing
json_data = '{"users": [{"name": "John", "age": 25}, {"name": "Jane", "age": 30}]}'
json_processor.execute_pipeline([json_data], "output.json")
print(f"JSON Stats: {json_processor.get_stats()}")

# Test XML processing
xml_data = "<users><user><name>John</name><age>25</age></user></users>"
xml_processor.execute_pipeline([xml_data], "output.xml")
print(f"XML Stats: {xml_processor.get_stats()}")

# ===============================================================================
# 6. MIXINS & COMPOSITION VS INHERITANCE
# ===============================================================================

print("\n" + "=" * 80)
print("6. MIXINS & COMPOSITION VS INHERITANCE")
print("=" * 80)

print("\n--- Real-Life Example: User Management System ---")

# Mixins - Small, focused classes that provide specific functionality
class TimestampMixin:
    """Mixin to add timestamp functionality"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def touch(self):
        """Update the timestamp"""
        self.updated_at = datetime.now()
        print(f"Timestamp updated: {self.updated_at}")
    
    def age_in_seconds(self):
        """Get age in seconds"""
        return (datetime.now() - self.created_at).total_seconds()

class LoggingMixin:
    """Mixin to add logging functionality"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_log = []
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger for this instance"""
        logger_name = f"{self.__class__.__name__}_{id(self)}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
    
    def log_action(self, action: str, details: str = ""):
        """Log an action"""
        timestamp = datetime.now()
        log_entry = {
            'timestamp': timestamp,
            'action': action,
            'details': details,
            'class': self.__class__.__name__
        }
        self.action_log.append(log_entry)
        self.logger.info(f"{action}: {details}")
    
    def get_action_history(self):
        """Get action history"""
        return self.action_log.copy()

class ValidationMixin:
    """Mixin to add validation functionality"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_rules = {}
        self.validation_errors = []
    
    def add_validation_rule(self, field: str, rule_func, error_message: str):
        """Add a validation rule"""
        if field not in self.validation_rules:
            self.validation_rules[field] = []
        
        self.validation_rules[field].append({
            'rule': rule_func,
            'message': error_message
        })
    
    def validate_field(self, field: str, value: Any) -> bool:
        """Validate a specific field"""
        if field not in self.validation_rules:
            return True
        
        field_valid = True
        for rule_info in self.validation_rules[field]:
            if not rule_info['rule'](value):
                self.validation_errors.append({
                    'field': field,
                    'value': value,
                    'message': rule_info['message']
                })
                field_valid = False
        
        return field_valid
    
    def validate_all(self) -> bool:
        """Validate all fields that have rules"""
        self.validation_errors = []
        all_valid = True
        
        for field in self.validation_rules:
            if hasattr(self, field):
                value = getattr(self, field)
                if not self.validate_field(field, value):
                    all_valid = False
        
        return all_valid
    
    def get_validation_errors(self):
        """Get validation errors"""
        return self.validation_errors.copy()

class CacheableMixin:
    """Mixin to add caching functionality"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}
        self._cache_ttl = {}  # Time to live for cache entries
        self.default_ttl = 300  # 5 minutes
    
    def cache_set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set cache value"""
        if ttl is None:
            ttl = self.default_ttl
        
        self._cache[key] = value
        self._cache_ttl[key] = datetime.now() + timedelta(seconds=ttl)
    
    def cache_get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        if key not in self._cache:
            return None
        
        # Check if expired
        if datetime.now() > self._cache_ttl[key]:
            del self._cache[key]
            del self._cache_ttl[key]
            return None
        
        return self._cache[key]
    
    def cache_clear(self):
        """Clear cache"""
        self._cache.clear()
        self._cache_ttl.clear()
    
    def cache_stats(self):
        """Get cache statistics"""
        active_entries = 0
        expired_entries = 0
        
        for key in self._cache:
            if datetime.now() <= self._cache_ttl[key]:
                active_entries += 1
            else:
                expired_entries += 1
        
        return {
            'active_entries': active_entries,
            'expired_entries': expired_entries,
            'total_entries': len(self._cache)
        }

# Base User class
class User:
    """Base user class"""
    
    def __init__(self, username: str, email: str):
        self.username = username
        self.email = email
        self.is_active = True
        super().__init__()  # Important for mixin cooperation
    
    def activate(self):
        """Activate user"""
        self.is_active = True
    
    def deactivate(self):
        """Deactivate user"""
        self.is_active = False
    
    def get_info(self):
        """Get user information"""
        return {
            'username': self.username,
            'email': self.email,
            'is_active': self.is_active
        }

# User with mixins
class EnhancedUser(User, TimestampMixin, LoggingMixin, ValidationMixin, CacheableMixin):
    """User class enhanced with multiple mixins"""
    
    def __init__(self, username: str, email: str):
        super().__init__(username, email)
        
        # Setup validation rules
        self.add_validation_rule(
            'username',
            lambda x: len(x) >= 3,
            "Username must be at least 3 characters"
        )
        self.add_validation_rule(
            'email',
            lambda x: '@' in x and '.' in x,
            "Email must be valid format"
        )
        
        self.log_action("user_created", f"User {username} created")
    
    def update_email(self, new_email: str):
        """Update user email with validation and logging"""
        old_email = self.email
        
        # Validate new email
        if not self.validate_field('email', new_email):
            self.log_action("email_update_failed", f"Invalid email: {new_email}")
            return False
        
        # Update email
        self.email = new_email
        self.touch()  # Update timestamp
        
        # Log action
        self.log_action("email_updated", f"Email changed from {old_email} to {new_email}")
        
        # Clear cache since user data changed
        self.cache_clear()
        
        return True
    
    def get_profile_data(self):
        """Get profile data with caching"""
        cache_key = "profile_data"
        cached_data = self.cache_get(cache_key)
        
        if cached_data:
            self.log_action("profile_data_cached", "Returned cached profile data")
            return cached_data
        
        # Generate profile data
        profile_data = {
            **self.get_info(),
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'age_seconds': self.age_in_seconds()
        }
        
        # Cache for future use
        self.cache_set(cache_key, profile_data)
        self.log_action("profile_data_generated", "Generated and cached profile data")
        
        return profile_data

# Composition example - alternative to inheritance
class UserManager:
    """User manager using composition instead of inheritance"""
    
    def __init__(self):
        self.users = {}
        self.logger = LoggingMixin()
        self.cache = CacheableMixin()
        self.validator = ValidationMixin()
        
        # Setup validation for user data
        self.validator.add_validation_rule(
            'username',
            lambda x: isinstance(x, str) and len(x) >= 3,
            "Username must be string with at least 3 characters"
        )
    
    def create_user(self, username: str, email: str) -> bool:
        """Create a new user using composition"""
        # Validate input
        if not self.validator.validate_field('username', username):
            return False
        
        # Check if user exists
        if username in self.users:
            self.logger.log_action("user_creation_failed", f"User {username} already exists")
            return False
        
        # Create user
        user = EnhancedUser(username, email)
        self.users[username] = user
        
        # Cache user
        self.cache.cache_set(f"user_{username}", user)
        
        # Log action
        self.logger.log_action("user_created", f"User {username} created via manager")
        
        return True
    
    def get_user(self, username: str) -> Optional[EnhancedUser]:
        """Get user using composition"""
        # Try cache first
        cached_user = self.cache.cache_get(f"user_{username}")
        if cached_user:
            self.logger.log_action("user_retrieved_cache", f"User {username} from cache")
            return cached_user
        
        # Get from storage
        user = self.users.get(username)
        if user:
            # Cache for future use
            self.cache.cache_set(f"user_{username}", user)
            self.logger.log_action("user_retrieved_storage", f"User {username} from storage")
        
        return user

# Example usage - Mixins and Composition
print("Demonstrating Mixins:")

# Create enhanced user with multiple mixins
user = EnhancedUser("john_doe", "john@example.com")
print(f"User created: {user.get_info()}")

# Use mixin functionality
time.sleep(1)  # Small delay for timestamp demo
user.update_email("john.doe@newdomain.com")

# Access profile data (first time - generates and caches)
profile1 = user.get_profile_data()
print(f"Profile data (first call): Generated new data")

# Access profile data (second time - from cache)
profile2 = user.get_profile_data()
print(f"Profile data (second call): From cache")

# Show action history
print(f"Action history: {len(user.get_action_history())} actions")

# Show cache stats
print(f"Cache stats: {user.cache_stats()}")

print("\nDemonstrating Composition:")

# Create user manager using composition
manager = UserManager()
manager.create_user("alice", "alice@example.com")
manager.create_user("bob", "bob@example.com")

# Retrieve users
alice = manager.get_user("alice")
bob = manager.get_user("bob")

print(f"Alice: {alice.get_info() if alice else 'Not found'}")
print(f"Bob: {bob.get_info() if bob else 'Not found'}")

print(f"\nMRO for EnhancedUser: {EnhancedUser.__mro__}")

# ===============================================================================
# 7. DIAMOND PROBLEM RESOLUTION
# ===============================================================================

print("\n" + "=" * 80)
print("7. DIAMOND PROBLEM RESOLUTION")
print("=" * 80)

print("\n--- Real-Life Example: Employee Management System ---")

class Person:
    """Base person class"""
    
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
        print(f"Person.__init__ called for {name}")
    
    def get_info(self):
        """Get basic person information"""
        return f"Person: {self.name}, Age: {self.age}"
    
    def introduce(self):
        """Introduce the person"""
        print(f"Hi, I'm {self.name}")

class Employee(Person):
    """Employee class"""
    
    def __init__(self, name: str, age: int, employee_id: str):
        super().__init__(name, age)
        self.employee_id = employee_id
        self.department = None
        print(f"Employee.__init__ called for {employee_id}")
    
    def get_info(self):
        """Get employee information"""
        base_info = super().get_info()
        return f"{base_info}, Employee ID: {self.employee_id}"
    
    def clock_in(self):
        """Clock in to work"""
        print(f"Employee {self.employee_id} clocked in")
    
    def introduce(self):
        """Introduce the employee"""
        super().introduce()
        print(f"I work here as employee {self.employee_id}")

class Manager(Person):
    """Manager class"""
    
    def __init__(self, name: str, age: int, team_size: int):
        super().__init__(name, age)
        self.team_size = team_size
        self.reports = []
        print(f"Manager.__init__ called for {name}")
    
    def get_info(self):
        """Get manager information"""
        base_info = super().get_info()
        return f"{base_info}, Team Size: {self.team_size}"
    
    def hold_meeting(self):
        """Hold a team meeting"""
        print(f"Manager {self.name} is holding a meeting with {self.team_size} people")
    
    def introduce(self):
        """Introduce the manager"""
        super().introduce()
        print(f"I manage a team of {self.team_size} people")

# Diamond Problem Example
class TeamLead(Employee, Manager):
    """Team lead - inherits from both Employee and Manager (Diamond Problem)"""
    
    def __init__(self, name: str, age: int, employee_id: str, team_size: int):
        # Using super() with multiple inheritance
        # Python's MRO resolves the diamond problem
        Employee.__init__(self, name, age, employee_id)
        Manager.__init__(self, name, age, team_size)
        
        self.is_technical_lead = True
        print(f"TeamLead.__init__ called for {name}")
    
    def get_info(self):
        """Get team lead information - resolves method conflicts"""
        # We need to be explicit about which parent's method to call
        employee_info = Employee.get_info(self)
        return f"{employee_info}, Team Size: {self.team_size}, Technical Lead: {self.is_technical_lead}"
    
    def introduce(self):
        """Introduce team lead - combines both roles"""
        print(f"Hi, I'm {self.name}")
        print(f"I'm employee {self.employee_id} and I lead a team of {self.team_size}")
    
    def conduct_code_review(self):
        """Team lead specific method"""
        print(f"Team Lead {self.name} is conducting code review")
    
    def assign_task(self, task: str, assignee: str):
        """Assign task to team member"""
        print(f"Team Lead {self.name} assigned '{task}' to {assignee}")

# Better Diamond Problem Resolution using explicit cooperation
class CooperativePerson:
    """Base person class designed for cooperative inheritance"""
    
    def __init__(self, name: str, age: int, **kwargs):
        self.name = name
        self.age = age
        print(f"CooperativePerson.__init__ called for {name}")
        super().__init__(**kwargs)  # Pass remaining kwargs up the chain
    
    def get_info(self):
        """Get basic person information"""
        return f"Person: {self.name}, Age: {self.age}"
    
    def introduce(self):
        """Introduce the person"""
        info = [f"Hi, I'm {self.name}"]
        return info

class CooperativeEmployee(CooperativePerson):
    """Employee class designed for cooperative inheritance"""
    
    def __init__(self, name: str, age: int, employee_id: str, **kwargs):
        self.employee_id = employee_id
        self.department = None
        print(f"CooperativeEmployee.__init__ called for {employee_id}")
        super().__init__(name=name, age=age, **kwargs)
    
    def get_info(self):
        """Get employee information"""
        base_info = super().get_info()
        return f"{base_info}, Employee ID: {self.employee_id}"
    
    def introduce(self):
        """Introduce the employee"""
        info = super().introduce()
        info.append(f"I work here as employee {self.employee_id}")
        return info

class CooperativeManager(CooperativePerson):
    """Manager class designed for cooperative inheritance"""
    
    def __init__(self, name: str, age: int, team_size: int, **kwargs):
        self.team_size = team_size
        self.reports = []
        print(f"CooperativeManager.__init__ called for {name}")
        super().__init__(name=name, age=age, **kwargs)
    
    def get_info(self):
        """Get manager information"""
        base_info = super().get_info()
        return f"{base_info}, Team Size: {self.team_size}"
    
    def introduce(self):
        """Introduce the manager"""
        info = super().introduce()
        info.append(f"I manage a team of {self.team_size} people")
        return info

class CooperativeTeamLead(CooperativeEmployee, CooperativeManager):
    """Team lead using cooperative inheritance"""
    
    def __init__(self, name: str, age: int, employee_id: str, team_size: int):
        self.is_technical_lead = True
        print(f"CooperativeTeamLead.__init__ called for {name}")
        super().__init__(
            name=name,
            age=age,
            employee_id=employee_id,
            team_size=team_size
        )
    
    def get_info(self):
        """Get team lead information"""
        base_info = super().get_info()
        return f"{base_info}, Technical Lead: {self.is_technical_lead}"
    
    def introduce(self):
        """Introduce team lead"""
        info_parts = super().introduce()
        return " | ".join(info_parts)

# Example usage - Diamond Problem
print("Demonstrating Diamond Problem:")

print("\n--- Standard Diamond Problem ---")
team_lead1 = TeamLead("Alice Johnson", 32, "EMP001", 5)
print(f"Team Lead Info: {team_lead1.get_info()}")
team_lead1.introduce()
print(f"MRO: {TeamLead.__mro__}")

print("\n--- Cooperative Inheritance Solution ---")
team_lead2 = CooperativeTeamLead("Bob Smith", 35, "EMP002", 8)
print(f"Cooperative Team Lead Info: {team_lead2.get_info()}")
print(f"Introduction: {team_lead2.introduce()}")
print(f"MRO: {CooperativeTeamLead.__mro__}")

# ===============================================================================
# 8. ADVANCED INHERITANCE PATTERNS
# ===============================================================================

print("\n" + "=" * 80)
print("8. ADVANCED INHERITANCE PATTERNS")
print("=" * 80)

print("\n--- Real-Life Example: Plugin Architecture ---")

# Template Method Pattern with Inheritance
class PluginBase(ABC):
    """Base class for all plugins using Template Method pattern"""
    
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.is_loaded = False
        self.dependencies = []
        self.config = {}
    
    def execute(self):
        """Template method defining the algorithm structure"""
        try:
            self.validate_dependencies()
            self.load_config()
            self.initialize()
            result = self.run()
            self.cleanup()
            return result
        except Exception as e:
            self.handle_error(e)
            raise
    
    def validate_dependencies(self):
        """Validate plugin dependencies"""
        for dep in self.dependencies:
            print(f"Validating dependency: {dep}")
            # In real implementation, check if dependency is available
    
    def load_config(self):
        """Load plugin configuration"""
        print(f"Loading configuration for {self.name}")
        # Override in subclasses for specific config loading
    
    @abstractmethod
    def initialize(self):
        """Initialize the plugin - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def run(self):
        """Run the plugin's main logic - must be implemented by subclasses"""
        pass
    
    def cleanup(self):
        """Cleanup resources - can be overridden by subclasses"""
        print(f"Cleaning up {self.name}")
    
    def handle_error(self, error: Exception):
        """Handle errors - can be overridden by subclasses"""
        print(f"Error in {self.name}: {error}")

class DataProcessingPlugin(PluginBase):
    """Base class for data processing plugins"""
    
    def __init__(self, name: str, version: str):
        super().__init__(name, version)
        self.input_data = None
        self.output_data = None
        self.processing_stats = {}
    
    def load_config(self):
        """Load data processing specific config"""
        super().load_config()
        self.config.update({
            'batch_size': 1000,
            'timeout': 300,
            'retry_count': 3
        })
    
    @abstractmethod
    def process_batch(self, batch_data):
        """Process a batch of data - implemented by specific processors"""
        pass
    
    def run(self):
        """Run data processing with batching"""
        if not self.input_data:
            raise ValueError("No input data provided")
        
        processed_count = 0
        batch_size = self.config.get('batch_size', 1000)
        
        # Process data in batches
        for i in range(0, len(self.input_data), batch_size):
            batch = self.input_data[i:i + batch_size]
            processed_batch = self.process_batch(batch)
            
            if self.output_data is None:
                self.output_data = []
            self.output_data.extend(processed_batch)
            processed_count += len(processed_batch)
        
        self.processing_stats = {
            'input_count': len(self.input_data),
            'output_count': processed_count,
            'batch_size': batch_size
        }
        
        return self.output_data

class TextProcessingPlugin(DataProcessingPlugin):
    """Concrete text processing plugin"""
    
    def __init__(self):
        super().__init__("TextProcessor", "1.0.0")
        self.dependencies = ["nltk", "regex"]
    
    def initialize(self):
        """Initialize text processing"""
        print("Initializing text processing plugin")
        self.config.update({
            'lowercase': True,
            'remove_punctuation': True,
            'min_word_length': 2
        })
    
    def process_batch(self, batch_data):
        """Process a batch of text data"""
        processed_batch = []
        
        for text in batch_data:
            processed_text = text
            
            # Apply configured transformations
            if self.config.get('lowercase', False):
                processed_text = processed_text.lower()
            
            if self.config.get('remove_punctuation', False):
                import string
                processed_text = processed_text.translate(
                    str.maketrans('', '', string.punctuation)
                )
            
            # Filter by minimum word length
            min_length = self.config.get('min_word_length', 1)
            words = [w for w in processed_text.split() if len(w) >= min_length]
            processed_text = ' '.join(words)
            
            processed_batch.append(processed_text)
        
        return processed_batch

class ImageProcessingPlugin(DataProcessingPlugin):
    """Concrete image processing plugin"""
    
    def __init__(self):
        super().__init__("ImageProcessor", "1.0.0")
        self.dependencies = ["PIL", "numpy"]
    
    def initialize(self):
        """Initialize image processing"""
        print("Initializing image processing plugin")
        self.config.update({
            'resize_width': 224,
            'resize_height': 224,
            'normalize': True
        })
    
    def process_batch(self, batch_data):
        """Process a batch of image data"""
        processed_batch = []
        
        for image_path in batch_data:
            # Simulate image processing
            processed_image = {
                'original_path': image_path,
                'width': self.config.get('resize_width', 224),
                'height': self.config.get('resize_height', 224),
                'normalized': self.config.get('normalize', True),
                'processed_at': datetime.now()
            }
            
            processed_batch.append(processed_image)
        
        return processed_batch

# Plugin Factory Pattern
class PluginFactory:
    """Factory for creating plugins"""
    
    _plugins = {
        'text': TextProcessingPlugin,
        'image': ImageProcessingPlugin
    }
    
    @classmethod
    def create_plugin(cls, plugin_type: str) -> PluginBase:
        """Create a plugin of the specified type"""
        if plugin_type not in cls._plugins:
            raise ValueError(f"Unknown plugin type: {plugin_type}")
        
        return cls._plugins[plugin_type]()
    
    @classmethod
    def register_plugin(cls, name: str, plugin_class):
        """Register a new plugin type"""
        cls._plugins[name] = plugin_class
    
    @classmethod
    def list_plugins(cls):
        """List available plugin types"""
        return list(cls._plugins.keys())

# Example usage - Advanced Patterns
print("Demonstrating Advanced Inheritance Patterns:")

# Create plugins using factory
text_plugin = PluginFactory.create_plugin('text')
image_plugin = PluginFactory.create_plugin('image')

# Process text data
text_data = [
    "Hello World! This is a test.",
    "PYTHON is awesome for data processing.",
    "Remove punctuation and normalize text."
]

text_plugin.input_data = text_data
text_result = text_plugin.execute()
print(f"Text processing result: {text_result[:2]}...")  # Show first 2 results
print(f"Text processing stats: {text_plugin.processing_stats}")

# Process image data
image_data = ["image1.jpg", "image2.png", "image3.gif"]
image_plugin.input_data = image_data
image_result = image_plugin.execute()
print(f"Image processing result count: {len(image_result)}")
print(f"Image processing stats: {image_plugin.processing_stats}")

print(f"Available plugins: {PluginFactory.list_plugins()}")

print("\n" + "=" * 80)
print("INHERITANCE ADVANCED CONCEPTS COMPLETE!")
print("=" * 80)
print("""
🎯 ADVANCED CONCEPTS COVERED:
✅ Abstract Base Classes (ABC)
✅ Mixins & Composition vs Inheritance
✅ Diamond Problem Resolution
✅ Advanced Inheritance Patterns

📝 REAL-LIFE EXAMPLES:
- Data Processing Pipeline (ABC)
- User Management System (Mixins)
- Employee Management System (Diamond Problem)
- Plugin Architecture (Advanced Patterns)

🚀 NEXT: Continue with encapsulation.py for data hiding and access control
""")
