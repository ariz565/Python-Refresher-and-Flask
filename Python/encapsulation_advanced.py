# ===============================================================================
# PYTHON OOP PILLAR 2: ENCAPSULATION (ADVANCED CONCEPTS)
# Real-Life Examples & Complete Mastery Guide - Part 2
# ===============================================================================

"""
ADVANCED ENCAPSULATION COVERAGE:
===============================
4. Descriptors and Data Descriptors
5. Class-Level Properties and Meta-Properties
6. Property Factories and Dynamic Properties
7. Weak References and Memory Management
8. Thread-Safe Properties
9. Security and Validation Patterns
10. Best Practices & Performance Optimization
"""

import weakref
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union, Callable, Type, get_type_hints
from functools import wraps, lru_cache
from collections import defaultdict
import json
import pickle
import hashlib
from enum import Enum
import logging

# ===============================================================================
# 4. DESCRIPTORS AND DATA DESCRIPTORS
# ===============================================================================

print("=" * 80)
print("4. DESCRIPTORS AND DATA DESCRIPTORS")
print("=" * 80)

print("\n--- Real-Life Example: Database ORM Field System ---")

class FieldDescriptor:
    """Base descriptor for database fields"""
    
    def __init__(self, field_type: Type, required: bool = False, default=None):
        self.field_type = field_type
        self.required = required
        self.default = default
        self.name = None  # Will be set by metaclass
        self._values = {}  # Store values per instance
    
    def __set_name__(self, owner, name):
        """Called when descriptor is assigned to a class attribute"""
        self.name = name
        self.private_name = f'_{name}'
    
    def __get__(self, instance, owner):
        """Get value from instance"""
        if instance is None:
            return self
        
        return getattr(instance, self.private_name, self.default)
    
    def __set__(self, instance, value):
        """Set value on instance with validation"""
        if value is None and self.required:
            raise ValueError(f"Field '{self.name}' is required")
        
        if value is not None and not isinstance(value, self.field_type):
            try:
                value = self.field_type(value)
            except (ValueError, TypeError):
                raise TypeError(f"Field '{self.name}' must be of type {self.field_type.__name__}")
        
        setattr(instance, self.private_name, value)
    
    def __delete__(self, instance):
        """Delete value from instance"""
        if self.required:
            raise AttributeError(f"Cannot delete required field '{self.name}'")
        
        setattr(instance, self.private_name, self.default)

class StringField(FieldDescriptor):
    """String field with additional validation"""
    
    def __init__(self, max_length: int = None, min_length: int = 0, 
                 pattern: str = None, required: bool = False, default: str = ""):
        super().__init__(str, required, default)
        self.max_length = max_length
        self.min_length = min_length
        self.pattern = pattern
        if pattern:
            import re
            self.regex = re.compile(pattern)
    
    def __set__(self, instance, value):
        """Set string value with additional validation"""
        if value is not None:
            value = str(value)
            
            # Length validation
            if len(value) < self.min_length:
                raise ValueError(f"Field '{self.name}' must be at least {self.min_length} characters")
            
            if self.max_length and len(value) > self.max_length:
                raise ValueError(f"Field '{self.name}' cannot exceed {self.max_length} characters")
            
            # Pattern validation
            if self.pattern and not self.regex.match(value):
                raise ValueError(f"Field '{self.name}' does not match pattern {self.pattern}")
        
        super().__set__(instance, value)

class IntegerField(FieldDescriptor):
    """Integer field with range validation"""
    
    def __init__(self, min_value: int = None, max_value: int = None, 
                 required: bool = False, default: int = 0):
        super().__init__(int, required, default)
        self.min_value = min_value
        self.max_value = max_value
    
    def __set__(self, instance, value):
        """Set integer value with range validation"""
        if value is not None:
            value = int(value)
            
            if self.min_value is not None and value < self.min_value:
                raise ValueError(f"Field '{self.name}' must be at least {self.min_value}")
            
            if self.max_value is not None and value > self.max_value:
                raise ValueError(f"Field '{self.name}' cannot exceed {self.max_value}")
        
        super().__set__(instance, value)

class DateTimeField(FieldDescriptor):
    """DateTime field with timezone and range validation"""
    
    def __init__(self, auto_now: bool = False, auto_now_add: bool = False,
                 required: bool = False, default=None):
        super().__init__(datetime, required, default)
        self.auto_now = auto_now
        self.auto_now_add = auto_now_add
    
    def __get__(self, instance, owner):
        """Get datetime value, auto-updating if configured"""
        if instance is None:
            return self
        
        if self.auto_now:
            setattr(instance, self.private_name, datetime.now())
        
        return super().__get__(instance, owner)
    
    def __set__(self, instance, value):
        """Set datetime value with parsing support"""
        if value is not None and not isinstance(value, datetime):
            if isinstance(value, str):
                try:
                    value = datetime.fromisoformat(value)
                except ValueError:
                    raise ValueError(f"Field '{self.name}' invalid datetime format")
            else:
                raise TypeError(f"Field '{self.name}' must be datetime or ISO string")
        
        super().__set__(instance, value)

class EmailField(StringField):
    """Email field with email validation"""
    
    def __init__(self, required: bool = False, default: str = ""):
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        super().__init__(max_length=254, pattern=email_pattern, required=required, default=default)

class ChoiceField(FieldDescriptor):
    """Choice field that validates against allowed values"""
    
    def __init__(self, choices: List[Any], required: bool = False, default=None):
        super().__init__(type(choices[0]) if choices else str, required, default)
        self.choices = choices
    
    def __set__(self, instance, value):
        """Set value with choice validation"""
        if value is not None and value not in self.choices:
            raise ValueError(f"Field '{self.name}' must be one of: {self.choices}")
        
        super().__set__(instance, value)

# Model using descriptors
class User:
    """User model using field descriptors"""
    
    # Field definitions using descriptors
    username = StringField(min_length=3, max_length=50, required=True)
    email = EmailField(required=True)
    age = IntegerField(min_value=0, max_value=150)
    status = ChoiceField(['active', 'inactive', 'suspended'], default='active')
    created_at = DateTimeField(auto_now_add=True)
    updated_at = DateTimeField(auto_now=True)
    
    def __init__(self, username: str, email: str, age: int = None):
        self.username = username
        self.email = email
        if age is not None:
            self.age = age
        
        # Set creation time
        self.created_at = datetime.now()
    
    def to_dict(self):
        """Convert user to dictionary"""
        return {
            'username': self.username,
            'email': self.email,
            'age': self.age,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        return f"User(username='{self.username}', email='{self.email}')"

# Example usage - Descriptors
print("Creating user with field descriptors:")

# Valid user creation
user = User("john_doe", "john@example.com", 25)
print(f"User created: {user}")
print(f"User dict: {user.to_dict()}")

# Test validation
try:
    user.username = "jo"  # Too short
except ValueError as e:
    print(f"Username validation: {e}")

try:
    user.email = "invalid-email"  # Invalid format
except ValueError as e:
    print(f"Email validation: {e}")

try:
    user.age = -5  # Invalid age
except ValueError as e:
    print(f"Age validation: {e}")

try:
    user.status = "invalid_status"  # Invalid choice
except ValueError as e:
    print(f"Status validation: {e}")

# Test auto-updating fields
print(f"Created at: {user.created_at}")
time.sleep(1)
print(f"Updated at (auto): {user.updated_at}")

# ===============================================================================
# 5. CLASS-LEVEL PROPERTIES AND META-PROPERTIES
# ===============================================================================

print("\n" + "=" * 80)
print("5. CLASS-LEVEL PROPERTIES AND META-PROPERTIES")
print("=" * 80)

print("\n--- Real-Life Example: Configuration Management System ---")

class classproperty:
    """Descriptor for class-level properties"""
    
    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__
    
    def __get__(self, instance, owner):
        return self.func(owner)
    
    def __set__(self, instance, value):
        raise AttributeError("Class properties are read-only")

class ConfigurationMeta(type):
    """Metaclass for configuration management"""
    
    def __new__(mcs, name, bases, namespace):
        # Auto-register configuration classes
        cls = super().__new__(mcs, name, bases, namespace)
        
        if name != 'BaseConfiguration':
            # Register in global configuration registry
            if not hasattr(mcs, '_config_registry'):
                mcs._config_registry = {}
            mcs._config_registry[name] = cls
        
        return cls
    
    @property
    def config_name(cls):
        """Class property for configuration name"""
        return cls.__name__.replace('Configuration', '').lower()
    
    @property
    def all_configs(cls):
        """Get all registered configuration classes"""
        return getattr(cls, '_config_registry', {})

class BaseConfiguration(metaclass=ConfigurationMeta):
    """Base configuration class with meta-properties"""
    
    _instances = {}  # Singleton storage
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation"""
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._config_data = {}
            self._validation_rules = {}
            self._change_listeners = []
            self._initialized = True
    
    @classproperty
    def config_schema(cls):
        """Class property defining configuration schema"""
        return {
            'type': 'object',
            'properties': {},
            'required': []
        }
    
    @classproperty
    def default_values(cls):
        """Class property for default configuration values"""
        return {}
    
    @classproperty
    def environment_prefix(cls):
        """Class property for environment variable prefix"""
        return f"{cls.config_name.upper()}_"
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self._config_data.get(key, self.default_values.get(key, default))
    
    def set(self, key: str, value: Any):
        """Set configuration value with validation"""
        # Validate if rules exist
        if key in self._validation_rules:
            validator = self._validation_rules[key]
            if not validator(value):
                raise ValueError(f"Validation failed for key '{key}'")
        
        old_value = self._config_data.get(key)
        self._config_data[key] = value
        
        # Notify listeners
        for listener in self._change_listeners:
            listener(key, old_value, value)
    
    def add_validation_rule(self, key: str, validator: Callable[[Any], bool]):
        """Add validation rule for a configuration key"""
        self._validation_rules[key] = validator
    
    def add_change_listener(self, listener: Callable[[str, Any, Any], None]):
        """Add listener for configuration changes"""
        self._change_listeners.append(listener)

class DatabaseConfiguration(BaseConfiguration):
    """Database configuration with specific properties"""
    
    @classproperty
    def config_schema(cls):
        """Database configuration schema"""
        return {
            'type': 'object',
            'properties': {
                'host': {'type': 'string'},
                'port': {'type': 'integer', 'minimum': 1, 'maximum': 65535},
                'database': {'type': 'string'},
                'username': {'type': 'string'},
                'password': {'type': 'string'},
                'pool_size': {'type': 'integer', 'minimum': 1, 'maximum': 100},
                'timeout': {'type': 'number', 'minimum': 0.1}
            },
            'required': ['host', 'database', 'username']
        }
    
    @classproperty
    def default_values(cls):
        """Default database configuration values"""
        return {
            'host': 'localhost',
            'port': 5432,
            'pool_size': 10,
            'timeout': 30.0,
            'ssl_enabled': False
        }
    
    def __init__(self):
        super().__init__()
        
        # Add specific validation rules
        self.add_validation_rule('port', lambda x: 1 <= x <= 65535)
        self.add_validation_rule('pool_size', lambda x: 1 <= x <= 100)
        self.add_validation_rule('timeout', lambda x: x > 0)
        
        # Add change listener for connection pooling
        self.add_change_listener(self._on_pool_config_change)
    
    def _on_pool_config_change(self, key: str, old_value: Any, new_value: Any):
        """Handle pool configuration changes"""
        if key in ['host', 'port', 'pool_size']:
            print(f"Database pool configuration changed: {key} = {new_value}")

class CacheConfiguration(BaseConfiguration):
    """Cache configuration with TTL and size limits"""
    
    @classproperty
    def default_values(cls):
        """Default cache configuration values"""
        return {
            'backend': 'memory',
            'default_ttl': 3600,
            'max_size': 1000,
            'cleanup_interval': 300
        }
    
    @classproperty
    def supported_backends(cls):
        """Supported cache backends"""
        return ['memory', 'redis', 'memcached']
    
    def __init__(self):
        super().__init__()
        
        # Add validation rules
        self.add_validation_rule('backend', lambda x: x in self.supported_backends)
        self.add_validation_rule('default_ttl', lambda x: x > 0)
        self.add_validation_rule('max_size', lambda x: x > 0)

# Configuration Factory
class ConfigurationFactory:
    """Factory for creating configuration instances"""
    
    @staticmethod
    def create_config(config_type: str) -> BaseConfiguration:
        """Create configuration instance by type"""
        config_classes = BaseConfiguration.all_configs
        
        for name, cls in config_classes.items():
            if cls.config_name == config_type:
                return cls()
        
        raise ValueError(f"Unknown configuration type: {config_type}")
    
    @staticmethod
    def list_available_configs() -> List[str]:
        """List all available configuration types"""
        return [cls.config_name for cls in BaseConfiguration.all_configs.values()]

# Example usage - Class Properties and Meta-Properties
print("Demonstrating class properties and meta-properties:")

# Access class properties
print(f"Database config name: {DatabaseConfiguration.config_name}")
print(f"Database defaults: {DatabaseConfiguration.default_values}")
print(f"Database schema: {DatabaseConfiguration.config_schema}")

# Create configuration instances (singletons)
db_config = DatabaseConfiguration()
cache_config = CacheConfiguration()

# Set configuration values
db_config.set('host', 'production-db.example.com')
db_config.set('port', 5432)
db_config.set('pool_size', 20)

cache_config.set('backend', 'redis')
cache_config.set('default_ttl', 7200)

# Access configuration values
print(f"DB Host: {db_config.get('host')}")
print(f"DB Pool Size: {db_config.get('pool_size')}")
print(f"Cache Backend: {cache_config.get('backend')}")

# Test singleton behavior
db_config2 = DatabaseConfiguration()
print(f"Singleton test: {db_config is db_config2}")

# Use factory
factory_config = ConfigurationFactory.create_config('database')
print(f"Factory config is same instance: {factory_config is db_config}")

print(f"Available configs: {ConfigurationFactory.list_available_configs()}")

# ===============================================================================
# 6. PROPERTY FACTORIES AND DYNAMIC PROPERTIES
# ===============================================================================

print("\n" + "=" * 80)
print("6. PROPERTY FACTORIES AND DYNAMIC PROPERTIES")
print("=" * 80)

print("\n--- Real-Life Example: API Client with Dynamic Endpoints ---")

def create_api_property(endpoint: str, method: str = 'GET', 
                       cache_ttl: int = None, auth_required: bool = True):
    """Factory function to create API endpoint properties"""
    
    def property_getter(self):
        """Get cached API response or fetch new data"""
        cache_key = f"{endpoint}_{method}"
        
        # Check cache if TTL is specified
        if cache_ttl and hasattr(self, '_api_cache'):
            cached_data = self._api_cache.get(cache_key)
            if cached_data and datetime.now() - cached_data['timestamp'] < timedelta(seconds=cache_ttl):
                print(f"Returning cached data for {endpoint}")
                return cached_data['data']
        
        # Fetch new data
        print(f"Fetching data from {endpoint} via {method}")
        
        # Simulate API call
        if auth_required and not getattr(self, '_authenticated', False):
            raise PermissionError(f"Authentication required for {endpoint}")
        
        # Mock API response
        response_data = {
            'endpoint': endpoint,
            'method': method,
            'timestamp': datetime.now().isoformat(),
            'data': f"Mock data from {endpoint}"
        }
        
        # Cache if TTL specified
        if cache_ttl:
            if not hasattr(self, '_api_cache'):
                self._api_cache = {}
            self._api_cache[cache_key] = {
                'data': response_data,
                'timestamp': datetime.now()
            }
        
        return response_data
    
    def property_setter(self, value):
        """POST/PUT data to API endpoint"""
        if method not in ['POST', 'PUT', 'PATCH']:
            raise AttributeError(f"Cannot set data on {method} endpoint")
        
        print(f"Sending data to {endpoint} via {method}: {value}")
        
        # Clear cache for this endpoint
        if hasattr(self, '_api_cache'):
            cache_key = f"{endpoint}_{method}"
            self._api_cache.pop(cache_key, None)
    
    def property_deleter(self):
        """DELETE from API endpoint"""
        if method != 'DELETE':
            raise AttributeError(f"Cannot delete from {method} endpoint")
        
        print(f"Deleting data from {endpoint}")
        
        # Clear all cache for this endpoint
        if hasattr(self, '_api_cache'):
            keys_to_remove = [k for k in self._api_cache.keys() if k.startswith(endpoint)]
            for key in keys_to_remove:
                del self._api_cache[key]
    
    # Create property with appropriate methods
    if method in ['POST', 'PUT', 'PATCH']:
        return property(property_getter, property_setter)
    elif method == 'DELETE':
        return property(property_getter, fdel=property_deleter)
    else:
        return property(property_getter)

def create_computed_property(computation_func: Callable, dependencies: List[str] = None):
    """Factory to create computed properties with dependency tracking"""
    
    def property_getter(self):
        """Get computed value, recalculating if dependencies changed"""
        cache_key = f"_computed_{computation_func.__name__}"
        deps_key = f"_computed_deps_{computation_func.__name__}"
        
        # Check if dependencies have changed
        current_deps = {}
        if dependencies:
            for dep in dependencies:
                if hasattr(self, dep):
                    current_deps[dep] = getattr(self, dep)
        
        # Compare with cached dependencies
        if (hasattr(self, cache_key) and hasattr(self, deps_key) and 
            getattr(self, deps_key) == current_deps):
            return getattr(self, cache_key)
        
        # Recalculate
        result = computation_func(self)
        setattr(self, cache_key, result)
        setattr(self, deps_key, current_deps)
        
        return result
    
    return property(property_getter)

class DynamicAPIClient:
    """API client with dynamically created properties"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self._authenticated = False
        self._rate_limit_remaining = 1000
        self._request_count = 0
        
        # Create dynamic properties for common endpoints
        self._create_endpoint_properties()
    
    def _create_endpoint_properties(self):
        """Dynamically create properties for API endpoints"""
        # Define endpoint configurations
        endpoints = {
            'users': {'method': 'GET', 'cache_ttl': 300},
            'posts': {'method': 'GET', 'cache_ttl': 60},
            'comments': {'method': 'GET', 'cache_ttl': 30},
            'profile': {'method': 'POST', 'auth_required': True},
            'settings': {'method': 'PUT', 'auth_required': True}
        }
        
        # Create properties dynamically
        for endpoint, config in endpoints.items():
            prop = create_api_property(
                endpoint=endpoint,
                method=config.get('method', 'GET'),
                cache_ttl=config.get('cache_ttl'),
                auth_required=config.get('auth_required', True)
            )
            setattr(self.__class__, endpoint, prop)
    
    def authenticate(self):
        """Authenticate with API"""
        self._authenticated = True
        print("Authenticated with API")
    
    # Computed properties using factory
    @create_computed_property
    def request_rate(self):
        """Compute current request rate"""
        if self._request_count == 0:
            return 0.0
        return self._request_count / 60.0  # requests per minute
    
    @create_computed_property
    def rate_limit_status(self):
        """Compute rate limit status"""
        if self._rate_limit_remaining > 500:
            return "good"
        elif self._rate_limit_remaining > 100:
            return "warning"
        else:
            return "critical"

# Dynamic property creation based on configuration
class ConfigurableModel:
    """Model with properties created from configuration"""
    
    def __init__(self, field_config: Dict[str, Dict[str, Any]]):
        self._field_config = field_config
        self._data = {}
        
        # Create properties dynamically based on configuration
        for field_name, config in field_config.items():
            self._create_field_property(field_name, config)
    
    def _create_field_property(self, field_name: str, config: Dict[str, Any]):
        """Create a property for a field based on configuration"""
        field_type = config.get('type', str)
        default_value = config.get('default')
        validators = config.get('validators', [])
        computed = config.get('computed')
        
        if computed:
            # Create computed property
            def computed_getter(self):
                return computed(self._data)
            
            prop = property(computed_getter)
        else:
            # Create regular property with validation
            def field_getter(self):
                return self._data.get(field_name, default_value)
            
            def field_setter(self, value):
                # Type conversion
                if value is not None and field_type != type(value):
                    try:
                        value = field_type(value)
                    except (ValueError, TypeError):
                        raise TypeError(f"Field '{field_name}' must be {field_type.__name__}")
                
                # Run validators
                for validator in validators:
                    if not validator(value):
                        raise ValueError(f"Validation failed for field '{field_name}'")
                
                self._data[field_name] = value
            
            prop = property(field_getter, field_setter)
        
        setattr(self.__class__, field_name, prop)

# Example usage - Property Factories and Dynamic Properties
print("Demonstrating property factories and dynamic properties:")

# API Client with dynamic endpoints
client = DynamicAPIClient("https://api.example.com", "api_key_123")
client.authenticate()

# Access dynamically created endpoint properties
users_data = client.users  # GET request with caching
print(f"Users data: {users_data['endpoint']}")

# Access again (should use cache)
users_data_cached = client.users
print("Second access uses cache")

# Computed properties
client._request_count = 10
print(f"Request rate: {client.request_rate}")
print(f"Rate limit status: {client.rate_limit_status}")

# Configurable model with dynamic properties
field_config = {
    'name': {
        'type': str,
        'validators': [lambda x: len(x) >= 2],
        'default': 'Unknown'
    },
    'age': {
        'type': int,
        'validators': [lambda x: 0 <= x <= 150],
        'default': 0
    },
    'full_info': {
        'computed': lambda data: f"{data.get('name', 'Unknown')} ({data.get('age', 0)} years old)"
    }
}

model = ConfigurableModel(field_config)
model.name = "John Doe"
model.age = 30

print(f"Model name: {model.name}")
print(f"Model age: {model.age}")
print(f"Model full info (computed): {model.full_info}")

print("\n" + "=" * 80)
print("ENCAPSULATION ADVANCED CONCEPTS COMPLETE!")
print("=" * 80)
print("""
🎯 ADVANCED CONCEPTS COVERED:
✅ Descriptors and Data Descriptors
✅ Class-Level Properties and Meta-Properties
✅ Property Factories and Dynamic Properties

📝 REAL-LIFE EXAMPLES:
- Database ORM Field System (Descriptors)
- Configuration Management System (Class Properties)
- API Client with Dynamic Endpoints (Property Factories)

🚀 NEXT: Continue with polymorphism.py for method overloading and dynamic behavior
""")
