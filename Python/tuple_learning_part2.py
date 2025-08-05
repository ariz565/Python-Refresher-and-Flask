# ===============================================================================
# COMPREHENSIVE PYTHON TUPLES LEARNING GUIDE - PART 2
# Continuing from tuple_learning.py
# ===============================================================================

"""
CONTINUATION OF SECTIONS:
=======================
6. Real-World Applications
7. Interview Problems & Solutions
8. Advanced Concepts for Experienced Developers
9. System Design with Tuples
10. Threading & Immutability Benefits
11. Best Practices & Common Pitfalls
12. Tuple vs Other Data Structures
"""

import time
import sys
from collections import namedtuple, defaultdict, Counter
import threading
import operator
from typing import Tuple, NamedTuple
import itertools
import functools
import json

# ===============================================================================
# 6. REAL-WORLD APPLICATIONS
# ===============================================================================

print("=" * 80)
print("6. REAL-WORLD APPLICATIONS")
print("=" * 80)

print("\n--- Application 1: Database Records ---")

# Representing database records as tuples
DatabaseRecord = namedtuple('DatabaseRecord', ['id', 'timestamp', 'user_id', 'action', 'data'])

# Simulating database records
records = [
    DatabaseRecord(1, '2025-01-01 10:00:00', 123, 'login', '{"ip": "192.168.1.1"}'),
    DatabaseRecord(2, '2025-01-01 10:05:00', 123, 'view_page', '{"page": "/dashboard"}'),
    DatabaseRecord(3, '2025-01-01 10:10:00', 456, 'login', '{"ip": "192.168.1.2"}'),
    DatabaseRecord(4, '2025-01-01 10:15:00', 123, 'logout', '{}')
]

# Analyzing records
user_actions = defaultdict(list)
for record in records:
    user_actions[record.user_id].append(record.action)

print(f"User 123 actions: {user_actions[123]}")
print(f"User 456 actions: {user_actions[456]}")

# Login records only
login_records = [r for r in records if r.action == 'login']
print(f"Login records: {len(login_records)}")

print("\n--- Application 2: Configuration Management ---")

# Configuration as named tuples
DatabaseConfig = namedtuple('DatabaseConfig', ['host', 'port', 'username', 'password', 'database'])
APIConfig = namedtuple('APIConfig', ['base_url', 'api_key', 'timeout', 'retries'])

# Environment-specific configurations
configs = {
    'development': {
        'database': DatabaseConfig('localhost', 5432, 'dev_user', 'dev_pass', 'dev_db'),
        'api': APIConfig('http://localhost:8000', 'dev_key', 30, 3)
    },
    'production': {
        'database': DatabaseConfig('prod.server.com', 5432, 'prod_user', 'prod_pass', 'prod_db'),
        'api': APIConfig('https://api.server.com', 'prod_key', 60, 5)
    }
}

def get_config(environment):
    return configs.get(environment, configs['development'])

dev_config = get_config('development')
print(f"Dev DB config: {dev_config['database']}")
print(f"Dev API config: {dev_config['api']}")

print("\n--- Application 3: Coordinate Systems ---")

# 2D and 3D coordinates
Point2D = namedtuple('Point2D', ['x', 'y'])
Point3D = namedtuple('Point3D', ['x', 'y', 'z'])

class GeometryCalculator:
    @staticmethod
    def distance_2d(p1: Point2D, p2: Point2D) -> float:
        return ((p2.x - p1.x)**2 + (p2.y - p1.y)**2)**0.5
    
    @staticmethod
    def distance_3d(p1: Point3D, p2: Point3D) -> float:
        return ((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)**0.5
    
    @staticmethod
    def midpoint_2d(p1: Point2D, p2: Point2D) -> Point2D:
        return Point2D((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)

# Example usage
point_a = Point2D(0, 0)
point_b = Point2D(3, 4)
point_c = Point3D(1, 2, 3)
point_d = Point3D(4, 6, 8)

calc = GeometryCalculator()
print(f"2D distance: {calc.distance_2d(point_a, point_b):.2f}")
print(f"3D distance: {calc.distance_3d(point_c, point_d):.2f}")
print(f"2D midpoint: {calc.midpoint_2d(point_a, point_b)}")

print("\n--- Application 4: Version Management ---")

Version = namedtuple('Version', ['major', 'minor', 'patch'])

class VersionManager:
    def __init__(self):
        self.versions = []
    
    def add_version(self, version_string):
        parts = version_string.split('.')
        version = Version(int(parts[0]), int(parts[1]), int(parts[2]))
        self.versions.append(version)
        return version
    
    def get_latest_version(self):
        if not self.versions:
            return None
        return max(self.versions)
    
    def is_compatible(self, v1: Version, v2: Version) -> bool:
        # Compatible if major version is same and v2 is newer or equal
        return v1.major == v2.major and v2 >= v1

# Example usage
vm = VersionManager()
vm.add_version("1.2.3")
vm.add_version("1.3.0")
vm.add_version("2.0.0")
vm.add_version("1.2.5")

latest = vm.get_latest_version()
print(f"Latest version: {latest}")

v1 = Version(1, 2, 3)
v2 = Version(1, 3, 0)
print(f"Versions {v1} and {v2} compatible: {vm.is_compatible(v1, v2)}")

# ===============================================================================
# 7. INTERVIEW PROBLEMS & SOLUTIONS
# ===============================================================================

print("\n" + "=" * 80)
print("7. INTERVIEW PROBLEMS & SOLUTIONS")
print("=" * 80)

print("\n--- Problem 1: Merge Intervals ---")

def merge_intervals(intervals):
    """Merge overlapping intervals represented as tuples"""
    if not intervals:
        return []
    
    # Sort intervals by start time
    sorted_intervals = sorted(intervals)
    merged = [sorted_intervals[0]]
    
    for current in sorted_intervals[1:]:
        last_merged = merged[-1]
        
        # If current interval overlaps with last merged
        if current[0] <= last_merged[1]:
            # Merge intervals
            merged[-1] = (last_merged[0], max(last_merged[1], current[1]))
        else:
            # No overlap, add current interval
            merged.append(current)
    
    return merged

# Example
intervals = [(1, 3), (2, 6), (8, 10), (15, 18)]
merged = merge_intervals(intervals)
print(f"Original intervals: {intervals}")
print(f"Merged intervals: {merged}")

print("\n--- Problem 2: Find Pairs with Target Sum ---")

def find_pairs_with_sum(numbers, target):
    """Find all pairs of tuples that sum to target"""
    seen = set()
    pairs = []
    
    for num in numbers:
        complement = target - num
        if complement in seen:
            pairs.append((min(num, complement), max(num, complement)))
        seen.add(num)
    
    return list(set(pairs))  # Remove duplicates

# Example
numbers = [1, 2, 3, 4, 5, 6]
target = 7
pairs = find_pairs_with_sum(numbers, target)
print(f"Numbers: {numbers}")
print(f"Pairs that sum to {target}: {pairs}")

print("\n--- Problem 3: Tuple Rotation ---")

def rotate_tuple(tup, positions):
    """Rotate tuple elements by given positions"""
    if not tup:
        return tup
    
    n = len(tup)
    positions = positions % n  # Handle positions > length
    
    return tup[positions:] + tup[:positions]

# Example
original = (1, 2, 3, 4, 5)
rotated_2 = rotate_tuple(original, 2)
rotated_neg1 = rotate_tuple(original, -1)

print(f"Original: {original}")
print(f"Rotated by 2: {rotated_2}")
print(f"Rotated by -1: {rotated_neg1}")

print("\n--- Problem 4: Longest Common Subsequence ---")

def longest_common_subsequence(seq1, seq2):
    """Find longest common subsequence of two tuples"""
    m, n = len(seq1), len(seq2)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Reconstruct LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[i-1] == seq2[j-1]:
            lcs.append(seq1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return tuple(reversed(lcs))

# Example
seq1 = (1, 2, 3, 4, 5)
seq2 = (2, 3, 5, 7, 8)
lcs = longest_common_subsequence(seq1, seq2)
print(f"Sequence 1: {seq1}")
print(f"Sequence 2: {seq2}")
print(f"Longest Common Subsequence: {lcs}")

print("\n--- Problem 5: Tuple Permutations ---")

def generate_permutations(tup):
    """Generate all permutations of a tuple"""
    return list(itertools.permutations(tup))

def unique_permutations(tup):
    """Generate unique permutations (handling duplicates)"""
    return list(set(itertools.permutations(tup)))

# Example
original = (1, 2, 3)
duplicates = (1, 1, 2)

perms = generate_permutations(original)
unique_perms = unique_permutations(duplicates)

print(f"Permutations of {original}: {perms}")
print(f"Unique permutations of {duplicates}: {unique_perms}")

# ===============================================================================
# 8. ADVANCED CONCEPTS FOR EXPERIENCED DEVELOPERS
# ===============================================================================

print("\n" + "=" * 80)
print("8. ADVANCED CONCEPTS FOR EXPERIENCED DEVELOPERS")
print("=" * 80)

print("\n--- Custom Tuple-like Classes ---")

class ImmutableSequence:
    """Custom immutable sequence using tuple internally"""
    
    def __init__(self, *items):
        self._items = tuple(items)
    
    def __getitem__(self, index):
        return self._items[index]
    
    def __len__(self):
        return len(self._items)
    
    def __iter__(self):
        return iter(self._items)
    
    def __repr__(self):
        return f"ImmutableSequence{self._items}"
    
    def __eq__(self, other):
        return isinstance(other, ImmutableSequence) and self._items == other._items
    
    def __hash__(self):
        return hash(self._items)
    
    def append(self, item):
        """Return new instance with item appended"""
        return ImmutableSequence(*self._items, item)
    
    def filter(self, predicate):
        """Return new instance with filtered items"""
        return ImmutableSequence(*filter(predicate, self._items))

# Example usage
seq = ImmutableSequence(1, 2, 3, 4, 5)
print(f"Original sequence: {seq}")

# Functional operations
seq_appended = seq.append(6)
seq_filtered = seq.filter(lambda x: x % 2 == 0)

print(f"After append(6): {seq_appended}")
print(f"After filtering evens: {seq_filtered}")
print(f"Original unchanged: {seq}")

print("\n--- Tuple Metaclasses ---")

class TupleMeta(type):
    """Metaclass for creating tuple-like classes"""
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        fields = kwargs.get('fields', [])
        
        # Add field properties
        for i, field in enumerate(fields):
            namespace[field] = property(lambda self, idx=i: self._data[idx])
        
        # Add __init__ method
        def __init__(self, *args):
            if len(args) != len(fields):
                raise ValueError(f"Expected {len(fields)} arguments, got {len(args)}")
            self._data = tuple(args)
        
        namespace['__init__'] = __init__
        namespace['_fields'] = tuple(fields)
        
        # Add other tuple-like methods
        namespace['__getitem__'] = lambda self, index: self._data[index]
        namespace['__len__'] = lambda self: len(self._data)
        namespace['__iter__'] = lambda self: iter(self._data)
        namespace['__repr__'] = lambda self: f"{name}{self._data}"
        namespace['__eq__'] = lambda self, other: (
            isinstance(other, self.__class__) and self._data == other._data
        )
        namespace['__hash__'] = lambda self: hash(self._data)
        
        return super().__new__(mcs, name, bases, namespace)

# Example usage
class Color(metaclass=TupleMeta, fields=['red', 'green', 'blue']):
    def brightness(self):
        return (self.red + self.green + self.blue) / 3

color = Color(255, 128, 64)
print(f"Color: {color}")
print(f"Red component: {color.red}")
print(f"Brightness: {color.brightness():.2f}")

print("\n--- Tuple Descriptors ---")

class TupleField:
    """Descriptor for tuple field validation"""
    
    def __init__(self, field_type, validator=None):
        self.field_type = field_type
        self.validator = validator
    
    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f'_{name}'
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.private_name)
    
    def __set__(self, instance, value):
        if not isinstance(value, self.field_type):
            raise TypeError(f"{self.name} must be {self.field_type.__name__}")
        
        if self.validator and not self.validator(value):
            raise ValueError(f"Invalid value for {self.name}: {value}")
        
        setattr(instance, self.private_name, value)

class ValidatedPoint:
    """Point class with validated tuple-like behavior"""
    
    x = TupleField(float, lambda v: v >= 0)
    y = TupleField(float, lambda v: v >= 0)
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self._data = (x, y)
    
    def __getitem__(self, index):
        return self._data[index]
    
    def __repr__(self):
        return f"ValidatedPoint({self.x}, {self.y})"

# Example usage
try:
    point = ValidatedPoint(3.0, 4.0)
    print(f"Valid point: {point}")
    
    # This will raise an error
    invalid_point = ValidatedPoint(-1.0, 2.0)
except ValueError as e:
    print(f"Validation error: {e}")

print("\n--- Functional Programming with Tuples ---")

# Tuple as a functor
def tuple_map(func, tup):
    """Apply function to each element of tuple"""
    return tuple(func(x) for x in tup)

def tuple_reduce(func, tup, initial=None):
    """Reduce tuple to single value"""
    return functools.reduce(func, tup, initial) if initial is not None else functools.reduce(func, tup)

def tuple_filter(predicate, tup):
    """Filter tuple elements"""
    return tuple(x for x in tup if predicate(x))

# Example functional operations
numbers = (1, 2, 3, 4, 5)

squared = tuple_map(lambda x: x**2, numbers)
sum_all = tuple_reduce(lambda x, y: x + y, numbers)
evens = tuple_filter(lambda x: x % 2 == 0, numbers)

print(f"Original: {numbers}")
print(f"Squared: {squared}")
print(f"Sum: {sum_all}")
print(f"Evens: {evens}")

# Monadic operations
def tuple_bind(tup, func):
    """Monadic bind for tuples"""
    result = []
    for item in tup:
        result.extend(func(item))
    return tuple(result)

# Example: expand each number to (number, number^2)
expanded = tuple_bind((1, 2, 3), lambda x: [x, x**2])
print(f"Expanded: {expanded}")

# ===============================================================================
# 9. SYSTEM DESIGN WITH TUPLES
# ===============================================================================

print("\n" + "=" * 80)
print("9. SYSTEM DESIGN WITH TUPLES")
print("=" * 80)

print("\n--- Event Sourcing with Tuples ---")

Event = namedtuple('Event', ['timestamp', 'event_type', 'entity_id', 'data'])

class EventStore:
    def __init__(self):
        self.events = []
        self.snapshots = {}
    
    def append_event(self, event):
        """Append event to store"""
        self.events.append(event)
    
    def get_events_for_entity(self, entity_id, after_timestamp=None):
        """Get all events for specific entity"""
        events = [e for e in self.events if e.entity_id == entity_id]
        if after_timestamp:
            events = [e for e in events if e.timestamp > after_timestamp]
        return events
    
    def create_snapshot(self, entity_id, state, timestamp):
        """Create snapshot of entity state"""
        self.snapshots[entity_id] = (timestamp, state)
    
    def get_latest_snapshot(self, entity_id):
        """Get latest snapshot for entity"""
        return self.snapshots.get(entity_id)

# Example usage
store = EventStore()

# Add events
events = [
    Event('2025-01-01T10:00:00', 'user_created', 'user_123', {'name': 'Alice', 'email': 'alice@example.com'}),
    Event('2025-01-01T10:05:00', 'email_changed', 'user_123', {'old_email': 'alice@example.com', 'new_email': 'alice.smith@example.com'}),
    Event('2025-01-01T10:10:00', 'user_deactivated', 'user_123', {'reason': 'user_request'})
]

for event in events:
    store.append_event(event)

user_events = store.get_events_for_entity('user_123')
print(f"Events for user_123: {len(user_events)}")
for event in user_events:
    print(f"  {event.timestamp}: {event.event_type}")

print("\n--- Configuration Management System ---")

ConfigValue = namedtuple('ConfigValue', ['key', 'value', 'environment', 'priority'])

class ConfigManager:
    def __init__(self):
        self.configs = []
        self.environment_priority = {
            'default': 0,
            'development': 1,
            'staging': 2,
            'production': 3
        }
    
    def add_config(self, key, value, environment='default', priority=None):
        """Add configuration value"""
        if priority is None:
            priority = self.environment_priority.get(environment, 0)
        
        config = ConfigValue(key, value, environment, priority)
        self.configs.append(config)
    
    def get_config(self, key, environment='production'):
        """Get configuration value for specific environment"""
        # Find all configs for this key
        matching_configs = [c for c in self.configs if c.key == key]
        
        if not matching_configs:
            return None
        
        # Filter by environment or lower priority environments
        applicable_configs = [
            c for c in matching_configs 
            if self.environment_priority.get(c.environment, 0) <= self.environment_priority.get(environment, 0)
        ]
        
        if not applicable_configs:
            return None
        
        # Return highest priority configuration
        return max(applicable_configs, key=lambda c: c.priority).value

# Example usage
config_mgr = ConfigManager()

# Add configurations
config_mgr.add_config('database_url', 'localhost:5432', 'default')
config_mgr.add_config('database_url', 'dev.db.com:5432', 'development')
config_mgr.add_config('database_url', 'prod.db.com:5432', 'production')
config_mgr.add_config('api_timeout', 30, 'default')
config_mgr.add_config('api_timeout', 5, 'development')

# Get configurations
prod_db = config_mgr.get_config('database_url', 'production')
dev_timeout = config_mgr.get_config('api_timeout', 'development')

print(f"Production database URL: {prod_db}")
print(f"Development API timeout: {dev_timeout}")

print("\n--- Caching System with Tuples ---")

CacheEntry = namedtuple('CacheEntry', ['key', 'value', 'timestamp', 'ttl'])

class TTLCache:
    def __init__(self):
        self.cache = {}
        self.access_order = []
    
    def put(self, key, value, ttl=3600):
        """Put value in cache with TTL"""
        timestamp = time.time()
        entry = CacheEntry(key, value, timestamp, ttl)
        self.cache[key] = entry
        
        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def get(self, key):
        """Get value from cache"""
        entry = self.cache.get(key)
        if not entry:
            return None
        
        # Check if expired
        if time.time() - entry.timestamp > entry.ttl:
            del self.cache[key]
            self.access_order.remove(key)
            return None
        
        # Update access order
        self.access_order.remove(key)
        self.access_order.append(key)
        
        return entry.value
    
    def cleanup_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry.timestamp > entry.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
            self.access_order.remove(key)
        
        return len(expired_keys)

# Example usage
cache = TTLCache()
cache.put('user_123', {'name': 'Alice', 'role': 'admin'}, ttl=60)
cache.put('user_456', {'name': 'Bob', 'role': 'user'}, ttl=30)

user_data = cache.get('user_123')
print(f"Cached user data: {user_data}")

# Simulate time passing and cleanup
time.sleep(0.1)
expired_count = cache.cleanup_expired()
print(f"Expired entries cleaned: {expired_count}")

print("\n" + "=" * 80)
print("TUPLE LEARNING GUIDE PART 2 COMPLETE")
print("Continue with tuple_learning_part3.py for remaining sections...")
print("=" * 80)
