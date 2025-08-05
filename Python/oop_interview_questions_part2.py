# ===============================================================================
# PYTHON OOP INTERVIEW QUESTIONS - PART 2
# Advanced Questions for Experienced Python Developers
# ===============================================================================

"""
ADVANCED OOP INTERVIEW COVERAGE - PART 2:
=========================================
6. Architecture & System Design Questions
7. Performance & Optimization Questions
8. Advanced OOP Concepts
9. Real-World Scenario Questions
10. Code Review & Best Practices
11. Threading & Concurrency with OOP
12. Testing & Mock Objects
"""

import time
import threading
import asyncio
import weakref
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Union, Protocol, Generic, TypeVar
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps, lru_cache
from collections import defaultdict, namedtuple
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch, MagicMock

print("=" * 100)
print("PYTHON OOP INTERVIEW QUESTIONS - PART 2 (ADVANCED)")
print("=" * 100)

# ===============================================================================
# 6. ARCHITECTURE & SYSTEM DESIGN QUESTIONS
# ===============================================================================

print("\n" + "=" * 80)
print("6. ARCHITECTURE & SYSTEM DESIGN QUESTIONS")
print("=" * 80)

print("""
Q14: Design a cache system that supports multiple eviction policies (LRU, LFU, TTL).
""")

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from collections import OrderedDict
import time
import threading

class EvictionPolicy(ABC):
    """Abstract base class for cache eviction policies"""
    
    @abstractmethod
    def on_get(self, key: str) -> None:
        """Called when a key is accessed"""
        pass
    
    @abstractmethod
    def on_put(self, key: str) -> None:
        """Called when a key is inserted"""
        pass
    
    @abstractmethod
    def evict(self, cache_data: Dict[str, Any], max_size: int) -> Optional[str]:
        """Return key to evict, or None if no eviction needed"""
        pass

class LRUPolicy(EvictionPolicy):
    """Least Recently Used eviction policy"""
    
    def __init__(self):
        self.access_order = OrderedDict()
    
    def on_get(self, key: str) -> None:
        if key in self.access_order:
            self.access_order.move_to_end(key)
    
    def on_put(self, key: str) -> None:
        self.access_order[key] = time.time()
    
    def evict(self, cache_data: Dict[str, Any], max_size: int) -> Optional[str]:
        if len(cache_data) >= max_size and self.access_order:
            # Remove least recently used (first item)
            lru_key = next(iter(self.access_order))
            del self.access_order[lru_key]
            return lru_key
        return None

class LFUPolicy(EvictionPolicy):
    """Least Frequently Used eviction policy"""
    
    def __init__(self):
        self.frequency = defaultdict(int)
        self.access_time = {}
    
    def on_get(self, key: str) -> None:
        self.frequency[key] += 1
        self.access_time[key] = time.time()
    
    def on_put(self, key: str) -> None:
        self.frequency[key] = 1
        self.access_time[key] = time.time()
    
    def evict(self, cache_data: Dict[str, Any], max_size: int) -> Optional[str]:
        if len(cache_data) >= max_size and self.frequency:
            # Find key with minimum frequency, break ties by access time
            lfu_key = min(self.frequency.keys(), 
                         key=lambda k: (self.frequency[k], self.access_time.get(k, 0)))
            del self.frequency[lfu_key]
            self.access_time.pop(lfu_key, None)
            return lfu_key
        return None

class TTLPolicy(EvictionPolicy):
    """Time To Live eviction policy"""
    
    def __init__(self, default_ttl: float = 300):  # 5 minutes default
        self.default_ttl = default_ttl
        self.expiry_times = {}
    
    def on_get(self, key: str) -> None:
        # Check if key has expired
        if key in self.expiry_times and time.time() > self.expiry_times[key]:
            del self.expiry_times[key]
    
    def on_put(self, key: str, ttl: Optional[float] = None) -> None:
        ttl = ttl or self.default_ttl
        self.expiry_times[key] = time.time() + ttl
    
    def evict(self, cache_data: Dict[str, Any], max_size: int) -> Optional[str]:
        # First, remove any expired keys
        current_time = time.time()
        expired_keys = [k for k, expiry in self.expiry_times.items() 
                       if current_time > expiry and k in cache_data]
        
        if expired_keys:
            # Return first expired key
            expired_key = expired_keys[0]
            del self.expiry_times[expired_key]
            return expired_key
        
        return None

class SmartCache:
    """Advanced cache with pluggable eviction policies"""
    
    def __init__(self, max_size: int = 100, eviction_policy: EvictionPolicy = None):
        self.max_size = max_size
        self.policy = eviction_policy or LRUPolicy()
        self._data = {}
        self._lock = threading.RLock()  # Thread-safe
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def get(self, key: str) -> Any:
        """Get value from cache"""
        with self._lock:
            if key in self._data:
                self.policy.on_get(key)
                self._stats['hits'] += 1
                return self._data[key]
            else:
                self._stats['misses'] += 1
                return None
    
    def put(self, key: str, value: Any, **kwargs) -> None:
        """Put value in cache"""
        with self._lock:
            # Check if eviction is needed
            evict_key = self.policy.evict(self._data, self.max_size)
            if evict_key and evict_key in self._data:
                del self._data[evict_key]
                self._stats['evictions'] += 1
            
            # Add new key
            self._data[key] = value
            
            # Handle TTL policy special case
            if isinstance(self.policy, TTLPolicy):
                self.policy.on_put(key, kwargs.get('ttl'))
            else:
                self.policy.on_put(key)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self._data)
    
    def clear(self) -> None:
        """Clear all cache data"""
        with self._lock:
            self._data.clear()
            # Reset policy state
            if hasattr(self.policy, 'access_order'):
                self.policy.access_order.clear()
            if hasattr(self.policy, 'frequency'):
                self.policy.frequency.clear()
            if hasattr(self.policy, 'expiry_times'):
                self.policy.expiry_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self._stats,
            'hit_rate': hit_rate,
            'size': self.size(),
            'max_size': self.max_size
        }
    
    def __repr__(self):
        return f"SmartCache(size={self.size()}, policy={self.policy.__class__.__name__})"

print("Testing cache system with different policies:")

# Test LRU Cache
print("\n--- Testing LRU Cache ---")
lru_cache = SmartCache(max_size=3, eviction_policy=LRUPolicy())

# Fill cache
for i in range(5):
    lru_cache.put(f"key{i}", f"value{i}")
    print(f"Added key{i}, cache size: {lru_cache.size()}")

# Test access pattern
lru_cache.get("key2")  # Make key2 most recently used
lru_cache.put("key5", "value5")  # Should evict least recently used

print(f"LRU Cache stats: {lru_cache.get_stats()}")

# Test LFU Cache
print("\n--- Testing LFU Cache ---")
lfu_cache = SmartCache(max_size=3, eviction_policy=LFUPolicy())

lfu_cache.put("A", "valueA")
lfu_cache.put("B", "valueB")
lfu_cache.put("C", "valueC")

# Access pattern: A=3 times, B=2 times, C=1 time
for _ in range(3):
    lfu_cache.get("A")
for _ in range(2):
    lfu_cache.get("B")
lfu_cache.get("C")

lfu_cache.put("D", "valueD")  # Should evict C (least frequent)
print(f"After adding D: {[k for k in lfu_cache._data.keys()]}")

# Test TTL Cache
print("\n--- Testing TTL Cache ---")
ttl_cache = SmartCache(max_size=5, eviction_policy=TTLPolicy(default_ttl=2))

ttl_cache.put("temp1", "value1", ttl=1)  # Expires in 1 second
ttl_cache.put("temp2", "value2", ttl=3)  # Expires in 3 seconds

print(f"Before expiry: temp1={ttl_cache.get('temp1')}, temp2={ttl_cache.get('temp2')}")

time.sleep(1.5)  # Wait for temp1 to expire

print(f"After 1.5s: temp1={ttl_cache.get('temp1')}, temp2={ttl_cache.get('temp2')}")

print("""
Q15: Design a plugin system with dynamic loading and dependency management.
""")

class PluginManager:
    """Advanced plugin system with dependency management"""
    
    def __init__(self):
        self._plugins = {}
        self._loaded_plugins = {}
        self._dependency_graph = {}
        self._hooks = defaultdict(list)
    
    def register_plugin(self, plugin_class):
        """Register a plugin class"""
        plugin_name = plugin_class.get_name()
        self._plugins[plugin_name] = plugin_class
        self._dependency_graph[plugin_name] = plugin_class.get_dependencies()
        print(f"Registered plugin: {plugin_name}")
    
    def load_plugin(self, plugin_name: str) -> bool:
        """Load a plugin and its dependencies"""
        if plugin_name in self._loaded_plugins:
            print(f"Plugin {plugin_name} already loaded")
            return True
        
        if plugin_name not in self._plugins:
            print(f"Plugin {plugin_name} not found")
            return False
        
        # Load dependencies first
        dependencies = self._dependency_graph.get(plugin_name, [])
        for dep in dependencies:
            if not self.load_plugin(dep):
                print(f"Failed to load dependency {dep} for {plugin_name}")
                return False
        
        # Load the plugin
        try:
            plugin_class = self._plugins[plugin_name]
            plugin_instance = plugin_class()
            
            # Initialize plugin
            plugin_instance.initialize(self)
            
            self._loaded_plugins[plugin_name] = plugin_instance
            print(f"Loaded plugin: {plugin_name}")
            
            # Register plugin hooks
            hooks = plugin_instance.get_hooks()
            for hook_name, callback in hooks.items():
                self._hooks[hook_name].append((plugin_name, callback))
            
            return True
            
        except Exception as e:
            print(f"Error loading plugin {plugin_name}: {e}")
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""
        if plugin_name not in self._loaded_plugins:
            print(f"Plugin {plugin_name} not loaded")
            return False
        
        # Check if other plugins depend on this one
        dependents = [name for name, deps in self._dependency_graph.items() 
                     if plugin_name in deps and name in self._loaded_plugins]
        
        if dependents:
            print(f"Cannot unload {plugin_name}: required by {dependents}")
            return False
        
        # Cleanup plugin
        plugin = self._loaded_plugins[plugin_name]
        plugin.cleanup()
        
        # Remove hooks
        for hook_name, callbacks in self._hooks.items():
            self._hooks[hook_name] = [(name, cb) for name, cb in callbacks 
                                    if name != plugin_name]
        
        del self._loaded_plugins[plugin_name]
        print(f"Unloaded plugin: {plugin_name}")
        return True
    
    def execute_hook(self, hook_name: str, *args, **kwargs):
        """Execute all callbacks registered for a hook"""
        results = {}
        for plugin_name, callback in self._hooks.get(hook_name, []):
            try:
                result = callback(*args, **kwargs)
                results[plugin_name] = result
            except Exception as e:
                print(f"Error in {plugin_name} hook {hook_name}: {e}")
                results[plugin_name] = None
        return results
    
    def get_loaded_plugins(self) -> List[str]:
        """Get list of loaded plugin names"""
        return list(self._loaded_plugins.keys())
    
    def get_plugin(self, plugin_name: str):
        """Get loaded plugin instance"""
        return self._loaded_plugins.get(plugin_name)

class BasePlugin(ABC):
    """Base class for all plugins"""
    
    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Return unique plugin name"""
        pass
    
    @classmethod
    def get_dependencies(cls) -> List[str]:
        """Return list of required plugin names"""
        return []
    
    @abstractmethod
    def initialize(self, plugin_manager):
        """Initialize the plugin"""
        pass
    
    def cleanup(self):
        """Cleanup when plugin is unloaded"""
        pass
    
    def get_hooks(self) -> Dict[str, callable]:
        """Return dictionary of hook_name -> callback"""
        return {}

# Example plugins
class LoggingPlugin(BasePlugin):
    @classmethod
    def get_name(cls):
        return "logging"
    
    def initialize(self, plugin_manager):
        self.log_file = "app.log"
        print(f"Logging plugin initialized, logging to {self.log_file}")
    
    def get_hooks(self):
        return {
            'app_start': self.log_app_start,
            'user_action': self.log_user_action,
            'app_stop': self.log_app_stop
        }
    
    def log_app_start(self):
        print(f"[LOG] Application started at {datetime.now()}")
    
    def log_user_action(self, action, user):
        print(f"[LOG] User {user} performed action: {action}")
    
    def log_app_stop(self):
        print(f"[LOG] Application stopped at {datetime.now()}")

class DatabasePlugin(BasePlugin):
    @classmethod
    def get_name(cls):
        return "database"
    
    @classmethod
    def get_dependencies(cls):
        return ["logging"]  # Depends on logging plugin
    
    def initialize(self, plugin_manager):
        self.connection = "mock_db_connection"
        print(f"Database plugin initialized, connected to {self.connection}")
    
    def get_hooks(self):
        return {
            'user_action': self.save_user_action
        }
    
    def save_user_action(self, action, user):
        print(f"[DB] Saving action '{action}' for user '{user}' to database")

class AuthenticationPlugin(BasePlugin):
    @classmethod
    def get_name(cls):
        return "auth"
    
    @classmethod
    def get_dependencies(cls):
        return ["database", "logging"]
    
    def initialize(self, plugin_manager):
        self.active_sessions = {}
        print("Authentication plugin initialized")
    
    def get_hooks(self):
        return {
            'user_login': self.handle_login,
            'user_logout': self.handle_logout
        }
    
    def handle_login(self, username, password):
        # Mock authentication
        if username and password:
            session_id = f"session_{len(self.active_sessions) + 1}"
            self.active_sessions[session_id] = username
            print(f"[AUTH] User {username} logged in, session: {session_id}")
            return session_id
        return None
    
    def handle_logout(self, session_id):
        if session_id in self.active_sessions:
            username = self.active_sessions.pop(session_id)
            print(f"[AUTH] User {username} logged out")

print("Testing plugin system:")

# Create plugin manager
pm = PluginManager()

# Register plugins
pm.register_plugin(LoggingPlugin)
pm.register_plugin(DatabasePlugin)
pm.register_plugin(AuthenticationPlugin)

# Load plugins (dependencies will be loaded automatically)
pm.load_plugin("auth")

print(f"\nLoaded plugins: {pm.get_loaded_plugins()}")

# Test hooks
print("\nTesting application flow with hooks:")
pm.execute_hook('app_start')

session = pm.execute_hook('user_login', 'john_doe', 'password123')
print(f"Login result: {session}")

pm.execute_hook('user_action', 'view_profile', 'john_doe')
pm.execute_hook('user_action', 'update_settings', 'john_doe')

pm.execute_hook('user_logout', 'session_1')
pm.execute_hook('app_stop')

# Test unloading
print(f"\nUnloading database plugin:")
pm.unload_plugin("database")  # Should fail due to auth dependency

print(f"Unloading auth plugin first:")
pm.unload_plugin("auth")
pm.unload_plugin("database")  # Should succeed now

# ===============================================================================
# 7. PERFORMANCE & OPTIMIZATION QUESTIONS
# ===============================================================================

print("\n" + "=" * 80)
print("7. PERFORMANCE & OPTIMIZATION QUESTIONS")
print("=" * 80)

print("""
Q16: Implement an object pool pattern for expensive-to-create objects with metrics.
""")

import time
import threading
from queue import Queue, Empty
from typing import TypeVar, Generic, Callable, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

T = TypeVar('T')

@dataclass
class PoolMetrics:
    """Metrics for object pool performance"""
    created_objects: int = 0
    destroyed_objects: int = 0
    borrowed_objects: int = 0
    returned_objects: int = 0
    pool_hits: int = 0
    pool_misses: int = 0
    current_pool_size: int = 0
    peak_pool_size: int = 0
    average_borrow_time: float = 0.0
    total_borrow_time: float = 0.0

class ObjectPool(Generic[T]):
    """High-performance object pool with metrics and validation"""
    
    def __init__(self, 
                 factory: Callable[[], T],
                 reset_func: Optional[Callable[[T], None]] = None,
                 validator: Optional[Callable[[T], bool]] = None,
                 max_size: int = 10,
                 min_size: int = 2,
                 max_idle_time: float = 300):  # 5 minutes
        
        self._factory = factory
        self._reset_func = reset_func
        self._validator = validator
        self._max_size = max_size
        self._min_size = min_size
        self._max_idle_time = max_idle_time
        
        self._pool = Queue(maxsize=max_size)
        self._borrowed = set()
        self._idle_times = {}
        self._lock = threading.RLock()
        self._metrics = PoolMetrics()
        
        # Pre-populate with minimum objects
        self._populate_pool()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _populate_pool(self):
        """Pre-populate pool with minimum objects"""
        with self._lock:
            current_size = self._pool.qsize()
            needed = self._min_size - current_size
            
            for _ in range(needed):
                if current_size < self._max_size:
                    obj = self._create_object()
                    self._pool.put((obj, time.time()))
                    current_size += 1
    
    def _create_object(self) -> T:
        """Create new object using factory"""
        obj = self._factory()
        self._metrics.created_objects += 1
        self._metrics.current_pool_size += 1
        self._metrics.peak_pool_size = max(self._metrics.peak_pool_size, 
                                         self._metrics.current_pool_size)
        return obj
    
    def _validate_object(self, obj: T) -> bool:
        """Validate object is still usable"""
        if self._validator:
            return self._validator(obj)
        return True
    
    def _reset_object(self, obj: T):
        """Reset object to clean state"""
        if self._reset_func:
            self._reset_func(obj)
    
    def borrow(self, timeout: float = 1.0) -> Optional[T]:
        """Borrow object from pool"""
        start_time = time.time()
        
        try:
            # Try to get from pool
            try:
                obj, idle_since = self._pool.get(timeout=timeout)
                self._metrics.pool_hits += 1
                
                # Validate object
                if not self._validate_object(obj):
                    # Object is invalid, create new one
                    self._metrics.destroyed_objects += 1
                    obj = self._create_object()
                
            except Empty:
                # Pool is empty, create new object
                self._metrics.pool_misses += 1
                obj = self._create_object()
            
            # Track borrowed object
            with self._lock:
                self._borrowed.add(id(obj))
                self._idle_times.pop(id(obj), None)
            
            # Update metrics
            borrow_time = time.time() - start_time
            self._metrics.borrowed_objects += 1
            self._metrics.total_borrow_time += borrow_time
            self._metrics.average_borrow_time = (
                self._metrics.total_borrow_time / self._metrics.borrowed_objects
            )
            
            return obj
            
        except Exception as e:
            print(f"Error borrowing object: {e}")
            return None
    
    def return_object(self, obj: T):
        """Return object to pool"""
        if obj is None:
            return
        
        obj_id = id(obj)
        
        with self._lock:
            if obj_id not in self._borrowed:
                print("Warning: Returning object that wasn't borrowed from this pool")
                return
            
            self._borrowed.remove(obj_id)
        
        # Reset object to clean state
        try:
            self._reset_object(obj)
        except Exception as e:
            print(f"Error resetting object: {e}")
            # Destroy corrupted object
            self._metrics.destroyed_objects += 1
            self._metrics.current_pool_size -= 1
            return
        
        # Return to pool if there's space
        try:
            current_time = time.time()
            self._pool.put((obj, current_time), block=False)
            self._idle_times[obj_id] = current_time
            self._metrics.returned_objects += 1
            
        except:
            # Pool is full, destroy object
            self._metrics.destroyed_objects += 1
            self._metrics.current_pool_size -= 1
    
    def _cleanup_worker(self):
        """Background thread to cleanup idle objects"""
        while True:
            try:
                time.sleep(60)  # Check every minute
                self._cleanup_idle_objects()
            except Exception as e:
                print(f"Cleanup error: {e}")
    
    def _cleanup_idle_objects(self):
        """Remove objects that have been idle too long"""
        current_time = time.time()
        cleanup_candidates = []
        
        # Collect idle objects
        while not self._pool.empty():
            try:
                obj, idle_since = self._pool.get(block=False)
                if current_time - idle_since > self._max_idle_time:
                    cleanup_candidates.append(obj)
                else:
                    # Put back if not too old
                    self._pool.put((obj, idle_since))
            except Empty:
                break
        
        # Clean up old objects but maintain minimum
        current_pool_size = self._pool.qsize()
        can_remove = max(0, current_pool_size - self._min_size)
        
        for i, obj in enumerate(cleanup_candidates):
            if i < can_remove:
                # Destroy object
                self._metrics.destroyed_objects += 1
                self._metrics.current_pool_size -= 1
                self._idle_times.pop(id(obj), None)
            else:
                # Keep object
                self._pool.put((obj, current_time))
    
    def get_metrics(self) -> PoolMetrics:
        """Get pool performance metrics"""
        self._metrics.current_pool_size = self._pool.qsize() + len(self._borrowed)
        return self._metrics
    
    def close(self):
        """Close pool and cleanup resources"""
        # Clear pool
        while not self._pool.empty():
            try:
                obj, _ = self._pool.get(block=False)
                self._metrics.destroyed_objects += 1
            except Empty:
                break
        
        self._borrowed.clear()
        self._idle_times.clear()

# Example: Database connection pool
class ExpensiveDatabase:
    """Simulates expensive-to-create database connection"""
    
    def __init__(self):
        print("Creating expensive database connection...")
        time.sleep(0.1)  # Simulate connection time
        self.connection_id = f"conn_{id(self)}"
        self.is_valid = True
        self.query_count = 0
    
    def execute_query(self, query: str):
        """Execute database query"""
        if not self.is_valid:
            raise RuntimeError("Connection is invalid")
        
        self.query_count += 1
        time.sleep(0.01)  # Simulate query time
        return f"Result for: {query} (conn: {self.connection_id})"
    
    def reset(self):
        """Reset connection state"""
        self.query_count = 0
        print(f"Reset connection {self.connection_id}")
    
    def is_healthy(self) -> bool:
        """Check if connection is still valid"""
        return self.is_valid and self.query_count < 100  # Expire after 100 queries

print("Testing object pool with database connections:")

# Create pool
db_pool = ObjectPool(
    factory=ExpensiveDatabase,
    reset_func=lambda db: db.reset(),
    validator=lambda db: db.is_healthy(),
    max_size=5,
    min_size=2,
    max_idle_time=10
)

def simulate_database_work(pool, worker_id: int):
    """Simulate database work by a worker"""
    for i in range(5):
        # Borrow connection
        conn = pool.borrow()
        if conn:
            try:
                # Use connection
                result = conn.execute_query(f"SELECT * FROM table_{worker_id}_{i}")
                print(f"Worker {worker_id}: {result}")
                time.sleep(0.05)  # Simulate processing
            finally:
                # Always return connection
                pool.return_object(conn)
        else:
            print(f"Worker {worker_id}: Failed to get connection")

# Simulate concurrent usage
print("\nSimulating concurrent database access:")
workers = []
for i in range(3):
    worker = threading.Thread(target=simulate_database_work, args=(db_pool, i))
    workers.append(worker)
    worker.start()

# Wait for workers to complete
for worker in workers:
    worker.join()

# Show metrics
metrics = db_pool.get_metrics()
print(f"\nPool Metrics:")
print(f"Created objects: {metrics.created_objects}")
print(f"Destroyed objects: {metrics.destroyed_objects}")
print(f"Borrowed objects: {metrics.borrowed_objects}")
print(f"Returned objects: {metrics.returned_objects}")
print(f"Pool hits: {metrics.pool_hits}")
print(f"Pool misses: {metrics.pool_misses}")
print(f"Average borrow time: {metrics.average_borrow_time:.4f}s")
print(f"Current pool size: {metrics.current_pool_size}")
print(f"Peak pool size: {metrics.peak_pool_size}")

# Cleanup
db_pool.close()

print("""
Q17: Implement a lazy evaluation system for expensive computations.
""")

from functools import wraps
from typing import Any, Callable, Optional
import weakref

class LazyProperty:
    """Descriptor that computes value only when first accessed"""
    
    def __init__(self, func: Callable):
        self.func = func
        self.name = func.__name__
        self.__doc__ = func.__doc__
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        
        # Check if value is already computed
        attr_name = f'_lazy_{self.name}'
        if hasattr(obj, attr_name):
            return getattr(obj, attr_name)
        
        # Compute and cache value
        print(f"Computing lazy property: {self.name}")
        value = self.func(obj)
        setattr(obj, attr_name, value)
        return value
    
    def __set__(self, obj, value):
        # Allow setting to override lazy computation
        attr_name = f'_lazy_{self.name}'
        setattr(obj, attr_name, value)
    
    def __delete__(self, obj):
        # Allow deletion to force recomputation
        attr_name = f'_lazy_{self.name}'
        if hasattr(obj, attr_name):
            delattr(obj, attr_name)

class LazyComputation:
    """Represents a computation that's deferred until needed"""
    
    def __init__(self, func: Callable, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._computed = False
        self._result = None
        self._error = None
    
    def compute(self):
        """Force computation and return result"""
        if not self._computed:
            try:
                print(f"Executing lazy computation: {self.func.__name__}")
                self._result = self.func(*self.args, **self.kwargs)
            except Exception as e:
                self._error = e
            finally:
                self._computed = True
        
        if self._error:
            raise self._error
        
        return self._result
    
    def is_computed(self) -> bool:
        """Check if computation has been performed"""
        return self._computed
    
    def __repr__(self):
        status = "computed" if self._computed else "pending"
        return f"LazyComputation({self.func.__name__}, {status})"

def lazy(func):
    """Decorator to make function calls lazy"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return LazyComputation(func, *args, **kwargs)
    return wrapper

class DataProcessor:
    """Example class using lazy evaluation"""
    
    def __init__(self, data: List[int]):
        self.data = data
        self._cache = {}
    
    @LazyProperty
    def mean(self) -> float:
        """Lazy computation of mean"""
        time.sleep(0.1)  # Simulate expensive computation
        return sum(self.data) / len(self.data)
    
    @LazyProperty
    def variance(self) -> float:
        """Lazy computation of variance (depends on mean)"""
        time.sleep(0.1)  # Simulate expensive computation
        mean_val = self.mean  # This will use cached value if available
        return sum((x - mean_val) ** 2 for x in self.data) / len(self.data)
    
    @LazyProperty
    def std_deviation(self) -> float:
        """Lazy computation of standard deviation"""
        time.sleep(0.1)  # Simulate expensive computation
        return self.variance ** 0.5
    
    @lazy
    def expensive_analysis(self, complexity: int) -> Dict[str, Any]:
        """Expensive analysis that's lazily computed"""
        time.sleep(complexity * 0.1)  # Simulate computation time
        
        result = {
            'mean': self.mean,
            'variance': self.variance,
            'std_dev': self.std_deviation,
            'complexity': complexity,
            'computed_at': datetime.now()
        }
        
        return result

print("Testing lazy evaluation system:")

# Create data processor
data = list(range(1, 101))  # 1 to 100
processor = DataProcessor(data)

print("\n--- Testing Lazy Properties ---")
print("Created processor, no computations yet")

# Access properties - should trigger computation
print(f"Mean: {processor.mean}")  # Computes mean
print(f"Variance: {processor.variance}")  # Computes variance (reuses mean)
print(f"Standard deviation: {processor.std_deviation}")  # Computes std_dev (reuses variance)

# Second access should use cached values
print("\nSecond access (should use cache):")
print(f"Mean: {processor.mean}")
print(f"Variance: {processor.variance}")

print("\n--- Testing Lazy Computations ---")

# Create lazy computations
analysis1 = processor.expensive_analysis(1)
analysis2 = processor.expensive_analysis(3)

print(f"Created lazy computations: {analysis1}, {analysis2}")
print(f"Analysis1 computed? {analysis1.is_computed()}")

# Force computation
result1 = analysis1.compute()
print(f"Analysis1 result: {result1}")
print(f"Analysis1 computed? {analysis1.is_computed()}")

# Second call should return cached result
print("\nSecond call to analysis1.compute() (cached):")
result1_cached = analysis1.compute()

class LazyList:
    """Lazy list that generates elements on demand"""
    
    def __init__(self, generator_func: Callable[[int], Any], size: Optional[int] = None):
        self.generator_func = generator_func
        self.size = size
        self._cache = {}
    
    def __getitem__(self, index: int):
        if self.size is not None and index >= self.size:
            raise IndexError("Index out of range")
        
        if index not in self._cache:
            print(f"Generating element at index {index}")
            self._cache[index] = self.generator_func(index)
        
        return self._cache[index]
    
    def __len__(self):
        if self.size is None:
            raise TypeError("LazyList with unlimited size has no len()")
        return self.size
    
    def __iter__(self):
        if self.size is None:
            # Infinite iterator
            index = 0
            while True:
                yield self[index]
                index += 1
        else:
            # Finite iterator
            for i in range(self.size):
                yield self[i]

def expensive_fibonacci(n: int) -> int:
    """Expensive Fibonacci computation"""
    time.sleep(0.01)  # Simulate computation time
    if n <= 1:
        return n
    return expensive_fibonacci(n - 1) + expensive_fibonacci(n - 2)

print("\n--- Testing Lazy List ---")

# Create lazy Fibonacci sequence
fib_lazy = LazyList(expensive_fibonacci, 10)

print("Created lazy Fibonacci list, no computations yet")

# Access specific elements
print(f"fib[5] = {fib_lazy[5]}")  # Computes only up to index 5
print(f"fib[3] = {fib_lazy[3]}")  # Uses cached value
print(f"fib[7] = {fib_lazy[7]}")  # Computes only index 7

# Iterate through list
print("\nIterating through first 8 elements:")
for i, value in enumerate(fib_lazy):
    if i >= 8:
        break
    print(f"fib[{i}] = {value}")

print("\n" + "=" * 80)
print("PART 2 COMPLETE - ARCHITECTURE & PERFORMANCE COVERED!")
print("=" * 80)
print("""
🎯 ADVANCED TOPICS COVERED IN PART 2:
✅ Architecture & System Design (Cache Systems, Plugin Architecture)
✅ Performance & Optimization (Object Pools, Lazy Evaluation)

📝 KEY ADVANCED CONCEPTS:
- Multi-policy cache systems (LRU, LFU, TTL)
- Plugin systems with dependency management
- Object pooling with metrics and validation
- Lazy evaluation patterns and descriptors
- Thread-safe implementations
- Performance monitoring and metrics

🚀 CONTINUE WITH PART 3: Threading, Testing, and Real-world Scenarios
""")
