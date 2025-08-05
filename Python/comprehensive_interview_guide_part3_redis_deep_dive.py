"""
COMPREHENSIVE INTERVIEW GUIDE - PART 3: REDIS DEEP DIVE
======================================================

This guide covers Redis fundamentals, data structures, persistence, clustering, and security.
Complete coverage of Redis as cache, session store, message broker, and database.

Author: Interview Preparation Guide
Date: August 2025
Technologies: Redis, Caching, Session Management, Pub/Sub
"""

# ===================================
# PART 3: REDIS FUNDAMENTALS
# ===================================

def redis_fundamentals_interview_questions():
    """
    Comprehensive Redis interview questions covering all Redis concepts
    """
    
    print("=" * 80)
    print("REDIS FUNDAMENTALS - INTERVIEW QUESTIONS & ANSWERS")
    print("=" * 80)
    
    questions_and_answers = [
        {
            "question": "1. How does Redis differ from traditional databases? Explain Redis data structures.",
            "answer": """
Redis is an in-memory data structure store that differs significantly from traditional databases.

REDIS vs TRADITIONAL DATABASES:

| Feature | Redis | Traditional DB (PostgreSQL/MySQL) |
|---------|-------|-----------------------------------|
| Storage | In-memory (RAM) | Disk-based |
| Speed | Extremely fast (μs) | Slower (ms) |
| Data Model | Key-value with structures | Relational (tables/rows) |
| Persistence | Optional (AOF/RDB) | Persistent by default |
| Queries | Simple key-based | Complex SQL queries |
| Transactions | Limited ACID | Full ACID compliance |
| Scalability | Horizontal (clustering) | Vertical primarily |
| Use Case | Cache, sessions, queues | Primary data storage |

REDIS DATA STRUCTURES:

1. STRINGS (most basic):
```python
import redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Basic string operations
r.set('key', 'value')
r.get('key')  # Returns b'value'
r.incr('counter')  # Atomic increment
r.expire('key', 3600)  # TTL in seconds

# String use cases: caching, counters, flags
```

2. HASHES (field-value pairs):
```python
# User profile storage
r.hset('user:1001', mapping={
    'name': 'John Doe',
    'email': 'john@example.com',
    'age': 30
})

r.hget('user:1001', 'name')  # Get single field
r.hgetall('user:1001')       # Get all fields
r.hincrby('user:1001', 'age', 1)  # Increment age

# Hash use cases: objects, user profiles, configs
```

3. LISTS (ordered collections):
```python
# Queue implementation
r.lpush('task_queue', 'task1', 'task2', 'task3')  # Add to left
r.rpop('task_queue')   # Remove from right (FIFO)
r.lrange('task_queue', 0, -1)  # Get all items

# Stack implementation
r.lpush('stack', 'item1', 'item2')  # Add to left
r.lpop('stack')        # Remove from left (LIFO)

# List use cases: queues, stacks, activity feeds
```

4. SETS (unique collections):
```python
# User interests
r.sadd('user:1001:interests', 'python', 'redis', 'django')
r.smembers('user:1001:interests')  # Get all members
r.sismember('user:1001:interests', 'python')  # Check membership

# Set operations
r.sadd('user:1002:interests', 'python', 'javascript', 'react')
r.sinter('user:1001:interests', 'user:1002:interests')  # Intersection
r.sunion('user:1001:interests', 'user:1002:interests')   # Union

# Set use cases: tags, followers, unique visitors
```

5. SORTED SETS (ordered by score):
```python
# Leaderboard
r.zadd('leaderboard', {'player1': 100, 'player2': 85, 'player3': 95})
r.zrange('leaderboard', 0, -1, withscores=True)  # Ascending order
r.zrevrange('leaderboard', 0, 2)  # Top 3 players

# Score operations
r.zincrby('leaderboard', 10, 'player1')  # Increase score
r.zrank('leaderboard', 'player1')       # Get rank

# Sorted set use cases: leaderboards, priority queues, time series
```

6. STREAMS (append-only log):
```python
# Event logging
r.xadd('events', {
    'user_id': '1001',
    'action': 'login',
    'timestamp': '1628123456'
})

# Read from stream
r.xread({'events': '$'}, block=1000)  # Block for 1 second

# Stream use cases: event sourcing, message queues, logs
```

ADVANCED DATA OPERATIONS:
```python
# Atomic operations with pipeline
pipe = r.pipeline()
pipe.set('key1', 'value1')
pipe.incr('counter')
pipe.lpush('list', 'item')
pipe.execute()  # Execute all at once

# Lua scripts for atomic complex operations
lua_script = '''
local current = redis.call('get', KEYS[1])
if current == false then
    return redis.call('set', KEYS[1], ARGV[1])
else
    return nil
end
'''
set_if_not_exists = r.register_script(lua_script)
result = set_if_not_exists(keys=['mykey'], args=['myvalue'])
```
            """,
            "follow_up": "When would you use each Redis data structure in a real application?"
        },
        
        {
            "question": "2. What happens if Redis crashes? Explain persistence mechanisms (RDB vs AOF).",
            "answer": """
Redis offers two persistence mechanisms to survive crashes and restarts.

DATA LOSS SCENARIOS:
- Without persistence: ALL data lost on crash/restart
- With RDB only: Data loss between snapshots possible
- With AOF only: Minimal data loss (configurable)
- With both: Maximum data safety

1. RDB (Redis Database Backup) - SNAPSHOTS:

Configuration:
```bash
# redis.conf
save 900 1      # Save if at least 1 key changed in 900 seconds
save 300 10     # Save if at least 10 keys changed in 300 seconds  
save 60 10000   # Save if at least 10000 keys changed in 60 seconds

# Manual snapshot
BGSAVE  # Non-blocking background save
SAVE    # Blocking save (not recommended for production)
```

RDB Advantages:
- Compact single file
- Faster restarts
- Better for backups
- Lower disk I/O during normal operation

RDB Disadvantages:
- Data loss between snapshots
- CPU intensive during save
- Not suitable for minimal data loss requirements

```python
# Python configuration for RDB
redis_conf = {
    'save': '900 1 300 10 60 10000',
    'rdbcompression': 'yes',
    'rdbchecksum': 'yes',
    'dbfilename': 'dump.rdb',
    'dir': '/var/lib/redis'
}
```

2. AOF (Append-Only File) - WRITE LOG:

Configuration:
```bash
# redis.conf
appendonly yes
appendfilename "appendonly.aof"

# Sync policy
appendfsync everysec    # Sync every second (recommended)
appendfsync always      # Sync every write (slowest, safest)
appendfsync no          # Let OS decide (fastest, least safe)

# AOF rewrite (compaction)
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
```

AOF Advantages:
- Minimal data loss (1 second max with everysec)
- Append-only (safer from corruption)
- Human-readable format
- Automatic background rewrite

AOF Disadvantages:
- Larger files than RDB
- Slower restarts
- Higher disk I/O
- Potentially slower than RDB

```python
# Python configuration for AOF
redis_conf = {
    'appendonly': 'yes',
    'appendfsync': 'everysec',
    'no-appendfsync-on-rewrite': 'no',
    'auto-aof-rewrite-percentage': '100',
    'auto-aof-rewrite-min-size': '64mb'
}
```

3. HYBRID PERSISTENCE (RDB + AOF):

```bash
# redis.conf - Best of both worlds
save 900 1
appendonly yes
appendfsync everysec

# RDB for fast restarts, AOF for minimal data loss
aof-use-rdb-preamble yes  # Use RDB format for base, AOF for recent changes
```

DISASTER RECOVERY STRATEGIES:

```python
import redis
import time
import subprocess

class RedisBackupManager:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        
    def create_backup(self):
        """Create RDB backup"""
        try:
            # Trigger background save
            self.redis_client.bgsave()
            
            # Wait for save to complete
            while self.redis_client.lastsave() == self.last_save_time:
                time.sleep(1)
            
            # Copy RDB file to backup location
            timestamp = int(time.time())
            subprocess.run([
                'cp', '/var/lib/redis/dump.rdb', 
                f'/backups/redis-backup-{timestamp}.rdb'
            ])
            
            return f'Backup created: redis-backup-{timestamp}.rdb'
        except Exception as e:
            return f'Backup failed: {e}'
    
    def restore_from_backup(self, backup_file):
        """Restore from RDB backup"""
        try:
            # Stop Redis
            subprocess.run(['systemctl', 'stop', 'redis'])
            
            # Replace RDB file
            subprocess.run(['cp', backup_file, '/var/lib/redis/dump.rdb'])
            
            # Start Redis
            subprocess.run(['systemctl', 'start', 'redis'])
            
            return 'Restore completed successfully'
        except Exception as e:
            return f'Restore failed: {e}'
    
    def get_memory_usage(self):
        """Monitor memory usage"""
        info = self.redis_client.info('memory')
        return {
            'used_memory': info['used_memory'],
            'used_memory_human': info['used_memory_human'],
            'used_memory_peak': info['used_memory_peak'],
            'used_memory_peak_human': info['used_memory_peak_human']
        }
```

PRODUCTION PERSISTENCE STRATEGY:

```python
# settings.py - Production Redis configuration
REDIS_PERSISTENCE_CONFIG = {
    # Use both RDB and AOF
    'save': '900 1 300 10 60 10000',  # RDB snapshots
    'appendonly': 'yes',               # AOF enabled
    'appendfsync': 'everysec',         # 1-second max data loss
    
    # Performance tuning
    'no-appendfsync-on-rewrite': 'no',
    'auto-aof-rewrite-percentage': '100',
    'auto-aof-rewrite-min-size': '64mb',
    
    # Hybrid mode
    'aof-use-rdb-preamble': 'yes',
    
    # Backup strategy
    'dir': '/var/lib/redis',
    'dbfilename': 'dump.rdb',
    'appendfilename': 'appendonly.aof'
}
```
            """,
            "follow_up": "How do you handle Redis failover in a production environment?"
        },
        
        {
            "question": "3. How does Redis handle concurrency and what are Redis transactions?",
            "answer": """
Redis handles concurrency using a single-threaded event loop model with atomic operations.

REDIS CONCURRENCY MODEL:

1. SINGLE-THREADED ARCHITECTURE:
```
Client 1 → |
Client 2 → | Event Loop | → Redis Operations (Atomic)
Client 3 → |
```

- All commands executed sequentially
- No race conditions between commands
- Atomic operations guaranteed
- High throughput due to no locking overhead

2. ATOMIC OPERATIONS:
```python
import redis
r = redis.Redis()

# These operations are atomic
r.incr('counter')           # Atomic increment
r.lpush('queue', 'item')    # Atomic list operation
r.sadd('set', 'member')     # Atomic set operation

# Complex atomic operation with Lua script
lua_script = '''
local key = KEYS[1]
local amount = tonumber(ARGV[1])
local current = tonumber(redis.call('get', key) or 0)

if current >= amount then
    redis.call('decrby', key, amount)
    return 1
else
    return 0
end
'''

withdraw_money = r.register_script(lua_script)
success = withdraw_money(keys=['account:balance'], args=[100])
```

3. REDIS TRANSACTIONS (MULTI/EXEC):

Basic Transaction:
```python
# Begin transaction
pipe = r.pipeline()
pipe.multi()

# Queue commands
pipe.set('key1', 'value1')
pipe.incr('counter')
pipe.lpush('list', 'item')

# Execute atomically
results = pipe.execute()

# All commands execute together or not at all
```

Conditional Transactions with WATCH:
```python
def transfer_money(from_account, to_account, amount):
    with r.pipeline() as pipe:
        while True:
            try:
                # Watch the accounts
                pipe.watch(from_account, to_account)
                
                # Get current balances
                from_balance = float(pipe.get(from_account) or 0)
                to_balance = float(pipe.get(to_account) or 0)
                
                # Check if transfer is valid
                if from_balance < amount:
                    pipe.unwatch()
                    return False, "Insufficient funds"
                
                # Start transaction
                pipe.multi()
                pipe.set(from_account, from_balance - amount)
                pipe.set(to_account, to_balance + amount)
                
                # Execute transaction
                pipe.execute()
                return True, "Transfer successful"
                
            except redis.WatchError:
                # Retry if watched keys were modified
                continue
```

OPTIMISTIC LOCKING PATTERN:
```python
class RedisCounter:
    def __init__(self, redis_client, key):
        self.redis = redis_client
        self.key = key
    
    def increment_with_check(self, max_value):
        """Increment counter but don't exceed max_value"""
        with self.redis.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(self.key)
                    current = int(pipe.get(self.key) or 0)
                    
                    if current >= max_value:
                        pipe.unwatch()
                        return False, "Max value reached"
                    
                    pipe.multi()
                    pipe.incr(self.key)
                    pipe.execute()
                    return True, current + 1
                    
                except redis.WatchError:
                    continue
```

4. PIPELINING FOR PERFORMANCE:

Without Pipelining (Multiple Round Trips):
```python
# Slow - 3 network round trips
r.set('key1', 'value1')
r.set('key2', 'value2')  
r.set('key3', 'value3')
```

With Pipelining (Single Round Trip):
```python
# Fast - 1 network round trip
pipe = r.pipeline()
pipe.set('key1', 'value1')
pipe.set('key2', 'value2')
pipe.set('key3', 'value3')
results = pipe.execute()
```

5. LUA SCRIPTS FOR COMPLEX ATOMICITY:

Rate Limiting Example:
```python
rate_limit_script = '''
local key = KEYS[1]
local window = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local current_time = tonumber(ARGV[3])

-- Clean old entries
redis.call('zremrangebyscore', key, 0, current_time - window)

-- Count current requests
local current_requests = redis.call('zcard', key)

if current_requests < limit then
    -- Allow request
    redis.call('zadd', key, current_time, current_time)
    redis.call('expire', key, window)
    return 1
else
    -- Rate limit exceeded
    return 0
end
'''

def is_rate_limited(user_id, window=60, limit=100):
    rate_limiter = r.register_script(rate_limit_script)
    current_time = int(time.time())
    
    result = rate_limiter(
        keys=[f'rate_limit:{user_id}'],
        args=[window, limit, current_time]
    )
    
    return result == 0  # True if rate limited
```

DISTRIBUTED LOCKING:
```python
import time
import uuid

class RedisDistributedLock:
    def __init__(self, redis_client, key, timeout=10):
        self.redis = redis_client
        self.key = f'lock:{key}'
        self.timeout = timeout
        self.identifier = str(uuid.uuid4())
    
    def acquire(self):
        """Acquire distributed lock"""
        end_time = time.time() + self.timeout
        
        while time.time() < end_time:
            if self.redis.set(self.key, self.identifier, nx=True, ex=self.timeout):
                return True
            time.sleep(0.001)
        
        return False
    
    def release(self):
        """Release distributed lock"""
        release_script = '''
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('del', KEYS[1])
        else
            return 0
        end
        '''
        
        unlock = self.redis.register_script(release_script)
        return unlock(keys=[self.key], args=[self.identifier])

# Usage
lock = RedisDistributedLock(r, 'critical_section')
if lock.acquire():
    try:
        # Critical section code
        pass
    finally:
        lock.release()
```
            """,
            "follow_up": "What are the limitations of Redis transactions compared to SQL databases?"
        }
    ]
    
    for qa in questions_and_answers:
        print(f"\nQ: {qa['question']}")
        print(f"A: {qa['answer']}")
        if qa.get('follow_up'):
            print(f"Follow-up: {qa['follow_up']}")
        print("-" * 80)

def redis_pub_sub_and_messaging():
    """
    Redis Pub/Sub, Streams, and messaging patterns
    """
    
    print("\n" + "=" * 80)
    print("REDIS PUB/SUB AND MESSAGING PATTERNS")
    print("=" * 80)
    
    print("""
1. REDIS PUB/SUB FUNDAMENTALS:

Basic Pub/Sub:
```python
import redis
import threading
import json

# Publisher
def publish_messages():
    r = redis.Redis()
    
    # Publish to channel
    r.publish('notifications', json.dumps({
        'type': 'user_signup',
        'user_id': 12345,
        'timestamp': time.time()
    }))
    
    # Publish to pattern
    r.publish('events:user:signup', 'User registered')
    r.publish('events:order:created', 'Order created')

# Subscriber
def subscribe_to_messages():
    r = redis.Redis()
    pubsub = r.pubsub()
    
    # Subscribe to specific channels
    pubsub.subscribe('notifications', 'alerts')
    
    # Subscribe to patterns
    pubsub.psubscribe('events:*', 'logs:*')
    
    for message in pubsub.listen():
        if message['type'] == 'message':
            print(f"Channel: {message['channel']}")
            print(f"Data: {message['data']}")
        elif message['type'] == 'pmessage':
            print(f"Pattern: {message['pattern']}")
            print(f"Channel: {message['channel']}")
            print(f"Data: {message['data']}")
```

Django Integration:
```python
# Django pub/sub service
import redis
from django.conf import settings
from django.utils import timezone

class NotificationService:
    def __init__(self):
        self.redis_client = redis.Redis.from_url(settings.REDIS_URL)
    
    def publish_user_event(self, event_type, user_id, data=None):
        message = {
            'event_type': event_type,
            'user_id': user_id,
            'data': data or {},
            'timestamp': timezone.now().isoformat()
        }
        
        # Publish to user-specific channel
        self.redis_client.publish(f'user:{user_id}', json.dumps(message))
        
        # Publish to general events channel
        self.redis_client.publish('events', json.dumps(message))
    
    def subscribe_to_user_events(self, user_id, callback):
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(f'user:{user_id}')
        
        for message in pubsub.listen():
            if message['type'] == 'message':
                callback(json.loads(message['data']))

# Usage in views
def user_signup(request):
    # Create user...
    user = User.objects.create_user(...)
    
    # Notify other services
    notification_service = NotificationService()
    notification_service.publish_user_event('signup', user.id, {
        'email': user.email,
        'plan': 'free'
    })
```

2. REDIS STREAMS (Advanced Messaging):

Stream Basics:
```python
import redis
r = redis.Redis()

# Add messages to stream
r.xadd('user_events', {
    'user_id': '123',
    'action': 'login',
    'ip': '192.168.1.1',
    'timestamp': int(time.time())
})

r.xadd('user_events', {
    'user_id': '456', 
    'action': 'purchase',
    'amount': '99.99',
    'product_id': 'prod_123'
})

# Read from stream
messages = r.xread({'user_events': '0'}, count=10)
for stream, msgs in messages:
    for msg_id, fields in msgs:
        print(f"ID: {msg_id}, Data: {fields}")

# Read new messages (blocking)
messages = r.xread({'user_events': '$'}, block=5000)  # Block for 5 seconds
```

Consumer Groups:
```python
# Create consumer group
r.xgroup_create('user_events', 'analytics_group', id='0', mkstream=True)

# Consumer in group
def process_analytics_events():
    while True:
        try:
            # Read messages for this consumer
            messages = r.xreadgroup(
                'analytics_group',      # Group name
                'consumer_1',           # Consumer name
                {'user_events': '>'},   # Stream and position
                count=10,               # Max messages
                block=1000              # Block timeout
            )
            
            for stream, msgs in messages:
                for msg_id, fields in msgs:
                    # Process message
                    process_user_event(fields)
                    
                    # Acknowledge message
                    r.xack('user_events', 'analytics_group', msg_id)
                    
        except Exception as e:
            print(f"Error processing events: {e}")

def process_user_event(event_data):
    # Analytics processing logic
    action = event_data[b'action'].decode()
    user_id = event_data[b'user_id'].decode()
    
    if action == 'login':
        update_login_stats(user_id)
    elif action == 'purchase':
        update_revenue_stats(user_id, event_data)
```

3. REAL-TIME FEATURES WITH WEBSOCKETS:

Django Channels + Redis:
```python
# consumers.py
import json
import redis
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.layers import get_channel_layer

class NotificationConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user_id = self.scope['user'].id
        self.group_name = f'user_{self.user_id}'
        
        # Join user group
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        await self.accept()
        
        # Start Redis subscriber in background
        asyncio.create_task(self.redis_subscriber())
    
    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )
    
    async def redis_subscriber(self):
        """Subscribe to Redis pub/sub for this user"""
        r = redis.Redis()
        pubsub = r.pubsub()
        pubsub.subscribe(f'user:{self.user_id}')
        
        for message in pubsub.listen():
            if message['type'] == 'message':
                await self.send(text_data=message['data'])

# views.py - Sending real-time notifications
def send_notification(user_id, message):
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f'user_{user_id}',
        {
            'type': 'notification_message',
            'message': message
        }
    )
```

4. MESSAGE QUEUE PATTERNS:

Work Queue Pattern:
```python
class RedisWorkQueue:
    def __init__(self, queue_name):
        self.redis = redis.Redis()
        self.queue_name = queue_name
    
    def add_job(self, job_data):
        """Add job to queue"""
        job_id = str(uuid.uuid4())
        job = {
            'id': job_id,
            'data': job_data,
            'created_at': time.time(),
            'status': 'pending'
        }
        
        # Add to queue
        self.redis.lpush(self.queue_name, json.dumps(job))
        
        # Store job details
        self.redis.hset(f'job:{job_id}', mapping=job)
        self.redis.expire(f'job:{job_id}', 3600)  # 1 hour TTL
        
        return job_id
    
    def get_job(self, timeout=10):
        """Get job from queue (blocking)"""
        result = self.redis.brpop(self.queue_name, timeout=timeout)
        if result:
            queue_name, job_data = result
            return json.loads(job_data)
        return None
    
    def mark_completed(self, job_id, result=None):
        """Mark job as completed"""
        self.redis.hset(f'job:{job_id}', mapping={
            'status': 'completed',
            'completed_at': time.time(),
            'result': json.dumps(result) if result else ''
        })

# Worker process
def worker():
    queue = RedisWorkQueue('email_queue')
    
    while True:
        job = queue.get_job()
        if job:
            try:
                # Process job
                result = send_email(job['data'])
                queue.mark_completed(job['id'], result)
            except Exception as e:
                queue.mark_failed(job['id'], str(e))
```

Priority Queue Pattern:
```python
class RedisPriorityQueue:
    def __init__(self, queue_name):
        self.redis = redis.Redis()
        self.queue_name = queue_name
    
    def add_job(self, job_data, priority=0):
        """Add job with priority (higher number = higher priority)"""
        job_id = str(uuid.uuid4())
        job = {
            'id': job_id,
            'data': job_data,
            'priority': priority,
            'created_at': time.time()
        }
        
        # Add to sorted set with priority as score
        self.redis.zadd(self.queue_name, {json.dumps(job): priority})
        return job_id
    
    def get_highest_priority_job(self):
        """Get highest priority job"""
        # Get highest score (rightmost) element
        result = self.redis.zpopmax(self.queue_name)
        if result:
            job_data, priority = result[0]
            return json.loads(job_data)
        return None
```
    """)

def redis_performance_and_scaling():
    """
    Redis performance optimization, memory management, and scaling strategies
    """
    
    print("\n" + "=" * 80)
    print("REDIS PERFORMANCE AND SCALING")
    print("=" * 80)
    
    print("""
1. MEMORY OPTIMIZATION:

Memory Analysis:
```bash
# Check memory usage
redis-cli info memory

# Memory usage by key patterns
redis-cli --bigkeys

# Sample keyspace
redis-cli --memkeys

# Memory usage of specific key
redis-cli memory usage mykey
```

Memory Policies:
```python
# redis.conf - Memory management
maxmemory 2gb
maxmemory-policy allkeys-lru

# Available policies:
# noeviction      - Return errors when memory limit reached
# allkeys-lru     - Evict least recently used keys  
# allkeys-lfu     - Evict least frequently used keys
# volatile-lru    - Evict LRU among keys with expire set
# volatile-lfu    - Evict LFU among keys with expire set
# volatile-random - Evict random keys with expire set
# volatile-ttl    - Evict keys with shortest TTL
```

Memory-Efficient Data Structures:
```python
# Use hashes for small objects instead of individual keys
# BAD - Many keys
r.set('user:1001:name', 'John')
r.set('user:1001:email', 'john@example.com') 
r.set('user:1001:age', 30)

# GOOD - Single hash
r.hset('user:1001', mapping={
    'name': 'John',
    'email': 'john@example.com',
    'age': 30
})

# Use appropriate data types
# BAD - String for simple counter
r.set('page_views', '1000000')

# GOOD - Use string with expiry for counters
r.setex('page_views', 3600, '1000000')  # 1 hour TTL

# GOOD - Use bitmap for boolean flags
r.setbit('user_online', user_id, 1)  # Mark user online
online = r.getbit('user_online', user_id)  # Check if online
```

2. PERFORMANCE MONITORING:

Redis Monitoring Script:
```python
import redis
import time
import psutil

class RedisMonitor:
    def __init__(self, host='localhost', port=6379):
        self.redis = redis.Redis(host=host, port=port)
    
    def get_performance_metrics(self):
        info = self.redis.info()
        
        return {
            # Memory metrics
            'used_memory': info['used_memory'],
            'used_memory_peak': info['used_memory_peak'],
            'memory_fragmentation_ratio': info.get('mem_fragmentation_ratio', 0),
            
            # Performance metrics
            'ops_per_sec': info.get('instantaneous_ops_per_sec', 0),
            'hits': info.get('keyspace_hits', 0),
            'misses': info.get('keyspace_misses', 0),
            'hit_rate': self.calculate_hit_rate(info),
            
            # Connection metrics
            'connected_clients': info['connected_clients'],
            'blocked_clients': info.get('blocked_clients', 0),
            
            # Persistence metrics
            'rdb_last_save_time': info.get('rdb_last_save_time', 0),
            'aof_enabled': info.get('aof_enabled', 0),
        }
    
    def calculate_hit_rate(self, info):
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0
    
    def check_slow_queries(self):
        """Get slow queries from Redis"""
        slow_log = self.redis.slowlog_get(10)
        return [
            {
                'id': entry['id'],
                'start_time': entry['start_time'],
                'duration': entry['duration'],
                'command': ' '.join(entry['command'])
            }
            for entry in slow_log
        ]
    
    def analyze_memory_usage(self):
        """Analyze memory usage patterns"""
        # Get sample of keys
        keys_sample = []
        for key in self.redis.scan_iter(count=1000):
            key_info = {
                'key': key.decode(),
                'type': self.redis.type(key).decode(),
                'memory': self.redis.memory_usage(key),
                'ttl': self.redis.ttl(key)
            }
            keys_sample.append(key_info)
        
        # Analyze patterns
        memory_by_type = {}
        for key_info in keys_sample:
            key_type = key_info['type']
            memory_by_type[key_type] = memory_by_type.get(key_type, 0) + key_info['memory']
        
        return {
            'total_keys_sampled': len(keys_sample),
            'memory_by_type': memory_by_type,
            'top_memory_consumers': sorted(
                keys_sample, 
                key=lambda x: x['memory'], 
                reverse=True
            )[:10]
        }
```

3. REDIS CLUSTERING:

Cluster Setup:
```bash
# Create cluster with 6 nodes (3 masters, 3 slaves)
redis-cli --cluster create \\
    127.0.0.1:7000 127.0.0.1:7001 127.0.0.1:7002 \\
    127.0.0.1:7003 127.0.0.1:7004 127.0.0.1:7005 \\
    --cluster-replicas 1

# Check cluster status
redis-cli -c -p 7000 cluster nodes
redis-cli -c -p 7000 cluster info
```

Python Cluster Client:
```python
from rediscluster import RedisCluster

# Cluster connection
startup_nodes = [
    {"host": "127.0.0.1", "port": "7000"},
    {"host": "127.0.0.1", "port": "7001"},
    {"host": "127.0.0.1", "port": "7002"}
]

rc = RedisCluster(startup_nodes=startup_nodes, decode_responses=True)

# Operations work the same as single Redis
rc.set('key1', 'value1')
rc.get('key1')

# Hash tags for multi-key operations
rc.mset({
    'user:{123}:name': 'John',
    'user:{123}:email': 'john@example.com'
})

# These keys will be on the same node due to hash tag {123}
```

4. REDIS SENTINEL (High Availability):

Sentinel Configuration:
```bash
# sentinel.conf
port 26379
sentinel monitor mymaster 127.0.0.1 6379 2
sentinel down-after-milliseconds mymaster 5000
sentinel failover-timeout mymaster 10000
sentinel parallel-syncs mymaster 1
```

Python Sentinel Client:
```python
from redis.sentinel import Sentinel

# Connect to Sentinel
sentinel = Sentinel([
    ('localhost', 26379),
    ('localhost', 26380),
    ('localhost', 26381)
])

# Get master and slave connections
master = sentinel.master_for('mymaster', socket_timeout=0.1)
slave = sentinel.slave_for('mymaster', socket_timeout=0.1)

# Write to master, read from slave
master.set('key', 'value')
value = slave.get('key')

# Automatic failover handling
def redis_operation_with_failover():
    try:
        return master.get('some_key')
    except redis.ConnectionError:
        # Sentinel will automatically promote new master
        time.sleep(1)
        return master.get('some_key')
```

5. CACHING STRATEGIES:

Cache-Aside Pattern:
```python
def get_user(user_id):
    # Try cache first
    cache_key = f'user:{user_id}'
    user_data = redis_client.get(cache_key)
    
    if user_data:
        return json.loads(user_data)
    
    # Cache miss - get from database
    user = User.objects.get(id=user_id)
    user_data = {
        'id': user.id,
        'name': user.name,
        'email': user.email
    }
    
    # Cache for 1 hour
    redis_client.setex(cache_key, 3600, json.dumps(user_data))
    return user_data
```

Write-Through Pattern:
```python
def update_user(user_id, data):
    # Update database
    user = User.objects.get(id=user_id)
    for key, value in data.items():
        setattr(user, key, value)
    user.save()
    
    # Update cache immediately
    cache_key = f'user:{user_id}'
    user_data = {
        'id': user.id,
        'name': user.name,
        'email': user.email
    }
    redis_client.setex(cache_key, 3600, json.dumps(user_data))
    
    return user
```

Write-Behind Pattern:
```python
def update_user_async(user_id, data):
    # Update cache immediately
    cache_key = f'user:{user_id}'
    cached_data = redis_client.get(cache_key)
    
    if cached_data:
        user_data = json.loads(cached_data)
        user_data.update(data)
        redis_client.setex(cache_key, 3600, json.dumps(user_data))
    
    # Queue database update for later
    update_user_db.delay(user_id, data)
```
    """)

if __name__ == "__main__":
    print("COMPREHENSIVE REDIS INTERVIEW GUIDE - PART 3")
    print("=" * 60)
    print("This guide covers Redis fundamentals, persistence, concurrency, and scaling")
    print("Use this as a comprehensive reference for Redis interviews")
    print("=" * 60)
    
    redis_fundamentals_interview_questions()
    redis_pub_sub_and_messaging()
    redis_performance_and_scaling()
    
    print("\n" + "=" * 80)
    print("END OF PART 3 - REDIS DEEP DIVE")
    print("Next: Part 4 - RabbitMQ and Apache Kafka")
    print("=" * 80)
