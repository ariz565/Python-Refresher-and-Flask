"""
COMPREHENSIVE INTERVIEW GUIDE - PART 4: RABBITMQ & APACHE KAFKA
==============================================================

This guide covers RabbitMQ messaging patterns, Apache Kafka streaming, and message broker comparisons.
Complete coverage of enterprise messaging systems and event streaming platforms.

Author: Interview Preparation Guide
Date: August 2025
Technologies: RabbitMQ, Apache Kafka, Message Brokers, Event Streaming
"""

# ===================================
# PART 4A: RABBITMQ FUNDAMENTALS
# ===================================

def rabbitmq_fundamentals_interview_questions():
    """
    Comprehensive RabbitMQ interview questions covering all messaging concepts
    """
    
    print("=" * 80)
    print("RABBITMQ FUNDAMENTALS - INTERVIEW QUESTIONS & ANSWERS")
    print("=" * 80)
    
    questions_and_answers = [
        {
            "question": "1. How does RabbitMQ work as a message broker? Explain exchanges, queues, and bindings.",
            "answer": """
RabbitMQ is a robust message broker that implements the Advanced Message Queuing Protocol (AMQP).
It ensures reliable message delivery between producers and consumers.

CORE CONCEPTS:

1. EXCHANGES - Message routing:
   - Receive messages from producers
   - Route messages to queues based on rules
   - Types: direct, fanout, topic, headers

2. QUEUES - Message storage:
   - Store messages until consumed
   - Can be durable (survive restarts)
   - Support priority and TTL

3. BINDINGS - Routing rules:
   - Connect exchanges to queues
   - Define routing keys and patterns

ARCHITECTURE FLOW:
```
Producer → Exchange → Binding → Queue → Consumer
```

EXCHANGE TYPES:

1. DIRECT EXCHANGE:
```python
import pika

# Setup connection
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Declare exchange and queue
channel.exchange_declare(exchange='direct_logs', exchange_type='direct')
channel.queue_declare(queue='critical_queue', durable=True)

# Bind queue to exchange with routing key
channel.queue_bind(
    exchange='direct_logs',
    queue='critical_queue', 
    routing_key='critical'
)

# Publish message
channel.basic_publish(
    exchange='direct_logs',
    routing_key='critical',
    body='Critical system error!',
    properties=pika.BasicProperties(delivery_mode=2)  # Persistent message
)

# Consumer
def process_critical_message(ch, method, properties, body):
    print(f"Processing critical: {body}")
    # Process message
    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_consume(
    queue='critical_queue',
    on_message_callback=process_critical_message
)
```

2. FANOUT EXCHANGE (Broadcast):
```python
# Publisher
channel.exchange_declare(exchange='notifications', exchange_type='fanout')

# Multiple services can bind their queues
channel.queue_declare(queue='email_service')
channel.queue_declare(queue='sms_service') 
channel.queue_declare(queue='push_service')

channel.queue_bind(exchange='notifications', queue='email_service')
channel.queue_bind(exchange='notifications', queue='sms_service')
channel.queue_bind(exchange='notifications', queue='push_service')

# One message goes to all bound queues
channel.basic_publish(
    exchange='notifications',
    routing_key='',  # Ignored in fanout
    body='User signed up!'
)
```

3. TOPIC EXCHANGE (Pattern matching):
```python
# Setup topic exchange
channel.exchange_declare(exchange='logs', exchange_type='topic')

# Bind with patterns
channel.queue_bind(exchange='logs', queue='all_logs', routing_key='#')
channel.queue_bind(exchange='logs', queue='error_logs', routing_key='*.error')
channel.queue_bind(exchange='logs', queue='user_logs', routing_key='user.*')

# Publish with specific routing keys
channel.basic_publish(exchange='logs', routing_key='user.login', body='User logged in')
channel.basic_publish(exchange='logs', routing_key='system.error', body='System error')
channel.basic_publish(exchange='logs', routing_key='user.error', body='User error')

# Routing patterns:
# * matches exactly one word
# # matches zero or more words
# user.* matches user.login, user.logout
# *.error matches user.error, system.error
# user.# matches user.login.success, user.logout.failed
```

CELERY WITH RABBITMQ:
```python
# settings.py
CELERY_BROKER_URL = 'amqp://guest:guest@localhost:5672//'

# Advanced RabbitMQ configuration
CELERY_TASK_ROUTES = {
    'myapp.tasks.send_email': {'queue': 'email_queue'},
    'myapp.tasks.process_image': {'queue': 'image_queue'},
    'myapp.tasks.generate_report': {'queue': 'report_queue'},
}

from kombu import Queue, Exchange

CELERY_TASK_QUEUES = (
    Queue('email_queue', 
          Exchange('tasks', type='direct'), 
          routing_key='email'),
    Queue('image_queue', 
          Exchange('tasks', type='direct'), 
          routing_key='image'),
    Queue('report_queue', 
          Exchange('tasks', type='direct'), 
          routing_key='report'),
)

# Priority queues
CELERY_TASK_QUEUES = (
    Queue('high_priority', 
          routing_key='high',
          queue_arguments={'x-max-priority': 10}),
    Queue('normal_priority', 
          routing_key='normal',
          queue_arguments={'x-max-priority': 5}),
)
```
            """,
            "follow_up": "What happens if RabbitMQ crashes with messages in queues?"
        },
        
        {
            "question": "2. How does RabbitMQ ensure message durability and handle acknowledgments?",
            "answer": '''
RabbitMQ provides multiple mechanisms to ensure message durability and prevent message loss.

MESSAGE DURABILITY LAYERS:

1. EXCHANGE DURABILITY:
```python
# Durable exchange (survives broker restart)
channel.exchange_declare(
    exchange='durable_exchange',
    exchange_type='direct',
    durable=True  # Survives restart
)
```

2. QUEUE DURABILITY:
```python
# Durable queue (survives broker restart)
channel.queue_declare(
    queue='durable_queue',
    durable=True,           # Queue survives restart
    auto_delete=False,      # Don't auto-delete when empty
    exclusive=False         # Allow multiple connections
)
```

3. MESSAGE PERSISTENCE:
```python
# Persistent message (written to disk)
channel.basic_publish(
    exchange='durable_exchange',
    routing_key='persistent',
    body='Important message',
    properties=pika.BasicProperties(
        delivery_mode=2  # Make message persistent
    )
)
```

ACKNOWLEDGMENT PATTERNS:

1. AUTOMATIC ACKNOWLEDGMENT (Auto-ack):
```python
# Risky - message deleted immediately on delivery
def auto_ack_consumer():
    def callback(ch, method, properties, body):
        print(f"Processing: {body}")
        # If processing fails here, message is lost!
        
    channel.basic_consume(
        queue='risky_queue',
        on_message_callback=callback,
        auto_ack=True  # Dangerous!
    )
```

2. MANUAL ACKNOWLEDGMENT (Recommended):
```python
def reliable_consumer():
    def safe_callback(ch, method, properties, body):
        try:
            # Process message
            result = process_message(body)
            
            # Only acknowledge after successful processing
            ch.basic_ack(delivery_tag=method.delivery_tag)
            print(f"Successfully processed: {body}")
            
        except Exception as e:
            print(f"Error processing message: {e}")
            
            # Reject and requeue for retry
            ch.basic_nack(
                delivery_tag=method.delivery_tag,
                requeue=True  # Put back in queue
            )
    
    channel.basic_consume(
        queue='reliable_queue',
        on_message_callback=safe_callback,
        auto_ack=False  # Manual acknowledgment
    )
```

3. CELERY ACKNOWLEDGMENT CONFIGURATION:
```python
# settings.py - Celery with RabbitMQ reliability
CELERY_TASK_ACKS_LATE = True  # Acknowledge after task completion
CELERY_WORKER_PREFETCH_MULTIPLIER = 1  # One task at a time per worker
CELERY_TASK_REJECT_ON_WORKER_LOST = True  # Requeue if worker dies

# Task-level acknowledgment control
@shared_task(acks_late=True, autoretry_for=(Exception,))
def reliable_task(data):
    try:
        return process_data(data)
    except Exception as exc:
        # Task will be retried automatically
        raise exc
```

ADVANCED RELIABILITY PATTERNS:

1. DEAD LETTER EXCHANGE:
```python
# Setup main queue with DLX
channel.queue_declare(
    queue='main_queue',
    durable=True,
    arguments={
        'x-dead-letter-exchange': 'dlx_exchange',
        'x-dead-letter-routing-key': 'failed',
        'x-message-ttl': 300000,  # 5 minutes TTL
        'x-max-retries': 3
    }
)

# Dead letter queue for failed messages
channel.exchange_declare(exchange='dlx_exchange', exchange_type='direct')
channel.queue_declare(queue='failed_queue', durable=True)
channel.queue_bind(
    exchange='dlx_exchange',
    queue='failed_queue',
    routing_key='failed'
)

def consumer_with_dlx():
    def process_with_retry(ch, method, properties, body):
        try:
            process_message(body)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            # Check retry count
            headers = properties.headers or {}
            retry_count = headers.get('x-retry-count', 0)
            
            if retry_count < 3:
                # Increment retry count and requeue
                headers['x-retry-count'] = retry_count + 1
                ch.basic_publish(
                    exchange='',
                    routing_key='main_queue',
                    body=body,
                    properties=pika.BasicProperties(headers=headers)
                )
                ch.basic_ack(delivery_tag=method.delivery_tag)
            else:
                # Max retries exceeded, send to DLX
                ch.basic_nack(
                    delivery_tag=method.delivery_tag,
                    requeue=False  # Will go to DLX
                )
```

2. PUBLISHER CONFIRMS:
```python
def reliable_publisher():
    # Enable publisher confirms
    channel.confirm_delivery()
    
    try:
        # Publish message
        if channel.basic_publish(
            exchange='reliable_exchange',
            routing_key='important',
            body='Critical data',
            properties=pika.BasicProperties(delivery_mode=2),
            mandatory=True  # Ensure queue exists
        ):
            print("Message confirmed by broker")
        else:
            print("Message not confirmed")
    except pika.exceptions.UnroutableError:
        print("Message could not be routed")
```

3. CLUSTERING FOR HIGH AVAILABILITY:
```bash
# Setup RabbitMQ cluster
rabbitmqctl join_cluster rabbit@node1
rabbitmqctl set_policy ha-all ".*" '{"ha-mode":"all"}'

# Mirror queues across all nodes
rabbitmqctl set_policy ha-two "^two\." '{"ha-mode":"exactly","ha-params":2}'
```

MONITORING RELIABILITY:
```python
import pika
import requests

class RabbitMQMonitor:
    def __init__(self, management_url='http://localhost:15672'):
        self.management_url = management_url
        self.auth = ('guest', 'guest')  # Default credentials
    
    def check_queue_health(self, queue_name):
        """Monitor queue health metrics"""
        response = requests.get(
            f'{self.management_url}/api/queues/%2f/{queue_name}',
            auth=self.auth
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                'messages': data.get('messages', 0),
                'messages_ready': data.get('messages_ready', 0),
                'messages_unacknowledged': data.get('messages_unacknowledged', 0),
                'consumers': data.get('consumers', 0),
                'memory': data.get('memory', 0)
            }
        return None
    
    def check_connection_health(self):
        """Check RabbitMQ connectivity"""
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters('localhost')
            )
            connection.close()
            return True
        except:
            return False
    
    def get_cluster_status(self):
        """Get cluster health status"""
        response = requests.get(
            f'{self.management_url}/api/nodes',
            auth=self.auth
        )
        
        if response.status_code == 200:
            nodes = response.json()
            return {
                'total_nodes': len(nodes),
                'running_nodes': len([n for n in nodes if n['running']]),
                'node_details': [
                    {
                        'name': n['name'],
                        'running': n['running'],
                        'memory_used': n.get('mem_used', 0),
                        'disk_free': n.get('disk_free', 0)
                    }
                    for n in nodes
                ]
            }
        return None
```
            ''',
            "follow_up": "How do you handle poison messages in RabbitMQ?"
        },
        
        {
            "question": "3. How do you scale RabbitMQ for high availability and performance?",
            "answer": '''
RabbitMQ scaling involves clustering, federation, and performance optimization strategies.

CLUSTERING STRATEGIES:

1. BASIC CLUSTER SETUP:
```bash
# Start multiple RabbitMQ nodes
# Node 1 (primary)
rabbitmq-server -detached

# Node 2 (join cluster)
rabbitmqctl stop_app
rabbitmqctl join_cluster rabbit@node1
rabbitmqctl start_app

# Node 3 (join cluster)  
rabbitmqctl stop_app
rabbitmqctl join_cluster rabbit@node1
rabbitmqctl start_app

# Check cluster status
rabbitmqctl cluster_status
```

2. QUEUE MIRRORING (High Availability):
```bash
# Mirror all queues across all nodes
rabbitmqctl set_policy ha-all ".*" '{"ha-mode":"all"}'

# Mirror specific queues across 2 nodes
rabbitmqctl set_policy ha-critical "critical-.*" '{"ha-mode":"exactly","ha-params":2}'

# Automatic synchronization
rabbitmqctl set_policy ha-sync ".*" '{"ha-mode":"all","ha-sync-mode":"automatic"}'
```

Python Cluster Client:
```python
import pika
import random

class RabbitMQClusterClient:
    def __init__(self, nodes):
        self.nodes = nodes
        self.connection = None
        self.channel = None
        self.connect()
    
    def connect(self):
        """Connect to available cluster node"""
        random.shuffle(self.nodes)  # Load balancing
        
        for node in self.nodes:
            try:
                self.connection = pika.BlockingConnection(
                    pika.ConnectionParameters(
                        host=node['host'],
                        port=node['port'],
                        heartbeat=600,
                        blocked_connection_timeout=300
                    )
                )
                self.channel = self.connection.channel()
                print(f"Connected to {node['host']}")
                return
            except Exception as e:
                print(f"Failed to connect to {node['host']}: {e}")
                continue
        
        raise Exception("Could not connect to any cluster node")
    
    def publish_with_failover(self, exchange, routing_key, body):
        """Publish with automatic failover"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                self.channel.basic_publish(
                    exchange=exchange,
                    routing_key=routing_key,
                    body=body,
                    properties=pika.BasicProperties(delivery_mode=2)
                )
                return True
            except Exception as e:
                print(f"Publish failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    self.connect()  # Reconnect and retry
        
        return False

# Usage
cluster_nodes = [
    {'host': 'rabbitmq-node1', 'port': 5672},
    {'host': 'rabbitmq-node2', 'port': 5672},
    {'host': 'rabbitmq-node3', 'port': 5672}
]

client = RabbitMQClusterClient(cluster_nodes)
```

3. FEDERATION (Geographic Distribution):
```bash
# Enable federation plugin
rabbitmq-plugins enable rabbitmq_federation

# Create federation upstream
rabbitmqctl set_parameter federation-upstream east '{"uri":"amqp://guest:guest@east.example.com:5672"}'

# Create federation policy
rabbitmqctl set_policy federate-me "^federated\." '{"federation-upstream-set":"all"}'
```

PERFORMANCE OPTIMIZATION:

1. CONNECTION POOLING:
```python
import pika
import threading
from queue import Queue

class RabbitMQConnectionPool:
    def __init__(self, max_connections=10, **connection_params):
        self.max_connections = max_connections
        self.connection_params = connection_params
        self.pool = Queue(maxsize=max_connections)
        self.lock = threading.Lock()
        
        # Pre-create connections
        for _ in range(max_connections):
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(**connection_params)
            )
            self.pool.put(connection)
    
    def get_connection(self):
        """Get connection from pool"""
        return self.pool.get()
    
    def return_connection(self, connection):
        """Return connection to pool"""
        if connection and not connection.is_closed:
            self.pool.put(connection)
    
    def publish_message(self, exchange, routing_key, body):
        """Publish using pooled connection"""
        connection = self.get_connection()
        try:
            channel = connection.channel()
            channel.basic_publish(
                exchange=exchange,
                routing_key=routing_key,
                body=body
            )
            channel.close()
        finally:
            self.return_connection(connection)

# Thread-safe publisher
pool = RabbitMQConnectionPool(
    max_connections=20,
    host='localhost',
    heartbeat=600
)
```

2. BATCH PROCESSING:
```python
def batch_publisher(messages, batch_size=100):
    """Publish messages in batches for better performance"""
    connection = pika.BlockingConnection(pika.ConnectionParameters())
    channel = connection.channel()
    
    # Enable publisher confirms for reliability
    channel.confirm_delivery()
    
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        
        for message in batch:
            channel.basic_publish(
                exchange=message['exchange'],
                routing_key=message['routing_key'],
                body=message['body'],
                properties=pika.BasicProperties(delivery_mode=2)
            )
        
        # Wait for batch confirmation
        if channel.basic_ack():
            print(f"Batch {i//batch_size + 1} confirmed")
    
    connection.close()
```

3. CONSUMER OPTIMIZATION:
```python
def optimized_consumer():
    connection = pika.BlockingConnection(pika.ConnectionParameters())
    channel = connection.channel()
    
    # Optimize prefetch for throughput
    channel.basic_qos(prefetch_count=10)  # Process 10 messages at once
    
    def process_message_batch(ch, method, properties, body):
        try:
            # Process message
            result = process_message(body)
            
            # Batch acknowledgment
            ch.basic_ack(delivery_tag=method.delivery_tag, multiple=False)
            
        except Exception as e:
            # Reject and requeue
            ch.basic_nack(
                delivery_tag=method.delivery_tag,
                requeue=True
            )
    
    channel.basic_consume(
        queue='high_throughput_queue',
        on_message_callback=process_message_batch
    )
    
    print("Starting optimized consumer...")
    channel.start_consuming()
```

4. LOAD BALANCING STRATEGIES:
```python
# Round-robin consumer distribution
def setup_worker_queues():
    connection = pika.BlockingConnection(pika.ConnectionParameters())
    channel = connection.channel()
    
    # Create worker-specific queues
    for worker_id in range(1, 6):  # 5 workers
        queue_name = f'worker_{worker_id}_queue'
        channel.queue_declare(queue=queue_name, durable=True)
        
        # Bind to common exchange with worker routing
        channel.queue_bind(
            exchange='work_distribution',
            queue=queue_name,
            routing_key=f'worker.{worker_id}'
        )

def distribute_work(tasks):
    """Distribute tasks across workers"""
    connection = pika.BlockingConnection(pika.ConnectionParameters())
    channel = connection.channel()
    
    worker_count = 5
    
    for i, task in enumerate(tasks):
        worker_id = (i % worker_count) + 1
        
        channel.basic_publish(
            exchange='work_distribution',
            routing_key=f'worker.{worker_id}',
            body=json.dumps(task),
            properties=pika.BasicProperties(delivery_mode=2)
        )
```

MONITORING AND ALERTING:
```python
class RabbitMQHealthCheck:
    def __init__(self, management_url, username, password):
        self.management_url = management_url
        self.auth = (username, password)
    
    def check_cluster_health(self):
        """Comprehensive cluster health check"""
        alerts = []
        
        # Check node status
        nodes_response = requests.get(
            f'{self.management_url}/api/nodes',
            auth=self.auth
        )
        
        if nodes_response.status_code == 200:
            nodes = nodes_response.json()
            down_nodes = [n for n in nodes if not n.get('running', False)]
            
            if down_nodes:
                alerts.append(f"Nodes down: {[n['name'] for n in down_nodes]}")
        
        # Check queue lengths
        queues_response = requests.get(
            f'{self.management_url}/api/queues',
            auth=self.auth
        )
        
        if queues_response.status_code == 200:
            queues = queues_response.json()
            
            for queue in queues:
                messages = queue.get('messages', 0)
                if messages > 10000:  # Alert threshold
                    alerts.append(f"Queue {queue['name']} has {messages} messages")
        
        # Check memory usage
        overview_response = requests.get(
            f'{self.management_url}/api/overview',
            auth=self.auth
        )
        
        if overview_response.status_code == 200:
            overview = overview_response.json()
            memory_used = overview.get('queue_totals', {}).get('memory', 0)
            if memory_used > 1000000000:  # 1GB threshold
                alerts.append(f"High memory usage: {memory_used} bytes")
        
        return alerts
```
            ''',
            "follow_up": "What's the difference between RabbitMQ clustering and federation?"
        }
    ]
    
    for qa in questions_and_answers:
        print(f"\nQ: {qa['question']}")
        print(f"A: {qa['answer']}")
        if qa.get('follow_up'):
            print(f"Follow-up: {qa['follow_up']}")
        print("-" * 80)

# ===================================
# PART 4B: APACHE KAFKA FUNDAMENTALS
# ===================================

def apache_kafka_interview_questions():
    """
    Comprehensive Apache Kafka interview questions covering streaming and event sourcing
    """
    
    print("\n" + "=" * 80)
    print("APACHE KAFKA - INTERVIEW QUESTIONS & ANSWERS")
    print("=" * 80)
    
    questions_and_answers = [
        {
            "question": "1. What is Apache Kafka and how does it differ from traditional message brokers?",
            "answer": '''
Apache Kafka is a distributed event streaming platform designed for high-throughput, 
fault-tolerant, and scalable real-time data processing.

KAFKA vs TRADITIONAL MESSAGE BROKERS:

| Feature | Kafka | RabbitMQ/Redis |
|---------|-------|----------------|
| Message Model | Event Log/Stream | Queue-based |
| Storage | Persistent disk log | Memory + optional persistence |
| Retention | Configurable (days/size) | Until consumed |
| Consumer Model | Pull-based | Push/Pull |
| Replay | Yes (offset-based) | No |
| Partitioning | Built-in | Manual sharding |
| Ordering | Per-partition | Per-queue |
| Throughput | Very High | High |
| Latency | Medium | Low |
| Use Case | Event streaming, logs | Task queues, RPC |

KAFKA CORE CONCEPTS:

1. TOPICS & PARTITIONS:
```python
from kafka import KafkaProducer, KafkaConsumer
import json

# Producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    key_serializer=lambda k: k.encode('utf-8') if k else None
)

# Send message to topic with key (determines partition)
producer.send(
    topic='user_events',
    key='user_123',  # Messages with same key go to same partition
    value={
        'user_id': 'user_123',
        'event': 'login',
        'timestamp': '2023-08-01T10:00:00Z'
    }
)

producer.flush()  # Ensure all messages are sent
```

2. CONSUMER GROUPS:
```python
# Consumer in group - automatic load balancing
consumer = KafkaConsumer(
    'user_events',
    bootstrap_servers=['localhost:9092'],
    group_id='analytics_group',  # Consumer group
    auto_offset_reset='earliest',  # Start from beginning
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

for message in consumer:
    print(f"Partition: {message.partition}")
    print(f"Offset: {message.offset}")
    print(f"Key: {message.key}")
    print(f"Value: {message.value}")
    
    # Process event
    process_user_event(message.value)
```

3. KAFKA WITH DJANGO:
```python
# Django Kafka integration
from django.conf import settings
from kafka import KafkaProducer
import json

class EventPublisher:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=settings.KAFKA_BROKERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',  # Wait for all replicas
            retries=3,   # Retry failed sends
            batch_size=16384,  # Batch messages for efficiency
            linger_ms=10       # Wait 10ms to batch more messages
        )
    
    def publish_user_event(self, user_id, event_type, data=None):
        event = {
            'user_id': user_id,
            'event_type': event_type,
            'data': data or {},
            'timestamp': timezone.now().isoformat(),
            'source': 'django_app'
        }
        
        future = self.producer.send(
            topic='user_events',
            key=str(user_id),
            value=event
        )
        
        # Optional: wait for confirmation
        try:
            record_metadata = future.get(timeout=10)
            print(f"Message sent to {record_metadata.topic} partition {record_metadata.partition}")
        except Exception as e:
            print(f"Failed to send message: {e}")

# Usage in Django views
def user_signup(request):
    # Create user
    user = User.objects.create_user(...)
    
    # Publish event
    publisher = EventPublisher()
    publisher.publish_user_event(
        user_id=user.id,
        event_type='user_signup',
        data={'email': user.email, 'plan': 'free'}
    )
```

4. CONSUMER PATTERNS:
```python
# Single consumer
def simple_consumer():
    consumer = KafkaConsumer(
        'orders',
        bootstrap_servers=['localhost:9092'],
        group_id='order_processor',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    
    for message in consumer:
        process_order(message.value)

# Multi-topic consumer
def multi_topic_consumer():
    consumer = KafkaConsumer(
        'orders', 'payments', 'shipments',
        bootstrap_servers=['localhost:9092'],
        group_id='order_pipeline'
    )
    
    for message in consumer:
        if message.topic == 'orders':
            process_order(message.value)
        elif message.topic == 'payments':
            process_payment(message.value)
        elif message.topic == 'shipments':
            process_shipment(message.value)

# Manual offset management
def manual_offset_consumer():
    consumer = KafkaConsumer(
        'critical_events',
        bootstrap_servers=['localhost:9092'],
        group_id='critical_processor',
        enable_auto_commit=False  # Manual commit
    )
    
    for message in consumer:
        try:
            process_critical_event(message.value)
            # Commit only after successful processing
            consumer.commit()
        except Exception as e:
            print(f"Error processing message: {e}")
            # Don't commit - message will be reprocessed
```

KAFKA STREAMING PATTERNS:

1. EVENT SOURCING:
```python
class UserEventStore:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    
    def store_event(self, user_id, event_type, event_data):
        event = {
            'event_id': str(uuid.uuid4()),
            'user_id': user_id,
            'event_type': event_type,
            'event_data': event_data,
            'timestamp': timezone.now().isoformat(),
            'version': 1
        }
        
        self.producer.send(
            topic=f'user_events_{user_id}',  # Per-user topic
            key=str(user_id),
            value=event
        )
    
    def replay_user_events(self, user_id, from_timestamp=None):
        """Rebuild user state from events"""
        consumer = KafkaConsumer(
            f'user_events_{user_id}',
            bootstrap_servers=['localhost:9092'],
            auto_offset_reset='earliest',
            group_id=f'replay_{user_id}_{int(time.time())}'
        )
        
        user_state = {}
        
        for message in consumer:
            event = json.loads(message.value.decode('utf-8'))
            
            if from_timestamp and event['timestamp'] < from_timestamp:
                continue
                
            # Apply event to rebuild state
            user_state = apply_event(user_state, event)
        
        return user_state
```

2. CQRS (Command Query Responsibility Segregation):
```python
# Command side - writes to Kafka
class OrderCommandHandler:
    def __init__(self):
        self.event_store = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    
    def create_order(self, order_data):
        event = {
            'event_type': 'OrderCreated',
            'order_id': order_data['order_id'],
            'customer_id': order_data['customer_id'],
            'items': order_data['items'],
            'total': order_data['total'],
            'timestamp': timezone.now().isoformat()
        }
        
        self.event_store.send('order_events', value=event)

# Query side - reads from Kafka to build projections
class OrderProjectionBuilder:
    def __init__(self):
        self.consumer = KafkaConsumer(
            'order_events',
            bootstrap_servers=['localhost:9092'],
            group_id='order_projections',
            auto_offset_reset='earliest'
        )
    
    def build_projections(self):
        for message in self.consumer:
            event = json.loads(message.value.decode('utf-8'))
            
            if event['event_type'] == 'OrderCreated':
                self.update_order_summary(event)
                self.update_customer_orders(event)
                self.update_product_sales(event)
    
    def update_order_summary(self, event):
        # Update order summary table/cache
        pass
    
    def update_customer_orders(self, event):
        # Update customer order history
        pass
    
    def update_product_sales(self, event):
        # Update product sales analytics
        pass
```
            ''',
            "follow_up": "When would you choose Kafka over RabbitMQ for your application?"
        }
    ]
    
    for qa in questions_and_answers:
        print(f"\nQ: {qa['question']}")
        print(f"A: {qa['answer']}")
        if qa.get('follow_up'):
            print(f"Follow-up: {qa['follow_up']}")
        print("-" * 80)

def message_broker_comparison():
    """
    Comprehensive comparison of Redis, RabbitMQ, and Apache Kafka
    """
    
    print("\n" + "=" * 80)
    print("MESSAGE BROKER COMPARISON: REDIS vs RABBITMQ vs KAFKA")
    print("=" * 80)
    
    print("""
DETAILED COMPARISON MATRIX:

| Feature | Redis | RabbitMQ | Apache Kafka |
|---------|-------|----------|--------------|
| **Primary Use Case** | Cache, Simple Queues | Reliable Messaging | Event Streaming |
| **Message Model** | Pub/Sub, Lists | AMQP Queues | Event Log |
| **Persistence** | Optional (AOF/RDB) | Durable Queues | Persistent Log |
| **Message Ordering** | Not Guaranteed | FIFO per Queue | FIFO per Partition |
| **Message Replay** | No | No | Yes (Offset-based) |
| **Throughput** | Very High | High | Very High |
| **Latency** | Very Low (μs) | Low (ms) | Medium (ms) |
| **Scalability** | Horizontal (Cluster) | Vertical + Cluster | Horizontal (Native) |
| **Consumer Groups** | No | No | Yes |
| **Load Balancing** | Manual | Built-in | Automatic |
| **Protocol** | Redis Protocol | AMQP | Kafka Protocol |
| **Routing** | Simple | Advanced (Exchanges) | Topic-based |
| **Message TTL** | Yes | Yes | Configurable Retention |
| **Dead Letter Queue** | Manual | Built-in | Manual |
| **Monitoring** | Redis CLI/Tools | Management UI | JMX/Tools |
| **Learning Curve** | Easy | Medium | Steep |
| **Operational Complexity** | Low | Medium | High |

WHEN TO USE EACH:

1. USE REDIS WHEN:
```python
# Cache-heavy applications
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
    }
}

# Session storage
SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'default'

# Simple pub/sub for real-time features
def redis_pub_sub_example():
    import redis
    r = redis.Redis()
    
    # Publisher
    r.publish('chat_room_1', 'Hello everyone!')
    
    # Subscriber
    pubsub = r.pubsub()
    pubsub.subscribe('chat_room_1')
    
    for message in pubsub.listen():
        if message['type'] == 'message':
            print(f"New message: {message['data']}")

# Rate limiting
def rate_limit_example():
    import time
    
    def is_rate_limited(user_id, limit=100, window=3600):
        key = f'rate_limit:{user_id}'
        current = r.incr(key)
        if current == 1:
            r.expire(key, window)
        return current > limit
```

2. USE RABBITMQ WHEN:
```python
# Reliable task processing
from celery import Celery

app = Celery('reliable_tasks')
app.config_from_object('django.conf:settings', namespace='CELERY')

# settings.py
CELERY_BROKER_URL = 'amqp://guest:guest@localhost:5672//'
CELERY_TASK_ACKS_LATE = True
CELERY_WORKER_PREFETCH_MULTIPLIER = 1

@app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3})
def critical_task(self, data):
    # Critical business logic that must not be lost
    return process_critical_data(data)

# Complex routing scenarios
def complex_routing_example():
    import pika
    
    connection = pika.BlockingConnection(pika.ConnectionParameters())
    channel = connection.channel()
    
    # Topic exchange for complex routing
    channel.exchange_declare(exchange='logs', exchange_type='topic')
    
    # Different services subscribe to different patterns
    channel.queue_bind(exchange='logs', queue='error_handler', routing_key='*.error')
    channel.queue_bind(exchange='logs', queue='audit_log', routing_key='user.*')
    channel.queue_bind(exchange='logs', queue='all_logs', routing_key='#')

# Microservices communication
def microservice_communication():
    # Service A publishes events
    channel.basic_publish(
        exchange='microservices',
        routing_key='user.created',
        body=json.dumps({'user_id': 123, 'email': 'user@example.com'})
    )
    
    # Service B, C, D can all react to user.created events
```

3. USE KAFKA WHEN:
```python
# Event-driven architecture
class EventDrivenSystem:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    
    def publish_domain_event(self, event_type, entity_id, event_data):
        event = {
            'event_type': event_type,
            'entity_id': entity_id,
            'event_data': event_data,
            'timestamp': timezone.now().isoformat(),
            'service': 'user_service'
        }
        
        self.producer.send('domain_events', value=event)

# Real-time analytics pipeline
def analytics_pipeline():
    # Producer: Web application logs
    def log_user_action(user_id, action, metadata):
        event = {
            'user_id': user_id,
            'action': action,
            'metadata': metadata,
            'timestamp': time.time()
        }
        producer.send('user_actions', value=event)
    
    # Consumer: Real-time analytics
    def process_analytics():
        consumer = KafkaConsumer(
            'user_actions',
            bootstrap_servers=['localhost:9092'],
            group_id='analytics_processor'
        )
        
        for message in consumer:
            event = json.loads(message.value.decode('utf-8'))
            update_real_time_dashboard(event)
            update_user_behavior_model(event)

# Data pipeline with replay capability
def data_pipeline_with_replay():
    # Can replay events from any point in time
    consumer = KafkaConsumer(
        'transaction_events',
        bootstrap_servers=['localhost:9092'],
        group_id='fraud_detection_v2',
        auto_offset_reset='earliest'  # Start from beginning
    )
    
    # Reprocess all historical data with new fraud detection algorithm
    for message in consumer:
        transaction = json.loads(message.value.decode('utf-8'))
        fraud_score = new_fraud_detection_algorithm(transaction)
        
        if fraud_score > 0.8:
            flag_suspicious_transaction(transaction)
```

HYBRID ARCHITECTURES:

```python
# Using multiple message brokers together
class HybridMessagingArchitecture:
    def __init__(self):
        # Redis for caching and real-time features
        self.redis_client = redis.Redis(host='localhost', port=6379)
        
        # RabbitMQ for reliable task processing
        self.rabbitmq_connection = pika.BlockingConnection(
            pika.ConnectionParameters('localhost')
        )
        self.rabbitmq_channel = self.rabbitmq_connection.channel()
        
        # Kafka for event streaming and analytics
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    
    def process_user_signup(self, user_data):
        # 1. Cache user data immediately (Redis)
        self.redis_client.setex(
            f"user:{user_data['id']}", 
            3600, 
            json.dumps(user_data)
        )
        
        # 2. Queue reliable tasks (RabbitMQ)
        self.rabbitmq_channel.basic_publish(
            exchange='tasks',
            routing_key='send_welcome_email',
            body=json.dumps({'user_id': user_data['id']}),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        
        # 3. Stream event for analytics (Kafka)
        self.kafka_producer.send(
            'user_events',
            value={
                'event_type': 'user_signup',
                'user_id': user_data['id'],
                'timestamp': timezone.now().isoformat(),
                'metadata': user_data
            }
        )
        
        # 4. Real-time notification (Redis Pub/Sub)
        self.redis_client.publish(
            'user_notifications',
            json.dumps({
                'type': 'welcome',
                'user_id': user_data['id']
            })
        )

# Decision matrix for choosing message broker
def choose_message_broker(requirements):
    scores = {
        'redis': 0,
        'rabbitmq': 0, 
        'kafka': 0
    }
    
    # Weighting based on requirements
    if requirements.get('low_latency', False):
        scores['redis'] += 3
        scores['rabbitmq'] += 2
        scores['kafka'] += 1
    
    if requirements.get('message_durability', False):
        scores['rabbitmq'] += 3
        scores['kafka'] += 3
        scores['redis'] += 1
    
    if requirements.get('high_throughput', False):
        scores['kafka'] += 3
        scores['redis'] += 3
        scores['rabbitmq'] += 2
    
    if requirements.get('message_replay', False):
        scores['kafka'] += 3
        scores['redis'] += 0
        scores['rabbitmq'] += 0
    
    if requirements.get('complex_routing', False):
        scores['rabbitmq'] += 3
        scores['kafka'] += 1
        scores['redis'] += 1
    
    if requirements.get('operational_simplicity', False):
        scores['redis'] += 3
        scores['rabbitmq'] += 2
        scores['kafka'] += 1
    
    return max(scores, key=scores.get)

# Example usage
requirements = {
    'low_latency': True,
    'message_durability': True,
    'high_throughput': False,
    'message_replay': False,
    'complex_routing': True,
    'operational_simplicity': False
}

recommended_broker = choose_message_broker(requirements)
print(f"Recommended broker: {recommended_broker}")
```
    """)

if __name__ == "__main__":
    print("COMPREHENSIVE MESSAGE BROKERS INTERVIEW GUIDE - PART 4")
    print("=" * 60)
    print("This guide covers RabbitMQ, Apache Kafka, and broker comparisons")
    print("Use this as a comprehensive reference for message broker interviews")
    print("=" * 60)
    
    rabbitmq_fundamentals_interview_questions()
    apache_kafka_interview_questions()
    message_broker_comparison()
    
    print("\n" + "=" * 80)
    print("END OF PART 4 - RABBITMQ & APACHE KAFKA")
    print("COMPREHENSIVE INTERVIEW GUIDE COMPLETE!")
    print("=" * 80)
