"""
COMPREHENSIVE INTERVIEW GUIDE - PART 2: CELERY & TASK QUEUE MANAGEMENT
======================================================================

This guide covers Celery fundamentals, task management, error handling, and advanced patterns.
Comprehensive coverage of asynchronous task processing with practical examples.

Author: Interview Preparation Guide
Date: August 2025
Technologies: Celery, Redis, RabbitMQ, Django
"""

# ===================================
# PART 2: CELERY FUNDAMENTALS
# ===================================

def celery_fundamentals_interview_questions():
    """
    Comprehensive Celery interview questions covering all aspects of task queue management
    """
    
    print("=" * 80)
    print("CELERY FUNDAMENTALS - INTERVIEW QUESTIONS & ANSWERS")
    print("=" * 80)
    
    questions_and_answers = [
        {
            "question": "1. What is Celery, and why is it used? Explain the architecture.",
            "answer": """
Celery is a distributed task queue system that allows you to run tasks asynchronously 
across multiple workers and machines. It's built on message passing and follows the 
producer-consumer pattern.

CORE COMPONENTS:
1. Producer: Django app that creates and sends tasks
2. Message Broker: Stores and routes tasks (Redis/RabbitMQ)
3. Worker: Processes that execute tasks
4. Result Backend: Stores task results (optional)

ARCHITECTURE FLOW:
```
Django App -> Message Broker -> Celery Worker -> Result Backend
(Producer)    (Redis/RabbitMQ)   (Consumer)      (Redis/DB)
```

BASIC SETUP:
```python
# celery.py
from celery import Celery

app = Celery('myproject')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

# settings.py
CELERY_BROKER_URL = 'redis://localhost:6379'
CELERY_RESULT_BACKEND = 'redis://localhost:6379'
CELERY_ACCEPT_CONTENT = ['application/json']
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TASK_SERIALIZER = 'json'

# tasks.py
from celery import shared_task
import time

@shared_task
def send_email_task(email, subject, message):
    time.sleep(2)  # Simulate email sending
    print(f"Email sent to {email}")
    return f"Email sent to {email}"

@shared_task
def process_image_task(image_path):
    # Heavy image processing
    time.sleep(10)
    return "Image processed successfully"
```

STARTING WORKERS:
```bash
# Basic worker
celery -A myproject worker --loglevel=info

# Multiple workers
celery -A myproject worker --loglevel=info --concurrency=4

# Specific queues
celery -A myproject worker --queues=high_priority,default
```

USE CASES:
- Email sending
- Image/video processing
- Data export/import
- Periodic reports
- Web scraping
- Machine learning model training
            """,
            "follow_up": "What happens if the message broker goes down?"
        },
        
        {
            "question": "2. How does Celery execute tasks asynchronously? Explain task states.",
            "answer": """
Celery executes tasks through a message passing system with different execution states.

TASK EXECUTION FLOW:
1. Task created and sent to broker
2. Worker picks up task from queue
3. Task execution begins
4. Result stored in backend (if configured)
5. Task completion/failure handled

TASK STATES:
```python
from celery import states

# Task states flow:
PENDING -> STARTED -> SUCCESS/FAILURE/RETRY
```

DETAILED TASK STATES:
```python
# Basic states
PENDING     # Task waiting for execution
STARTED     # Task execution has begun (requires task_track_started=True)
SUCCESS     # Task executed successfully
FAILURE     # Task execution failed
RETRY       # Task failed but will be retried
REVOKED     # Task was revoked/cancelled

# Custom states
from celery import current_task

@shared_task(bind=True)
def long_running_task(self):
    self.update_state(
        state='PROGRESS',
        meta={'current': 0, 'total': 100}
    )
    
    for i in range(100):
        # Do work
        time.sleep(0.1)
        self.update_state(
            state='PROGRESS',
            meta={'current': i+1, 'total': 100}
        )
    
    return {'status': 'Complete', 'result': 'Task finished'}
```

TASK TRACKING AND MONITORING:
```python
# In views.py
from django.http import JsonResponse
from .tasks import long_running_task

def start_task(request):
    task = long_running_task.delay()
    return JsonResponse({'task_id': task.id})

def task_status(request, task_id):
    task = long_running_task.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Task is waiting to be processed'
        }
    elif task.state == 'PROGRESS':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
        }
    elif task.state == 'SUCCESS':
        response = {
            'state': task.state,
            'result': task.result
        }
    else:  # FAILURE
        response = {
            'state': task.state,
            'error': str(task.info)
        }
    
    return JsonResponse(response)
```

TASK CONFIGURATION:
```python
# Task options
@shared_task(
    bind=True,                    # Access to self (task instance)
    autoretry_for=(Exception,),   # Auto-retry on these exceptions
    retry_kwargs={'max_retries': 3, 'countdown': 60},
    retry_backoff=True,           # Exponential backoff
    retry_backoff_max=700,        # Max backoff time
    retry_jitter=False            # Add randomness to backoff
)
def robust_task(self, data):
    try:
        # Task logic here
        return process_data(data)
    except SpecificException as exc:
        # Custom retry logic
        raise self.retry(exc=exc, countdown=60, max_retries=3)
```
            """,
            "follow_up": "How do you handle task failures and implement retry logic?"
        },
        
        {
            "question": "3. What are the different Celery message brokers and their trade-offs?",
            "answer": """
Celery supports multiple message brokers, each with specific advantages and trade-offs.

BROKER COMPARISON:

| Feature | Redis | RabbitMQ | Amazon SQS |
|---------|-------|----------|------------|
| Speed | Very Fast (in-memory) | Fast | Moderate |
| Persistence | Optional (AOF/RDB) | Durable queues | Persistent |
| Reliability | Good | Excellent | Excellent |
| Setup Complexity | Simple | Moderate | Minimal |
| Scalability | Horizontal | Clustering | Auto-scale |
| Message Ordering | Not guaranteed | FIFO available | FIFO available |
| Cost | Free/Self-hosted | Free/Self-hosted | Pay-per-use |

1. REDIS AS BROKER:
```python
# settings.py
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

# Redis configuration for persistence
CELERY_BROKER_TRANSPORT_OPTIONS = {
    'visibility_timeout': 3600,
    'fanout_prefix': True,
    'fanout_patterns': True
}

# Pros:
- Extremely fast (in-memory)
- Simple setup and configuration
- Good for development and lightweight production
- Built-in pub/sub capabilities

# Cons:
- Potential message loss if not configured for persistence
- Memory limited
- Single-threaded (though fast)
```

2. RABBITMQ AS BROKER:
```python
# settings.py
CELERY_BROKER_URL = 'amqp://guest:guest@localhost:5672//'

# RabbitMQ specific settings
CELERY_TASK_ROUTES = {
    'myapp.tasks.critical_task': {'queue': 'critical'},
    'myapp.tasks.normal_task': {'queue': 'normal'},
}

# Queue configuration
from kombu import Queue

CELERY_TASK_QUEUES = (
    Queue('critical', routing_key='critical'),
    Queue('normal', routing_key='normal'),
    Queue('low_priority', routing_key='low'),
)

# Pros:
- Message durability and persistence
- Advanced routing capabilities
- Excellent reliability
- Support for priority queues
- Message acknowledgments

# Cons:
- More complex setup
- Requires additional service management
- Higher memory usage
```

3. AMAZON SQS AS BROKER:
```python
# settings.py
CELERY_BROKER_URL = 'sqs://AKIAIOSFODNN7EXAMPLE:wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY@'

# SQS specific settings
CELERY_BROKER_TRANSPORT_OPTIONS = {
    'region': 'us-east-1',
    'visibility_timeout': 240,
    'polling_interval': 1,
}

# Pros:
- Fully managed service
- Auto-scaling
- High availability
- No infrastructure management

# Cons:
- Higher latency
- Cost per message
- AWS vendor lock-in
- Limited message size
```

CHOOSING THE RIGHT BROKER:

Development:
```python
# Use Redis for development
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_TASK_ALWAYS_EAGER = True  # Execute tasks synchronously
```

Production (High Reliability):
```python
# Use RabbitMQ for production
CELERY_BROKER_URL = 'amqp://user:pass@rabbitmq:5672//'
CELERY_TASK_ACKS_LATE = True
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
```

Cloud/Microservices:
```python
# Use SQS for cloud deployments
CELERY_BROKER_URL = 'sqs://access_key:secret_key@'
```
            """,
            "follow_up": "How do you ensure message durability in each broker?"
        }
    ]
    
    for qa in questions_and_answers:
        print(f"\nQ: {qa['question']}")
        print(f"A: {qa['answer']}")
        if qa.get('follow_up'):
            print(f"Follow-up: {qa['follow_up']}")
        print("-" * 80)

def celery_advanced_patterns():
    """
    Advanced Celery patterns, task dependencies, and workflow management
    """
    
    print("\n" + "=" * 80)
    print("CELERY ADVANCED PATTERNS")
    print("=" * 80)
    
    print("""
1. TASK DEPENDENCIES AND WORKFLOWS:

CHAINS (Sequential Execution):
```python
from celery import chain, group, chord
from .tasks import add, multiply, send_notification

# Simple chain
workflow = chain(add.s(2, 2), multiply.s(4), send_notification.s())
result = workflow()

# More complex chain
def process_order_workflow(order_id):
    workflow = chain(
        validate_order.s(order_id),
        process_payment.s(),
        update_inventory.s(),
        send_confirmation_email.s(),
        generate_invoice.s()
    )
    return workflow.apply_async()
```

GROUPS (Parallel Execution):
```python
# Execute tasks in parallel
parallel_tasks = group(
    process_image.s('image1.jpg'),
    process_image.s('image2.jpg'),
    process_image.s('image3.jpg'),
)
result = parallel_tasks.apply_async()

# Wait for all to complete
results = result.get()
```

CHORDS (Parallel + Callback):
```python
# Process multiple items, then aggregate results
def bulk_process_workflow(items):
    # Process all items in parallel
    parallel_jobs = group(process_item.s(item) for item in items)
    
    # When all complete, aggregate results
    workflow = chord(parallel_jobs)(aggregate_results.s())
    return workflow.apply_async()

@shared_task
def aggregate_results(results):
    total = sum(results)
    send_summary_report.delay(total)
    return total
```

MAP-REDUCE PATTERN:
```python
# Map phase - parallel processing
@shared_task
def map_task(data_chunk):
    return [process_item(item) for item in data_chunk]

# Reduce phase - aggregate results
@shared_task
def reduce_task(results):
    return sum(results)

def map_reduce_workflow(large_dataset):
    # Split data into chunks
    chunks = [large_dataset[i:i+100] for i in range(0, len(large_dataset), 100)]
    
    # Map phase
    map_jobs = group(map_task.s(chunk) for chunk in chunks)
    
    # Reduce phase
    workflow = chord(map_jobs)(reduce_task.s())
    return workflow.apply_async()
```

2. ERROR HANDLING AND RETRY STRATEGIES:

CUSTOM RETRY LOGIC:
```python
from celery.exceptions import Retry, MaxRetriesExceededError
import random

@shared_task(bind=True, max_retries=5)
def unreliable_api_call(self, url, data):
    try:
        response = requests.post(url, json=data, timeout=10)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.Timeout as exc:
        # Retry immediately on timeout
        raise self.retry(exc=exc, countdown=1)
    
    except requests.exceptions.ConnectionError as exc:
        # Exponential backoff for connection errors
        countdown = 2 ** self.request.retries
        raise self.retry(exc=exc, countdown=countdown)
    
    except requests.exceptions.HTTPError as exc:
        if exc.response.status_code >= 500:
            # Retry server errors with jitter
            countdown = (2 ** self.request.retries) + random.uniform(0, 1)
            raise self.retry(exc=exc, countdown=countdown)
        else:
            # Don't retry client errors
            raise exc
    
    except MaxRetriesExceededError:
        # Send alert when all retries exhausted
        send_alert.delay(f"Task failed after {self.max_retries} retries")
        raise
```

DEAD LETTER QUEUE PATTERN:
```python
@shared_task(bind=True)
def process_with_dlq(self, data):
    try:
        return process_data(data)
    except Exception as exc:
        if self.request.retries >= self.max_retries:
            # Send to dead letter queue
            dead_letter_task.delay(data, str(exc))
            return f"Sent to DLQ: {exc}"
        raise self.retry(exc=exc)

@shared_task
def dead_letter_task(failed_data, error_message):
    # Log failure, send alert, store for manual review
    logger.error(f"Task failed permanently: {error_message}")
    FailedTask.objects.create(
        data=failed_data,
        error=error_message,
        timestamp=timezone.now()
    )
```

3. TASK ROUTING AND PRIORITIZATION:

QUEUE-BASED ROUTING:
```python
# settings.py
from kombu import Queue

CELERY_TASK_ROUTES = {
    'myapp.tasks.critical_task': {'queue': 'critical'},
    'myapp.tasks.normal_task': {'queue': 'normal'},
    'myapp.tasks.background_task': {'queue': 'background'},
}

CELERY_TASK_QUEUES = (
    Queue('critical', routing_key='critical'),
    Queue('normal', routing_key='normal'),
    Queue('background', routing_key='background'),
)

# Priority queues (RabbitMQ only)
CELERY_TASK_QUEUES = (
    Queue('high_priority', routing_key='high', queue_arguments={'x-max-priority': 10}),
    Queue('normal', routing_key='normal', queue_arguments={'x-max-priority': 5}),
)
```

WORKER SPECIALIZATION:
```bash
# Start workers for specific queues
celery -A myproject worker --queues=critical --hostname=critical@%h
celery -A myproject worker --queues=normal,background --hostname=general@%h

# CPU-intensive worker
celery -A myproject worker --queues=cpu_heavy --concurrency=2

# I/O intensive worker
celery -A myproject worker --queues=io_heavy --concurrency=20
```

4. MONITORING AND DEBUGGING:

TASK MONITORING:
```python
# Custom task base class with monitoring
from celery import Task
from celery.signals import task_prerun, task_postrun, task_failure

class MonitoredTask(Task):
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"Task {task_id} succeeded: {retval}")
    
    def on_failure(self, exc, task_id, args, kwargs, traceback):
        logger.error(f"Task {task_id} failed: {exc}")
        send_alert.delay(f"Task failure: {task_id} - {exc}")
    
    def on_retry(self, exc, task_id, args, kwargs, traceback):
        logger.warning(f"Task {task_id} retrying: {exc}")

@shared_task(base=MonitoredTask)
def monitored_task(data):
    return process_data(data)
```

TASK INSPECTION:
```python
from celery import current_app

# Get active tasks
i = current_app.control.inspect()
active_tasks = i.active()
scheduled_tasks = i.scheduled()
reserved_tasks = i.reserved()

# Revoke tasks
current_app.control.revoke(task_id, terminate=True)

# Task stats
stats = i.stats()
```
    """)

def celery_configuration_best_practices():
    """
    Celery configuration best practices for production environments
    """
    
    print("\n" + "=" * 80)
    print("CELERY PRODUCTION CONFIGURATION")
    print("=" * 80)
    
    print("""
PRODUCTION SETTINGS:

```python
# settings.py - Production Celery Configuration

# Broker settings
CELERY_BROKER_URL = 'redis://redis:6379/0'
CELERY_RESULT_BACKEND = 'redis://redis:6379/0'

# Task settings
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TIMEZONE = 'UTC'
CELERY_ENABLE_UTC = True

# Task execution settings
CELERY_TASK_ACKS_LATE = True  # Acknowledge after task completion
CELERY_WORKER_PREFETCH_MULTIPLIER = 1  # One task per worker
CELERY_TASK_REJECT_ON_WORKER_LOST = True

# Task routing
CELERY_TASK_ROUTES = {
    'myapp.tasks.send_email': {'queue': 'emails'},
    'myapp.tasks.process_image': {'queue': 'images'},
    'myapp.tasks.generate_report': {'queue': 'reports'},
}

# Worker settings
CELERY_WORKER_CONCURRENCY = 4  # Based on CPU cores
CELERY_WORKER_MAX_TASKS_PER_CHILD = 1000  # Restart worker after N tasks
CELERY_WORKER_DISABLE_RATE_LIMITS = False

# Monitoring
CELERY_SEND_TASK_EVENTS = True
CELERY_TASK_SEND_SENT_EVENT = True

# Error handling
CELERY_TASK_SOFT_TIME_LIMIT = 300  # 5 minutes
CELERY_TASK_TIME_LIMIT = 360      # 6 minutes (hard limit)

# Result backend settings
CELERY_RESULT_EXPIRES = 3600  # 1 hour

# Security
CELERY_WORKER_HIJACK_ROOT_LOGGER = False
CELERY_WORKER_LOG_COLOR = False

# Beat scheduler (for periodic tasks)
CELERY_BEAT_SCHEDULE = {
    'cleanup-old-results': {
        'task': 'myapp.tasks.cleanup_old_results',
        'schedule': crontab(minute=0, hour=2),  # 2 AM daily
    },
    'generate-daily-report': {
        'task': 'myapp.tasks.generate_daily_report',
        'schedule': crontab(minute=0, hour=8),  # 8 AM daily
    },
}
```

DOCKER CONFIGURATION:

```dockerfile
# Dockerfile for Celery worker
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Create celery user
RUN groupadd -r celery && useradd -r -g celery celery
RUN chown -R celery:celery /app
USER celery

CMD ["celery", "-A", "myproject", "worker", "--loglevel=info"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  worker:
    build: .
    command: celery -A myproject worker --loglevel=info
    volumes:
      - .:/app
    depends_on:
      - redis
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0

  beat:
    build: .
    command: celery -A myproject beat --loglevel=info
    volumes:
      - .:/app
    depends_on:
      - redis

  flower:
    build: .
    command: celery -A myproject flower
    ports:
      - "5555:5555"
    depends_on:
      - redis
```

MONITORING WITH FLOWER:

```python
# Install: pip install flower

# Start Flower
celery -A myproject flower

# Access web UI at http://localhost:5555

# Flower configuration
FLOWER_BASIC_AUTH = ['admin:password']
FLOWER_PORT = 5555
FLOWER_ADDRESS = '0.0.0.0'
```

SYSTEMD SERVICE CONFIGURATION:

```ini
# /etc/systemd/system/celery.service
[Unit]
Description=Celery Service
After=network.target

[Service]
Type=forking
User=celery
Group=celery
EnvironmentFile=/etc/conf.d/celery
WorkingDirectory=/opt/myproject
ExecStart=/opt/myproject/venv/bin/celery multi start worker1 \
    -A myproject --pidfile=/var/run/celery/%n.pid \
    --logfile=/var/log/celery/%n%I.log --loglevel=INFO
ExecStop=/opt/myproject/venv/bin/celery multi stopwait worker1 \
    --pidfile=/var/run/celery/%n.pid
ExecReload=/opt/myproject/venv/bin/celery multi restart worker1 \
    -A myproject --pidfile=/var/run/celery/%n.pid \
    --logfile=/var/log/celery/%n%I.log --loglevel=INFO

[Install]
WantedBy=multi-user.target
```

HEALTH CHECKS AND MONITORING:

```python
# Health check endpoint
from django.http import JsonResponse
from celery import current_app

def celery_health_check(request):
    try:
        # Check if workers are available
        i = current_app.control.inspect()
        stats = i.stats()
        
        if not stats:
            return JsonResponse({'status': 'unhealthy', 'error': 'No workers available'})
        
        # Check broker connection
        from celery import current_app
        current_app.connection().ensure_connection(max_retries=3)
        
        return JsonResponse({
            'status': 'healthy',
            'workers': len(stats),
            'broker': 'connected'
        })
    
    except Exception as e:
        return JsonResponse({
            'status': 'unhealthy',
            'error': str(e)
        }, status=503)
```
    """)

if __name__ == "__main__":
    print("COMPREHENSIVE CELERY INTERVIEW GUIDE - PART 2")
    print("=" * 60)
    print("This guide covers Celery fundamentals, advanced patterns, and production setup")
    print("Use this as a comprehensive reference for Celery interviews")
    print("=" * 60)
    
    celery_fundamentals_interview_questions()
    celery_advanced_patterns()
    celery_configuration_best_practices()
    
    print("\n" + "=" * 80)
    print("END OF PART 2 - CELERY & TASK QUEUE MANAGEMENT")
    print("Next: Part 3 - Redis Deep Dive")
    print("=" * 80)
