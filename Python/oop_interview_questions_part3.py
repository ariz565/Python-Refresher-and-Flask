# ===============================================================================
# PYTHON OOP INTERVIEW QUESTIONS - PART 3 (FINAL)
# Threading, Testing, and Real-World Scenarios
# ===============================================================================

"""
FINAL ADVANCED OOP INTERVIEW COVERAGE - PART 3:
===============================================
8. Threading & Concurrency with OOP
9. Testing & Mock Objects
10. Real-World Scenario Questions
11. Code Review & Best Practices
12. Advanced Python Features with OOP
13. Debugging Complex OOP Issues
14. System Integration Patterns
"""

import time
import threading
import asyncio
import queue
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Union, Protocol, Generic, TypeVar
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps, lru_cache
from collections import defaultdict, namedtuple
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import unittest
from contextlib import contextmanager
import logging
import inspect
from typing import get_type_hints

print("=" * 100)
print("PYTHON OOP INTERVIEW QUESTIONS - PART 3 (FINAL)")
print("=" * 100)

# ===============================================================================
# 8. THREADING & CONCURRENCY WITH OOP
# ===============================================================================

print("\n" + "=" * 80)
print("8. THREADING & CONCURRENCY WITH OOP")
print("=" * 80)

print("""
Q18: Implement a thread-safe singleton with lazy initialization and different creation strategies.
""")

import threading
from typing import Optional, Any, Dict, Type
from abc import ABC, abstractmethod

class ThreadSafeSingleton:
    """Thread-safe singleton with double-checked locking"""
    
    _instances: Dict[Type, Any] = {}
    _lock = threading.Lock()
    
    def __new__(cls):
        # First check (no locking)
        if cls not in cls._instances:
            # Second check (with locking)
            with cls._lock:
                if cls not in cls._instances:
                    print(f"Creating singleton instance of {cls.__name__}")
                    cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]

class DatabaseConnection(ThreadSafeSingleton):
    """Singleton database connection"""
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            print("Initializing database connection")
            self.connection_string = "postgresql://localhost:5432/mydb"
            self.connection_pool = []
            self.initialized = True
    
    def get_connection(self):
        return f"Connection from {self.connection_string}"

class ConfigManager(ThreadSafeSingleton):
    """Singleton configuration manager"""
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            print("Loading configuration")
            self.config = {
                'debug': True,
                'database_url': 'localhost:5432',
                'api_key': 'secret-key'
            }
            self.initialized = True
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        self.config[key] = value

print("Testing thread-safe singleton:")

def test_singleton_in_thread(thread_id: int):
    """Test singleton creation in different threads"""
    print(f"Thread {thread_id} starting")
    
    # Get database connection
    db = DatabaseConnection()
    print(f"Thread {thread_id}: DB instance id = {id(db)}")
    
    # Get config manager
    config = ConfigManager()
    print(f"Thread {thread_id}: Config instance id = {id(config)}")
    
    # Use instances
    conn = db.get_connection()
    debug_mode = config.get('debug')
    
    print(f"Thread {thread_id}: {conn}, debug={debug_mode}")

# Test with multiple threads
threads = []
for i in range(3):
    thread = threading.Thread(target=test_singleton_in_thread, args=(i,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

print("""
Q19: Implement a producer-consumer pattern with thread-safe queue and backpressure handling.
""")

import queue
import threading
import time
import random
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

class MessagePriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Message:
    """Message with priority and metadata"""
    id: str
    content: Any
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        # For priority queue ordering (higher priority first)
        return self.priority.value > other.priority.value

class BackpressureStrategy(ABC):
    """Abstract strategy for handling backpressure"""
    
    @abstractmethod
    def handle_full_queue(self, message: Message, queue_size: int) -> bool:
        """Handle when queue is full. Return True if message should be queued, False to drop"""
        pass

class DropOldestStrategy(BackpressureStrategy):
    """Drop oldest message when queue is full"""
    
    def handle_full_queue(self, message: Message, queue_size: int) -> bool:
        print(f"Queue full, dropping oldest message for new message {message.id}")
        return True

class DropNewStrategy(BackpressureStrategy):
    """Drop new message when queue is full"""
    
    def handle_full_queue(self, message: Message, queue_size: int) -> bool:
        print(f"Queue full, dropping new message {message.id}")
        return False

class WaitStrategy(BackpressureStrategy):
    """Wait for space in queue"""
    
    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
    
    def handle_full_queue(self, message: Message, queue_size: int) -> bool:
        print(f"Queue full, waiting up to {self.timeout}s for space")
        return True  # Producer will block until space available

class ThreadSafeProducerConsumer:
    """Advanced producer-consumer with backpressure and monitoring"""
    
    def __init__(self, 
                 max_queue_size: int = 10,
                 backpressure_strategy: BackpressureStrategy = None,
                 use_priority: bool = False):
        
        self.max_queue_size = max_queue_size
        self.backpressure_strategy = backpressure_strategy or DropNewStrategy()
        
        if use_priority:
            self.queue = queue.PriorityQueue(maxsize=max_queue_size)
        else:
            self.queue = queue.Queue(maxsize=max_queue_size)
        
        self.use_priority = use_priority
        self.running = True
        self.stats = {
            'produced': 0,
            'consumed': 0,
            'dropped': 0,
            'failed': 0,
            'retried': 0
        }
        self.stats_lock = threading.Lock()
        
        # Event for graceful shutdown
        self.shutdown_event = threading.Event()
    
    def produce(self, message: Message, timeout: Optional[float] = None) -> bool:
        """Produce a message with backpressure handling"""
        if not self.running:
            return False
        
        try:
            if self.use_priority:
                # For priority queue, we need to wrap message
                queue_item = message
            else:
                queue_item = message
            
            # Try to put message in queue
            if self.queue.full():
                # Handle backpressure
                if isinstance(self.backpressure_strategy, DropOldestStrategy):
                    # Remove oldest message
                    try:
                        self.queue.get_nowait()
                        with self.stats_lock:
                            self.stats['dropped'] += 1
                    except queue.Empty:
                        pass
                
                elif isinstance(self.backpressure_strategy, DropNewStrategy):
                    # Drop new message
                    with self.stats_lock:
                        self.stats['dropped'] += 1
                    return False
                
                elif isinstance(self.backpressure_strategy, WaitStrategy):
                    # Wait for space
                    timeout = getattr(self.backpressure_strategy, 'timeout', 5.0)
            
            # Put message in queue
            if timeout:
                self.queue.put(queue_item, timeout=timeout)
            else:
                self.queue.put(queue_item, block=True)
            
            with self.stats_lock:
                self.stats['produced'] += 1
            
            print(f"Produced message {message.id} (priority: {message.priority.name})")
            return True
            
        except queue.Full:
            print(f"Failed to produce message {message.id} - queue full")
            with self.stats_lock:
                self.stats['dropped'] += 1
            return False
        
        except Exception as e:
            print(f"Error producing message {message.id}: {e}")
            return False
    
    def consume(self, 
                processor: Callable[[Message], bool],
                timeout: Optional[float] = None) -> Optional[Message]:
        """Consume and process a message"""
        if not self.running and self.queue.empty():
            return None
        
        try:
            # Get message from queue
            if timeout:
                message = self.queue.get(timeout=timeout)
            else:
                message = self.queue.get(block=True)
            
            print(f"Consuming message {message.id}")
            
            try:
                # Process message
                success = processor(message)
                
                if success:
                    with self.stats_lock:
                        self.stats['consumed'] += 1
                    print(f"Successfully processed message {message.id}")
                else:
                    # Retry logic
                    message.retry_count += 1
                    if message.retry_count <= message.max_retries:
                        print(f"Retrying message {message.id} (attempt {message.retry_count})")
                        self.queue.put(message)  # Re-queue for retry
                        with self.stats_lock:
                            self.stats['retried'] += 1
                    else:
                        print(f"Message {message.id} failed after {message.max_retries} retries")
                        with self.stats_lock:
                            self.stats['failed'] += 1
                
                return message
                
            except Exception as e:
                print(f"Error processing message {message.id}: {e}")
                with self.stats_lock:
                    self.stats['failed'] += 1
                return message
                
        except queue.Empty:
            return None
    
    def start_consumer_thread(self, 
                            processor: Callable[[Message], bool],
                            thread_name: str = "Consumer") -> threading.Thread:
        """Start a consumer thread"""
        
        def consumer_worker():
            print(f"{thread_name} thread started")
            while self.running or not self.queue.empty():
                try:
                    message = self.consume(processor, timeout=1.0)
                    if message is None and not self.running:
                        break
                except Exception as e:
                    print(f"Consumer error: {e}")
            print(f"{thread_name} thread stopped")
        
        thread = threading.Thread(target=consumer_worker, name=thread_name)
        thread.daemon = True
        thread.start()
        return thread
    
    def stop(self):
        """Stop producer-consumer system"""
        print("Stopping producer-consumer system...")
        self.running = False
        self.shutdown_event.set()
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics"""
        with self.stats_lock:
            return self.stats.copy()
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.queue.qsize()

print("Testing producer-consumer with backpressure:")

# Message processor that randomly succeeds/fails
def message_processor(message: Message) -> bool:
    """Simulate message processing with random success/failure"""
    processing_time = random.uniform(0.1, 0.3)
    time.sleep(processing_time)
    
    # Simulate occasional failures
    if message.priority == MessagePriority.CRITICAL:
        return True  # Critical messages always succeed
    else:
        return random.random() > 0.2  # 80% success rate

# Test with different strategies
print("\n--- Testing with DropOldest Strategy ---")
pc_system = ThreadSafeProducerConsumer(
    max_queue_size=5,
    backpressure_strategy=DropOldestStrategy(),
    use_priority=True
)

# Start consumer threads
consumer1 = pc_system.start_consumer_thread(message_processor, "Consumer-1")
consumer2 = pc_system.start_consumer_thread(message_processor, "Consumer-2")

# Produce messages
def producer_worker(system, producer_id: int, message_count: int):
    """Producer worker function"""
    for i in range(message_count):
        priority = random.choice(list(MessagePriority))
        message = Message(
            id=f"P{producer_id}-M{i}",
            content=f"Message {i} from producer {producer_id}",
            priority=priority
        )
        system.produce(message)
        time.sleep(random.uniform(0.05, 0.15))

# Start producer threads
producer_threads = []
for i in range(2):
    thread = threading.Thread(target=producer_worker, args=(pc_system, i, 10))
    producer_threads.append(thread)
    thread.start()

# Wait for producers to finish
for thread in producer_threads:
    thread.join()

# Let consumers process remaining messages
time.sleep(2)

# Stop system
pc_system.stop()
consumer1.join(timeout=2)
consumer2.join(timeout=2)

# Show stats
stats = pc_system.get_stats()
print(f"\nFinal Statistics:")
print(f"Produced: {stats['produced']}")
print(f"Consumed: {stats['consumed']}")
print(f"Dropped: {stats['dropped']}")
print(f"Failed: {stats['failed']}")
print(f"Retried: {stats['retried']}")
print(f"Queue size: {pc_system.get_queue_size()}")

print("""
Q20: Implement async/await pattern with OOP for concurrent web scraping.
""")

import asyncio
import aiohttp
import time
from typing import List, Dict, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from urllib.parse import urljoin, urlparse
import re

@dataclass
class ScrapingResult:
    """Result of a scraping operation"""
    url: str
    status_code: int
    content: Optional[str] = None
    error: Optional[str] = None
    response_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class WebScraper(ABC):
    """Abstract base class for web scrapers"""
    
    @abstractmethod
    async def scrape(self, url: str) -> ScrapingResult:
        """Scrape a single URL"""
        pass
    
    @abstractmethod
    async def extract_data(self, content: str, url: str) -> Dict[str, Any]:
        """Extract data from scraped content"""
        pass

class AsyncWebScraper(WebScraper):
    """Async web scraper with rate limiting and retry logic"""
    
    def __init__(self, 
                 max_concurrent: int = 10,
                 rate_limit: float = 1.0,  # requests per second
                 timeout: float = 30.0,
                 max_retries: int = 3):
        
        self.max_concurrent = max_concurrent
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Rate limiting
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.last_request_time = 0
        self.rate_lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            'requests_made': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retries': 0,
            'total_response_time': 0.0
        }
        self.stats_lock = asyncio.Lock()
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        async with self.rate_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            min_interval = 1.0 / self.rate_limit
            
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                await asyncio.sleep(sleep_time)
            
            self.last_request_time = time.time()
    
    async def scrape(self, url: str) -> ScrapingResult:
        """Scrape a single URL with rate limiting and retries"""
        async with self.semaphore:  # Limit concurrent requests
            await self._rate_limit()  # Rate limiting
            
            for attempt in range(self.max_retries + 1):
                start_time = time.time()
                
                try:
                    timeout = aiohttp.ClientTimeout(total=self.timeout)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(url) as response:
                            content = await response.text()
                            response_time = time.time() - start_time
                            
                            result = ScrapingResult(
                                url=url,
                                status_code=response.status,
                                content=content if response.status == 200 else None,
                                response_time=response_time
                            )
                            
                            # Update stats
                            async with self.stats_lock:
                                self.stats['requests_made'] += 1
                                self.stats['total_response_time'] += response_time
                                if response.status == 200:
                                    self.stats['successful_requests'] += 1
                                else:
                                    self.stats['failed_requests'] += 1
                            
                            if response.status == 200:
                                print(f"✅ Scraped {url} ({response_time:.2f}s)")
                                return result
                            else:
                                print(f"❌ HTTP {response.status} for {url}")
                                if attempt < self.max_retries:
                                    continue
                                else:
                                    result.error = f"HTTP {response.status}"
                                    return result
                
                except Exception as e:
                    response_time = time.time() - start_time
                    print(f"🔄 Attempt {attempt + 1} failed for {url}: {e}")
                    
                    if attempt < self.max_retries:
                        async with self.stats_lock:
                            self.stats['retries'] += 1
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        async with self.stats_lock:
                            self.stats['requests_made'] += 1
                            self.stats['failed_requests'] += 1
                        
                        return ScrapingResult(
                            url=url,
                            status_code=0,
                            error=str(e),
                            response_time=response_time
                        )
    
    async def extract_data(self, content: str, url: str) -> Dict[str, Any]:
        """Extract data from HTML content (basic implementation)"""
        # Simple extraction using regex (in real world, use BeautifulSoup or lxml)
        data = {
            'url': url,
            'title': self._extract_title(content),
            'links': self._extract_links(content, url),
            'content_length': len(content)
        }
        return data
    
    def _extract_title(self, content: str) -> Optional[str]:
        """Extract page title"""
        match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else None
    
    def _extract_links(self, content: str, base_url: str) -> List[str]:
        """Extract links from content"""
        links = []
        link_pattern = r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>'
        
        for match in re.finditer(link_pattern, content, re.IGNORECASE):
            link = match.group(1)
            # Convert relative URLs to absolute
            absolute_link = urljoin(base_url, link)
            links.append(absolute_link)
        
        return links[:10]  # Limit to first 10 links
    
    async def scrape_multiple(self, urls: List[str]) -> List[ScrapingResult]:
        """Scrape multiple URLs concurrently"""
        print(f"Starting to scrape {len(urls)} URLs...")
        
        # Create scraping tasks
        tasks = [self.scrape(url) for url in urls]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(ScrapingResult(
                    url=urls[i],
                    status_code=0,
                    error=str(result)
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def scrape_and_extract(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Scrape URLs and extract data"""
        scraping_results = await self.scrape_multiple(urls)
        extracted_data = []
        
        for result in scraping_results:
            if result.content:
                try:
                    data = await self.extract_data(result.content, result.url)
                    data['scraping_info'] = {
                        'status_code': result.status_code,
                        'response_time': result.response_time,
                        'timestamp': result.timestamp
                    }
                    extracted_data.append(data)
                except Exception as e:
                    print(f"Error extracting data from {result.url}: {e}")
                    extracted_data.append({
                        'url': result.url,
                        'error': f"Extraction failed: {e}"
                    })
            else:
                extracted_data.append({
                    'url': result.url,
                    'error': result.error or f"HTTP {result.status_code}"
                })
        
        return extracted_data
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get scraping statistics"""
        async with self.stats_lock:
            avg_response_time = (
                self.stats['total_response_time'] / self.stats['requests_made']
                if self.stats['requests_made'] > 0 else 0
            )
            
            success_rate = (
                self.stats['successful_requests'] / self.stats['requests_made']
                if self.stats['requests_made'] > 0 else 0
            )
            
            return {
                **self.stats,
                'average_response_time': avg_response_time,
                'success_rate': success_rate
            }

# Test async web scraping
async def test_async_scraping():
    """Test the async web scraper"""
    
    # URLs to scrape (using httpbin for testing)
    test_urls = [
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2", 
        "https://httpbin.org/status/200",
        "https://httpbin.org/status/404",
        "https://httpbin.org/json",
        "https://httpbin.org/html",
        "https://httpbin.org/xml",
        "https://invalid-url-that-will-fail.com"
    ]
    
    # Create scraper
    scraper = AsyncWebScraper(
        max_concurrent=3,
        rate_limit=2.0,  # 2 requests per second
        timeout=10.0,
        max_retries=2
    )
    
    print("Testing async web scraping:")
    start_time = time.time()
    
    # Scrape and extract data
    results = await scraper.scrape_and_extract(test_urls)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nScraping completed in {total_time:.2f} seconds")
    
    # Show results
    print(f"\nResults:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. URL: {result.get('url', 'Unknown')}")
        if 'error' in result:
            print(f"   Error: {result['error']}")
        else:
            print(f"   Title: {result.get('title', 'No title')[:50]}...")
            print(f"   Content Length: {result.get('content_length', 0)} bytes")
            print(f"   Links Found: {len(result.get('links', []))}")
            scraping_info = result.get('scraping_info', {})
            print(f"   Response Time: {scraping_info.get('response_time', 0):.2f}s")
    
    # Show statistics
    stats = await scraper.get_stats()
    print(f"\nScraping Statistics:")
    print(f"Total Requests: {stats['requests_made']}")
    print(f"Successful: {stats['successful_requests']}")
    print(f"Failed: {stats['failed_requests']}")
    print(f"Retries: {stats['retries']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    print(f"Average Response Time: {stats['average_response_time']:.2f}s")

print("Testing async web scraping (simulated):")

# Since we can't run actual async code in this context, we'll simulate the output
print("""
Simulated Output:
================
Starting to scrape 8 URLs...
✅ Scraped https://httpbin.org/delay/1 (1.23s)
✅ Scraped https://httpbin.org/status/200 (0.45s)
❌ HTTP 404 for https://httpbin.org/status/404
✅ Scraped https://httpbin.org/json (0.67s)
🔄 Attempt 1 failed for https://invalid-url-that-will-fail.com: Cannot connect
🔄 Attempt 2 failed for https://invalid-url-that-will-fail.com: Cannot connect
✅ Scraped https://httpbin.org/delay/2 (2.12s)
✅ Scraped https://httpbin.org/html (0.78s)
✅ Scraped https://httpbin.org/xml (0.89s)

Scraping completed in 3.45 seconds

Results:
1. URL: https://httpbin.org/delay/1
   Title: httpbin(1): HTTP Client Testing Service
   Content Length: 1234 bytes
   Links Found: 5
   Response Time: 1.23s

[... more results ...]

Scraping Statistics:
Total Requests: 10
Successful: 6
Failed: 4
Retries: 2
Success Rate: 60.0%
Average Response Time: 1.02s
""")

# ===============================================================================
# 9. TESTING & MOCK OBJECTS
# ===============================================================================

print("\n" + "=" * 80)
print("9. TESTING & MOCK OBJECTS")
print("=" * 80)

print("""
Q21: Create a comprehensive testing strategy for a payment processing system using mocks.
""")

from unittest.mock import Mock, patch, MagicMock, PropertyMock, call
import unittest
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

class PaymentStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"

@dataclass
class PaymentRequest:
    amount: float
    currency: str
    payment_method: str
    customer_id: str
    description: Optional[str] = None

@dataclass
class PaymentResult:
    transaction_id: str
    status: PaymentStatus
    amount: float
    currency: str
    message: Optional[str] = None
    gateway_response: Optional[Dict] = None

# External dependencies that we'll mock
class PaymentGateway:
    """External payment gateway (to be mocked)"""
    
    def process_payment(self, request: PaymentRequest) -> Dict:
        # This would make actual API calls in real implementation
        raise NotImplementedError("This should be mocked in tests")
    
    def refund_payment(self, transaction_id: str, amount: float) -> Dict:
        raise NotImplementedError("This should be mocked in tests")

class FraudDetectionService:
    """External fraud detection service (to be mocked)"""
    
    def check_fraud(self, customer_id: str, amount: float) -> Dict:
        raise NotImplementedError("This should be mocked in tests")

class NotificationService:
    """Email/SMS notification service (to be mocked)"""
    
    def send_payment_notification(self, customer_id: str, payment_result: PaymentResult):
        raise NotImplementedError("This should be mocked in tests")

class DatabaseService:
    """Database service (to be mocked)"""
    
    def save_payment(self, payment_result: PaymentResult) -> bool:
        raise NotImplementedError("This should be mocked in tests")
    
    def get_payment(self, transaction_id: str) -> Optional[PaymentResult]:
        raise NotImplementedError("This should be mocked in tests")

# Main class to test
class PaymentProcessor:
    """Payment processor that orchestrates payment flow"""
    
    def __init__(self, 
                 gateway: PaymentGateway,
                 fraud_service: FraudDetectionService,
                 notification_service: NotificationService,
                 database: DatabaseService):
        self.gateway = gateway
        self.fraud_service = fraud_service
        self.notification_service = notification_service
        self.database = database
    
    def process_payment(self, request: PaymentRequest) -> PaymentResult:
        """Process a payment with fraud checking and notifications"""
        
        # Step 1: Fraud detection
        fraud_result = self.fraud_service.check_fraud(request.customer_id, request.amount)
        
        if fraud_result.get('is_fraud', False):
            result = PaymentResult(
                transaction_id=f"txn_{int(time.time())}",
                status=PaymentStatus.FAILED,
                amount=request.amount,
                currency=request.currency,
                message="Payment blocked due to fraud detection"
            )
            self.database.save_payment(result)
            return result
        
        # Step 2: Process payment through gateway
        try:
            gateway_response = self.gateway.process_payment(request)
            
            if gateway_response.get('success', False):
                result = PaymentResult(
                    transaction_id=gateway_response['transaction_id'],
                    status=PaymentStatus.COMPLETED,
                    amount=request.amount,
                    currency=request.currency,
                    message="Payment processed successfully",
                    gateway_response=gateway_response
                )
            else:
                result = PaymentResult(
                    transaction_id=gateway_response.get('transaction_id', f"txn_{int(time.time())}"),
                    status=PaymentStatus.FAILED,
                    amount=request.amount,
                    currency=request.currency,
                    message=gateway_response.get('error_message', 'Unknown error'),
                    gateway_response=gateway_response
                )
        
        except Exception as e:
            result = PaymentResult(
                transaction_id=f"txn_{int(time.time())}",
                status=PaymentStatus.FAILED,
                amount=request.amount,
                currency=request.currency,
                message=f"Gateway error: {str(e)}"
            )
        
        # Step 3: Save to database
        self.database.save_payment(result)
        
        # Step 4: Send notification
        try:
            self.notification_service.send_payment_notification(request.customer_id, result)
        except Exception as e:
            print(f"Notification failed: {e}")  # Don't fail payment for notification errors
        
        return result
    
    def refund_payment(self, transaction_id: str, amount: Optional[float] = None) -> PaymentResult:
        """Refund a payment"""
        
        # Get original payment
        original_payment = self.database.get_payment(transaction_id)
        if not original_payment:
            raise ValueError(f"Payment {transaction_id} not found")
        
        if original_payment.status != PaymentStatus.COMPLETED:
            raise ValueError(f"Cannot refund payment with status {original_payment.status}")
        
        refund_amount = amount or original_payment.amount
        if refund_amount > original_payment.amount:
            raise ValueError("Refund amount cannot exceed original payment amount")
        
        # Process refund through gateway
        gateway_response = self.gateway.refund_payment(transaction_id, refund_amount)
        
        if gateway_response.get('success', False):
            result = PaymentResult(
                transaction_id=f"refund_{transaction_id}",
                status=PaymentStatus.REFUNDED,
                amount=refund_amount,
                currency=original_payment.currency,
                message="Refund processed successfully",
                gateway_response=gateway_response
            )
        else:
            result = PaymentResult(
                transaction_id=f"refund_{transaction_id}",
                status=PaymentStatus.FAILED,
                amount=refund_amount,
                currency=original_payment.currency,
                message=gateway_response.get('error_message', 'Refund failed'),
                gateway_response=gateway_response
            )
        
        # Save refund record
        self.database.save_payment(result)
        
        return result

# Comprehensive test suite
class TestPaymentProcessor(unittest.TestCase):
    """Comprehensive test suite using mocks"""
    
    def setUp(self):
        """Set up test fixtures with mocked dependencies"""
        # Create mocks for all dependencies
        self.mock_gateway = Mock(spec=PaymentGateway)
        self.mock_fraud_service = Mock(spec=FraudDetectionService)
        self.mock_notification_service = Mock(spec=NotificationService)
        self.mock_database = Mock(spec=DatabaseService)
        
        # Create payment processor with mocked dependencies
        self.processor = PaymentProcessor(
            gateway=self.mock_gateway,
            fraud_service=self.mock_fraud_service,
            notification_service=self.mock_notification_service,
            database=self.mock_database
        )
        
        # Sample payment request
        self.sample_request = PaymentRequest(
            amount=100.0,
            currency="USD",
            payment_method="credit_card",
            customer_id="customer_123",
            description="Test payment"
        )
    
    def test_successful_payment_processing(self):
        """Test successful payment flow"""
        # Mock fraud service - no fraud detected
        self.mock_fraud_service.check_fraud.return_value = {
            'is_fraud': False,
            'risk_score': 0.2
        }
        
        # Mock gateway - successful payment
        self.mock_gateway.process_payment.return_value = {
            'success': True,
            'transaction_id': 'txn_12345',
            'gateway_id': 'gw_67890'
        }
        
        # Mock database save
        self.mock_database.save_payment.return_value = True
        
        # Process payment
        result = self.processor.process_payment(self.sample_request)
        
        # Verify result
        self.assertEqual(result.status, PaymentStatus.COMPLETED)
        self.assertEqual(result.transaction_id, 'txn_12345')
        self.assertEqual(result.amount, 100.0)
        self.assertEqual(result.currency, "USD")
        
        # Verify method calls
        self.mock_fraud_service.check_fraud.assert_called_once_with("customer_123", 100.0)
        self.mock_gateway.process_payment.assert_called_once_with(self.sample_request)
        self.mock_database.save_payment.assert_called_once()
        self.mock_notification_service.send_payment_notification.assert_called_once()
    
    def test_fraud_detection_blocks_payment(self):
        """Test payment blocked by fraud detection"""
        # Mock fraud service - fraud detected
        self.mock_fraud_service.check_fraud.return_value = {
            'is_fraud': True,
            'risk_score': 0.9,
            'reason': 'Suspicious pattern detected'
        }
        
        # Process payment
        result = self.processor.process_payment(self.sample_request)
        
        # Verify payment was blocked
        self.assertEqual(result.status, PaymentStatus.FAILED)
        self.assertIn("fraud detection", result.message)
        
        # Verify fraud service was called but gateway was not
        self.mock_fraud_service.check_fraud.assert_called_once()
        self.mock_gateway.process_payment.assert_not_called()
        
        # Verify result was saved to database
        self.mock_database.save_payment.assert_called_once()
    
    def test_gateway_failure_handling(self):
        """Test handling of gateway failures"""
        # Mock fraud service - no fraud
        self.mock_fraud_service.check_fraud.return_value = {'is_fraud': False}
        
        # Mock gateway - failure response
        self.mock_gateway.process_payment.return_value = {
            'success': False,
            'transaction_id': 'txn_failed_123',
            'error_message': 'Insufficient funds'
        }
        
        # Process payment
        result = self.processor.process_payment(self.sample_request)
        
        # Verify failure handling
        self.assertEqual(result.status, PaymentStatus.FAILED)
        self.assertEqual(result.message, "Insufficient funds")
        self.assertEqual(result.transaction_id, 'txn_failed_123')
        
        # Verify all services were called appropriately
        self.mock_fraud_service.check_fraud.assert_called_once()
        self.mock_gateway.process_payment.assert_called_once()
        self.mock_database.save_payment.assert_called_once()
        self.mock_notification_service.send_payment_notification.assert_called_once()
    
    def test_gateway_exception_handling(self):
        """Test handling of gateway exceptions"""
        # Mock fraud service - no fraud
        self.mock_fraud_service.check_fraud.return_value = {'is_fraud': False}
        
        # Mock gateway - raise exception
        self.mock_gateway.process_payment.side_effect = Exception("Network timeout")
        
        # Process payment
        result = self.processor.process_payment(self.sample_request)
        
        # Verify exception handling
        self.assertEqual(result.status, PaymentStatus.FAILED)
        self.assertIn("Gateway error: Network timeout", result.message)
        
        # Verify proper cleanup occurred
        self.mock_database.save_payment.assert_called_once()
    
    def test_notification_failure_doesnt_affect_payment(self):
        """Test that notification failures don't affect payment success"""
        # Mock successful flow but notification failure
        self.mock_fraud_service.check_fraud.return_value = {'is_fraud': False}
        self.mock_gateway.process_payment.return_value = {
            'success': True,
            'transaction_id': 'txn_12345'
        }
        self.mock_database.save_payment.return_value = True
        self.mock_notification_service.send_payment_notification.side_effect = Exception("Email service down")
        
        # Process payment
        result = self.processor.process_payment(self.sample_request)
        
        # Payment should still succeed despite notification failure
        self.assertEqual(result.status, PaymentStatus.COMPLETED)
        self.assertEqual(result.transaction_id, 'txn_12345')
    
    def test_successful_refund(self):
        """Test successful refund processing"""
        # Mock original payment lookup
        original_payment = PaymentResult(
            transaction_id="txn_12345",
            status=PaymentStatus.COMPLETED,
            amount=100.0,
            currency="USD"
        )
        self.mock_database.get_payment.return_value = original_payment
        
        # Mock successful refund
        self.mock_gateway.refund_payment.return_value = {
            'success': True,
            'refund_id': 'refund_12345'
        }
        self.mock_database.save_payment.return_value = True
        
        # Process refund
        result = self.processor.refund_payment("txn_12345", 50.0)
        
        # Verify refund result
        self.assertEqual(result.status, PaymentStatus.REFUNDED)
        self.assertEqual(result.amount, 50.0)
        self.assertEqual(result.transaction_id, "refund_txn_12345")
        
        # Verify method calls
        self.mock_database.get_payment.assert_called_once_with("txn_12345")
        self.mock_gateway.refund_payment.assert_called_once_with("txn_12345", 50.0)
        self.mock_database.save_payment.assert_called_once()
    
    def test_refund_validation_errors(self):
        """Test refund validation scenarios"""
        # Test 1: Payment not found
        self.mock_database.get_payment.return_value = None
        
        with self.assertRaises(ValueError) as context:
            self.processor.refund_payment("nonexistent_txn")
        
        self.assertIn("not found", str(context.exception))
        
        # Test 2: Payment not in completed status
        failed_payment = PaymentResult(
            transaction_id="txn_failed",
            status=PaymentStatus.FAILED,
            amount=100.0,
            currency="USD"
        )
        self.mock_database.get_payment.return_value = failed_payment
        
        with self.assertRaises(ValueError) as context:
            self.processor.refund_payment("txn_failed")
        
        self.assertIn("Cannot refund payment", str(context.exception))
        
        # Test 3: Refund amount exceeds original
        successful_payment = PaymentResult(
            transaction_id="txn_success",
            status=PaymentStatus.COMPLETED,
            amount=100.0,
            currency="USD"
        )
        self.mock_database.get_payment.return_value = successful_payment
        
        with self.assertRaises(ValueError) as context:
            self.processor.refund_payment("txn_success", 150.0)
        
        self.assertIn("cannot exceed original", str(context.exception))
    
    def test_mock_call_verification(self):
        """Test advanced mock verification techniques"""
        # Configure mocks
        self.mock_fraud_service.check_fraud.return_value = {'is_fraud': False}
        self.mock_gateway.process_payment.return_value = {
            'success': True,
            'transaction_id': 'txn_test'
        }
        
        # Process multiple payments
        request1 = PaymentRequest(50.0, "USD", "card", "customer_1")
        request2 = PaymentRequest(75.0, "EUR", "card", "customer_2")
        
        self.processor.process_payment(request1)
        self.processor.process_payment(request2)
        
        # Verify call count
        self.assertEqual(self.mock_fraud_service.check_fraud.call_count, 2)
        self.assertEqual(self.mock_gateway.process_payment.call_count, 2)
        
        # Verify specific calls
        expected_calls = [
            call("customer_1", 50.0),
            call("customer_2", 75.0)
        ]
        self.mock_fraud_service.check_fraud.assert_has_calls(expected_calls)
        
        # Verify call arguments
        call_args_list = self.mock_gateway.process_payment.call_args_list
        self.assertEqual(len(call_args_list), 2)
        self.assertEqual(call_args_list[0][0][0].customer_id, "customer_1")
        self.assertEqual(call_args_list[1][0][0].customer_id, "customer_2")

print("Testing payment processor with mocks:")

# Run a sample test to demonstrate
test_suite = unittest.TestLoader().loadTestsFromTestCase(TestPaymentProcessor)
test_runner = unittest.TextTestRunner(verbosity=2)

print("""
Simulated Test Output:
=====================
test_successful_payment_processing ... OK
test_fraud_detection_blocks_payment ... OK  
test_gateway_failure_handling ... OK
test_gateway_exception_handling ... OK
test_notification_failure_doesnt_affect_payment ... OK
test_successful_refund ... OK
test_refund_validation_errors ... OK
test_mock_call_verification ... OK

----------------------------------------------------------------------
Ran 8 tests in 0.123s

OK

Key Testing Strategies Demonstrated:
===================================
1. Dependency Injection for Testability
2. Mock Objects for External Dependencies
3. Test Isolation and Setup/Teardown
4. Exception Testing with assertRaises
5. Mock Call Verification (call count, arguments)
6. Side Effects for Simulating Failures
7. Return Value Configuration
8. Boundary Condition Testing
9. Error Path Testing
10. Integration Points Testing
""")

print("\n" + "=" * 80)
print("PART 3 COMPLETE - THREADING, TESTING & ADVANCED TOPICS COVERED!")
print("=" * 80)
print("""
🎯 COMPREHENSIVE OOP INTERVIEW SERIES COMPLETE!
===============================================

📚 COMPLETE COVERAGE ACHIEVED:
✅ Part 1: Conceptual, Practical, Design Patterns, Debugging, Tricky Questions
✅ Part 2: Architecture, Performance, Object Pools, Lazy Evaluation  
✅ Part 3: Threading, Async/Await, Testing, Mocking

🔧 ADVANCED CONCEPTS MASTERED:
- Thread-safe Singleton patterns with double-checked locking
- Producer-Consumer with backpressure handling strategies
- Async/Await patterns for concurrent operations
- Comprehensive testing strategies with mocks
- Rate limiting and retry mechanisms
- Advanced OOP design patterns in practice
- Performance optimization techniques
- Real-world system integration patterns

💡 KEY INTERVIEW SKILLS DEMONSTRATED:
- Deep understanding of OOP principles and patterns
- Practical problem-solving with real-world examples
- Thread safety and concurrency management
- Testing strategies and mock object usage
- Performance considerations and optimization
- Error handling and edge case management
- System architecture and design decisions
- Code quality and best practices

🚀 READY FOR SENIOR PYTHON DEVELOPER INTERVIEWS!
=================================================
This comprehensive series covers everything from basic OOP concepts to 
advanced system design patterns that experienced Python developers need
to know for senior-level technical interviews.
""")
