# ===============================================================================
# PYTHON OOP PILLAR 4: ABSTRACTION
# Real-Life Examples & Complete Mastery Guide
# ===============================================================================

"""
COMPREHENSIVE ABSTRACTION COVERAGE:
==================================
1. Abstract Base Classes (ABC) Deep Dive
2. Interface Design Patterns
3. Abstract Methods and Properties
4. Template Method Pattern
5. Strategy Pattern with Abstraction
6. Factory Pattern Abstractions
7. Adapter and Bridge Patterns
8. Facade Pattern for Complexity Hiding
9. Command Pattern Abstraction
10. System Architecture Abstraction
"""

import abc
from abc import ABC, abstractmethod, abstractproperty
from typing import Protocol, List, Dict, Any, Optional, Union, Callable, Type
from enum import Enum, auto
from datetime import datetime, timedelta
import json
import threading
import queue
import time
from functools import wraps
import logging

# ===============================================================================
# 1. ABSTRACT BASE CLASSES (ABC) DEEP DIVE
# ===============================================================================

print("=" * 80)
print("1. ABSTRACT BASE CLASSES (ABC) DEEP DIVE")
print("=" * 80)

print("\n--- Real-Life Example: Payment Gateway System ---")

class PaymentStatus(Enum):
    """Payment status enumeration"""
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    REFUNDED = auto()
    CANCELLED = auto()

class PaymentResult:
    """Payment result data structure"""
    
    def __init__(self, transaction_id: str, status: PaymentStatus, 
                 amount: float, message: str = "", details: Dict[str, Any] = None):
        self.transaction_id = transaction_id
        self.status = status
        self.amount = amount
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()

class PaymentGateway(ABC):
    """Abstract base class for payment gateways"""
    
    def __init__(self, gateway_name: str, config: Dict[str, Any]):
        self.gateway_name = gateway_name
        self.config = config
        self._is_initialized = False
        self._connection_pool = None
        self._rate_limiter = None
    
    @abstractmethod
    def initialize_connection(self) -> bool:
        """Initialize connection to payment gateway"""
        pass
    
    @abstractmethod
    def validate_payment_data(self, payment_data: Dict[str, Any]) -> bool:
        """Validate payment data specific to gateway"""
        pass
    
    @abstractmethod
    def process_payment(self, amount: float, payment_data: Dict[str, Any]) -> PaymentResult:
        """Process payment through the gateway"""
        pass
    
    @abstractmethod
    def refund_payment(self, transaction_id: str, amount: float = None) -> PaymentResult:
        """Refund a payment"""
        pass
    
    @abstractmethod
    def get_transaction_status(self, transaction_id: str) -> PaymentResult:
        """Get status of a transaction"""
        pass
    
    @property
    @abstractmethod
    def supported_currencies(self) -> List[str]:
        """Return list of supported currencies"""
        pass
    
    @property
    @abstractmethod
    def max_transaction_amount(self) -> float:
        """Maximum transaction amount supported"""
        pass
    
    @property
    @abstractmethod
    def transaction_fee_rate(self) -> float:
        """Transaction fee rate (as decimal)"""
        pass
    
    # Concrete methods that use abstract methods
    def process_payment_with_validation(self, amount: float, currency: str, 
                                      payment_data: Dict[str, Any]) -> PaymentResult:
        """Template method that validates and processes payment"""
        
        # Pre-validation
        if not self._is_initialized:
            if not self.initialize_connection():
                return PaymentResult(
                    "INIT_ERROR", PaymentStatus.FAILED, amount,
                    "Failed to initialize payment gateway"
                )
        
        # Currency validation
        if currency not in self.supported_currencies:
            return PaymentResult(
                "CURRENCY_ERROR", PaymentStatus.FAILED, amount,
                f"Currency {currency} not supported"
            )
        
        # Amount validation
        if amount > self.max_transaction_amount:
            return PaymentResult(
                "AMOUNT_ERROR", PaymentStatus.FAILED, amount,
                f"Amount exceeds maximum of {self.max_transaction_amount}"
            )
        
        # Gateway-specific validation
        if not self.validate_payment_data(payment_data):
            return PaymentResult(
                "VALIDATION_ERROR", PaymentStatus.FAILED, amount,
                "Payment data validation failed"
            )
        
        # Calculate fees
        fee = amount * self.transaction_fee_rate
        total_amount = amount + fee
        
        # Process payment
        result = self.process_payment(total_amount, payment_data)
        result.details['original_amount'] = amount
        result.details['fee'] = fee
        result.details['currency'] = currency
        
        return result
    
    def get_gateway_info(self) -> Dict[str, Any]:
        """Get gateway information"""
        return {
            'name': self.gateway_name,
            'supported_currencies': self.supported_currencies,
            'max_amount': self.max_transaction_amount,
            'fee_rate': self.transaction_fee_rate,
            'is_initialized': self._is_initialized
        }

class StripeGateway(PaymentGateway):
    """Concrete implementation for Stripe payment gateway"""
    
    def __init__(self, api_key: str, webhook_secret: str):
        config = {
            'api_key': api_key,
            'webhook_secret': webhook_secret,
            'api_url': 'https://api.stripe.com/v1'
        }
        super().__init__("Stripe", config)
    
    def initialize_connection(self) -> bool:
        """Initialize Stripe connection"""
        print(f"Initializing Stripe connection...")
        # Simulate API connection
        if self.config.get('api_key'):
            self._is_initialized = True
            print("Stripe connection initialized successfully")
            return True
        return False
    
    def validate_payment_data(self, payment_data: Dict[str, Any]) -> bool:
        """Validate Stripe-specific payment data"""
        required_fields = ['card_number', 'exp_month', 'exp_year', 'cvc']
        
        for field in required_fields:
            if field not in payment_data:
                print(f"Missing required field: {field}")
                return False
        
        # Validate card number (simplified)
        card_number = payment_data['card_number'].replace(' ', '')
        if not card_number.isdigit() or len(card_number) < 13:
            print("Invalid card number")
            return False
        
        return True
    
    def process_payment(self, amount: float, payment_data: Dict[str, Any]) -> PaymentResult:
        """Process payment through Stripe"""
        print(f"Processing ${amount:.2f} payment through Stripe...")
        
        # Simulate API call
        transaction_id = f"stripe_{int(time.time())}"
        
        # Simulate success/failure (90% success rate)
        import random
        if random.random() < 0.9:
            return PaymentResult(
                transaction_id, PaymentStatus.COMPLETED, amount,
                "Payment processed successfully",
                {'payment_method': 'stripe', 'processor': 'stripe'}
            )
        else:
            return PaymentResult(
                transaction_id, PaymentStatus.FAILED, amount,
                "Payment declined by bank",
                {'decline_code': 'insufficient_funds'}
            )
    
    def refund_payment(self, transaction_id: str, amount: float = None) -> PaymentResult:
        """Refund Stripe payment"""
        print(f"Processing refund for transaction {transaction_id}")
        
        refund_id = f"refund_{transaction_id}"
        return PaymentResult(
            refund_id, PaymentStatus.REFUNDED, amount or 0,
            "Refund processed successfully"
        )
    
    def get_transaction_status(self, transaction_id: str) -> PaymentResult:
        """Get Stripe transaction status"""
        print(f"Checking status for transaction {transaction_id}")
        
        # Simulate status lookup
        return PaymentResult(
            transaction_id, PaymentStatus.COMPLETED, 0,
            "Transaction found"
        )
    
    @property
    def supported_currencies(self) -> List[str]:
        """Stripe supported currencies"""
        return ['USD', 'EUR', 'GBP', 'CAD', 'AUD', 'JPY']
    
    @property
    def max_transaction_amount(self) -> float:
        """Stripe maximum transaction amount"""
        return 999999.99
    
    @property
    def transaction_fee_rate(self) -> float:
        """Stripe transaction fee rate"""
        return 0.029  # 2.9%

class PayPalGateway(PaymentGateway):
    """Concrete implementation for PayPal payment gateway"""
    
    def __init__(self, client_id: str, client_secret: str, sandbox: bool = True):
        config = {
            'client_id': client_id,
            'client_secret': client_secret,
            'sandbox': sandbox,
            'api_url': 'https://api.sandbox.paypal.com' if sandbox else 'https://api.paypal.com'
        }
        super().__init__("PayPal", config)
    
    def initialize_connection(self) -> bool:
        """Initialize PayPal connection"""
        print(f"Initializing PayPal connection (sandbox: {self.config['sandbox']})...")
        # Simulate OAuth token acquisition
        if self.config.get('client_id') and self.config.get('client_secret'):
            self._is_initialized = True
            print("PayPal connection initialized successfully")
            return True
        return False
    
    def validate_payment_data(self, payment_data: Dict[str, Any]) -> bool:
        """Validate PayPal-specific payment data"""
        # PayPal can use email or card
        if 'paypal_email' in payment_data:
            email = payment_data['paypal_email']
            if '@' not in email or '.' not in email:
                print("Invalid PayPal email")
                return False
            return True
        
        # Alternative: card payment through PayPal
        if 'card_number' in payment_data:
            return self._validate_card_data(payment_data)
        
        print("Missing PayPal email or card information")
        return False
    
    def _validate_card_data(self, payment_data: Dict[str, Any]) -> bool:
        """Validate card data for PayPal"""
        required_fields = ['card_number', 'exp_month', 'exp_year']
        return all(field in payment_data for field in required_fields)
    
    def process_payment(self, amount: float, payment_data: Dict[str, Any]) -> PaymentResult:
        """Process payment through PayPal"""
        print(f"Processing ${amount:.2f} payment through PayPal...")
        
        transaction_id = f"paypal_{int(time.time())}"
        
        # Simulate PayPal processing
        import random
        if random.random() < 0.95:  # PayPal has higher success rate
            return PaymentResult(
                transaction_id, PaymentStatus.COMPLETED, amount,
                "PayPal payment completed",
                {'payment_method': 'paypal', 'processor': 'paypal'}
            )
        else:
            return PaymentResult(
                transaction_id, PaymentStatus.FAILED, amount,
                "PayPal payment failed",
                {'error_code': 'PAYMENT_DENIED'}
            )
    
    def refund_payment(self, transaction_id: str, amount: float = None) -> PaymentResult:
        """Refund PayPal payment"""
        print(f"Processing PayPal refund for {transaction_id}")
        
        refund_id = f"paypal_refund_{transaction_id}"
        return PaymentResult(
            refund_id, PaymentStatus.REFUNDED, amount or 0,
            "PayPal refund processed"
        )
    
    def get_transaction_status(self, transaction_id: str) -> PaymentResult:
        """Get PayPal transaction status"""
        print(f"Checking PayPal status for {transaction_id}")
        
        return PaymentResult(
            transaction_id, PaymentStatus.COMPLETED, 0,
            "PayPal transaction found"
        )
    
    @property
    def supported_currencies(self) -> List[str]:
        """PayPal supported currencies"""
        return ['USD', 'EUR', 'GBP', 'CAD', 'AUD', 'JPY', 'CHF', 'SEK']
    
    @property
    def max_transaction_amount(self) -> float:
        """PayPal maximum transaction amount"""
        return 60000.00  # PayPal has lower limits
    
    @property
    def transaction_fee_rate(self) -> float:
        """PayPal transaction fee rate"""
        return 0.034  # 3.4%

# Payment processor that uses abstractions
class PaymentProcessor:
    """High-level payment processor using abstract gateways"""
    
    def __init__(self):
        self.gateways: Dict[str, PaymentGateway] = {}
        self.default_gateway = None
    
    def register_gateway(self, gateway: PaymentGateway, is_default: bool = False):
        """Register a payment gateway"""
        self.gateways[gateway.gateway_name] = gateway
        if is_default or not self.default_gateway:
            self.default_gateway = gateway.gateway_name
        print(f"Registered gateway: {gateway.gateway_name}")
    
    def process_payment(self, amount: float, currency: str, payment_data: Dict[str, Any],
                       gateway_name: str = None) -> PaymentResult:
        """Process payment using specified or default gateway"""
        
        gateway_name = gateway_name or self.default_gateway
        if gateway_name not in self.gateways:
            return PaymentResult(
                "GATEWAY_ERROR", PaymentStatus.FAILED, amount,
                f"Gateway {gateway_name} not available"
            )
        
        gateway = self.gateways[gateway_name]
        return gateway.process_payment_with_validation(amount, currency, payment_data)
    
    def get_best_gateway_for_amount(self, amount: float, currency: str) -> Optional[str]:
        """Get best gateway for amount based on fees and limits"""
        best_gateway = None
        lowest_fee = float('inf')
        
        for name, gateway in self.gateways.items():
            if (currency in gateway.supported_currencies and 
                amount <= gateway.max_transaction_amount):
                
                fee = amount * gateway.transaction_fee_rate
                if fee < lowest_fee:
                    lowest_fee = fee
                    best_gateway = name
        
        return best_gateway

# Example usage - Abstract Base Classes
print("Demonstrating Abstract Base Classes:")

# Create payment processor
processor = PaymentProcessor()

# Register gateways
stripe_gateway = StripeGateway("sk_test_123", "whsec_456")
paypal_gateway = PayPalGateway("client_123", "secret_456", sandbox=True)

processor.register_gateway(stripe_gateway, is_default=True)
processor.register_gateway(paypal_gateway)

# Process payments
stripe_payment_data = {
    'card_number': '4242424242424242',
    'exp_month': '12',
    'exp_year': '2025',
    'cvc': '123'
}

paypal_payment_data = {
    'paypal_email': 'user@example.com'
}

# Process with default gateway (Stripe)
result1 = processor.process_payment(100.0, 'USD', stripe_payment_data)
print(f"Payment result: {result1.status.name} - {result1.message}")

# Process with specific gateway (PayPal)
result2 = processor.process_payment(50.0, 'EUR', paypal_payment_data, 'PayPal')
print(f"PayPal result: {result2.status.name} - {result2.message}")

# Find best gateway for amount
best_gateway = processor.get_best_gateway_for_amount(1000.0, 'USD')
print(f"Best gateway for $1000: {best_gateway}")

# ===============================================================================
# 2. INTERFACE DESIGN PATTERNS
# ===============================================================================

print("\n" + "=" * 80)
print("2. INTERFACE DESIGN PATTERNS")
print("=" * 80)

print("\n--- Real-Life Example: Cloud Storage System ---")

# Interface segregation principle - multiple small interfaces
class Readable(Protocol):
    """Interface for readable storage"""
    
    def read(self, path: str) -> bytes:
        """Read file contents"""
        ...
    
    def exists(self, path: str) -> bool:
        """Check if file exists"""
        ...
    
    def get_size(self, path: str) -> int:
        """Get file size"""
        ...

class Writable(Protocol):
    """Interface for writable storage"""
    
    def write(self, path: str, data: bytes) -> bool:
        """Write file contents"""
        ...
    
    def delete(self, path: str) -> bool:
        """Delete file"""
        ...

class Listable(Protocol):
    """Interface for listing storage contents"""
    
    def list_files(self, path: str) -> List[str]:
        """List files in directory"""
        ...
    
    def list_directories(self, path: str) -> List[str]:
        """List subdirectories"""
        ...

class Searchable(Protocol):
    """Interface for searching storage"""
    
    def search_files(self, pattern: str) -> List[str]:
        """Search for files matching pattern"""
        ...
    
    def find_by_extension(self, extension: str) -> List[str]:
        """Find files by extension"""
        ...

class Versioned(Protocol):
    """Interface for version control"""
    
    def create_version(self, path: str) -> str:
        """Create new version of file"""
        ...
    
    def list_versions(self, path: str) -> List[str]:
        """List all versions of file"""
        ...
    
    def restore_version(self, path: str, version: str) -> bool:
        """Restore specific version"""
        ...

# Base storage abstraction
class CloudStorage(ABC):
    """Abstract base for cloud storage providers"""
    
    def __init__(self, provider_name: str, config: Dict[str, Any]):
        self.provider_name = provider_name
        self.config = config
        self._connection = None
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to storage provider"""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from storage provider"""
        pass
    
    @abstractmethod
    def get_quota_info(self) -> Dict[str, Any]:
        """Get storage quota information"""
        pass

# Concrete implementations with different interface combinations
class AmazonS3Storage(CloudStorage, Readable, Writable, Listable, Searchable):
    """Amazon S3 storage implementation"""
    
    def __init__(self, access_key: str, secret_key: str, bucket: str):
        config = {
            'access_key': access_key,
            'secret_key': secret_key,
            'bucket': bucket,
            'region': 'us-east-1'
        }
        super().__init__("Amazon S3", config)
        self._files = {}  # Simulate file storage
    
    def connect(self) -> bool:
        """Connect to S3"""
        print(f"Connecting to S3 bucket: {self.config['bucket']}")
        self._connection = "s3_connection"
        return True
    
    def disconnect(self) -> bool:
        """Disconnect from S3"""
        print("Disconnecting from S3")
        self._connection = None
        return True
    
    def get_quota_info(self) -> Dict[str, Any]:
        """Get S3 quota info"""
        return {
            'used_bytes': sum(len(data) for data in self._files.values()),
            'total_bytes': 1000000000,  # 1GB
            'provider': 'Amazon S3'
        }
    
    # Readable interface
    def read(self, path: str) -> bytes:
        """Read from S3"""
        if path in self._files:
            print(f"Reading {path} from S3")
            return self._files[path]
        raise FileNotFoundError(f"File {path} not found in S3")
    
    def exists(self, path: str) -> bool:
        """Check if file exists in S3"""
        return path in self._files
    
    def get_size(self, path: str) -> int:
        """Get file size in S3"""
        if path in self._files:
            return len(self._files[path])
        return 0
    
    # Writable interface
    def write(self, path: str, data: bytes) -> bool:
        """Write to S3"""
        print(f"Writing {len(data)} bytes to {path} in S3")
        self._files[path] = data
        return True
    
    def delete(self, path: str) -> bool:
        """Delete from S3"""
        if path in self._files:
            print(f"Deleting {path} from S3")
            del self._files[path]
            return True
        return False
    
    # Listable interface
    def list_files(self, path: str) -> List[str]:
        """List files in S3 path"""
        return [f for f in self._files.keys() if f.startswith(path) and '/' not in f[len(path):]]
    
    def list_directories(self, path: str) -> List[str]:
        """List directories in S3 path"""
        dirs = set()
        for f in self._files.keys():
            if f.startswith(path):
                remaining = f[len(path):]
                if '/' in remaining:
                    dirs.add(remaining.split('/')[0])
        return list(dirs)
    
    # Searchable interface
    def search_files(self, pattern: str) -> List[str]:
        """Search files in S3"""
        import re
        regex = re.compile(pattern)
        return [f for f in self._files.keys() if regex.search(f)]
    
    def find_by_extension(self, extension: str) -> List[str]:
        """Find files by extension in S3"""
        return [f for f in self._files.keys() if f.endswith(extension)]

class GoogleDriveStorage(CloudStorage, Readable, Writable, Listable, Versioned):
    """Google Drive storage with versioning"""
    
    def __init__(self, api_key: str, folder_id: str):
        config = {
            'api_key': api_key,
            'folder_id': folder_id
        }
        super().__init__("Google Drive", config)
        self._files = {}
        self._versions = {}  # Track file versions
    
    def connect(self) -> bool:
        """Connect to Google Drive"""
        print("Connecting to Google Drive API")
        self._connection = "drive_connection"
        return True
    
    def disconnect(self) -> bool:
        """Disconnect from Google Drive"""
        print("Disconnecting from Google Drive")
        self._connection = None
        return True
    
    def get_quota_info(self) -> Dict[str, Any]:
        """Get Drive quota info"""
        return {
            'used_bytes': sum(len(data) for data in self._files.values()),
            'total_bytes': 15000000000,  # 15GB
            'provider': 'Google Drive'
        }
    
    # Readable interface
    def read(self, path: str) -> bytes:
        """Read from Google Drive"""
        if path in self._files:
            print(f"Reading {path} from Google Drive")
            return self._files[path]
        raise FileNotFoundError(f"File {path} not found in Google Drive")
    
    def exists(self, path: str) -> bool:
        """Check if file exists in Google Drive"""
        return path in self._files
    
    def get_size(self, path: str) -> int:
        """Get file size in Google Drive"""
        if path in self._files:
            return len(self._files[path])
        return 0
    
    # Writable interface
    def write(self, path: str, data: bytes) -> bool:
        """Write to Google Drive with versioning"""
        print(f"Writing {len(data)} bytes to {path} in Google Drive")
        
        # Save previous version if file exists
        if path in self._files:
            self.create_version(path)
        
        self._files[path] = data
        return True
    
    def delete(self, path: str) -> bool:
        """Delete from Google Drive"""
        if path in self._files:
            print(f"Moving {path} to Google Drive trash")
            del self._files[path]
            # Also clean up versions
            if path in self._versions:
                del self._versions[path]
            return True
        return False
    
    # Listable interface
    def list_files(self, path: str) -> List[str]:
        """List files in Google Drive path"""
        return [f for f in self._files.keys() if f.startswith(path)]
    
    def list_directories(self, path: str) -> List[str]:
        """List directories in Google Drive path"""
        dirs = set()
        for f in self._files.keys():
            if f.startswith(path):
                remaining = f[len(path):]
                if '/' in remaining:
                    dirs.add(remaining.split('/')[0])
        return list(dirs)
    
    # Versioned interface
    def create_version(self, path: str) -> str:
        """Create new version in Google Drive"""
        if path not in self._files:
            return ""
        
        if path not in self._versions:
            self._versions[path] = []
        
        version_id = f"v{len(self._versions[path]) + 1}_{int(time.time())}"
        self._versions[path].append({
            'version_id': version_id,
            'data': self._files[path],
            'timestamp': datetime.now()
        })
        
        print(f"Created version {version_id} for {path}")
        return version_id
    
    def list_versions(self, path: str) -> List[str]:
        """List versions in Google Drive"""
        if path in self._versions:
            return [v['version_id'] for v in self._versions[path]]
        return []
    
    def restore_version(self, path: str, version: str) -> bool:
        """Restore version in Google Drive"""
        if path in self._versions:
            for v in self._versions[path]:
                if v['version_id'] == version:
                    self._files[path] = v['data']
                    print(f"Restored {path} to version {version}")
                    return True
        return False

# Storage manager using interface abstractions
class StorageManager:
    """Manager that works with any storage provider through interfaces"""
    
    def __init__(self):
        self.storages: Dict[str, CloudStorage] = {}
        self.default_storage = None
    
    def register_storage(self, storage: CloudStorage, is_default: bool = False):
        """Register a storage provider"""
        self.storages[storage.provider_name] = storage
        storage.connect()
        
        if is_default or not self.default_storage:
            self.default_storage = storage.provider_name
        
        print(f"Registered storage: {storage.provider_name}")
    
    def copy_file(self, source_storage: str, source_path: str,
                  dest_storage: str, dest_path: str) -> bool:
        """Copy file between different storage providers"""
        
        if source_storage not in self.storages or dest_storage not in self.storages:
            return False
        
        source = self.storages[source_storage]
        dest = self.storages[dest_storage]
        
        # Check if source supports reading and dest supports writing
        if not isinstance(source, Readable) or not isinstance(dest, Writable):
            print("Storage providers don't support required operations")
            return False
        
        try:
            # Read from source
            data = source.read(source_path)
            
            # Write to destination
            return dest.write(dest_path, data)
        
        except Exception as e:
            print(f"Copy failed: {e}")
            return False
    
    def search_across_storages(self, pattern: str) -> Dict[str, List[str]]:
        """Search for files across all searchable storages"""
        results = {}
        
        for name, storage in self.storages.items():
            if isinstance(storage, Searchable):
                results[name] = storage.search_files(pattern)
            else:
                print(f"Storage {name} doesn't support searching")
        
        return results
    
    def backup_with_versioning(self, storage_name: str, path: str) -> bool:
        """Create backup version if storage supports versioning"""
        if storage_name not in self.storages:
            return False
        
        storage = self.storages[storage_name]
        if isinstance(storage, Versioned):
            version = storage.create_version(path)
            print(f"Created backup version: {version}")
            return True
        else:
            print(f"Storage {storage_name} doesn't support versioning")
            return False

# Example usage - Interface Design Patterns
print("Demonstrating Interface Design Patterns:")

# Create storage manager
manager = StorageManager()

# Register different storage providers with different capabilities
s3_storage = AmazonS3Storage("access_key", "secret_key", "my-bucket")
drive_storage = GoogleDriveStorage("api_key", "folder_id")

manager.register_storage(s3_storage, is_default=True)
manager.register_storage(drive_storage)

# Use storage through interfaces
# Write data to both storages
test_data = b"This is test file content"
s3_storage.write("test.txt", test_data)
drive_storage.write("documents/test.txt", test_data)

# Copy between storages (using interfaces)
success = manager.copy_file("Amazon S3", "test.txt", "Google Drive", "backup/test.txt")
print(f"Copy operation success: {success}")

# Search across storages (only works for Searchable storages)
search_results = manager.search_across_storages("test.*")
print(f"Search results: {search_results}")

# Use versioning (only works for Versioned storages)
drive_storage.write("documents/test.txt", b"Updated content")
version_success = manager.backup_with_versioning("Google Drive", "documents/test.txt")

# List versions
versions = drive_storage.list_versions("documents/test.txt")
print(f"Available versions: {versions}")

print("\n" + "=" * 80)
print("ABSTRACTION PART 1 COMPLETE!")
print("=" * 80)
print("""
🎯 CONCEPTS COVERED:
✅ Abstract Base Classes (ABC) Deep Dive
✅ Interface Design Patterns
✅ Interface Segregation Principle

📝 REAL-LIFE EXAMPLES:
- Payment Gateway System (ABC Deep Dive)
- Cloud Storage System (Interface Design)

🚀 NEXT: Continue with abstraction_advanced.py for Design Patterns and Architecture
""")
