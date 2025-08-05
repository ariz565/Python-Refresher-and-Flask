"""
COMPREHENSIVE INTERVIEW GUIDE - PART 1: DJANGO FUNDAMENTALS
===========================================================

This guide covers Django fundamentals, ORM, security, and core concepts.
Based on comprehensive analysis of interview questions and best practices.

Author: Interview Preparation Guide
Date: August 2025
Technologies: Django, Python
"""

# ===================================
# PART 1: DJANGO FUNDAMENTALS
# ===================================

def django_fundamentals_interview_questions():
    """
    Comprehensive Django fundamentals interview questions and detailed answers
    """
    
    print("=" * 80)
    print("DJANGO FUNDAMENTALS - INTERVIEW QUESTIONS & ANSWERS")
    print("=" * 80)
    
    questions_and_answers = [
        {
            "question": "1. How does Django handle asynchronous tasks?",
            "answer": """
Django itself is primarily synchronous, but it has support for asynchronous views, 
middleware, and database operations as of Django 3.1+. However, Django does not 
provide built-in background task execution.

Key Points:
- Django 3.1+ supports async views and middleware
- For background tasks, use Celery (most common)
- Async views: async def my_view(request)
- ASGI deployment for async support (Daphne, Uvicorn)

Example Async View:
```python
import asyncio
from django.http import JsonResponse

async def async_view(request):
    await asyncio.sleep(1)  # Simulate async operation
    return JsonResponse({'message': 'Async response'})
```

For background tasks, integrate with Celery:
```python
from celery import shared_task

@shared_task
def send_email_task(email, message):
    # Background email sending
    pass
```
            """,
            "follow_up": "What are the differences between sync and async Django views?"
        },
        
        {
            "question": "2. What are Django signals, and how do they compare to Celery tasks?",
            "answer": """
Django signals allow decoupled components of a Django application to communicate 
when certain events occur.

DJANGO SIGNALS:
- Synchronous execution within request-response cycle
- Triggered by Django events (pre_save, post_save, etc.)
- Good for: logging, cache invalidation, simple notifications

CELERY TASKS:
- Asynchronous execution in background
- Prevents blocking user-facing operations
- Good for: sending emails, processing files, heavy computations

Example Django Signal:
```python
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)
```

Example Celery Task:
```python
from celery import shared_task

@shared_task
def process_image(image_path):
    # Heavy image processing in background
    pass
```

COMPARISON:
| Feature | Django Signals | Celery Tasks |
|---------|---------------|--------------|
| Execution | Synchronous | Asynchronous |
| Performance Impact | Blocks request | Non-blocking |
| Error Handling | Affects request | Isolated |
| Use Cases | Simple operations | Heavy operations |
            """,
            "follow_up": "When would you choose signals over Celery tasks?"
        },
        
        {
            "question": "3. How would you scale a Django application handling high traffic?",
            "answer": """
Scaling Django for high traffic requires multiple strategies:

1. DATABASE OPTIMIZATION:
   - Use database indexing
   - Avoid N+1 queries (use select_related, prefetch_related)
   - Database connection pooling
   - Read replicas for read-heavy operations

2. CACHING STRATEGIES:
   - Redis/Memcached for frequently accessed data
   - Database query caching
   - Template fragment caching
   - Full page caching with Varnish

3. LOAD BALANCING:
   - Multiple application servers behind load balancer
   - Nginx/HAProxy for load distribution
   - Auto-scaling with cloud providers

4. ASYNCHRONOUS PROCESSING:
   - Celery for background tasks
   - Message queues (Redis/RabbitMQ)
   - Separate worker servers

5. DEPLOYMENT OPTIMIZATION:
   - WSGI servers (Gunicorn, uWSGI)
   - ASGI servers for async support (Daphne, Uvicorn)
   - CDN for static files
   - Database partitioning/sharding

Example Caching Configuration:
```python
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

# View-level caching
from django.views.decorators.cache import cache_page

@cache_page(60 * 15)  # Cache for 15 minutes
def my_view(request):
    return render(request, 'template.html')
```

Database Optimization:
```python
# Bad - N+1 query problem
for book in Book.objects.all():
    print(book.author.name)  # Each iteration hits database

# Good - Use select_related
for book in Book.objects.select_related('author'):
    print(book.author.name)  # Single query with JOIN
```
            """,
            "follow_up": "What's the difference between select_related and prefetch_related?"
        },
        
        {
            "question": "4. How does Django's ORM interact with databases?",
            "answer": """
Django ORM (Object-Relational Mapper) provides an abstraction layer between 
Python objects and database tables.

KEY FEATURES:
1. QuerySet API - Lazy evaluation
2. Model-to-table mapping
3. Automatic SQL generation
4. Transaction support
5. Connection pooling

LAZY EVALUATION:
QuerySets are not executed until explicitly evaluated:
```python
# This doesn't hit the database yet
users = User.objects.filter(is_active=True)

# These operations trigger database queries
list(users)          # Evaluate to list
users.count()        # Count query
for user in users:   # Iteration
    print(user.name)
```

QUERYSET METHODS:
```python
# Filtering
User.objects.filter(age__gte=18)
User.objects.exclude(is_staff=True)

# Ordering
User.objects.order_by('-created_at')

# Aggregation
from django.db.models import Count, Avg
User.objects.aggregate(total=Count('id'), avg_age=Avg('age'))

# Annotation
User.objects.annotate(post_count=Count('posts'))
```

RELATIONSHIP QUERIES:
```python
# Forward relationship
book = Book.objects.select_related('author').get(id=1)
print(book.author.name)  # No additional query

# Reverse relationship
author = Author.objects.prefetch_related('books').get(id=1)
for book in author.books.all():  # No additional queries
    print(book.title)
```

TRANSACTION SUPPORT:
```python
from django.db import transaction

@transaction.atomic
def transfer_money(from_account, to_account, amount):
    from_account.balance -= amount
    from_account.save()
    to_account.balance += amount
    to_account.save()
```
            """,
            "follow_up": "Explain the difference between .get() and .filter().first()"
        },
        
        {
            "question": "5. What happens when a database transaction fails in Django?",
            "answer": """
Django transaction failure handling depends on whether you're using atomic transactions:

WITHOUT ATOMIC TRANSACTIONS:
- Partial changes may persist
- Can lead to data inconsistency
- Each save() is a separate transaction

WITH ATOMIC TRANSACTIONS:
- All changes are rolled back
- Data consistency maintained
- Transaction boundary management

Example with atomic decorator:
```python
from django.db import transaction

@transaction.atomic
def create_user_with_profile(username, email):
    try:
        user = User.objects.create(username=username, email=email)
        profile = UserProfile.objects.create(user=user, bio="New user")
        return user
    except Exception as e:
        # Automatic rollback - no partial data
        raise e
```

Example with atomic context manager:
```python
def transfer_money(from_acc, to_acc, amount):
    try:
        with transaction.atomic():
            if from_acc.balance < amount:
                raise ValueError("Insufficient funds")
            
            from_acc.balance -= amount
            from_acc.save()
            
            to_acc.balance += amount
            to_acc.save()
            
            # Log transaction
            Transaction.objects.create(
                from_account=from_acc,
                to_account=to_acc,
                amount=amount
            )
    except Exception as e:
        # All changes rolled back automatically
        print(f"Transaction failed: {e}")
```

SAVEPOINTS (Nested Transactions):
```python
def complex_operation():
    with transaction.atomic():  # Outer transaction
        create_user()
        
        try:
            with transaction.atomic():  # Savepoint
                risky_operation()
        except RiskyError:
            # Only inner transaction rolled back
            pass
        
        final_operation()  # Still executes
```

TRANSACTION ISOLATION LEVELS:
```python
from django.db import transaction

with transaction.atomic():
    # Default isolation level (usually READ COMMITTED)
    pass

# Custom isolation level
from django.db import connections
connection = connections['default']
with connection.cursor() as cursor:
    cursor.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
```
            """,
            "follow_up": "What are Django's transaction isolation levels?"
        }
    ]
    
    for qa in questions_and_answers:
        print(f"\nQ: {qa['question']}")
        print(f"A: {qa['answer']}")
        if qa.get('follow_up'):
            print(f"Follow-up: {qa['follow_up']}")
        print("-" * 80)

def django_orm_deep_dive():
    """
    Deep dive into Django ORM concepts, relationships, and best practices
    """
    
    print("\n" + "=" * 80)
    print("DJANGO ORM - DEEP DIVE")
    print("=" * 80)
    
    print("""
1. MODEL RELATIONSHIPS:

FOREIGNKEY (One-to-Many):
```python
class Author(models.Model):
    name = models.CharField(max_length=100)
    
class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.ForeignKey(Author, on_delete=models.CASCADE)
    
# Usage
author = Author.objects.get(id=1)
books = author.book_set.all()  # Reverse relationship
```

MANYTOMANY:
```python
class Book(models.Model):
    title = models.CharField(max_length=200)
    genres = models.ManyToManyField('Genre')
    
class Genre(models.Model):
    name = models.CharField(max_length=50)
    
# Usage
book = Book.objects.get(id=1)
book.genres.add(genre1, genre2)
book.genres.remove(genre1)
```

ONETOONE:
```python
class User(models.Model):
    username = models.CharField(max_length=50)
    
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    bio = models.TextField()
    
# Usage
user = User.objects.get(id=1)
profile = user.userprofile  # Direct access
```

2. FIELD OPTIONS:

UNIQUE vs NULL vs BLANK:
```python
class MyModel(models.Model):
    # unique=True: No duplicates in database
    email = models.EmailField(unique=True)
    
    # null=True: Database can store NULL
    # blank=True: Form validation allows empty
    optional_field = models.CharField(max_length=100, null=True, blank=True)
    
    # For text fields, use blank=True, not null=True
    description = models.TextField(blank=True)  # Empty string, not NULL
```

3. QUERYSET OPTIMIZATION:

SELECT_RELATED (Forward ForeignKey/OneToOne):
```python
# Bad - N+1 queries
books = Book.objects.all()
for book in books:
    print(book.author.name)  # Database hit for each book

# Good - Single query with JOIN
books = Book.objects.select_related('author')
for book in books:
    print(book.author.name)  # No additional queries
```

PREFETCH_RELATED (Reverse ForeignKey/ManyToMany):
```python
# Bad - N+1 queries
authors = Author.objects.all()
for author in authors:
    for book in author.book_set.all():  # Query for each author
        print(book.title)

# Good - 2 queries total
authors = Author.objects.prefetch_related('book_set')
for author in authors:
    for book in author.book_set.all():  # No additional queries
        print(book.title)
```

ONLY/DEFER:
```python
# Load only specific fields
users = User.objects.only('username', 'email')

# Defer heavy fields
articles = Article.objects.defer('content')  # Don't load content field
```

4. AGGREGATION AND ANNOTATION:
```python
from django.db.models import Count, Avg, Sum, Max, Min

# Aggregation (single result)
stats = Book.objects.aggregate(
    total_books=Count('id'),
    avg_pages=Avg('pages'),
    max_price=Max('price')
)

# Annotation (per-object result)
authors = Author.objects.annotate(
    book_count=Count('book'),
    avg_book_price=Avg('book__price')
).filter(book_count__gt=5)
```

5. CUSTOM MANAGERS AND QUERYSETS:
```python
class PublishedBookManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(is_published=True)

class BookQuerySet(models.QuerySet):
    def published(self):
        return self.filter(is_published=True)
    
    def by_author(self, author):
        return self.filter(author=author)

class Book(models.Model):
    title = models.CharField(max_length=200)
    is_published = models.BooleanField(default=False)
    
    objects = models.Manager()  # Default manager
    published = PublishedBookManager()  # Custom manager
    
    # Using custom QuerySet
    objects = BookQuerySet.as_manager()

# Usage
published_books = Book.published.all()
author_books = Book.objects.published().by_author(author)
```
    """)

def django_security_guide():
    """
    Comprehensive Django security best practices and interview questions
    """
    
    print("\n" + "=" * 80)
    print("DJANGO SECURITY - COMPREHENSIVE GUIDE")
    print("=" * 80)
    
    print("""
1. CSRF (Cross-Site Request Forgery) PROTECTION:

How CSRF attacks work:
- Malicious site tricks user into submitting form to your site
- User's browser sends cookies automatically
- Attacker performs actions as authenticated user

Django's CSRF Protection:
```python
# In templates
<form method="post">
    {% csrf_token %}
    <!-- form fields -->
</form>

# In views (automatic for most cases)
from django.views.decorators.csrf import csrf_exempt, csrf_protect

@csrf_exempt  # Disable CSRF (be careful!)
def api_view(request):
    pass

# For AJAX requests
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

$.ajaxSetup({
    beforeSend: function(xhr, settings) {
        if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
            xhr.setRequestHeader("X-CSRFToken", getCookie('csrftoken'));
        }
    }
});
```

2. XSS (Cross-Site Scripting) PREVENTION:

Django automatically escapes template variables:
```python
# In template - automatically escaped
{{ user_input }}  # Safe from XSS

# Manual escaping
from django.utils.html import escape
safe_text = escape(user_input)

# Mark as safe (be very careful!)
from django.utils.safestring import mark_safe
html_content = mark_safe("<strong>Bold text</strong>")

# In templates
{{ html_content|safe }}  # Renders HTML
{{ user_input|escape }}  # Force escape
```

Content Security Policy (CSP):
```python
# settings.py
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True

# Custom middleware for CSP
class CSPMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        response['Content-Security-Policy'] = "default-src 'self'"
        return response
```

3. SQL INJECTION PREVENTION:

Django ORM automatically prevents SQL injection:
```python
# Safe - uses parameterized queries
User.objects.filter(username=username)
User.objects.raw("SELECT * FROM users WHERE username = %s", [username])

# Dangerous - never do this!
User.objects.extra(where=["username = '%s'" % username])  # Vulnerable!

# Safe raw SQL
from django.db import connection
cursor = connection.cursor()
cursor.execute("SELECT * FROM users WHERE username = %s", [username])
```

4. AUTHENTICATION AND AUTHORIZATION:

Password Security:
```python
# settings.py
PASSWORD_HASHERS = [
    'django.contrib.auth.hashers.Argon2PasswordHasher',
    'django.contrib.auth.hashers.PBKDF2PasswordHasher',
    'django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher',
    'django.contrib.auth.hashers.BCryptSHA256PasswordHasher',
]

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
        'OPTIONS': {'min_length': 8,}
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]
```

Permission and User Management:
```python
# Custom user model
from django.contrib.auth.models import AbstractUser

class CustomUser(AbstractUser):
    email = models.EmailField(unique=True)
    is_verified = models.BooleanField(default=False)

# Permission decorators
from django.contrib.auth.decorators import login_required, permission_required

@login_required
@permission_required('myapp.can_edit', raise_exception=True)
def edit_view(request):
    pass

# Class-based view permissions
from django.contrib.auth.mixins import LoginRequiredMixin, PermissionRequiredMixin

class EditView(LoginRequiredMixin, PermissionRequiredMixin, UpdateView):
    permission_required = 'myapp.can_edit'
    model = MyModel
```

5. HTTPS AND SECURITY HEADERS:

```python
# settings.py - Production security settings
SECURE_SSL_REDIRECT = True
SECURE_HSTS_SECONDS = 31536000  # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True
X_FRAME_OPTIONS = 'DENY'

# Session security
SESSION_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_AGE = 3600  # 1 hour

# CSRF cookie security
CSRF_COOKIE_SECURE = True
CSRF_COOKIE_HTTPONLY = True
```

6. FILE UPLOAD SECURITY:

```python
import os
from django.core.exceptions import ValidationError

def validate_file_extension(value):
    ext = os.path.splitext(value.name)[1]
    valid_extensions = ['.pdf', '.doc', '.docx', '.jpg', '.png']
    if not ext.lower() in valid_extensions:
        raise ValidationError('Unsupported file extension.')

class Document(models.Model):
    title = models.CharField(max_length=100)
    file = models.FileField(
        upload_to='documents/',
        validators=[validate_file_extension]
    )

# File size validation
def validate_file_size(value):
    filesize = value.size
    if filesize > 10 * 1024 * 1024:  # 10MB
        raise ValidationError("File too large. Max size is 10MB.")
```
    """)

if __name__ == "__main__":
    print("COMPREHENSIVE DJANGO INTERVIEW GUIDE - PART 1")
    print("=" * 60)
    print("This guide covers Django fundamentals, ORM, and security")
    print("Use this as a comprehensive reference for Django interviews")
    print("=" * 60)
    
    django_fundamentals_interview_questions()
    django_orm_deep_dive()
    django_security_guide()
    
    print("\n" + "=" * 80)
    print("END OF PART 1 - DJANGO FUNDAMENTALS")
    print("Next: Part 2 - Celery and Task Queue Management")
    print("=" * 80)
