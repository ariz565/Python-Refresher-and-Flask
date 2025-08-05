"""
Django REST Framework (DRF) - Complete Mastery Guide - PART 3
Testing, Performance, Best Practices, and Interview Questions
"""

print("=" * 80)
print("DJANGO REST FRAMEWORK - PART 3: TESTING & BEST PRACTICES")
print("=" * 80)

# ============================================================================
# SECTION 8: TESTING DRF APIS
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 8: TESTING DRF APIS")
print("=" * 60)

# Question 8: DRF Testing
print("\n8. How do you test Django REST Framework APIs?")
print("-" * 48)
print("""
DRF TESTING OVERVIEW:

🧪 TESTING IMPORTANCE:
• Ensures API reliability and correctness
• Prevents regressions during development
• Documents expected behavior
• Validates security and permissions
• Tests edge cases and error handling

DRF TEST COMPONENTS:
1. APITestCase: DRF's test case class
2. APIClient: HTTP client for testing
3. Test Factories: Generate test data
4. Mock Objects: Simulate external dependencies
5. Test Coverage: Measure test completeness

TESTING LEVELS:
1. Unit Tests: Individual components
2. Integration Tests: Component interactions
3. End-to-End Tests: Full API workflows
4. Performance Tests: Load and stress testing

TEST TYPES:
• Authentication Tests
• Permission Tests  
• Serializer Tests
• View/ViewSet Tests
• Filter Tests
• Pagination Tests
• Throttling Tests

INTERVIEW QUESTIONS:
Q: "How do you test DRF permissions?"
A: "Create test users with different roles, test all CRUD operations,
   verify permission behavior for each user type and action."

Q: "What's the difference between APITestCase and TestCase?"
A: "APITestCase provides DRF-specific features like APIClient,
   authentication helpers, and DRF response assertions."
""")

def demonstrate_drf_testing():
    """Demonstrate comprehensive DRF testing patterns"""
    print("DRF Testing Complete Guide:")
    
    print("\n1. Basic Test Setup:")
    print("-" * 22)
    print("""
# test_models.py
from django.test import TestCase
from django.contrib.auth.models import User
from .models import Book, Category, Author

class BookModelTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.category = Category.objects.create(
            name='Fiction',
            description='Fiction books'
        )
        self.author = Author.objects.create(
            name='Test Author',
            email='author@example.com'
        )
        
    def test_book_creation(self):
        '''Test book model creation'''
        book = Book.objects.create(
            title='Test Book',
            isbn='1234567890123',
            published_date='2023-01-01',
            pages=200,
            price=29.99,
            category=self.category,
            created_by=self.user
        )
        
        self.assertEqual(book.title, 'Test Book')
        self.assertEqual(book.isbn, '1234567890123')
        self.assertEqual(str(book), 'Test Book')
        self.assertTrue(book.created_at)
        self.assertEqual(book.created_by, self.user)
    
    def test_book_str_representation(self):
        '''Test book string representation'''
        book = Book.objects.create(
            title='Another Test Book',
            isbn='9876543210987',
            published_date='2023-02-01',
            pages=300,
            price=39.99,
            category=self.category,
            created_by=self.user
        )
        self.assertEqual(str(book), 'Another Test Book')
    
    def test_book_relationships(self):
        '''Test book model relationships'''
        book = Book.objects.create(
            title='Relationship Test',
            isbn='1111111111111',
            published_date='2023-03-01',
            pages=400,
            price=49.99,
            category=self.category,
            created_by=self.user
        )
        book.authors.add(self.author)
        
        self.assertEqual(book.category, self.category)
        self.assertEqual(book.created_by, self.user)
        self.assertIn(self.author, book.authors.all())
        self.assertIn(book, self.author.books.all())
    """)
    
    print("\n2. Serializer Testing:")
    print("-" * 25)
    print("""
# test_serializers.py
from django.test import TestCase
from django.contrib.auth.models import User
from rest_framework.test import APITestCase
from .models import Book, Category
from .serializers import BookSerializer, BookCreateSerializer

class BookSerializerTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.category = Category.objects.create(
            name='Test Category'
        )
        self.book_data = {
            'title': 'Test Book',
            'isbn': '1234567890123',
            'published_date': '2023-01-01',
            'pages': 200,
            'price': 29.99,
            'category': self.category.id,
        }
    
    def test_book_serializer_valid_data(self):
        '''Test serializer with valid data'''
        serializer = BookCreateSerializer(data=self.book_data)
        self.assertTrue(serializer.is_valid())
        
        # Test serialized data
        book = Book.objects.create(
            created_by=self.user,
            category=self.category,
            **self.book_data
        )
        serializer = BookSerializer(book)
        
        self.assertEqual(serializer.data['title'], 'Test Book')
        self.assertEqual(serializer.data['isbn'], '1234567890123')
        self.assertIn('created_at', serializer.data)
    
    def test_book_serializer_invalid_data(self):
        '''Test serializer with invalid data'''
        invalid_data = self.book_data.copy()
        invalid_data['isbn'] = '123'  # Too short
        
        serializer = BookCreateSerializer(data=invalid_data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('isbn', serializer.errors)
    
    def test_book_serializer_validation(self):
        '''Test custom validation methods'''
        # Test negative price
        invalid_data = self.book_data.copy()
        invalid_data['price'] = -10
        
        serializer = BookCreateSerializer(data=invalid_data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('price', serializer.errors)
        
        # Test negative pages
        invalid_data = self.book_data.copy()
        invalid_data['pages'] = -100
        
        serializer = BookCreateSerializer(data=invalid_data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('pages', serializer.errors)
    
    def test_book_serializer_create(self):
        '''Test serializer create method'''
        serializer = BookCreateSerializer(data=self.book_data)
        self.assertTrue(serializer.is_valid())
        
        # Mock request context
        from unittest.mock import Mock
        mock_request = Mock()
        mock_request.user = self.user
        serializer.context = {'request': mock_request}
        
        book = serializer.save()
        self.assertEqual(book.title, 'Test Book')
        self.assertEqual(book.created_by, self.user)
    
    def test_nested_serializer(self):
        '''Test nested serializer behavior'''
        from .serializers import BookDetailSerializer
        
        book = Book.objects.create(
            created_by=self.user,
            category=self.category,
            **self.book_data
        )
        
        serializer = BookDetailSerializer(book)
        data = serializer.data
        
        self.assertIn('category', data)
        self.assertEqual(data['category']['name'], 'Test Category')
        self.assertIn('created_by', data)
    """)
    
    print("\n3. API View Testing:")
    print("-" * 24)
    print("""
# test_views.py
from django.urls import reverse
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.test import APITestCase, APIClient
from rest_framework.authtoken.models import Token
from .models import Book, Category

class BookAPITest(APITestCase):
    def setUp(self):
        '''Set up test data'''
        self.client = APIClient()
        
        # Create test users
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.admin_user = User.objects.create_superuser(
            username='admin',
            email='admin@example.com',
            password='adminpass123'
        )
        
        # Create authentication tokens
        self.user_token = Token.objects.create(user=self.user)
        self.admin_token = Token.objects.create(user=self.admin_user)
        
        # Create test data
        self.category = Category.objects.create(
            name='Test Category',
            description='Test description'
        )
        
        self.book = Book.objects.create(
            title='Test Book',
            isbn='1234567890123',
            published_date='2023-01-01',
            pages=200,
            price=29.99,
            category=self.category,
            created_by=self.user
        )
        
        # API URLs
        self.books_url = reverse('book-list')
        self.book_detail_url = reverse('book-detail', kwargs={'pk': self.book.pk})
    
    def test_get_book_list_unauthenticated(self):
        '''Test getting book list without authentication'''
        response = self.client.get(self.books_url)
        
        # Depending on your permission settings
        if hasattr(self, 'ALLOW_ANONYMOUS_READ'):
            self.assertEqual(response.status_code, status.HTTP_200_OK)
        else:
            self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
    
    def test_get_book_list_authenticated(self):
        '''Test getting book list with authentication'''
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.user_token.key)
        response = self.client.get(self.books_url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('results', response.data)
        self.assertEqual(len(response.data['results']), 1)
        self.assertEqual(response.data['results'][0]['title'], 'Test Book')
    
    def test_get_book_detail(self):
        '''Test getting book detail'''
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.user_token.key)
        response = self.client.get(self.book_detail_url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['title'], 'Test Book')
        self.assertEqual(response.data['isbn'], '1234567890123')
    
    def test_create_book_authenticated(self):
        '''Test creating a book with authentication'''
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.user_token.key)
        
        data = {
            'title': 'New Test Book',
            'isbn': '9876543210987',
            'published_date': '2023-02-01',
            'pages': 300,
            'price': 39.99,
            'category': self.category.id,
        }
        
        response = self.client.post(self.books_url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['title'], 'New Test Book')
        self.assertEqual(Book.objects.count(), 2)
        
        # Verify the book was created with correct user
        new_book = Book.objects.get(title='New Test Book')
        self.assertEqual(new_book.created_by, self.user)
    
    def test_create_book_unauthenticated(self):
        '''Test creating a book without authentication'''
        data = {
            'title': 'Unauthorized Book',
            'isbn': '1111111111111',
            'published_date': '2023-03-01',
            'pages': 100,
            'price': 19.99,
            'category': self.category.id,
        }
        
        response = self.client.post(self.books_url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
    
    def test_update_book_owner(self):
        '''Test updating book by owner'''
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.user_token.key)
        
        data = {
            'title': 'Updated Test Book',
            'price': 35.99,
        }
        
        response = self.client.patch(self.book_detail_url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['title'], 'Updated Test Book')
        self.assertEqual(float(response.data['price']), 35.99)
        
        # Verify database was updated
        self.book.refresh_from_db()
        self.assertEqual(self.book.title, 'Updated Test Book')
    
    def test_update_book_non_owner(self):
        '''Test updating book by non-owner'''
        other_user = User.objects.create_user(
            username='otheruser',
            password='otherpass123'
        )
        other_token = Token.objects.create(user=other_user)
        
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + other_token.key)
        
        data = {'title': 'Hacked Book'}
        response = self.client.patch(self.book_detail_url, data, format='json')
        
        # Should be forbidden
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
    
    def test_delete_book_owner(self):
        '''Test deleting book by owner'''
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.user_token.key)
        
        response = self.client.delete(self.book_detail_url)
        
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEqual(Book.objects.count(), 0)
    
    def test_delete_book_admin(self):
        '''Test deleting book by admin'''
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.admin_token.key)
        
        response = self.client.delete(self.book_detail_url)
        
        # Admin should be able to delete any book
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
    
    def test_book_validation_errors(self):
        '''Test API validation errors'''
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.user_token.key)
        
        # Test invalid data
        invalid_data = {
            'title': '',  # Empty title
            'isbn': '123',  # Too short
            'pages': -100,  # Negative pages
            'price': -10,  # Negative price
        }
        
        response = self.client.post(self.books_url, invalid_data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('title', response.data)
        self.assertIn('isbn', response.data)
        self.assertIn('pages', response.data)
        self.assertIn('price', response.data)
    """)
    
    print("\n4. Advanced Testing Patterns:")
    print("-" * 35)
    print("""
# test_advanced.py
from django.test import TestCase, override_settings
from django.core.cache import cache
from unittest.mock import patch, Mock
from rest_framework.test import APITestCase
from .models import Book

class AdvancedAPITest(APITestCase):
    
    def setUp(self):
        cache.clear()  # Clear cache before each test
    
    def tearDown(self):
        cache.clear()  # Clear cache after each test
    
    @override_settings(DEBUG=True)
    def test_with_debug_mode(self):
        '''Test behavior in debug mode'''
        # Test specific to debug mode
        pass
    
    @override_settings(
        CACHES={
            'default': {
                'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
            }
        }
    )
    def test_without_cache(self):
        '''Test behavior without caching'''
        # Test when cache is disabled
        pass
    
    @patch('myapp.services.external_api_call')
    def test_external_api_integration(self, mock_api_call):
        '''Test with mocked external API'''
        # Mock external API response
        mock_api_call.return_value = {
            'status': 'success',
            'data': {'isbn_verified': True}
        }
        
        # Test your code that calls the external API
        # Verify the mock was called correctly
        self.assertTrue(mock_api_call.called)
    
    def test_throttling(self):
        '''Test API throttling limits'''
        self.client.force_authenticate(user=self.user)
        
        # Make requests up to the limit
        for i in range(100):  # Assuming 100/hour limit
            response = self.client.get(self.books_url)
            if response.status_code == 429:  # Too Many Requests
                break
        
        # Next request should be throttled
        response = self.client.get(self.books_url)
        self.assertEqual(response.status_code, status.HTTP_429_TOO_MANY_REQUESTS)
    
    def test_pagination(self):
        '''Test API pagination'''
        # Create multiple books
        for i in range(25):
            Book.objects.create(
                title=f'Book {i}',
                isbn=f'123456789012{i:02d}',
                published_date='2023-01-01',
                pages=200,
                price=29.99,
                category=self.category,
                created_by=self.user
            )
        
        self.client.force_authenticate(user=self.user)
        response = self.client.get(self.books_url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('count', response.data)
        self.assertIn('next', response.data)
        self.assertIn('results', response.data)
        self.assertEqual(response.data['count'], 26)  # 25 + 1 from setUp
    
    def test_filtering(self):
        '''Test API filtering'''
        # Create books in different categories
        fiction_category = Category.objects.create(name='Fiction')
        non_fiction_category = Category.objects.create(name='Non-Fiction')
        
        Book.objects.create(
            title='Fiction Book',
            isbn='1111111111111',
            category=fiction_category,
            created_by=self.user,
            published_date='2023-01-01',
            pages=200,
            price=29.99
        )
        
        Book.objects.create(
            title='Non-Fiction Book', 
            isbn='2222222222222',
            category=non_fiction_category,
            created_by=self.user,
            published_date='2023-01-01',
            pages=300,
            price=39.99
        )
        
        self.client.force_authenticate(user=self.user)
        
        # Test category filtering
        response = self.client.get(f'{self.books_url}?category={fiction_category.id}')
        self.assertEqual(len(response.data['results']), 1)
        self.assertEqual(response.data['results'][0]['title'], 'Fiction Book')
        
        # Test search
        response = self.client.get(f'{self.books_url}?search=Fiction')
        self.assertEqual(len(response.data['results']), 1)
    
    def test_performance(self):
        '''Test API performance'''
        import time
        from django.test.utils import override_settings
        
        # Create many books
        books = []
        for i in range(100):
            books.append(Book(
                title=f'Performance Book {i}',
                isbn=f'999999999999{i:02d}',
                category=self.category,
                created_by=self.user,
                published_date='2023-01-01',
                pages=200,
                price=29.99
            ))
        Book.objects.bulk_create(books)
        
        self.client.force_authenticate(user=self.user)
        
        # Measure response time
        start_time = time.time()
        response = self.client.get(self.books_url)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertLess(response_time, 1.0)  # Should respond within 1 second

# Test factories using factory_boy
import factory
from factory.django import DjangoModelFactory

class UserFactory(DjangoModelFactory):
    class Meta:
        model = User
    
    username = factory.Sequence(lambda n: f"user{n}")
    email = factory.LazyAttribute(lambda obj: f"{obj.username}@example.com")
    first_name = factory.Faker('first_name')
    last_name = factory.Faker('last_name')

class CategoryFactory(DjangoModelFactory):
    class Meta:
        model = Category
    
    name = factory.Faker('word')
    description = factory.Faker('text', max_nb_chars=200)

class BookFactory(DjangoModelFactory):
    class Meta:
        model = Book
    
    title = factory.Faker('sentence', nb_words=3)
    isbn = factory.Faker('isbn13')
    published_date = factory.Faker('date')
    pages = factory.Faker('random_int', min=50, max=1000)
    price = factory.Faker('pydecimal', left_digits=3, right_digits=2, positive=True)
    category = factory.SubFactory(CategoryFactory)
    created_by = factory.SubFactory(UserFactory)

# Using factories in tests
class TestWithFactories(APITestCase):
    def test_book_factory(self):
        '''Test using factory to create test data'''
        book = BookFactory()
        
        self.assertTrue(book.title)
        self.assertTrue(book.isbn)
        self.assertTrue(book.category)
        self.assertTrue(book.created_by)
    
    def test_multiple_books(self):
        '''Test creating multiple books with factory'''
        books = BookFactory.create_batch(5)
        
        self.assertEqual(len(books), 5)
        self.assertEqual(Book.objects.count(), 5)
    """)

demonstrate_drf_testing()

# ============================================================================
# SECTION 9: PERFORMANCE OPTIMIZATION
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 9: PERFORMANCE OPTIMIZATION")
print("=" * 60)

# Question 9: Performance Optimization
print("\n9. How do you optimize Django REST Framework performance?")
print("-" * 62)
print("""
DRF PERFORMANCE OPTIMIZATION:

⚡ PERFORMANCE BOTTLENECKS:
• N+1 queries (most common)
• Unoptimized database queries
• Large serialized datasets
• Missing database indexes
• Inefficient pagination
• Unnecessary data transfer
• Lack of caching

OPTIMIZATION STRATEGIES:
1. Database Optimization
2. Query Optimization
3. Serialization Optimization
4. Caching Strategies
5. Pagination Optimization
6. Field Selection
7. Compression

QUERY OPTIMIZATION:
• select_related() for ForeignKey
• prefetch_related() for ManyToMany
• only() and defer() for field selection
• Database indexes
• Query analysis with django-debug-toolbar

CACHING LEVELS:
• Response Caching
• QuerySet Caching
• Template Fragment Caching
• Database Query Caching
• External API Caching

INTERVIEW QUESTIONS:
Q: "How do you solve N+1 query problems in DRF?"
A: "Use select_related() for ForeignKey and prefetch_related()
   for ManyToMany relationships. Override get_queryset() in views."

Q: "What caching strategies work best with DRF?"
A: "Response caching for read-heavy endpoints, queryset caching
   for expensive queries, and Redis for distributed caching."
""")

def demonstrate_performance_optimization():
    """Demonstrate DRF performance optimization techniques"""
    print("Performance Optimization Complete Guide:")
    
    print("\n1. Query Optimization:")
    print("-" * 25)
    print("""
# PROBLEM: N+1 Query Issue
class BookListView(generics.ListAPIView):
    '''BAD: This creates N+1 queries'''
    queryset = Book.objects.all()  # 1 query
    serializer_class = BookSerializer
    
    # For each book, serializer accesses:
    # - book.category.name (N queries)
    # - book.created_by.username (N queries)
    # - book.authors.all() (N queries)

# SOLUTION: Optimized Queries
class OptimizedBookListView(generics.ListAPIView):
    '''GOOD: Optimized with select_related and prefetch_related'''
    serializer_class = BookSerializer
    
    def get_queryset(self):
        return Book.objects.select_related(
            'category',        # ForeignKey - use select_related
            'created_by'       # ForeignKey - use select_related
        ).prefetch_related(
            'authors',         # ManyToMany - use prefetch_related
            'tags',           # ManyToMany - use prefetch_related
            'reviews__reviewer'  # Nested relationship
        ).order_by('-created_at')

# Advanced Prefetching
from django.db.models import Prefetch

class AdvancedOptimizedView(generics.ListAPIView):
    def get_queryset(self):
        # Custom prefetch with filtering
        active_reviews = Prefetch(
            'reviews',
            queryset=Review.objects.filter(is_active=True).select_related('reviewer'),
            to_attr='active_reviews'
        )
        
        recent_comments = Prefetch(
            'comments',
            queryset=Comment.objects.filter(
                created_at__gte=timezone.now() - timedelta(days=30)
            ).select_related('author'),
            to_attr='recent_comments'
        )
        
        return Book.objects.select_related(
            'category', 'created_by'
        ).prefetch_related(
            'authors',
            active_reviews,
            recent_comments
        ).only(
            # Only fetch required fields
            'id', 'title', 'isbn', 'price', 'published_date',
            'category__name', 'created_by__username'
        )

# Field Selection Optimization
class FieldOptimizedViewSet(viewsets.ModelViewSet):
    serializer_class = BookSerializer
    
    def get_queryset(self):
        queryset = Book.objects.select_related('category', 'created_by')
        
        # Different optimizations for different actions
        if self.action == 'list':
            # List view - minimal fields
            return queryset.only(
                'id', 'title', 'price', 'published_date',
                'category__name'
            )
        elif self.action == 'retrieve':
            # Detail view - all fields with relationships
            return queryset.prefetch_related(
                'authors', 'tags', 'reviews__reviewer'
            )
        else:
            return queryset

# Database Indexes (in models.py)
class Book(models.Model):
    title = models.CharField(max_length=200, db_index=True)
    isbn = models.CharField(max_length=13, unique=True, db_index=True)
    published_date = models.DateField(db_index=True)
    price = models.DecimalField(max_digits=10, decimal_places=2, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['published_date', 'price']),
            models.Index(fields=['category', 'created_at']),
            models.Index(fields=['created_by', 'published_date']),
        ]
        ordering = ['-created_at']  # Default ordering
    """)
    
    print("\n2. Serialization Optimization:")
    print("-" * 34)
    print("""
# Optimized Serializers
class OptimizedBookListSerializer(serializers.ModelSerializer):
    '''Lightweight serializer for list views'''
    category_name = serializers.CharField(source='category.name', read_only=True)
    author_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Book
        fields = [
            'id', 'title', 'price', 'published_date',
            'category_name', 'author_count'
        ]
    
    def get_author_count(self, obj):
        # Efficient if prefetched
        return obj.authors.count()

class OptimizedBookDetailSerializer(serializers.ModelSerializer):
    '''Full serializer for detail views'''
    category = CategorySerializer(read_only=True)
    authors = AuthorSerializer(many=True, read_only=True)
    review_stats = serializers.SerializerMethodField()
    
    class Meta:
        model = Book
        fields = '__all__'
    
    def get_review_stats(self, obj):
        # Use prefetched data if available
        if hasattr(obj, 'active_reviews'):
            reviews = obj.active_reviews
        else:
            reviews = obj.reviews.all()
        
        if reviews:
            ratings = [review.rating for review in reviews]
            return {
                'count': len(ratings),
                'average': sum(ratings) / len(ratings),
                'max': max(ratings),
                'min': min(ratings)
            }
        return {'count': 0, 'average': None, 'max': None, 'min': None}

# Dynamic Field Selection
class DynamicFieldsSerializer(serializers.ModelSerializer):
    '''Serializer that allows dynamic field selection'''
    
    def __init__(self, *args, **kwargs):
        # Extract fields from context
        fields = None
        context = kwargs.get('context', {})
        request = context.get('request')
        
        if request:
            fields = request.query_params.get('fields')
        
        if fields:
            fields = fields.split(',')
            
        super().__init__(*args, **kwargs)
        
        if fields:
            # Drop unwanted fields
            allowed = set(fields)
            existing = set(self.fields)
            for field_name in existing - allowed:
                self.fields.pop(field_name)

class BookDynamicSerializer(DynamicFieldsSerializer):
    class Meta:
        model = Book
        fields = '__all__'

# Usage: GET /api/books/?fields=id,title,price

# Cached Properties in Serializers
class CachedBookSerializer(serializers.ModelSerializer):
    expensive_calculation = serializers.SerializerMethodField()
    
    def get_expensive_calculation(self, obj):
        # Cache expensive calculations
        cache_key = f'book_calc_{obj.id}'
        result = cache.get(cache_key)
        
        if result is None:
            # Perform expensive calculation
            result = self.perform_expensive_calculation(obj)
            cache.set(cache_key, result, 3600)  # Cache for 1 hour
        
        return result
    
    def perform_expensive_calculation(self, obj):
        # Simulate expensive operation
        return sum(review.rating for review in obj.reviews.all())
    """)
    
    print("\n3. Caching Strategies:")
    print("-" * 26)
    print("""
# Response Caching
from django.views.decorators.cache import cache_page
from django.utils.decorators import method_decorator
from rest_framework.response import Response

@method_decorator(cache_page(60 * 15), name='list')  # 15 minutes
class CachedBookListView(generics.ListAPIView):
    queryset = Book.objects.all()
    serializer_class = BookSerializer

# ViewSet with Cached Actions
class CachedBookViewSet(viewsets.ModelViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
    
    @method_decorator(cache_page(60 * 15))
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)
    
    @method_decorator(cache_page(60 * 30))
    def retrieve(self, request, *args, **kwargs):
        return super().retrieve(request, *args, **kwargs)

# Custom Caching in Views
from django.core.cache import cache

class SmartCachedBookViewSet(viewsets.ModelViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
    
    def list(self, request, *args, **kwargs):
        # Create cache key based on query parameters
        cache_key = self.get_cache_key(request)
        cached_response = cache.get(cache_key)
        
        if cached_response:
            return Response(cached_response)
        
        response = super().list(request, *args, **kwargs)
        
        # Cache successful responses
        if response.status_code == 200:
            cache.set(cache_key, response.data, 60 * 15)
        
        return response
    
    def get_cache_key(self, request):
        # Include relevant query parameters in cache key
        query_params = request.query_params
        cache_params = {
            'page': query_params.get('page', '1'),
            'category': query_params.get('category', ''),
            'search': query_params.get('search', ''),
            'ordering': query_params.get('ordering', ''),
        }
        
        cache_key = 'book_list_' + '_'.join(f'{k}:{v}' for k, v in cache_params.items())
        return cache_key
    
    def perform_create(self, serializer):
        # Invalidate cache on creation
        self.invalidate_list_cache()
        super().perform_create(serializer)
    
    def perform_update(self, serializer):
        # Invalidate cache on update
        self.invalidate_list_cache()
        self.invalidate_detail_cache(serializer.instance.pk)
        super().perform_update(serializer)
    
    def invalidate_list_cache(self):
        # Invalidate all list caches
        cache.delete_many(['book_list_*'])
    
    def invalidate_detail_cache(self, pk):
        cache.delete(f'book_detail_{pk}')

# QuerySet Caching
class CachedQuerySetMixin:
    cache_timeout = 60 * 15  # 15 minutes
    
    def get_queryset(self):
        cache_key = self.get_queryset_cache_key()
        cached_queryset = cache.get(cache_key)
        
        if cached_queryset is None:
            cached_queryset = list(super().get_queryset())
            cache.set(cache_key, cached_queryset, self.cache_timeout)
        
        return cached_queryset
    
    def get_queryset_cache_key(self):
        return f'{self.__class__.__name__}_queryset'

# External API Caching
import requests
from django.core.cache import cache

class ExternalAPIService:
    @staticmethod
    def get_book_metadata(isbn):
        cache_key = f'book_metadata_{isbn}'
        metadata = cache.get(cache_key)
        
        if metadata is None:
            try:
                response = requests.get(f'https://api.books.com/isbn/{isbn}')
                metadata = response.json()
                cache.set(cache_key, metadata, 60 * 60 * 24)  # Cache for 24 hours
            except requests.RequestException:
                metadata = {}
        
        return metadata

# Redis Configuration for Production
# settings.py
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'SERIALIZER': 'django_redis.serializers.json.JSONSerializer',
            'COMPRESSOR': 'django_redis.compressors.zlib.ZlibCompressor',
        },
        'KEY_PREFIX': 'drf_api',
        'TIMEOUT': 60 * 15,  # 15 minutes default
    }
}
    """)

def demonstrate_advanced_performance():
    """Demonstrate advanced performance optimization techniques"""
    print("Advanced Performance Optimization:")
    
    print("\n4. Advanced Performance Techniques:")
    print("-" * 40)
   
# Database Connection Pooling
# settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'your_db',
        'USER': 'your_user',
        'PASSWORD': 'your_password',
        'HOST': 'localhost',
        'PORT': '5432',
        'OPTIONS': {
            'MAX_CONNS': 20,
            'OPTIONS': {
                'MAX_CONNS': 20,
            }
        },
    }
}

# Async Views (Django 4.1+)
from django.http import JsonResponse
import asyncio

class AsyncBookView(View):
    async def get(self, request):
        # Async database operations
        books = await sync_to_async(list)(
            Book.objects.select_related('category').all()
        )
        
        # Async external API calls
        tasks = [self.get_external_data(book.isbn) for book in books]
        external_data = await asyncio.gather(*tasks)
        
        return JsonResponse({'books': books, 'external': external_data})
    
    async def get_external_data(self, isbn):
        # Async HTTP request
        async with aiohttp.ClientSession() as session:
            async with session.get(f'https://api.books.com/{isbn}') as response:
                return await response.json()

# Bulk Operations
class BulkBookViewSet(viewsets.ModelViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
    
    @action(detail=False, methods=['post'])
    def bulk_create(self, request):
        '''Create multiple books in one operation'''
        serializer = self.get_serializer(data=request.data, many=True)
        serializer.is_valid(raise_exception=True)
        
        # Use bulk_create for better performance
        books = [Book(**item) for item in serializer.validated_data]
        Book.objects.bulk_create(books)
        
        return Response({'created': len(books)}, status=status.HTTP_201_CREATED)
    
    @action(detail=False, methods=['patch'])
    def bulk_update(self, request):
        '''Update multiple books in one operation'''
        book_updates = request.data
        
        # Use bulk_update for better performance
        books_to_update = []
        for update_data in book_updates:
            book_id = update_data.pop('id')
            try:
                book = Book.objects.get(id=book_id)
                for key, value in update_data.items():
                    setattr(book, key, value)
                books_to_update.append(book)
            except Book.DoesNotExist:
                continue
        
        Book.objects.bulk_update(books_to_update, update_data.keys())
        
        return Response({'updated': len(books_to_update)})

# Database Raw Queries for Complex Operations
class AnalyticsView(APIView):
    def get(self, request):
        # Use raw SQL for complex analytics
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    c.name as category,
                    COUNT(b.id) as book_count,
                    AVG(b.price) as avg_price,
                    AVG(r.rating) as avg_rating
                FROM books_book b
                JOIN books_category c ON b.category_id = c.id
                LEFT JOIN books_review r ON b.id = r.book_id
                GROUP BY c.id, c.name
                ORDER BY book_count DESC
            """)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'category': row[0],
                    'book_count': row[1],
                    'avg_price': float(row[2]) if row[2] else None,
                    'avg_rating': float(row[3]) if row[3] else None,
                })
        
        return Response(results)

# Response Compression
# settings.py
MIDDLEWARE = [
    'django.middleware.gzip.GZipMiddleware',  # Add this
    # ... other middleware
]

# Custom compression for large datasets
from django.http import HttpResponse
import gzip
import json

class CompressedJSONResponse(HttpResponse):
    def __init__(self, data, **kwargs):
        content = json.dumps(data)
        
        # Compress if content is large
        if len(content) > 1024:  # 1KB threshold
            content = gzip.compress(content.encode('utf-8'))
            kwargs['content_type'] = 'application/json'
            kwargs['headers'] = kwargs.get('headers', {})
            kwargs['headers']['Content-Encoding'] = 'gzip'
        
        super().__init__(content, **kwargs)

# Performance Monitoring
import time
import logging

logger = logging.getLogger(__name__)

class PerformanceMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        start_time = time.time()
        
        response = self.get_response(request)
        
        duration = time.time() - start_time
        
        # Log slow requests
        if duration > 1.0:  # Slower than 1 second
            logger.warning(
                f"Slow request: {request.path} took {duration:.2f}s"
            )
        
        # Add performance header
        response['X-Response-Time'] = f'{duration:.3f}'
        
        return response
    """)

demonstrate_advanced_performance()

print("\n" + "=" * 80)
print("COMPREHENSIVE DRF INTERVIEW QUESTIONS & ANSWERS")
print("=" * 80)

print("""
🎯 TOP 50 DRF INTERVIEW QUESTIONS WITH DETAILED ANSWERS:

FUNDAMENTALS (Questions 1-10):
===============================

Q1: "What is Django REST Framework and why use it?"
A1: "DRF is a powerful toolkit for building Web APIs in Django. Use it for:
    • Rapid API development with minimal code
    • Built-in authentication and permissions
    • Automatic serialization/deserialization  
    • Browsable API for testing
    • Production-ready features like throttling, pagination"

Q2: "Explain the difference between Serializer and ModelSerializer?"
A2: "Serializer requires manual field definition and create/update methods.
    ModelSerializer automatically generates fields from model and provides
    default create/update. Use Serializer for complex custom logic,
    ModelSerializer for standard CRUD operations."

Q3: "What are ViewSets and how do they differ from APIView?"
A3: "ViewSets group related views together and provide automatic routing.
    APIView gives maximum control but requires manual implementation.
    ViewSets promote DRY principle, APIView offers flexibility."

Q4: "How does DRF handle authentication vs permissions?"
A4: "Authentication identifies WHO the user is (TokenAuth, SessionAuth).
    Permissions determine WHAT they can do (IsAuthenticated, custom).
    Authentication runs first, then permissions check access."

Q5: "Explain the serialization process in DRF?"
A5: "Python Object → Serializer → Python Dict → JSON
    For input: JSON → Python Dict → Serializer → Python Object
    Includes validation, transformation, and relationship handling."

ADVANCED CONCEPTS (Questions 11-25):
====================================

Q11: "How do you optimize DRF performance for large datasets?"
A11: "Use select_related() for ForeignKeys, prefetch_related() for M2M,
     implement proper pagination, add database indexes, use caching,
     optimize serializers with only required fields."

Q12: "What's the difference between throttling and rate limiting?"
A12: "Same concept - throttling controls request frequency per user/IP.
     DRF provides AnonRateThrottle, UserRateThrottle, ScopedRateThrottle.
     Prevents abuse and ensures fair resource usage."

Q13: "How do you implement custom authentication in DRF?"
A13: "Extend BaseAuthentication class, implement authenticate() method
     that returns (user, auth) tuple or None. Handle authentication
     headers, validate tokens, and raise AuthenticationFailed for errors."

Q14: "Explain pagination strategies in DRF?"
A14: "PageNumberPagination for standard pagination with page numbers.
     LimitOffsetPagination for offset-based pagination.
     CursorPagination for large datasets with cursor-based navigation.
     Choose based on data size and UI requirements."

Q15: "How do you handle nested relationships in serializers?"
A15: "Use nested serializers, depth parameter, or SerializerMethodField.
     For write operations, handle nested data in create/update methods.
     Use PrimaryKeyRelatedField for simple relationships."

REAL-WORLD SCENARIOS (Questions 26-40):
=======================================

Q26: "How do you implement file upload in DRF?"
A26: "Use FileField or ImageField in serializer, handle files in views
     with request.FILES, validate file types/sizes, store in media
     directory or cloud storage, return file URLs in responses."

Q27: "How do you test DRF APIs comprehensively?"
A27: "Use APITestCase and APIClient, test all CRUD operations,
     test authentication/permissions, validate error responses,
     test edge cases, mock external dependencies, check performance."

Q28: "How do you implement API versioning in DRF?"
A28: "URL versioning: /api/v1/books/, Header versioning: Accept-version,
     Query parameter: ?version=v1, Hostname: v1.api.example.com
     Configure in settings with DEFAULT_VERSION and ALLOWED_VERSIONS."

Q29: "How do you handle bulk operations efficiently?"
A29: "Use bulk_create() and bulk_update() for database operations,
     implement bulk endpoints in ViewSets, validate bulk data,
     handle partial failures gracefully, use transactions."

Q30: "How do you implement real-time features with DRF?"
A30: "Use Django Channels for WebSockets, implement async views,
     use signals for real-time updates, integrate with Redis/RabbitMQ
     for message queuing, use Server-Sent Events for live data."

TROUBLESHOOTING (Questions 41-50):
==================================

Q41: "How do you debug N+1 query problems in DRF?"
A41: "Use django-debug-toolbar to identify queries, implement
     select_related() and prefetch_related(), optimize serializers,
     use only() and defer() for field selection."

Q42: "How do you handle CORS in DRF APIs?"
A42: "Install django-cors-headers, add to middleware, configure
     CORS_ALLOWED_ORIGINS, CORS_ALLOW_CREDENTIALS, handle preflight
     requests, secure for production environments."

Q43: "How do you implement custom error handling?"
A43: "Create custom exception handler, override exception_handler(),
     return consistent error format, log errors, handle different
     exception types (ValidationError, PermissionDenied, etc.)"

Q44: "How do you secure DRF APIs in production?"
A44: "Use HTTPS, implement proper authentication, validate all inputs,
     use CORS properly, implement rate limiting, sanitize responses,
     keep dependencies updated, use environment variables for secrets."

Q45: "How do you monitor DRF API performance?"
A45: "Use APM tools (New Relic, DataDog), implement custom metrics,
     log slow queries, monitor response times, set up alerts,
     use database query analysis tools."

SYSTEM DESIGN QUESTIONS:
=======================

Q46: "Design a scalable book recommendation API?"
A46: "Use microservices architecture, implement caching layers,
     use async processing for ML recommendations, implement proper
     pagination, use CDN for static content, design for horizontal scaling."

Q47: "How would you handle millions of API requests per day?"
A47: "Implement caching at multiple levels, use load balancers,
     database read replicas, CDN for static content, async processing
     for heavy operations, proper monitoring and auto-scaling."

Q48: "Design authentication for a multi-tenant SaaS API?"
A48: "Use JWT with tenant information, implement tenant isolation,
     use middleware for tenant context, design proper data models,
     implement tenant-specific rate limiting and permissions."

Q49: "How do you ensure API backward compatibility?"
A49: "Use API versioning, deprecate features gradually, maintain
     old versions, use feature flags, communicate changes clearly,
     implement proper testing for all versions."

Q50: "Design a robust API error handling system?"
A50: "Create consistent error response format, implement proper
     HTTP status codes, log all errors with context, provide
     helpful error messages, implement retry mechanisms for clients."

KEY TAKEAWAYS FOR INTERVIEWS:
============================
✅ Understand DRF architecture and components thoroughly
✅ Know when to use different serializer types and view classes
✅ Understand performance optimization techniques
✅ Be familiar with authentication and permission patterns
✅ Know how to test APIs properly
✅ Understand security best practices
✅ Be able to design scalable API architectures
✅ Know common debugging techniques
✅ Understand caching strategies
✅ Be familiar with real-world deployment considerations

HANDS-ON PRACTICE:
=================
🔧 Build a complete API with authentication
🔧 Implement complex filtering and search
🔧 Practice performance optimization
🔧 Write comprehensive tests
🔧 Deploy to production environment
🔧 Monitor and debug real issues
🔧 Implement advanced features like WebSockets
🔧 Work with external API integrations
""")

print("\n" + "=" * 80)
print("END OF COMPREHENSIVE DRF MASTERY GUIDE")
print("You now have complete knowledge of Django REST Framework!")
print("Practice these concepts to become a DRF expert! 🚀")
print("=" * 80)
