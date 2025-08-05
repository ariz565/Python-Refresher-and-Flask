"""
Django REST Framework (DRF) - Complete Mastery Guide
Comprehensive coverage of all DRF concepts with interview questions
From basics to advanced patterns with detailed explanations
"""

print("=" * 80)
print("DJANGO REST FRAMEWORK - COMPLETE MASTERY GUIDE")
print("=" * 80)

# ============================================================================
# SECTION 1: DRF FUNDAMENTALS AND ARCHITECTURE
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 1: DRF FUNDAMENTALS AND ARCHITECTURE")
print("=" * 60)

# Question 1: What is Django REST Framework?
print("\n1. What is Django REST Framework and why use it?")
print("-" * 51)
print("""
DJANGO REST FRAMEWORK (DRF) OVERVIEW:

🔧 DEFINITION:
Django REST Framework is a powerful and flexible toolkit for building 
Web APIs in Django. It's built on top of Django and follows REST principles.

🏗️ CORE ARCHITECTURE:
┌─────────────────────────────────────────┐
│              DRF Layer                  │  ← API Views, Serializers
├─────────────────────────────────────────┤
│             Django Layer                │  ← Models, URLs, Middleware
├─────────────────────────────────────────┤
│             Database Layer              │  ← PostgreSQL, MySQL, etc.
└─────────────────────────────────────────┘

KEY COMPONENTS:
🎯 Serializers: Data conversion and validation
🎯 Views/ViewSets: Request handling and business logic
🎯 Routers: Automatic URL routing
🎯 Authentication: Multiple authentication schemes
🎯 Permissions: Fine-grained access control
🎯 Throttling: Rate limiting requests
🎯 Filtering: Data filtering and searching
🎯 Pagination: Result pagination
🎯 Content Negotiation: Multiple response formats

DRF vs DJANGO VIEWS:
Feature              | Django Views    | DRF Views
---------------------|-----------------|----------------
API Development      | Manual          | Built-in support
Serialization        | Manual          | Automatic
Authentication       | Manual          | Multiple schemes
Permissions          | Manual          | Declarative
Documentation        | Manual          | Browsable API
Content Types        | HTML only       | JSON, XML, etc.
HTTP Methods         | Manual          | Automatic routing

WHY USE DRF?
✅ Rapid API development
✅ Built-in authentication and permissions
✅ Automatic serialization/deserialization
✅ Browsable API for testing
✅ Flexible and customizable
✅ Excellent documentation
✅ Large community and ecosystem
✅ Production-ready features

REAL-WORLD APPLICATIONS:
🌐 Social Media APIs (Twitter, Instagram)
🛍️ E-commerce Platforms (Shopify, Amazon)
📱 Mobile App Backends
🏢 Enterprise APIs
🎮 Gaming Platforms
📊 Data Analytics Dashboards

INTERVIEW INSIGHT:
DRF abstracts away the complexity of building REST APIs while 
maintaining Django's philosophy of "Don't Repeat Yourself".
""")

def demonstrate_drf_basics():
    """Demonstrate DRF basic concepts"""
    print("DRF Fundamentals Demo:")
    
    print("\n1. Basic DRF Setup:")
    print("-" * 22)
    print("""
# settings.py
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third party apps
    'rest_framework',
    'django_filters',
    'corsheaders',
    
    # Local apps
    'apps.users',
    'apps.blog',
    'apps.api',
]

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.TokenAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
        'rest_framework.renderers.BrowsableAPIRenderer',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
    'DEFAULT_FILTER_BACKENDS': [
        'django_filters.rest_framework.DjangoFilterBackend',
        'rest_framework.filters.SearchFilter',
        'rest_framework.filters.OrderingFilter',
    ],
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle'
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '100/hour',
        'user': '1000/hour'
    }
}
    """)
    
    print("\n2. Simple API Example:")
    print("-" * 26)
    print("""
# models.py
from django.db import models
from django.contrib.auth.models import User

class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.CharField(max_length=100)
    isbn = models.CharField(max_length=13, unique=True)
    published_date = models.DateField()
    pages = models.IntegerField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.title

# serializers.py
from rest_framework import serializers
from .models import Book

class BookSerializer(serializers.ModelSerializer):
    class Meta:
        model = Book
        fields = '__all__'
        read_only_fields = ('created_by', 'created_at', 'updated_at')

# views.py
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated
from .models import Book
from .serializers import BookSerializer

class BookViewSet(viewsets.ModelViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
    permission_classes = [IsAuthenticated]
    
    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)

# urls.py
from rest_framework.routers import DefaultRouter
from .views import BookViewSet

router = DefaultRouter()
router.register(r'books', BookViewSet)
urlpatterns = router.urls
    """)

demonstrate_drf_basics()

# ============================================================================
# SECTION 2: SERIALIZERS - COMPLETE GUIDE
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 2: SERIALIZERS - COMPLETE GUIDE")
print("=" * 60)

# Question 2: Serializers Deep Dive
print("\n2. What are DRF Serializers and how do they work?")
print("-" * 52)
print("""
SERIALIZERS OVERVIEW:

🔄 PURPOSE:
Serializers convert complex data types (Django models, querysets) into 
Python data types that can be rendered into JSON, XML, etc., and vice versa.

SERIALIZER TYPES:
1. Serializer (Base class)
2. ModelSerializer (Most common)
3. HyperlinkedModelSerializer (With hyperlinks)
4. ListSerializer (For lists)
5. BaseSerializer (Custom implementations)

SERIALIZATION PROCESS:
Python Object → Serializer → Python Dict → JSON/XML

DESERIALIZATION PROCESS:
JSON/XML → Python Dict → Serializer → Python Object

KEY FEATURES:
🔧 Data Validation
🔧 Data Transformation
🔧 Nested Relationships
🔧 Custom Fields
🔧 Method Fields
🔧 Write/Read-only Fields
🔧 Field-level Validation
🔧 Object-level Validation

INTERVIEW QUESTIONS:
Q: "What's the difference between Serializer and ModelSerializer?"
A: "Serializer requires manual field definition and validation. 
   ModelSerializer automatically generates fields based on model 
   and provides default implementations for create() and update()."

Q: "How does DRF handle nested relationships?"
A: "Through nested serializers, depth parameter, or custom 
   SerializerMethodField for complex transformations."
""")

def demonstrate_serializers():
    """Demonstrate comprehensive serializer examples"""
    print("Serializers Complete Guide:")
    
    print("\n1. Basic Serializer Types:")
    print("-" * 30)
    print("""
# Basic Serializer (Manual field definition)
from rest_framework import serializers

class BookBasicSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=200)
    author = serializers.CharField(max_length=100)
    isbn = serializers.CharField(max_length=13)
    published_date = serializers.DateField()
    pages = serializers.IntegerField()
    price = serializers.DecimalField(max_digits=10, decimal_places=2)
    
    def create(self, validated_data):
        return Book.objects.create(**validated_data)
    
    def update(self, instance, validated_data):
        instance.title = validated_data.get('title', instance.title)
        instance.author = validated_data.get('author', instance.author)
        instance.isbn = validated_data.get('isbn', instance.isbn)
        instance.published_date = validated_data.get('published_date', instance.published_date)
        instance.pages = validated_data.get('pages', instance.pages)
        instance.price = validated_data.get('price', instance.price)
        instance.save()
        return instance

# ModelSerializer (Automatic field generation)
class BookModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = Book
        fields = '__all__'  # or list specific fields
        # fields = ['title', 'author', 'isbn', 'published_date']
        # exclude = ['created_at', 'updated_at']
        read_only_fields = ('created_by', 'created_at', 'updated_at')
        extra_kwargs = {
            'price': {'write_only': True},
            'isbn': {'validators': []}  # Remove default validators
        }

# HyperlinkedModelSerializer (URLs instead of IDs)
class BookHyperlinkedSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Book
        fields = '__all__'
        extra_kwargs = {
            'url': {'view_name': 'book-detail', 'lookup_field': 'pk'}
        }
    """)
    
    print("\n2. Field Types and Validation:")
    print("-" * 35)
    print("""
# All DRF Field Types
class ComprehensiveSerializer(serializers.Serializer):
    # Basic Fields
    char_field = serializers.CharField(max_length=100)
    email_field = serializers.EmailField()
    regex_field = serializers.RegexField(regex=r'^[a-zA-Z0-9]+$')
    slug_field = serializers.SlugField()
    url_field = serializers.URLField()
    
    # Numeric Fields
    integer_field = serializers.IntegerField(min_value=0, max_value=100)
    float_field = serializers.FloatField()
    decimal_field = serializers.DecimalField(max_digits=10, decimal_places=2)
    
    # Date/Time Fields
    date_field = serializers.DateField()
    datetime_field = serializers.DateTimeField()
    time_field = serializers.TimeField()
    duration_field = serializers.DurationField()
    
    # Boolean Fields
    boolean_field = serializers.BooleanField()
    null_boolean_field = serializers.NullBooleanField()
    
    # Choice Fields
    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('published', 'Published'),
        ('archived', 'Archived'),
    ]
    choice_field = serializers.ChoiceField(choices=STATUS_CHOICES)
    multiple_choice_field = serializers.MultipleChoiceField(choices=STATUS_CHOICES)
    
    # File Fields
    file_field = serializers.FileField()
    image_field = serializers.ImageField()
    
    # Composite Fields
    list_field = serializers.ListField(child=serializers.CharField())
    dict_field = serializers.DictField()
    json_field = serializers.JSONField()
    
    # Relationship Fields
    primary_key_field = serializers.PrimaryKeyRelatedField(queryset=User.objects.all())
    string_related_field = serializers.StringRelatedField()
    slug_related_field = serializers.SlugRelatedField(slug_field='username', queryset=User.objects.all())
    
    # Custom Fields
    custom_field = serializers.SerializerMethodField()
    
    def get_custom_field(self, obj):
        return f"Custom: {obj.title}"

# Field-level Validation
class BookWithValidationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Book
        fields = '__all__'
    
    def validate_isbn(self, value):
        '''Validate ISBN format'''
        if len(value) != 13:
            raise serializers.ValidationError("ISBN must be 13 characters long")
        if not value.isdigit():
            raise serializers.ValidationError("ISBN must contain only digits")
        return value
    
    def validate_pages(self, value):
        '''Validate page count'''
        if value <= 0:
            raise serializers.ValidationError("Pages must be greater than 0")
        if value > 10000:
            raise serializers.ValidationError("Pages cannot exceed 10,000")
        return value
    
    def validate_price(self, value):
        '''Validate price'''
        if value < 0:
            raise serializers.ValidationError("Price cannot be negative")
        return value
    
    # Object-level Validation
    def validate(self, data):
        '''Cross-field validation'''
        if data.get('pages', 0) > 1000 and data.get('price', 0) < 10:
            raise serializers.ValidationError(
                "Books with more than 1000 pages should cost at least $10"
            )
        
        # Check if ISBN already exists (for create)
        if not self.instance and Book.objects.filter(isbn=data.get('isbn')).exists():
            raise serializers.ValidationError("Book with this ISBN already exists")
        
        return data
    """)
    
    print("\n3. Nested Relationships:")
    print("-" * 29)
    print("""
# Models for nested relationships
class Author(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    bio = models.TextField(blank=True)

class Category(models.Model):
    name = models.CharField(max_length=50)
    description = models.TextField(blank=True)

class Book(models.Model):
    title = models.CharField(max_length=200)
    authors = models.ManyToManyField(Author, related_name='books')
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    isbn = models.CharField(max_length=13, unique=True)
    published_date = models.DateField()

class Review(models.Model):
    book = models.ForeignKey(Book, related_name='reviews', on_delete=models.CASCADE)
    reviewer = models.ForeignKey(User, on_delete=models.CASCADE)
    rating = models.IntegerField(choices=[(i, i) for i in range(1, 6)])
    comment = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

# Nested Serializers
class AuthorSerializer(serializers.ModelSerializer):
    book_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Author
        fields = ['id', 'name', 'email', 'bio', 'book_count']
    
    def get_book_count(self, obj):
        return obj.books.count()

class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = ['id', 'name', 'description']

class ReviewSerializer(serializers.ModelSerializer):
    reviewer_name = serializers.CharField(source='reviewer.username', read_only=True)
    
    class Meta:
        model = Review
        fields = ['id', 'reviewer', 'reviewer_name', 'rating', 'comment', 'created_at']
        read_only_fields = ('reviewer', 'created_at')

# Main serializer with nested relationships
class BookDetailSerializer(serializers.ModelSerializer):
    authors = AuthorSerializer(many=True, read_only=True)
    author_ids = serializers.ListField(
        child=serializers.IntegerField(),
        write_only=True,
        required=False
    )
    category = CategorySerializer(read_only=True)
    category_id = serializers.IntegerField(write_only=True)
    reviews = ReviewSerializer(many=True, read_only=True)
    average_rating = serializers.SerializerMethodField()
    review_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Book
        fields = [
            'id', 'title', 'isbn', 'published_date',
            'authors', 'author_ids', 'category', 'category_id',
            'reviews', 'average_rating', 'review_count'
        ]
    
    def get_average_rating(self, obj):
        reviews = obj.reviews.all()
        if reviews:
            return sum(review.rating for review in reviews) / len(reviews)
        return None
    
    def get_review_count(self, obj):
        return obj.reviews.count()
    
    def create(self, validated_data):
        author_ids = validated_data.pop('author_ids', [])
        book = Book.objects.create(**validated_data)
        if author_ids:
            book.authors.set(author_ids)
        return book
    
    def update(self, instance, validated_data):
        author_ids = validated_data.pop('author_ids', None)
        
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()
        
        if author_ids is not None:
            instance.authors.set(author_ids)
        
        return instance

# Using depth parameter for automatic nesting
class BookWithDepthSerializer(serializers.ModelSerializer):
    class Meta:
        model = Book
        fields = '__all__'
        depth = 2  # Automatically nest 2 levels deep
    """)
    
    print("\n4. Advanced Serializer Patterns:")
    print("-" * 37)
    print("""
# Dynamic Fields Serializer
class DynamicFieldsModelSerializer(serializers.ModelSerializer):
    '''A ModelSerializer that allows dynamic field selection'''
    
    def __init__(self, *args, **kwargs):
        # Extract fields parameter
        fields = kwargs.pop('fields', None)
        super().__init__(*args, **kwargs)
        
        if fields is not None:
            # Drop any fields that are not specified
            allowed = set(fields)
            existing = set(self.fields)
            for field_name in existing - allowed:
                self.fields.pop(field_name)

class BookDynamicSerializer(DynamicFieldsModelSerializer):
    class Meta:
        model = Book
        fields = '__all__'

# Usage: BookDynamicSerializer(instance, fields=('title', 'author', 'price'))

# Conditional Serialization
class ConditionalBookSerializer(serializers.ModelSerializer):
    price = serializers.SerializerMethodField()
    detailed_info = serializers.SerializerMethodField()
    
    class Meta:
        model = Book
        fields = ['title', 'author', 'price', 'detailed_info']
    
    def get_price(self, obj):
        # Only show price to authenticated users
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            return obj.price
        return None
    
    def get_detailed_info(self, obj):
        # Show detailed info only to staff
        request = self.context.get('request')
        if request and request.user.is_staff:
            return {
                'isbn': obj.isbn,
                'pages': obj.pages,
                'created_at': obj.created_at
            }
        return None

# Polymorphic Serialization
class MediaSerializer(serializers.ModelSerializer):
    class Meta:
        model = Media
        fields = ['id', 'title', 'type']
    
    def to_representation(self, instance):
        if instance.type == 'book':
            return BookSerializer(instance).data
        elif instance.type == 'movie':
            return MovieSerializer(instance).data
        return super().to_representation(instance)

# Custom create/update logic
class BookAdvancedSerializer(serializers.ModelSerializer):
    tags = serializers.ListField(
        child=serializers.CharField(max_length=50),
        write_only=True,
        required=False
    )
    
    class Meta:
        model = Book
        fields = '__all__'
    
    def create(self, validated_data):
        tags_data = validated_data.pop('tags', [])
        book = Book.objects.create(**validated_data)
        
        # Handle tags
        for tag_name in tags_data:
            tag, created = Tag.objects.get_or_create(name=tag_name)
            book.tags.add(tag)
        
        # Send notification
        self.send_creation_notification(book)
        
        return book
    
    def update(self, instance, validated_data):
        tags_data = validated_data.pop('tags', None)
        
        # Update basic fields
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()
        
        # Update tags if provided
        if tags_data is not None:
            instance.tags.clear()
            for tag_name in tags_data:
                tag, created = Tag.objects.get_or_create(name=tag_name)
                instance.tags.add(tag)
        
        return instance
    
    def send_creation_notification(self, book):
        # Custom logic for notifications
        pass
    """)

demonstrate_serializers()

# ============================================================================
# SECTION 3: VIEWS AND VIEWSETS
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 3: VIEWS AND VIEWSETS")
print("=" * 60)

# Question 3: Views and ViewSets
print("\n3. What are DRF Views and ViewSets? How do they differ?")
print("-" * 57)
print("""
DRF VIEWS AND VIEWSETS:

VIEW HIERARCHY:
┌─────────────────────────────────────────┐
│               APIView                   │  ← Base class
├─────────────────────────────────────────┤
│          Generic Views                  │  ← ListAPIView, etc.
├─────────────────────────────────────────┤
│             ViewSets                    │  ← ModelViewSet, etc.
└─────────────────────────────────────────┘

VIEW TYPES:

1. APIView (Base class):
   • Most flexible, requires manual implementation
   • Direct control over HTTP methods
   • Custom logic for each endpoint

2. Generic Views:
   • Pre-built common patterns
   • ListAPIView, CreateAPIView, RetrieveAPIView
   • UpdateAPIView, DestroyAPIView
   • ListCreateAPIView, RetrieveUpdateDestroyAPIView

3. ViewSets:
   • Group related views together
   • ModelViewSet (full CRUD)
   • ReadOnlyModelViewSet (List + Retrieve)
   • Custom ViewSets

COMPARISON:
Feature         | APIView    | Generic Views | ViewSets
----------------|------------|---------------|----------
Flexibility     | High       | Medium        | Medium
Code Reuse      | Low        | Medium        | High
URL Routing     | Manual     | Manual        | Automatic
HTTP Methods    | Manual     | Automatic     | Automatic
CRUD Operations | Manual     | Semi-auto     | Automatic

WHEN TO USE:
• APIView: Complex custom logic, non-standard operations
• Generic Views: Standard patterns with some customization
• ViewSets: Standard CRUD operations, RESTful APIs

INTERVIEW INSIGHT:
ViewSets promote DRY principle by grouping related functionality,
while APIView provides maximum flexibility for complex scenarios.
""")

def demonstrate_views_and_viewsets():
    """Demonstrate all types of DRF views"""
    print("Views and ViewSets Complete Guide:")
    
    print("\n1. APIView (Base Class):")
    print("-" * 28)
    print("""
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import Http404

class BookAPIView(APIView):
    '''
    List all books, or create a new book
    '''
    permission_classes = [IsAuthenticated]
    
    def get(self, request, format=None):
        books = Book.objects.all()
        serializer = BookSerializer(books, many=True)
        return Response(serializer.data)
    
    def post(self, request, format=None):
        serializer = BookSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(created_by=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class BookDetailAPIView(APIView):
    '''
    Retrieve, update or delete a book instance
    '''
    
    def get_object(self, pk):
        try:
            return Book.objects.get(pk=pk)
        except Book.DoesNotExist:
            raise Http404
    
    def get(self, request, pk, format=None):
        book = self.get_object(pk)
        serializer = BookSerializer(book)
        return Response(serializer.data)
    
    def put(self, request, pk, format=None):
        book = self.get_object(pk)
        serializer = BookSerializer(book, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def patch(self, request, pk, format=None):
        book = self.get_object(pk)
        serializer = BookSerializer(book, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def delete(self, request, pk, format=None):
        book = self.get_object(pk)
        book.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
    """)
    
    print("\n2. Generic Views:")
    print("-" * 21)
    print("""
from rest_framework import generics

# Individual Generic Views
class BookListAPIView(generics.ListAPIView):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['category', 'authors']
    search_fields = ['title', 'author']
    ordering_fields = ['title', 'published_date', 'price']
    ordering = ['-published_date']

class BookCreateAPIView(generics.CreateAPIView):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
    permission_classes = [IsAuthenticated]
    
    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)

class BookRetrieveAPIView(generics.RetrieveAPIView):
    queryset = Book.objects.all()
    serializer_class = BookDetailSerializer
    lookup_field = 'pk'

class BookUpdateAPIView(generics.UpdateAPIView):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
    permission_classes = [IsAuthenticated, IsOwnerOrReadOnly]

class BookDestroyAPIView(generics.DestroyAPIView):
    queryset = Book.objects.all()
    permission_classes = [IsAuthenticated, IsOwnerOrReadOnly]

# Combined Generic Views
class BookListCreateAPIView(generics.ListCreateAPIView):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
    permission_classes = [IsAuthenticated]
    
    def get_serializer_class(self):
        if self.request.method == 'POST':
            return BookCreateSerializer
        return BookListSerializer
    
    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)

class BookRetrieveUpdateDestroyAPIView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
    permission_classes = [IsAuthenticated, IsOwnerOrReadOnly]
    
    def get_serializer_class(self):
        if self.request.method in ['PUT', 'PATCH']:
            return BookUpdateSerializer
        return BookDetailSerializer
    """)
    
    print("\n3. ViewSets:")
    print("-" * 15)
    print("""
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

# ModelViewSet (Full CRUD)
class BookViewSet(viewsets.ModelViewSet):
    '''
    A viewset that provides default `create()`, `retrieve()`, `update()`,
    `partial_update()`, `destroy()` and `list()` actions.
    '''
    queryset = Book.objects.all()
    serializer_class = BookSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['category', 'authors']
    search_fields = ['title', 'author']
    ordering_fields = ['title', 'published_date', 'price']
    
    def get_serializer_class(self):
        '''Return different serializers for different actions'''
        if self.action == 'list':
            return BookListSerializer
        elif self.action == 'create':
            return BookCreateSerializer
        elif self.action in ['update', 'partial_update']:
            return BookUpdateSerializer
        return BookDetailSerializer
    
    def get_permissions(self):
        '''Different permissions for different actions'''
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            permission_classes = [IsAuthenticated, IsOwnerOrReadOnly]
        else:
            permission_classes = [AllowAny]
        return [permission() for permission in permission_classes]
    
    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)
    
    # Custom actions
    @action(detail=True, methods=['post'])
    def set_favorite(self, request, pk=None):
        '''Mark book as favorite'''
        book = self.get_object()
        user = request.user
        
        favorite, created = Favorite.objects.get_or_create(
            user=user, book=book
        )
        
        if created:
            return Response({'status': 'favorite set'})
        else:
            return Response({'status': 'already favorite'})
    
    @action(detail=True, methods=['delete'])
    def remove_favorite(self, request, pk=None):
        '''Remove book from favorites'''
        book = self.get_object()
        user = request.user
        
        try:
            favorite = Favorite.objects.get(user=user, book=book)
            favorite.delete()
            return Response({'status': 'favorite removed'})
        except Favorite.DoesNotExist:
            return Response({'status': 'not in favorites'})
    
    @action(detail=False)
    def my_books(self, request):
        '''Get current user's books'''
        books = Book.objects.filter(created_by=request.user)
        serializer = self.get_serializer(books, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def categories(self, request):
        '''Get all categories with book counts'''
        from django.db.models import Count
        categories = Category.objects.annotate(
            book_count=Count('book')
        ).values('id', 'name', 'book_count')
        return Response(categories)
    
    @action(detail=True, methods=['post'])
    def add_review(self, request, pk=None):
        '''Add a review to a book'''
        book = self.get_object()
        serializer = ReviewSerializer(data=request.data)
        
        if serializer.is_valid():
            serializer.save(book=book, reviewer=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# ReadOnlyModelViewSet
class CategoryViewSet(viewsets.ReadOnlyModelViewSet):
    '''
    A viewset that provides default `list()` and `retrieve()` actions.
    '''
    queryset = Category.objects.all()
    serializer_class = CategorySerializer
    permission_classes = [AllowAny]

# Custom ViewSet
class BookAnalyticsViewSet(viewsets.ViewSet):
    '''
    A simple ViewSet for custom analytics endpoints
    '''
    permission_classes = [IsAuthenticated, IsAdminUser]
    
    def list(self, request):
        '''Get analytics overview'''
        total_books = Book.objects.count()
        total_authors = Author.objects.count()
        total_reviews = Review.objects.count()
        
        data = {
            'total_books': total_books,
            'total_authors': total_authors,
            'total_reviews': total_reviews,
            'books_by_category': self.get_books_by_category(),
            'top_rated_books': self.get_top_rated_books(),
        }
        return Response(data)
    
    def retrieve(self, request, pk=None):
        '''Get detailed analytics for a specific book'''
        try:
            book = Book.objects.get(pk=pk)
            analytics = {
                'book_id': book.id,
                'title': book.title,
                'total_reviews': book.reviews.count(),
                'average_rating': self.calculate_average_rating(book),
                'views_count': getattr(book, 'views_count', 0),
            }
            return Response(analytics)
        except Book.DoesNotExist:
            return Response({'error': 'Book not found'}, 
                          status=status.HTTP_404_NOT_FOUND)
    
    @action(detail=False)
    def popular_books(self, request):
        '''Get most popular books'''
        # Custom logic to determine popularity
        popular_books = Book.objects.annotate(
            review_count=Count('reviews')
        ).order_by('-review_count')[:10]
        
        serializer = BookListSerializer(popular_books, many=True)
        return Response(serializer.data)
    
    def get_books_by_category(self):
        from django.db.models import Count
        return list(Category.objects.annotate(
            count=Count('book')
        ).values('name', 'count'))
    
    def get_top_rated_books(self):
        # Implementation for top rated books
        pass
    
    def calculate_average_rating(self, book):
        reviews = book.reviews.all()
        if reviews:
            return sum(review.rating for review in reviews) / len(reviews)
        return 0
    """)

demonstrate_views_and_viewsets()

# ============================================================================
# SECTION 4: AUTHENTICATION AND PERMISSIONS
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 4: AUTHENTICATION AND PERMISSIONS")
print("=" * 60)

# Question 4: Authentication and Permissions
print("\n4. How does DRF handle authentication and permissions?")
print("-" * 58)
print("""
AUTHENTICATION IN DRF:

🔐 AUTHENTICATION vs AUTHORIZATION:
• Authentication: "Who is this user?" (Identity verification)
• Authorization: "What can this user do?" (Permission checking)

AUTHENTICATION CLASSES:
1. SessionAuthentication: Django session-based
2. TokenAuthentication: Token-based (simple)
3. BasicAuthentication: HTTP Basic auth
4. JSONWebTokenAuthentication: JWT (third-party)
5. OAuth2Authentication: OAuth2 (third-party)
6. Custom Authentication: Your own implementation

PERMISSION CLASSES:
1. AllowAny: Open access
2. IsAuthenticated: Logged-in users only
3. IsAdminUser: Admin users only
4. IsAuthenticatedOrReadOnly: Read for all, write for authenticated
5. DjangoModelPermissions: Django's built-in permissions
6. Custom Permissions: Your own rules

AUTHENTICATION FLOW:
Request → Authentication Classes → User Object → Permission Classes → View

PERMISSION LEVELS:
1. Global: settings.py DEFAULT_PERMISSION_CLASSES
2. View-level: permission_classes attribute
3. Object-level: has_object_permission method

INTERVIEW QUESTIONS:
Q: "What's the difference between authentication and permissions?"
A: "Authentication identifies the user, permissions determine what 
   they can do. Authentication runs first, then permissions."

Q: "How do you implement custom authentication?"
A: "Extend BaseAuthentication class and implement authenticate() method
   that returns (user, auth) tuple or None."
""")

def demonstrate_authentication_permissions():
    """Demonstrate authentication and permissions"""
    print("Authentication and Permissions Guide:")
    
    print("\n1. Authentication Setup:")
    print("-" * 28)
    print("""
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.TokenAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
}

# Token Authentication Setup
INSTALLED_APPS = [
    # ...
    'rest_framework.authtoken',
]

# Create tokens for existing users
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token

class Command(BaseCommand):
    def handle(self, *args, **options):
        for user in User.objects.all():
            Token.objects.get_or_create(user=user)

# Token creation on user registration
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token

@receiver(post_save, sender=User)
def create_auth_token(sender, instance=None, created=False, **kwargs):
    if created:
        Token.objects.create(user=instance)
    """)
    
    print("\n2. Authentication Views:")
    print("-" * 28)
    print("""
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.authtoken.models import Token
from rest_framework.response import Response

# Custom token authentication view
class CustomAuthToken(ObtainAuthToken):
    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(
            data=request.data,
            context={'request': request}
        )
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        token, created = Token.objects.get_or_create(user=user)
        
        return Response({
            'token': token.key,
            'user_id': user.pk,
            'username': user.username,
            'email': user.email,
            'is_staff': user.is_staff,
        })

# Login/Logout views
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth import authenticate, login, logout

@api_view(['POST'])
@permission_classes([])
def login_view(request):
    username = request.data.get('username')
    password = request.data.get('password')
    
    if username and password:
        user = authenticate(username=username, password=password)
        if user:
            login(request, user)
            token, created = Token.objects.get_or_create(user=user)
            return Response({
                'token': token.key,
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                }
            })
        else:
            return Response(
                {'error': 'Invalid credentials'}, 
                status=status.HTTP_401_UNAUTHORIZED
            )
    else:
        return Response(
            {'error': 'Username and password required'}, 
            status=status.HTTP_400_BAD_REQUEST
        )

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_view(request):
    try:
        request.user.auth_token.delete()
        logout(request)
        return Response({'message': 'Successfully logged out'})
    except:
        return Response({'error': 'Error logging out'})

# User registration
from rest_framework import serializers
from django.contrib.auth.models import User

class UserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)
    password_confirm = serializers.CharField(write_only=True)
    
    class Meta:
        model = User
        fields = ('username', 'email', 'password', 'password_confirm', 
                 'first_name', 'last_name')
    
    def validate(self, data):
        if data['password'] != data['password_confirm']:
            raise serializers.ValidationError("Passwords don't match")
        return data
    
    def create(self, validated_data):
        validated_data.pop('password_confirm')
        user = User.objects.create_user(**validated_data)
        return user

@api_view(['POST'])
@permission_classes([])
def register_view(request):
    serializer = UserRegistrationSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        token, created = Token.objects.get_or_create(user=user)
        return Response({
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
            },
            'token': token.key
        }, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    """)
    
    print("\n3. Custom Authentication:")
    print("-" * 30)
    print("""
from rest_framework.authentication import BaseAuthentication
from rest_framework import exceptions
from django.contrib.auth.models import User
import jwt
from django.conf import settings

# JWT Authentication
class JWTAuthentication(BaseAuthentication):
    def authenticate(self, request):
        auth_header = request.META.get('HTTP_AUTHORIZATION')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return None
        
        token = auth_header.split(' ')[1]
        
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=['HS256'])
            user = User.objects.get(id=payload['user_id'])
            return (user, token)
        except jwt.ExpiredSignatureError:
            raise exceptions.AuthenticationFailed('Token has expired')
        except jwt.InvalidTokenError:
            raise exceptions.AuthenticationFailed('Invalid token')
        except User.DoesNotExist:
            raise exceptions.AuthenticationFailed('User not found')

# API Key Authentication
from rest_framework.authentication import BaseAuthentication
from rest_framework import exceptions
from .models import APIKey

class APIKeyAuthentication(BaseAuthentication):
    def authenticate(self, request):
        api_key = request.META.get('HTTP_X_API_KEY')
        
        if not api_key:
            return None
        
        try:
            key_obj = APIKey.objects.select_related('user').get(
                key=api_key, 
                is_active=True
            )
            return (key_obj.user, api_key)
        except APIKey.DoesNotExist:
            raise exceptions.AuthenticationFailed('Invalid API key')

# Custom User Authentication with Multiple Fields
class MultiFieldAuthentication(BaseAuthentication):
    def authenticate(self, request):
        identifier = request.data.get('identifier')  # username or email
        password = request.data.get('password')
        
        if not identifier or not password:
            return None
        
        # Try to find user by username or email
        user = None
        if '@' in identifier:
            try:
                user = User.objects.get(email=identifier)
            except User.DoesNotExist:
                pass
        else:
            try:
                user = User.objects.get(username=identifier)
            except User.DoesNotExist:
                pass
        
        if user and user.check_password(password):
            return (user, None)
        
        raise exceptions.AuthenticationFailed('Invalid credentials')
    """)
    
    print("\n4. Permission Classes:")
    print("-" * 25)
    print("""
from rest_framework import permissions

# Built-in Permission Classes Usage
class BookViewSet(viewsets.ModelViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
    
    def get_permissions(self):
        '''Different permissions for different actions'''
        if self.action == 'list':
            permission_classes = [permissions.AllowAny]
        elif self.action == 'create':
            permission_classes = [permissions.IsAuthenticated]
        elif self.action in ['update', 'partial_update', 'destroy']:
            permission_classes = [permissions.IsAuthenticated, IsOwnerOrReadOnly]
        else:
            permission_classes = [permissions.IsAuthenticatedOrReadOnly]
        
        return [permission() for permission in permission_classes]

# Custom Permission Classes
class IsOwnerOrReadOnly(permissions.BasePermission):
    '''
    Custom permission to only allow owners to edit their objects.
    '''
    
    def has_object_permission(self, request, view, obj):
        # Read permissions for any request
        if request.method in permissions.SAFE_METHODS:
            return True
        
        # Write permissions only to the owner
        return obj.created_by == request.user

class IsAuthorOrReadOnly(permissions.BasePermission):
    '''
    Custom permission for book authors
    '''
    
    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS:
            return True
        
        # Check if user is one of the book's authors
        return request.user in obj.authors.all()

class IsAdminOrOwnerOrReadOnly(permissions.BasePermission):
    '''
    Permission for admin users or owners
    '''
    
    def has_permission(self, request, view):
        if request.method in permissions.SAFE_METHODS:
            return True
        return request.user and request.user.is_authenticated
    
    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS:
            return True
        
        # Admin users can do anything
        if request.user.is_staff:
            return True
        
        # Owner can edit their own objects
        return obj.created_by == request.user

class IsInGroupOrReadOnly(permissions.BasePermission):
    '''
    Permission based on user groups
    '''
    required_group = 'editors'
    
    def has_permission(self, request, view):
        if request.method in permissions.SAFE_METHODS:
            return True
        
        if not request.user or not request.user.is_authenticated:
            return False
        
        return request.user.groups.filter(name=self.required_group).exists()

class TimeBasedPermission(permissions.BasePermission):
    '''
    Permission that changes based on time
    '''
    
    def has_permission(self, request, view):
        from datetime import datetime, time
        
        # Only allow writes during business hours
        if request.method not in permissions.SAFE_METHODS:
            now = datetime.now().time()
            start_time = time(9, 0)  # 9 AM
            end_time = time(17, 0)   # 5 PM
            
            if not (start_time <= now <= end_time):
                return False
        
        return True

class SubscriptionPermission(permissions.BasePermission):
    '''
    Permission based on user subscription status
    '''
    
    def has_permission(self, request, view):
        if not request.user or not request.user.is_authenticated:
            return False
        
        # Check if user has active subscription
        try:
            subscription = request.user.subscription
            return subscription.is_active
        except:
            return False

# Complex Permission Logic
class ComplexBookPermission(permissions.BasePermission):
    '''
    Complex permission logic for books
    '''
    
    def has_permission(self, request, view):
        # Basic authentication check
        if not request.user or not request.user.is_authenticated:
            if request.method in permissions.SAFE_METHODS:
                return True
            return False
        
        # Additional checks for non-safe methods
        if request.method not in permissions.SAFE_METHODS:
            # Check user's book creation limit
            if view.action == 'create':
                user_book_count = Book.objects.filter(
                    created_by=request.user
                ).count()
                if user_book_count >= 10 and not request.user.is_staff:
                    return False
        
        return True
    
    def has_object_permission(self, request, view, obj):
        # Read permissions for published books
        if request.method in permissions.SAFE_METHODS:
            if obj.status == 'published':
                return True
            # Draft books only visible to author and staff
            return obj.created_by == request.user or request.user.is_staff
        
        # Write permissions
        if request.user.is_staff:
            return True
        
        if obj.created_by == request.user:
            # Authors can edit their own books
            return True
        
        # Collaborators can edit if they're in the authors list
        if hasattr(obj, 'authors') and request.user in obj.authors.all():
            return True
        
        return False
    """)

demonstrate_authentication_permissions()

print("\n" + "=" * 80)
print("PART 1 COMPLETE - Continue to Part 2 for Advanced DRF Concepts")
print("=" * 80)
