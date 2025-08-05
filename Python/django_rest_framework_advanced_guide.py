"""
Django REST Framework (DRF) - Complete Mastery Guide - PART 2
Advanced DRF concepts: Filtering, Pagination, Testing, Performance
"""

print("=" * 80)
print("DJANGO REST FRAMEWORK - PART 2: ADVANCED CONCEPTS")
print("=" * 80)

# ============================================================================
# SECTION 5: FILTERING, SEARCHING, AND ORDERING
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 5: FILTERING, SEARCHING, AND ORDERING")
print("=" * 60)

# Question 5: Filtering and Searching
print("\n5. How do you implement filtering, searching, and ordering in DRF?")
print("-" * 71)
print("""
FILTERING IN DRF:

🔍 FILTER BACKENDS:
1. DjangoFilterBackend: Field-based filtering
2. SearchFilter: Full-text search
3. OrderingFilter: Result ordering
4. Custom Filter Backends: Your own logic

FILTERING LEVELS:
1. URL Parameters: ?category=fiction&author=tolkien
2. Query Parameters: ?search=python&ordering=-created_at
3. Custom Filtering: Complex business logic

DJANGO-FILTER INTEGRATION:
• Provides advanced filtering capabilities
• Supports range filters, date filters, choice filters
• Custom filter classes and methods
• Integration with DRF filter backends

SEARCH vs FILTER:
• Filtering: Exact matches, field-based
• Searching: Partial matches, full-text search across fields
• Ordering: Sort results by specific fields

FILTER TYPES:
1. Exact: field=value
2. Icontains: field__icontains=value (case-insensitive)
3. Range: field__range=min,max
4. Date: field__date=2023-01-01
5. In: field__in=value1,value2
6. Boolean: field=true/false
7. Null: field__isnull=true/false

INTERVIEW QUESTIONS:
Q: "How do you implement complex filtering in DRF?"
A: "Use django-filter with custom FilterSet classes, override 
   get_queryset() method, or create custom filter backends."

Q: "What's the difference between search and filter?"
A: "Search is full-text across multiple fields, filter is 
   exact/partial matches on specific fields."
""")

def demonstrate_filtering_searching():
    """Demonstrate comprehensive filtering and searching"""
    print("Filtering, Searching, and Ordering Guide:")
    
    print("\n1. Basic Filter Setup:")
    print("-" * 26)
    print("""
# settings.py
INSTALLED_APPS = [
    # ...
    'django_filters',
]

REST_FRAMEWORK = {
    'DEFAULT_FILTER_BACKENDS': [
        'django_filters.rest_framework.DjangoFilterBackend',
        'rest_framework.filters.SearchFilter',
        'rest_framework.filters.OrderingFilter',
    ],
}

# Basic filtering in views
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter

class BookViewSet(viewsets.ModelViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    
    # Simple field filtering
    filterset_fields = ['category', 'authors', 'published_date']
    
    # Search across multiple fields
    search_fields = ['title', 'description', 'authors__name']
    
    # Allow ordering by these fields
    ordering_fields = ['title', 'published_date', 'price', 'created_at']
    ordering = ['-created_at']  # Default ordering

# Usage examples:
# GET /api/books/?category=fiction
# GET /api/books/?search=python programming
# GET /api/books/?ordering=-published_date
# GET /api/books/?category=fiction&search=python&ordering=title
    """)
    
    print("\n2. Advanced Filtering with django-filter:")
    print("-" * 42)
    print("""
import django_filters
from django_filters import rest_framework as filters

class BookFilter(filters.FilterSet):
    # Exact filtering
    category = filters.ModelChoiceFilter(queryset=Category.objects.all())
    
    # Case-insensitive partial matching
    title = filters.CharFilter(lookup_expr='icontains')
    author_name = filters.CharFilter(
        field_name='authors__name', 
        lookup_expr='icontains'
    )
    
    # Range filtering
    price_min = filters.NumberFilter(field_name='price', lookup_expr='gte')
    price_max = filters.NumberFilter(field_name='price', lookup_expr='lte')
    price_range = filters.RangeFilter(field_name='price')
    
    # Date filtering
    published_after = filters.DateFilter(
        field_name='published_date', 
        lookup_expr='gte'
    )
    published_before = filters.DateFilter(
        field_name='published_date', 
        lookup_expr='lte'
    )
    published_year = filters.NumberFilter(
        field_name='published_date', 
        lookup_expr='year'
    )
    
    # Boolean filtering
    is_available = filters.BooleanFilter(field_name='is_available')
    
    # Choice filtering
    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('published', 'Published'),
        ('archived', 'Archived'),
    ]
    status = filters.ChoiceFilter(choices=STATUS_CHOICES)
    
    # Multiple choice filtering
    categories = filters.ModelMultipleChoiceFilter(
        field_name='category',
        queryset=Category.objects.all()
    )
    
    # Custom method filtering
    has_reviews = filters.BooleanFilter(method='filter_has_reviews')
    min_rating = filters.NumberFilter(method='filter_min_rating')
    
    class Meta:
        model = Book
        fields = {
            'title': ['exact', 'icontains'],
            'price': ['exact', 'gte', 'lte', 'range'],
            'published_date': ['exact', 'year', 'month', 'day', 'gte', 'lte'],
            'is_available': ['exact'],
        }
    
    def filter_has_reviews(self, queryset, name, value):
        '''Filter books that have/don't have reviews'''
        if value:
            return queryset.filter(reviews__isnull=False).distinct()
        return queryset.filter(reviews__isnull=True)
    
    def filter_min_rating(self, queryset, name, value):
        '''Filter books with minimum average rating'''
        from django.db.models import Avg
        return queryset.annotate(
            avg_rating=Avg('reviews__rating')
        ).filter(avg_rating__gte=value)

# Use the filter in ViewSet
class BookViewSet(viewsets.ModelViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
    filterset_class = BookFilter
    search_fields = ['title', 'description', 'authors__name']
    ordering_fields = ['title', 'published_date', 'price']

# Advanced filtering examples:
# GET /api/books/?price_min=10&price_max=50
# GET /api/books/?published_year=2023
# GET /api/books/?has_reviews=true&min_rating=4
# GET /api/books/?categories=1,2,3
    """)
    
    print("\n3. Custom Filter Backends:")
    print("-" * 31)
    print("""
from rest_framework.filters import BaseFilterBackend

class CustomBookFilterBackend(BaseFilterBackend):
    '''
    Custom filter backend for complex business logic
    '''
    
    def filter_queryset(self, request, queryset, view):
        # User's preference-based filtering
        user = request.user
        
        if user.is_authenticated and hasattr(user, 'profile'):
            profile = user.profile
            
            # Filter by user's preferred categories
            if profile.preferred_categories.exists():
                queryset = queryset.filter(
                    category__in=profile.preferred_categories.all()
                )
            
            # Filter by user's language preference
            if profile.preferred_language:
                queryset = queryset.filter(
                    language=profile.preferred_language
                )
            
            # Filter by user's reading level
            if profile.reading_level:
                queryset = queryset.filter(
                    difficulty_level__lte=profile.reading_level
                )
        
        # Location-based filtering
        location = request.query_params.get('location')
        if location:
            queryset = queryset.filter(
                available_locations__icontains=location
            )
        
        # Availability filtering
        available_only = request.query_params.get('available_only')
        if available_only == 'true':
            queryset = queryset.filter(
                copies_available__gt=0
            )
        
        return queryset
    
    def get_schema_fields(self, view):
        # Define the schema for API documentation
        fields = []
        if hasattr(view, 'get_queryset'):
            fields.append(
                coreapi.Field(
                    name='location',
                    location='query',
                    required=False,
                    type='string',
                    description='Filter by location'
                )
            )
        return fields

class GeographicFilterBackend(BaseFilterBackend):
    '''
    Filter by geographic location
    '''
    
    def filter_queryset(self, request, queryset, view):
        lat = request.query_params.get('lat')
        lng = request.query_params.get('lng')
        radius = request.query_params.get('radius', 10)  # Default 10km
        
        if lat and lng:
            from django.contrib.gis.geos import Point
            from django.contrib.gis.measure import Distance
            
            location = Point(float(lng), float(lat), srid=4326)
            queryset = queryset.filter(
                location__distance_lte=(location, Distance(km=radius))
            )
        
        return queryset

# Use custom backends
class BookViewSet(viewsets.ModelViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
    filter_backends = [
        DjangoFilterBackend,
        SearchFilter,
        OrderingFilter,
        CustomBookFilterBackend,
        GeographicFilterBackend,
    ]
    filterset_class = BookFilter
    """)
    
    print("\n4. Advanced Search Implementation:")
    print("-" * 38)
    print("""
# Full-text search with PostgreSQL
from django.contrib.postgres.search import (
    SearchVector, SearchQuery, SearchRank
)

class PostgreSQLSearchBackend(BaseFilterBackend):
    '''
    Advanced full-text search using PostgreSQL
    '''
    
    def filter_queryset(self, request, queryset, view):
        search = request.query_params.get('search')
        if not search:
            return queryset
        
        # Create search vectors for multiple fields
        search_vector = (
            SearchVector('title', weight='A') +
            SearchVector('description', weight='B') +
            SearchVector('authors__name', weight='C')
        )
        
        search_query = SearchQuery(search)
        
        return queryset.annotate(
            search=search_vector,
            rank=SearchRank(search_vector, search_query)
        ).filter(search=search_query).order_by('-rank')

# Elasticsearch integration (requires elasticsearch-dsl-drf)
from elasticsearch_dsl import Document, Text, Date, Integer, Keyword
from elasticsearch_dsl.connections import connections

# Elasticsearch document
class BookDocument(Document):
    title = Text(analyzer='standard')
    description = Text(analyzer='standard')
    author_names = Text(analyzer='standard')
    category = Keyword()
    published_date = Date()
    price = Integer()
    
    class Index:
        name = 'books'
        settings = {
            'number_of_shards': 1,
            'number_of_replicas': 0
        }

class ElasticsearchFilterBackend(BaseFilterBackend):
    '''
    Elasticsearch-powered search backend
    '''
    
    def filter_queryset(self, request, queryset, view):
        search = request.query_params.get('search')
        if not search:
            return queryset
        
        # Elasticsearch query
        s = BookDocument.search()
        s = s.query("multi_match", query=search, fields=[
            'title^3',  # Title has higher weight
            'description^2',
            'author_names'
        ])
        
        # Get book IDs from Elasticsearch
        response = s.execute()
        book_ids = [hit.meta.id for hit in response]
        
        # Filter Django queryset by IDs
        return queryset.filter(id__in=book_ids)

# Fuzzy search implementation
class FuzzySearchBackend(BaseFilterBackend):
    '''
    Fuzzy search for handling typos
    '''
    
    def filter_queryset(self, request, queryset, view):
        search = request.query_params.get('fuzzy_search')
        if not search:
            return queryset
        
        from fuzzywuzzy import fuzz, process
        
        # Get all book titles for fuzzy matching
        books = list(queryset.values('id', 'title'))
        titles = [book['title'] for book in books]
        
        # Find fuzzy matches
        matches = process.extract(search, titles, limit=20, scorer=fuzz.ratio)
        matched_titles = [match[0] for match in matches if match[1] >= 70]
        
        return queryset.filter(title__in=matched_titles)

# Combined search view
class AdvancedBookSearchView(generics.ListAPIView):
    serializer_class = BookSerializer
    filter_backends = [
        PostgreSQLSearchBackend,
        ElasticsearchFilterBackend,
        FuzzySearchBackend,
        DjangoFilterBackend,
        OrderingFilter,
    ]
    filterset_class = BookFilter
    ordering_fields = ['relevance', 'published_date', 'price']
    
    def get_queryset(self):
        queryset = Book.objects.select_related('category').prefetch_related('authors')
        
        # Apply search type based on parameter
        search_type = self.request.query_params.get('search_type', 'standard')
        
        if search_type == 'fuzzy':
            # Use fuzzy search
            pass
        elif search_type == 'elasticsearch':
            # Use Elasticsearch
            pass
        elif search_type == 'postgresql':
            # Use PostgreSQL full-text search
            pass
        
        return queryset
    """)

demonstrate_filtering_searching()

# ============================================================================
# SECTION 6: PAGINATION
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 6: PAGINATION")
print("=" * 60)

# Question 6: Pagination
print("\n6. How do you implement pagination in DRF?")
print("-" * 46)
print("""
PAGINATION IN DRF:

📄 WHY PAGINATION?
• Improved performance (smaller datasets)
• Better user experience (faster loading)
• Reduced memory usage
• Better mobile app performance
• SEO benefits for web APIs

PAGINATION TYPES:
1. PageNumberPagination: ?page=2
2. LimitOffsetPagination: ?limit=10&offset=20
3. CursorPagination: ?cursor=xyz (for large datasets)
4. Custom Pagination: Your own implementation

PAGINATION COMPONENTS:
• page_size: Number of items per page
• page_query_param: Query parameter name
• max_page_size: Maximum allowed page size
• last_page_strings: Strings for last page

PERFORMANCE CONSIDERATIONS:
• Use CursorPagination for large datasets
• Avoid COUNT() queries when possible
• Use select_related() and prefetch_related()
• Consider database indexing

INTERVIEW QUESTIONS:
Q: "When would you use CursorPagination over PageNumberPagination?"
A: "CursorPagination for large datasets, real-time data, or when 
   data changes frequently. PageNumberPagination for static data 
   where users need to jump to specific pages."

Q: "How do you handle pagination with filtering?"
A: "DRF automatically applies pagination after filtering. Ensure 
   proper database indexing on filtered fields for performance."
""")

def demonstrate_pagination():
    """Demonstrate all pagination types and custom implementations"""
    print("Pagination Complete Guide:")
    
    print("\n1. Built-in Pagination Classes:")
    print("-" * 36)
    print("""
# settings.py - Global pagination settings
REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20
}

# 1. PageNumberPagination
from rest_framework.pagination import PageNumberPagination

class StandardPageNumberPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100
    page_query_param = 'page'
    
    def get_paginated_response(self, data):
        return Response({
            'count': self.page.paginator.count,
            'next': self.get_next_link(),
            'previous': self.get_previous_link(),
            'total_pages': self.page.paginator.num_pages,
            'current_page': self.page.number,
            'page_size': self.get_page_size(self.request),
            'results': data
        })

# 2. LimitOffsetPagination
from rest_framework.pagination import LimitOffsetPagination

class StandardLimitOffsetPagination(LimitOffsetPagination):
    default_limit = 20
    limit_query_param = 'limit'
    offset_query_param = 'offset'
    max_limit = 100
    
    def get_paginated_response(self, data):
        return Response({
            'count': self.count,
            'next': self.get_next_link(),
            'previous': self.get_previous_link(),
            'limit': self.get_limit(self.request),
            'offset': self.get_offset(self.request),
            'results': data
        })

# 3. CursorPagination
from rest_framework.pagination import CursorPagination

class StandardCursorPagination(CursorPagination):
    page_size = 20
    ordering = '-created_at'  # Required field for cursor pagination
    cursor_query_param = 'cursor'
    page_size_query_param = 'page_size'
    max_page_size = 100
    
    def get_paginated_response(self, data):
        return Response({
            'next': self.get_next_link(),
            'previous': self.get_previous_link(),
            'page_size': self.get_page_size(self.request),
            'results': data
        })

# Usage in views
class BookViewSet(viewsets.ModelViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
    pagination_class = StandardPageNumberPagination
    
    def get_pagination_class(self):
        '''Dynamic pagination based on request'''
        pagination_type = self.request.query_params.get('pagination_type')
        
        if pagination_type == 'cursor':
            return StandardCursorPagination
        elif pagination_type == 'limit_offset':
            return StandardLimitOffsetPagination
        return StandardPageNumberPagination
    """)
    
    print("\n2. Custom Pagination Classes:")
    print("-" * 33)
    print("""
from rest_framework.pagination import BasePagination
from rest_framework.response import Response
from collections import OrderedDict

class CustomPageNumberPagination(PageNumberPagination):
    '''Enhanced page number pagination with metadata'''
    
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 200
    
    def get_paginated_response(self, data):
        return Response(OrderedDict([
            ('pagination', OrderedDict([
                ('count', self.page.paginator.count),
                ('total_pages', self.page.paginator.num_pages),
                ('current_page', self.page.number),
                ('page_size', self.get_page_size(self.request)),
                ('has_next', self.page.has_next()),
                ('has_previous', self.page.has_previous()),
                ('next_page', self.page.next_page_number() if self.page.has_next() else None),
                ('previous_page', self.page.previous_page_number() if self.page.has_previous() else None),
            ])),
            ('links', OrderedDict([
                ('next', self.get_next_link()),
                ('previous', self.get_previous_link()),
            ])),
            ('results', data)
        ]))

class HeaderBasedPagination(PageNumberPagination):
    '''Pagination with information in headers'''
    
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100
    
    def get_paginated_response(self, data):
        response = Response(data)
        response['X-Total-Count'] = self.page.paginator.count
        response['X-Total-Pages'] = self.page.paginator.num_pages
        response['X-Current-Page'] = self.page.number
        response['X-Page-Size'] = self.get_page_size(self.request)
        
        if self.get_next_link():
            response['X-Next-Page'] = self.get_next_link()
        if self.get_previous_link():
            response['X-Previous-Page'] = self.get_previous_link()
        
        return response

class SeekPagination(BasePagination):
    '''Seek pagination for high-performance scenarios'''
    
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100
    ordering = '-id'  # Must be unique and ordered field
    
    def paginate_queryset(self, queryset, request, view=None):
        self.page_size = self.get_page_size(request)
        if not self.page_size:
            return None
        
        self.request = request
        
        # Get the seek parameter (last seen ID)
        seek_id = request.query_params.get('seek')
        
        if seek_id:
            # Filter to get records after the seek point
            if self.ordering.startswith('-'):
                field_name = self.ordering[1:]
                queryset = queryset.filter(**{f'{field_name}__lt': seek_id})
            else:
                queryset = queryset.filter(**{f'{self.ordering}__gt': seek_id})
        
        # Order and limit
        queryset = queryset.order_by(self.ordering)
        
        # Get one extra to check if there are more results
        results = list(queryset[:self.page_size + 1])
        
        self.has_next = len(results) > self.page_size
        if self.has_next:
            results = results[:-1]
        
        self.results = results
        return results
    
    def get_paginated_response(self, data):
        next_seek = None
        if self.has_next and self.results:
            if self.ordering.startswith('-'):
                field_name = self.ordering[1:]
            else:
                field_name = self.ordering
            next_seek = getattr(self.results[-1], field_name)
        
        return Response({
            'next_seek': next_seek,
            'has_next': self.has_next,
            'page_size': self.page_size,
            'results': data
        })
    
    def get_page_size(self, request):
        if self.page_size_query_param:
            try:
                return int(request.query_params[self.page_size_query_param])
            except (KeyError, ValueError):
                pass
        return self.page_size

class MetadataEnhancedPagination(PageNumberPagination):
    '''Pagination with rich metadata'''
    
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100
    
    def get_paginated_response(self, data):
        # Calculate additional metadata
        total_count = self.page.paginator.count
        current_page = self.page.number
        total_pages = self.page.paginator.num_pages
        page_size = self.get_page_size(self.request)
        
        start_index = (current_page - 1) * page_size + 1
        end_index = min(current_page * page_size, total_count)
        
        # Page range for pagination UI
        page_range = self.get_page_range(current_page, total_pages)
        
        return Response({
            'metadata': {
                'pagination': {
                    'count': total_count,
                    'page': current_page,
                    'pages': total_pages,
                    'page_size': page_size,
                    'start_index': start_index,
                    'end_index': end_index,
                    'has_next': self.page.has_next(),
                    'has_previous': self.page.has_previous(),
                    'page_range': page_range,
                }
            },
            'links': {
                'next': self.get_next_link(),
                'previous': self.get_previous_link(),
                'first': self.get_first_link(),
                'last': self.get_last_link(),
            },
            'data': data
        })
    
    def get_page_range(self, current_page, total_pages, window=5):
        '''Get page range for pagination UI'''
        start = max(1, current_page - window // 2)
        end = min(total_pages + 1, start + window)
        
        if end - start < window:
            start = max(1, end - window)
        
        return list(range(start, end))
    
    def get_first_link(self):
        if not self.page.has_previous():
            return None
        
        url = self.request.build_absolute_uri()
        return self.replace_query_param(url, self.page_query_param, 1)
    
    def get_last_link(self):
        if not self.page.has_next():
            return None
        
        url = self.request.build_absolute_uri()
        return self.replace_query_param(
            url, 
            self.page_query_param, 
            self.page.paginator.num_pages
        )
    """)
    
    print("\n3. Performance-Optimized Pagination:")
    print("-" * 40)
    print("""
class OptimizedCursorPagination(CursorPagination):
    '''High-performance cursor pagination'''
    
    page_size = 50
    ordering = '-created_at'
    
    def paginate_queryset(self, queryset, request, view=None):
        # Optimize queryset with select_related and prefetch_related
        if hasattr(view, 'get_optimized_queryset'):
            queryset = view.get_optimized_queryset(queryset)
        
        return super().paginate_queryset(queryset, request, view)

class CountlessPageNumberPagination(PageNumberPagination):
    '''Pagination without expensive COUNT() queries'''
    
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100
    
    def get_count(self, queryset):
        '''Skip count for performance'''
        return None
    
    def get_paginated_response(self, data):
        return Response({
            'next': self.get_next_link(),
            'previous': self.get_previous_link(),
            'page_size': self.get_page_size(self.request),
            'results': data
        })
    
    def get_next_link(self):
        if len(self.page.object_list) < self.get_page_size(self.request):
            return None
        return super().get_next_link()

# Optimized ViewSet with pagination
class OptimizedBookViewSet(viewsets.ModelViewSet):
    serializer_class = BookSerializer
    pagination_class = OptimizedCursorPagination
    
    def get_queryset(self):
        return Book.objects.select_related(
            'category', 'created_by'
        ).prefetch_related(
            'authors', 'tags'
        ).order_by('-created_at')
    
    def get_optimized_queryset(self, queryset):
        '''Additional optimizations for pagination'''
        # Only select required fields for list view
        if self.action == 'list':
            return queryset.only(
                'id', 'title', 'price', 'published_date',
                'category__name', 'created_at'
            )
        return queryset

# Infinite scroll pagination for mobile apps
class InfiniteScrollPagination(CursorPagination):
    '''Optimized for infinite scroll UIs'''
    
    page_size = 20
    ordering = '-created_at'
    
    def get_paginated_response(self, data):
        return Response({
            'has_more': self.has_next,
            'next_cursor': self.get_next_link(),
            'results': data
        })

# Database-specific optimizations
class PostgreSQLOptimizedPagination(PageNumberPagination):
    '''PostgreSQL-specific optimizations'''
    
    page_size = 20
    
    def paginate_queryset(self, queryset, request, view=None):
        # Use PostgreSQL's LIMIT/OFFSET optimization
        page_size = self.get_page_size(request)
        if not page_size:
            return None
        
        paginator = self.django_paginator_class(queryset, page_size)
        page_number = request.query_params.get(self.page_query_param, 1)
        
        try:
            self.page = paginator.page(page_number)
        except InvalidPage as exc:
            msg = self.invalid_page_message.format(
                page_number=page_number, message=str(exc)
            )
            raise NotFound(msg)
        
        # Use PostgreSQL window functions for better performance
        if hasattr(queryset.model, 'get_windowed_queryset'):
            offset = (self.page.number - 1) * page_size
            return queryset.model.get_windowed_queryset(
                queryset, offset, page_size
            )
        
        return list(self.page)
    """)

demonstrate_pagination()

# ============================================================================
# SECTION 7: THROTTLING AND RATE LIMITING
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 7: THROTTLING AND RATE LIMITING")
print("=" * 60)

# Question 7: Throttling
print("\n7. How do you implement throttling and rate limiting in DRF?")
print("-" * 64)
print("""
THROTTLING IN DRF:

🚦 PURPOSE OF THROTTLING:
• Prevent API abuse and DoS attacks
• Ensure fair usage among users
• Protect server resources
• Implement business rate limits
• Control costs for external services

THROTTLE TYPES:
1. AnonRateThrottle: Anonymous users
2. UserRateThrottle: Authenticated users  
3. ScopedRateThrottle: Specific endpoints
4. Custom Throttles: Business-specific logic

THROTTLE RATES:
• Format: 'number/period'
• Periods: sec, min, hour, day
• Examples: '100/hour', '1000/day', '5/min'

THROTTLE SCOPE:
• Global: Applied to all views
• View-level: Applied to specific views
• Action-level: Applied to specific actions

STORAGE BACKENDS:
• Memory: Fast but not persistent
• Cache: Redis/Memcached (recommended)
• Database: Persistent but slower

INTERVIEW QUESTIONS:
Q: "How do you implement different rate limits for different user types?"
A: "Use custom throttle classes that check user attributes like 
   subscription type, user role, or premium status."

Q: "What's the difference between throttling and caching?"
A: "Throttling limits request frequency, caching stores responses 
   to avoid processing. Both improve performance differently."
""")

def demonstrate_throttling():
    """Demonstrate comprehensive throttling implementations"""
    print("Throttling and Rate Limiting Guide:")
    
    print("\n1. Basic Throttling Setup:")
    print("-" * 30)
    print("""
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle'
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '100/hour',
        'user': '1000/hour',
        'premium': '5000/hour',
        'api': '2000/hour',
        'upload': '10/hour',
        'search': '60/min',
    }
}

# Cache configuration for throttling
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

# Basic throttling in views
from rest_framework.throttling import UserRateThrottle, AnonRateThrottle

class BookViewSet(viewsets.ModelViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
    throttle_classes = [UserRateThrottle, AnonRateThrottle]
    throttle_scope = 'api'
    
    def get_throttles(self):
        '''Dynamic throttling based on action'''
        if self.action == 'create':
            throttle_classes = [UploadRateThrottle]
        elif self.action == 'list':
            throttle_classes = [SearchRateThrottle]
        else:
            throttle_classes = self.throttle_classes
        
        return [throttle() for throttle in throttle_classes]
    """)
    
    print("\n2. Custom Throttle Classes:")
    print("-" * 31)
    print("""
from rest_framework.throttling import BaseThrottle, UserRateThrottle
import time

class PremiumUserThrottle(UserRateThrottle):
    '''Higher rate limits for premium users'''
    
    def get_cache_key(self, request, view):
        if request.user.is_authenticated:
            # Check if user has premium subscription
            if hasattr(request.user, 'subscription') and request.user.subscription.is_premium:
                # Use premium rate limit
                return f'premium_{request.user.pk}'
            else:
                # Use standard user rate limit
                return f'user_{request.user.pk}'
        return None
    
    def get_rate(self):
        '''Get rate based on user type'''
        # This would be set based on cache key prefix
        return '5000/hour'  # Premium rate

class IPBasedThrottle(BaseThrottle):
    '''Throttle based on IP address'''
    
    def __init__(self):
        self.rate = '100/hour'
        self.num_requests, self.duration = self.parse_rate(self.rate)
        self.cache = caches['default']
    
    def get_cache_key(self, request, view):
        return f'ip_throttle_{self.get_client_ip(request)}'
    
    def get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip
    
    def allow_request(self, request, view):
        key = self.get_cache_key(request, view)
        if key is None:
            return True
        
        history = self.cache.get(key, [])
        now = time.time()
        
        # Remove old entries
        while history and history[-1] <= now - self.duration:
            history.pop()
        
        if len(history) >= self.num_requests:
            return False
        
        # Add current request
        history.insert(0, now)
        self.cache.set(key, history, self.duration)
        return True
    
    def wait(self):
        '''Time to wait before next request'''
        key = self.get_cache_key(self.request, self.view)
        history = self.cache.get(key, [])
        if history:
            remaining_duration = self.duration - (time.time() - history[-1])
            return max(0, remaining_duration)
        return 0

class UserTypeThrottle(UserRateThrottle):
    '''Different rates based on user type'''
    
    RATE_MAP = {
        'basic': '100/hour',
        'premium': '1000/hour',
        'enterprise': '10000/hour',
        'admin': '50000/hour',
    }
    
    def get_rate(self):
        if not self.request.user.is_authenticated:
            return '50/hour'  # Anonymous users
        
        user_type = getattr(self.request.user, 'user_type', 'basic')
        return self.RATE_MAP.get(user_type, '100/hour')
    
    def get_cache_key(self, request, view):
        if request.user.is_authenticated:
            user_type = getattr(request.user, 'user_type', 'basic')
            return f'{user_type}_{request.user.pk}'
        return f'anon_{self.get_client_ip(request)}'

class TimeBasedThrottle(BaseThrottle):
    '''Different rates based on time of day'''
    
    def __init__(self):
        self.peak_rate = '50/hour'      # 9 AM - 5 PM
        self.off_peak_rate = '200/hour'  # Other times
    
    def allow_request(self, request, view):
        from datetime import datetime
        
        current_hour = datetime.now().hour
        
        # Peak hours: 9 AM to 5 PM
        if 9 <= current_hour <= 17:
            rate = self.peak_rate
        else:
            rate = self.off_peak_rate
        
        # Create a temporary throttle instance with appropriate rate
        throttle = UserRateThrottle()
        throttle.rate = rate
        throttle.num_requests, throttle.duration = throttle.parse_rate(rate)
        
        return throttle.allow_request(request, view)

class APIKeyThrottle(BaseThrottle):
    '''Throttle based on API key limits'''
    
    def get_cache_key(self, request, view):
        api_key = request.META.get('HTTP_X_API_KEY')
        if api_key:
            return f'apikey_{api_key}'
        return None
    
    def allow_request(self, request, view):
        api_key = request.META.get('HTTP_X_API_KEY')
        if not api_key:
            return False
        
        try:
            # Get API key object with rate limits
            from .models import APIKey
            key_obj = APIKey.objects.get(key=api_key, is_active=True)
            
            # Check if rate limit exceeded
            cache_key = self.get_cache_key(request, view)
            current_count = cache.get(cache_key, 0)
            
            if current_count >= key_obj.rate_limit_per_hour:
                return False
            
            # Increment counter
            cache.set(cache_key, current_count + 1, 3600)  # 1 hour
            return True
            
        except APIKey.DoesNotExist:
            return False

class BurstThrottle(BaseThrottle):
    '''Allow bursts with recovery time'''
    
    def __init__(self):
        self.burst_rate = '10/min'      # 10 requests per minute
        self.sustained_rate = '100/hour' # 100 requests per hour
        self.bucket_size = 10
        self.refill_rate = 100 / 3600    # tokens per second
    
    def allow_request(self, request, view):
        key = self.get_cache_key(request, view)
        if not key:
            return True
        
        now = time.time()
        bucket_data = cache.get(key, {'tokens': self.bucket_size, 'last_update': now})
        
        # Calculate tokens to add
        time_passed = now - bucket_data['last_update']
        tokens_to_add = time_passed * self.refill_rate
        
        # Update bucket
        bucket_data['tokens'] = min(
            self.bucket_size,
            bucket_data['tokens'] + tokens_to_add
        )
        bucket_data['last_update'] = now
        
        # Check if request can be served
        if bucket_data['tokens'] >= 1:
            bucket_data['tokens'] -= 1
            cache.set(key, bucket_data, 3600)
            return True
        
        cache.set(key, bucket_data, 3600)
        return False
    
    def get_cache_key(self, request, view):
        if request.user.is_authenticated:
            return f'burst_{request.user.pk}'
        return f'burst_anon_{self.get_client_ip(request)}'
    """)
    
    print("\n3. Advanced Throttling Patterns:")
    print("-" * 37)
    print("""
# Sliding window throttle
class SlidingWindowThrottle(BaseThrottle):
    '''More accurate sliding window rate limiting'''
    
    def __init__(self):
        self.rate = '100/hour'
        self.window_size = 3600  # 1 hour in seconds
        self.num_requests = 100
    
    def allow_request(self, request, view):
        key = self.get_cache_key(request, view)
        if not key:
            return True
        
        now = time.time()
        
        # Get current window data
        window_key = f'{key}_{int(now // self.window_size)}'
        current_count = cache.get(window_key, 0)
        
        # Get previous window data
        prev_window_key = f'{key}_{int((now - self.window_size) // self.window_size)}'
        prev_count = cache.get(prev_window_key, 0)
        
        # Calculate weighted count based on overlap
        elapsed_in_current = now % self.window_size
        weight = elapsed_in_current / self.window_size
        weighted_count = current_count + (prev_count * (1 - weight))
        
        if weighted_count >= self.num_requests:
            return False
        
        # Increment current window
        cache.set(window_key, current_count + 1, self.window_size * 2)
        return True

# Hierarchical throttling
class HierarchicalThrottle(BaseThrottle):
    '''Multiple throttle levels (per-second, per-minute, per-hour)'''
    
    def __init__(self):
        self.throttles = [
            ('second', 5, 1),      # 5 per second
            ('minute', 60, 60),    # 60 per minute  
            ('hour', 1000, 3600),  # 1000 per hour
        ]
    
    def allow_request(self, request, view):
        key_base = self.get_cache_key(request, view)
        if not key_base:
            return True
        
        now = time.time()
        
        for level, limit, window in self.throttles:
            key = f'{key_base}_{level}_{int(now // window)}'
            current_count = cache.get(key, 0)
            
            if current_count >= limit:
                return False
        
        # All levels passed, increment all counters
        for level, limit, window in self.throttles:
            key = f'{key_base}_{level}_{int(now // window)}'
            current_count = cache.get(key, 0)
            cache.set(key, current_count + 1, window * 2)
        
        return True

# Adaptive throttling
class AdaptiveThrottle(BaseThrottle):
    '''Adjust rates based on system load'''
    
    def __init__(self):
        self.base_rate = 100
        self.min_rate = 10
        self.max_rate = 500
    
    def get_current_rate(self):
        '''Calculate current rate based on system metrics'''
        import psutil
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # Calculate load factor (0.1 to 2.0)
        load_factor = 1.0
        
        if cpu_percent > 80:
            load_factor *= 0.5  # Reduce rate by 50%
        elif cpu_percent < 30:
            load_factor *= 1.5  # Increase rate by 50%
        
        if memory_percent > 85:
            load_factor *= 0.3  # Severe reduction
        
        # Apply load factor
        adjusted_rate = int(self.base_rate * load_factor)
        return max(self.min_rate, min(self.max_rate, adjusted_rate))
    
    def allow_request(self, request, view):
        current_rate = self.get_current_rate()
        
        # Use UserRateThrottle with dynamic rate
        throttle = UserRateThrottle()
        throttle.rate = f'{current_rate}/hour'
        throttle.num_requests, throttle.duration = throttle.parse_rate(throttle.rate)
        
        return throttle.allow_request(request, view)

# Geographic throttling
class GeographicThrottle(BaseThrottle):
    '''Different rates for different regions'''
    
    REGION_RATES = {
        'US': '1000/hour',
        'EU': '800/hour', 
        'ASIA': '500/hour',
        'OTHER': '200/hour',
    }
    
    def get_user_region(self, request):
        '''Determine user region from IP or user profile'''
        # This could use GeoIP or user profile data
        ip = self.get_client_ip(request)
        
        # Mock implementation
        if ip.startswith('192.168'):
            return 'US'
        return 'OTHER'
    
    def allow_request(self, request, view):
        region = self.get_user_region(request)
        rate = self.REGION_RATES.get(region, '100/hour')
        
        throttle = UserRateThrottle()
        throttle.rate = rate
        throttle.num_requests, throttle.duration = throttle.parse_rate(rate)
        
        return throttle.allow_request(request, view)

# Usage in views
class AdvancedBookViewSet(viewsets.ModelViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
    
    def get_throttles(self):
        '''Complex throttling logic'''
        throttles = []
        
        # Base throttling
        if self.request.user.is_authenticated:
            throttles.append(PremiumUserThrottle())
        else:
            throttles.append(AnonRateThrottle())
        
        # Action-specific throttling
        if self.action == 'create':
            throttles.append(BurstThrottle())
        elif self.action in ['list', 'retrieve']:
            throttles.append(HierarchicalThrottle())
        
        # Geographic throttling
        throttles.append(GeographicThrottle())
        
        # Adaptive throttling during peak hours
        from datetime import datetime
        if 9 <= datetime.now().hour <= 17:
            throttles.append(AdaptiveThrottle())
        
        return throttles
    """)

demonstrate_throttling()

print("\n" + "=" * 80)
print("PART 2 COMPLETE - Continue to Part 3 for Testing, Performance & Best Practices")
print("=" * 80)
