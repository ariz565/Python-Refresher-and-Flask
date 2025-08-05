"""
Complete Python Web Development & Data Science Interview Questions
Covering web frameworks, APIs, databases, data science libraries, and deployment
Based on comprehensive Python interview preparation material
"""

print("=" * 80)
print("COMPLETE PYTHON WEB DEVELOPMENT & DATA SCIENCE QUESTIONS")
print("=" * 80)

# ============================================================================
# SECTION 1: WEB DEVELOPMENT FRAMEWORKS
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 1: WEB DEVELOPMENT FRAMEWORKS")
print("=" * 50)

# Question 1: Flask Framework Fundamentals
print("\n1. How do you create a web application using Flask?")
print("-" * 49)
print("""
Flask is a lightweight web framework for Python:

BASIC CONCEPTS:
• WSGI application framework
• Minimal and flexible
• Uses decorators for routing
• Built-in development server
• Template engine (Jinja2)

CORE COMPONENTS:
• Application instance
• Routes and view functions
• Request and response objects
• Templates and static files
• Configuration management

BASIC STRUCTURE:
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/user/<name>')
def user(name):
    return f'Hello, {name}!'

if __name__ == '__main__':
    app.run(debug=True)
""")

def demonstrate_flask_concepts():
    """Demonstrate Flask concepts (conceptual - no actual server)"""
    print("Flask Framework Demo (Conceptual):")
    print("""
    ROUTING EXAMPLES:
    
    @app.route('/')                          # GET request to root
    @app.route('/users', methods=['POST'])   # POST request
    @app.route('/user/<int:id>')            # URL parameter
    @app.route('/search')                    # Query parameters
    
    REQUEST HANDLING:
    
    @app.route('/api/data', methods=['POST'])
    def handle_data():
        data = request.get_json()
        name = request.form.get('name')
        file = request.files.get('upload')
        return jsonify({'status': 'success'})
    
    TEMPLATES:
    
    @app.route('/profile/<username>')
    def profile(username):
        user_data = {'name': username, 'email': f'{username}@example.com'}
        return render_template('profile.html', user=user_data)
    
    ERROR HANDLING:
    
    @app.errorhandler(404)
    def not_found(error):
        return render_template('404.html'), 404
    
    CONFIGURATION:
    
    app.config['SECRET_KEY'] = 'your-secret-key'
    app.config['DATABASE_URI'] = 'sqlite:///app.db'
    """)

demonstrate_flask_concepts()

# Question 2: Django Framework Fundamentals
print("\n2. How does Django framework work?")
print("-" * 35)
print("""
Django is a high-level web framework following MVT pattern:

ARCHITECTURE:
• Model-View-Template (MVT) pattern
• Object-Relational Mapping (ORM)
• URL dispatcher
• Template engine
• Admin interface

KEY FEATURES:
• Built-in authentication
• Automatic admin interface
• Database migrations
• Security features
• Internationalization

PROJECT STRUCTURE:
myproject/
    manage.py
    myproject/
        __init__.py
        settings.py
        urls.py
        wsgi.py
    myapp/
        __init__.py
        models.py
        views.py
        urls.py
        templates/
        static/

MODELS (models.py):
from django.db import models

class User(models.Model):
    username = models.CharField(max_length=100)
    email = models.EmailField()
    created_at = models.DateTimeField(auto_now_add=True)

VIEWS (views.py):
from django.shortcuts import render
from django.http import JsonResponse

def user_list(request):
    users = User.objects.all()
    return render(request, 'users.html', {'users': users})

URLS (urls.py):
from django.urls import path
from . import views

urlpatterns = [
    path('users/', views.user_list, name='user_list'),
    path('api/users/', views.user_api, name='user_api'),
]
""")

def demonstrate_django_concepts():
    """Demonstrate Django concepts (conceptual)"""
    print("Django Framework Demo (Conceptual):")
    print("""
    ORM EXAMPLES:
    
    # Create
    user = User.objects.create(username='john', email='john@example.com')
    
    # Read
    users = User.objects.all()
    user = User.objects.get(id=1)
    active_users = User.objects.filter(is_active=True)
    
    # Update
    user.email = 'newemail@example.com'
    user.save()
    
    # Delete
    user.delete()
    
    CLASS-BASED VIEWS:
    
    from django.views.generic import ListView, CreateView
    
    class UserListView(ListView):
        model = User
        template_name = 'users.html'
        context_object_name = 'users'
    
    FORMS:
    
    from django import forms
    
    class UserForm(forms.ModelForm):
        class Meta:
            model = User
            fields = ['username', 'email']
    
    MIDDLEWARE:
    
    class CustomMiddleware:
        def __init__(self, get_response):
            self.get_response = get_response
        
        def __call__(self, request):
            # Process request
            response = self.get_response(request)
            # Process response
            return response
    """)

demonstrate_django_concepts()

# Question 3: RESTful APIs with Python
print("\n3. How do you create RESTful APIs in Python?")
print("-" * 45)
print("""
RESTful APIs follow REST architectural principles:

REST PRINCIPLES:
• Stateless communication
• Uniform interface
• Client-server architecture
• Cacheable responses
• Layered system

HTTP METHODS:
• GET: Retrieve data
• POST: Create new resource
• PUT: Update entire resource
• PATCH: Partial update
• DELETE: Remove resource

STATUS CODES:
• 200: OK
• 201: Created
• 400: Bad Request
• 401: Unauthorized
• 404: Not Found
• 500: Internal Server Error

API DESIGN:
• Use nouns for endpoints
• Consistent naming conventions
• Version your API
• Proper error handling
• Documentation
""")

def demonstrate_rest_api():
    """Demonstrate REST API concepts"""
    print("RESTful API Demo (Conceptual):")
    print("""
    FLASK REST API EXAMPLE:
    
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    # Sample data
    users = [
        {'id': 1, 'name': 'John', 'email': 'john@example.com'},
        {'id': 2, 'name': 'Jane', 'email': 'jane@example.com'}
    ]
    
    # GET /api/users - Get all users
    @app.route('/api/users', methods=['GET'])
    def get_users():
        return jsonify(users)
    
    # GET /api/users/<id> - Get specific user
    @app.route('/api/users/<int:user_id>', methods=['GET'])
    def get_user(user_id):
        user = next((u for u in users if u['id'] == user_id), None)
        if user:
            return jsonify(user)
        return jsonify({'error': 'User not found'}), 404
    
    # POST /api/users - Create new user
    @app.route('/api/users', methods=['POST'])
    def create_user():
        data = request.get_json()
        new_user = {
            'id': len(users) + 1,
            'name': data['name'],
            'email': data['email']
        }
        users.append(new_user)
        return jsonify(new_user), 201
    
    # PUT /api/users/<id> - Update user
    @app.route('/api/users/<int:user_id>', methods=['PUT'])
    def update_user(user_id):
        user = next((u for u in users if u['id'] == user_id), None)
        if user:
            data = request.get_json()
            user.update(data)
            return jsonify(user)
        return jsonify({'error': 'User not found'}), 404
    
    # DELETE /api/users/<id> - Delete user
    @app.route('/api/users/<int:user_id>', methods=['DELETE'])
    def delete_user(user_id):
        global users
        users = [u for u in users if u['id'] != user_id]
        return '', 204
    
    DJANGO REST FRAMEWORK:
    
    from rest_framework import serializers, viewsets
    from rest_framework.response import Response
    
    class UserSerializer(serializers.ModelSerializer):
        class Meta:
            model = User
            fields = ['id', 'username', 'email']
    
    class UserViewSet(viewsets.ModelViewSet):
        queryset = User.objects.all()
        serializer_class = UserSerializer
    """)

demonstrate_rest_api()

# ============================================================================
# SECTION 2: DATABASE INTEGRATION
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 2: DATABASE INTEGRATION")
print("=" * 50)

# Question 4: Database Connectivity
print("\n4. How do you connect to databases in Python?")
print("-" * 46)
print("""
Python provides various ways to connect to databases:

SQLITE (Built-in):
• sqlite3 module
• Lightweight, file-based database
• No server required
• Good for development and small applications

POSTGRESQL:
• psycopg2 library
• Most popular PostgreSQL adapter
• Supports advanced PostgreSQL features

MYSQL:
• mysql-connector-python
• PyMySQL (pure Python)
• MySQLdb (C extension)

SQLALCHEMY ORM:
• Database-agnostic ORM
• High-level abstraction
• Migration support
• Relationship mapping
""")

import sqlite3
import tempfile
import os

def demonstrate_database_connectivity():
    """Demonstrate database connectivity"""
    print("Database Connectivity Demo:")
    
    # Create temporary database
    db_path = os.path.join(tempfile.gettempdir(), "demo.db")
    
    print("\n1. SQLite connection and operations:")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            age INTEGER
        )
    ''')
    
    # Insert data
    users_data = [
        ('Alice', 'alice@example.com', 30),
        ('Bob', 'bob@example.com', 25),
        ('Charlie', 'charlie@example.com', 35)
    ]
    
    cursor.executemany('INSERT OR IGNORE INTO users (name, email, age) VALUES (?, ?, ?)', users_data)
    print(f"Inserted {cursor.rowcount} users")
    
    # Query data
    cursor.execute('SELECT * FROM users ORDER BY age')
    users = cursor.fetchall()
    
    print("\nAll users:")
    for user in users:
        print(f"ID: {user[0]}, Name: {user[1]}, Email: {user[2]}, Age: {user[3]}")
    
    # Update data
    cursor.execute('UPDATE users SET age = ? WHERE name = ?', (31, 'Alice'))
    print(f"\nUpdated {cursor.rowcount} record(s)")
    
    # Query with condition
    cursor.execute('SELECT name, age FROM users WHERE age > ?', (30,))
    older_users = cursor.fetchall()
    
    print("\nUsers older than 30:")
    for user in older_users:
        print(f"Name: {user[0]}, Age: {user[1]}")
    
    # Commit and close
    conn.commit()
    cursor.close()
    conn.close()
    
    print("\n2. SQLAlchemy ORM example (conceptual):")
    print("""
    from sqlalchemy import create_engine, Column, Integer, String
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    
    Base = declarative_base()
    
    class User(Base):
        __tablename__ = 'users'
        
        id = Column(Integer, primary_key=True)
        name = Column(String(50), nullable=False)
        email = Column(String(100), unique=True, nullable=False)
        age = Column(Integer)
    
    # Create engine and session
    engine = create_engine('sqlite:///example.db')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Create user
    new_user = User(name='Alice', email='alice@example.com', age=30)
    session.add(new_user)
    session.commit()
    
    # Query users
    users = session.query(User).filter(User.age > 25).all()
    
    # Update user
    user = session.query(User).filter(User.name == 'Alice').first()
    user.age = 31
    session.commit()
    
    # Delete user
    session.delete(user)
    session.commit()
    """)
    
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)

demonstrate_database_connectivity()

# Question 5: Database Migrations and Schema Management
print("\n5. How do you handle database migrations in Python?")
print("-" * 54)
print("""
Database migrations manage schema changes over time:

MIGRATION CONCEPTS:
• Version control for database schema
• Incremental changes
• Rollback capabilities
• Team collaboration
• Environment consistency

ALEMBIC (SQLAlchemy):
• Database migration tool
• Auto-generates migration scripts
• Supports various databases
• Integrates with SQLAlchemy

DJANGO MIGRATIONS:
• Built-in migration system
• Automatic migration generation
• Dependency tracking
• Data migrations support

MIGRATION WORKFLOW:
1. Make model changes
2. Generate migration
3. Review migration script
4. Apply migration
5. Test changes
""")

def demonstrate_migrations():
    """Demonstrate migration concepts"""
    print("Database Migrations Demo (Conceptual):")
    print("""
    ALEMBIC WORKFLOW:
    
    # Initialize Alembic in project
    $ alembic init alembic
    
    # Generate migration
    $ alembic revision --autogenerate -m "Add user table"
    
    # Apply migration
    $ alembic upgrade head
    
    # Rollback migration
    $ alembic downgrade -1
    
    MIGRATION SCRIPT EXAMPLE:
    
    from alembic import op
    import sqlalchemy as sa
    
    def upgrade():
        op.create_table('users',
            sa.Column('id', sa.Integer(), primary_key=True),
            sa.Column('name', sa.String(50), nullable=False),
            sa.Column('email', sa.String(100), unique=True),
            sa.Column('created_at', sa.DateTime(), default=sa.func.now())
        )
    
    def downgrade():
        op.drop_table('users')
    
    DJANGO MIGRATIONS:
    
    # Generate migrations
    $ python manage.py makemigrations
    
    # Apply migrations
    $ python manage.py migrate
    
    # Show migration status
    $ python manage.py showmigrations
    
    # Custom migration
    $ python manage.py makemigrations --empty myapp
    
    DATA MIGRATION EXAMPLE:
    
    from django.db import migrations
    
    def populate_users(apps, schema_editor):
        User = apps.get_model('myapp', 'User')
        User.objects.create(name='Admin', email='admin@example.com')
    
    class Migration(migrations.Migration):
        dependencies = [
            ('myapp', '0001_initial'),
        ]
    
        operations = [
            migrations.RunPython(populate_users),
        ]
    """)

demonstrate_migrations()

# ============================================================================
# SECTION 3: DATA SCIENCE LIBRARIES
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 3: DATA SCIENCE LIBRARIES")
print("=" * 50)

# Question 6: NumPy for Numerical Computing
print("\n6. How do you use NumPy for numerical computing?")
print("-" * 48)
print("""
NumPy provides efficient numerical computing in Python:

KEY FEATURES:
• N-dimensional arrays (ndarray)
• Element-wise operations
• Broadcasting
• Linear algebra functions
• Random number generation
• Integration with C/C++/Fortran

ARRAY OPERATIONS:
• Creation: zeros, ones, arange, linspace
• Indexing and slicing
• Reshaping and resizing
• Mathematical operations
• Statistical functions

PERFORMANCE:
• Vectorized operations
• Memory efficient
• Compiled C code
• Broadcasting rules
""")

def demonstrate_numpy_concepts():
    """Demonstrate NumPy concepts (conceptual)"""
    print("NumPy Demo (Conceptual - requires numpy installation):")
    print("""
    ARRAY CREATION:
    
    import numpy as np
    
    # From lists
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([[1, 2], [3, 4]])
    
    # Built-in functions
    zeros = np.zeros((3, 4))
    ones = np.ones((2, 3))
    range_arr = np.arange(0, 10, 2)
    linspace = np.linspace(0, 1, 5)
    
    ARRAY OPERATIONS:
    
    # Element-wise operations
    result = arr1 * 2
    result = arr1 + arr2  # Broadcasting
    
    # Mathematical functions
    sqrt_arr = np.sqrt(arr1)
    sin_arr = np.sin(arr1)
    
    # Statistical operations
    mean = np.mean(arr1)
    std = np.std(arr1)
    max_val = np.max(arr1)
    
    INDEXING AND SLICING:
    
    # Basic indexing
    element = arr1[0]
    slice_arr = arr1[1:4]
    
    # Boolean indexing
    mask = arr1 > 3
    filtered = arr1[mask]
    
    # Fancy indexing
    indices = [0, 2, 4]
    selected = arr1[indices]
    
    LINEAR ALGEBRA:
    
    # Matrix operations
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    # Matrix multiplication
    C = np.dot(A, B)
    C = A @ B  # Python 3.5+
    
    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Inverse matrix
    A_inv = np.linalg.inv(A)
    """)

demonstrate_numpy_concepts()

# Question 7: Pandas for Data Manipulation
print("\n7. How do you use Pandas for data manipulation?")
print("-" * 47)
print("""
Pandas provides powerful data structures and analysis tools:

KEY DATA STRUCTURES:
• Series: 1-dimensional labeled array
• DataFrame: 2-dimensional labeled data structure
• Index: Immutable array-like object

CORE FUNCTIONALITY:
• Data loading and saving
• Data cleaning and transformation
• Merging and joining
• Grouping and aggregation
• Time series analysis

DATA I/O:
• CSV, Excel, JSON, SQL
• Web scraping
• APIs and web services
• Big data formats (Parquet, HDF5)
""")

def demonstrate_pandas_concepts():
    """Demonstrate Pandas concepts (conceptual)"""
    print("Pandas Demo (Conceptual - requires pandas installation):")
    print("""
    DATA STRUCTURES:
    
    import pandas as pd
    
    # Series
    s = pd.Series([1, 3, 5, np.nan, 6, 8])
    
    # DataFrame
    df = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': ['a', 'b', 'c', 'd'],
        'C': [1.1, 2.2, 3.3, 4.4]
    })
    
    DATA LOADING:
    
    # CSV files
    df = pd.read_csv('data.csv')
    df.to_csv('output.csv', index=False)
    
    # Excel files
    df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
    
    # JSON data
    df = pd.read_json('data.json')
    
    # SQL databases
    df = pd.read_sql('SELECT * FROM users', connection)
    
    DATA EXPLORATION:
    
    # Basic info
    df.head()          # First 5 rows
    df.tail()          # Last 5 rows
    df.info()          # Data types and memory usage
    df.describe()      # Statistical summary
    df.shape           # Dimensions
    
    # Column operations
    df.columns         # Column names
    df['A']           # Select column
    df[['A', 'B']]    # Select multiple columns
    
    DATA FILTERING:
    
    # Boolean indexing
    filtered = df[df['A'] > 2]
    
    # Multiple conditions
    filtered = df[(df['A'] > 1) & (df['C'] < 3.0)]
    
    # Query method
    filtered = df.query('A > 2 and C < 3.0')
    
    DATA MANIPULATION:
    
    # Adding columns
    df['D'] = df['A'] * df['C']
    
    # Dropping columns/rows
    df = df.drop('B', axis=1)
    df = df.drop(0, axis=0)
    
    # Grouping and aggregation
    grouped = df.groupby('B').agg({
        'A': 'mean',
        'C': ['sum', 'count']
    })
    
    # Merging DataFrames
    df1 = pd.DataFrame({'key': ['A', 'B'], 'value': [1, 2]})
    df2 = pd.DataFrame({'key': ['A', 'C'], 'value': [3, 4]})
    merged = pd.merge(df1, df2, on='key', how='inner')
    """)

demonstrate_pandas_concepts()

# Question 8: Matplotlib for Data Visualization
print("\n8. How do you create visualizations with Matplotlib?")
print("-" * 52)
print("""
Matplotlib is the foundational plotting library for Python:

PLOT TYPES:
• Line plots: plot()
• Scatter plots: scatter()
• Bar plots: bar(), barh()
• Histograms: hist()
• Box plots: boxplot()
• Heatmaps: imshow()

CUSTOMIZATION:
• Colors and styles
• Labels and titles
• Legends and annotations
• Subplots and layouts
• Axis formatting

OBJECT-ORIENTED INTERFACE:
• Figure and Axes objects
• Better control and customization
• Recommended for complex plots
""")

def demonstrate_matplotlib_concepts():
    """Demonstrate Matplotlib concepts (conceptual)"""
    print("Matplotlib Demo (Conceptual - requires matplotlib installation):")
    print("""
    BASIC PLOTTING:
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Line plot
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='sin(x)')
    plt.plot(x, np.cos(x), label='cos(x)')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Trigonometric Functions')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Scatter plot
    x = np.random.randn(100)
    y = np.random.randn(100)
    colors = np.random.rand(100)
    
    plt.scatter(x, y, c=colors, alpha=0.6)
    plt.colorbar()
    plt.title('Random Scatter Plot')
    plt.show()
    
    SUBPLOTS:
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Line plot
    axes[0, 0].plot(x, y)
    axes[0, 0].set_title('Line Plot')
    
    # Plot 2: Histogram
    axes[0, 1].hist(np.random.randn(1000), bins=30)
    axes[0, 1].set_title('Histogram')
    
    # Plot 3: Bar plot
    categories = ['A', 'B', 'C', 'D']
    values = [23, 45, 56, 78]
    axes[1, 0].bar(categories, values)
    axes[1, 0].set_title('Bar Plot')
    
    # Plot 4: Box plot
    data = [np.random.randn(100) for _ in range(4)]
    axes[1, 1].boxplot(data)
    axes[1, 1].set_title('Box Plot')
    
    plt.tight_layout()
    plt.show()
    
    CUSTOMIZATION:
    
    # Custom style
    plt.style.use('seaborn')
    
    # Custom colors and markers
    plt.plot(x, y, color='red', linestyle='--', marker='o', markersize=3)
    
    # Annotations
    plt.annotate('Maximum', xy=(5, 1), xytext=(6, 0.5),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # Multiple y-axes
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ax1.plot(x, y, 'b-')
    ax2.plot(x, y**2, 'r-')
    
    ax1.set_ylabel('Linear', color='b')
    ax2.set_ylabel('Quadratic', color='r')
    """)

demonstrate_matplotlib_concepts()

# ============================================================================
# SECTION 4: DEPLOYMENT AND PRODUCTION
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 4: DEPLOYMENT AND PRODUCTION")
print("=" * 50)

# Question 9: Application Deployment
print("\n9. How do you deploy Python applications?")
print("-" * 42)
print("""
DEPLOYMENT OPTIONS:

TRADITIONAL SERVERS:
• Linux servers (Ubuntu, CentOS)
• Web servers (Nginx, Apache)
• WSGI servers (Gunicorn, uWSGI)
• Process managers (systemd, supervisor)

CLOUD PLATFORMS:
• AWS (EC2, Lambda, Elastic Beanstalk)
• Google Cloud (App Engine, Cloud Run)
• Microsoft Azure (App Service)
• DigitalOcean, Linode

CONTAINERIZATION:
• Docker containers
• Kubernetes orchestration
• Docker Compose for development
• Container registries

PLATFORM-AS-A-SERVICE:
• Heroku
• Railway
• Render
• Vercel (for serverless)

SERVERLESS:
• AWS Lambda
• Google Cloud Functions
• Azure Functions
• Vercel Functions
""")

def demonstrate_deployment_concepts():
    """Demonstrate deployment concepts"""
    print("Deployment Concepts Demo:")
    print("""
    DOCKERFILE EXAMPLE:
    
    FROM python:3.9-slim
    
    WORKDIR /app
    
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    COPY . .
    
    EXPOSE 5000
    
    CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
    
    DOCKER-COMPOSE.YML:
    
    version: '3.8'
    services:
      web:
        build: .
        ports:
          - "5000:5000"
        environment:
          - FLASK_ENV=production
        depends_on:
          - db
      
      db:
        image: postgres:13
        environment:
          POSTGRES_DB: myapp
          POSTGRES_USER: user
          POSTGRES_PASSWORD: password
        volumes:
          - postgres_data:/var/lib/postgresql/data
    
    volumes:
      postgres_data:
    
    NGINX CONFIGURATION:
    
    server {
        listen 80;
        server_name example.com;
        
        location / {
            proxy_pass http://127.0.0.1:5000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        
        location /static {
            alias /app/static;
        }
    }
    
    SYSTEMD SERVICE:
    
    [Unit]
    Description=My Python App
    After=network.target
    
    [Service]
    User=www-data
    Group=www-data
    WorkingDirectory=/app
    Environment=PATH=/app/venv/bin
    ExecStart=/app/venv/bin/gunicorn --bind 127.0.0.1:5000 app:app
    Restart=always
    
    [Install]
    WantedBy=multi-user.target
    """)

demonstrate_deployment_concepts()

# Question 10: Performance Optimization and Monitoring
print("\n10. How do you optimize and monitor Python applications?")
print("-" * 62)
print("""
PERFORMANCE OPTIMIZATION:

CODE OPTIMIZATION:
• Use appropriate data structures
• Avoid premature optimization
• Profile before optimizing
• Cache expensive operations
• Use generators for large datasets

DATABASE OPTIMIZATION:
• Query optimization
• Database indexing
• Connection pooling
• Query caching
• Database partitioning

CACHING STRATEGIES:
• In-memory caching (Redis, Memcached)
• Application-level caching
• HTTP caching headers
• CDN for static content

MONITORING TOOLS:
• Application Performance Monitoring (APM)
• Log aggregation and analysis
• Error tracking
• Metrics collection
• Health checks

PRODUCTION BEST PRACTICES:
• Environment configuration
• Secret management
• Graceful degradation
• Circuit breakers
• Rate limiting
""")

def demonstrate_optimization_monitoring():
    """Demonstrate optimization and monitoring concepts"""
    print("Optimization and Monitoring Demo:")
    print("""
    CACHING EXAMPLE:
    
    from functools import lru_cache
    import redis
    
    # Function-level caching
    @lru_cache(maxsize=128)
    def expensive_function(param):
        # Expensive computation
        return result
    
    # Redis caching
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    def cached_database_query(query_id):
        cache_key = f"query:{query_id}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
        
        result = database.execute_query(query_id)
        redis_client.setex(cache_key, 300, json.dumps(result))  # 5 min TTL
        return result
    
    LOGGING CONFIGURATION:
    
    import logging
    import logging.config
    
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'detailed',
            },
            'file': {
                'class': 'logging.FileHandler',
                'filename': 'app.log',
                'level': 'DEBUG',
                'formatter': 'detailed',
            },
        },
        'loggers': {
            '': {
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
                'propagate': False,
            },
        },
    }
    
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    
    HEALTH CHECK ENDPOINT:
    
    @app.route('/health')
    def health_check():
        try:
            # Check database connection
            db.session.execute('SELECT 1')
            
            # Check external services
            response = requests.get('http://external-api.com/health', timeout=5)
            response.raise_for_status()
            
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'checks': {
                    'database': 'ok',
                    'external_api': 'ok'
                }
            }), 200
            
        except Exception as e:
            return jsonify({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }), 503
    
    ERROR HANDLING AND MONITORING:
    
    import sentry_sdk
    from sentry_sdk.integrations.flask import FlaskIntegration
    
    sentry_sdk.init(
        dsn="your-sentry-dsn",
        integrations=[FlaskIntegration()],
        traces_sample_rate=1.0
    )
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        sentry_sdk.capture_exception(e)
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500
    """)

demonstrate_optimization_monitoring()

print("\n" + "=" * 80)
print("END OF WEB DEVELOPMENT & DATA SCIENCE SECTION")
print("This completes the comprehensive Python interview preparation covering:")
print("• Exception handling and file operations")
print("• Standard library and testing frameworks") 
print("• Web development and data science")
print("• Database integration and deployment")
print("• Performance optimization and monitoring")
print("=" * 80)
