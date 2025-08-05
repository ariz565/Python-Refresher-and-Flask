"""
FastAPI vs Flask vs Django - Complete Architecture Guide
Learn web framework architectures, project structures, and best practices
Covers FastAPI, Flask, Django, and Django REST Framework with interviews
"""

print("=" * 80)
print("WEB FRAMEWORK ARCHITECTURES - COMPLETE INTERVIEW GUIDE")
print("=" * 80)

# ============================================================================
# SECTION 1: FASTAPI ARCHITECTURE DEEP DIVE
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 1: FASTAPI ARCHITECTURE DEEP DIVE")
print("=" * 60)

# Question 1: FastAPI Architecture Overview
print("\n1. What is FastAPI architecture and how does it work?")
print("-" * 55)
print("""
FASTAPI ARCHITECTURE OVERVIEW:

🏗️ CORE ARCHITECTURE:
FastAPI is built on top of Starlette (ASGI framework) and Pydantic (data validation).

ARCHITECTURE LAYERS:
┌─────────────────────────────────────────┐
│             Application Layer           │  ← Business Logic
├─────────────────────────────────────────┤
│              FastAPI Layer              │  ← Routing, Validation
├─────────────────────────────────────────┤
│             Starlette Layer             │  ← ASGI, Middleware
├─────────────────────────────────────────┤
│              Uvicorn Layer              │  ← ASGI Server
└─────────────────────────────────────────┘

KEY COMPONENTS:
🔧 Starlette: High-performance ASGI framework
📊 Pydantic: Data validation and serialization
🚀 Uvicorn: Lightning-fast ASGI server
📝 OpenAPI: Automatic API documentation
🔒 Security: Built-in authentication utilities

ASGI vs WSGI:
Feature          | ASGI (FastAPI)    | WSGI (Flask/Django)
-----------------|-------------------|--------------------
Async Support    | ✅ Native         | ❌ No (thread-based)
Performance      | ⭐⭐⭐⭐⭐         | ⭐⭐⭐
WebSocket        | ✅ Built-in       | ❌ Requires extensions
Concurrency      | Async/await       | Threading/processes
Memory Usage     | Lower             | Higher

FASTAPI ADVANTAGES:
✅ Automatic API documentation (OpenAPI/Swagger)
✅ Type hints for better development experience
✅ Built-in data validation and serialization
✅ High performance (comparable to NodeJS/Go)
✅ Modern Python features (async/await)
✅ Excellent IDE support with autocomplete
✅ Built-in security utilities

WHEN TO USE FASTAPI:
🎯 Building REST APIs and microservices
🎯 Machine learning model serving
🎯 Real-time applications with WebSockets
🎯 High-performance data processing APIs
🎯 Modern applications requiring async processing
🎯 Projects where automatic documentation is valuable

INTERVIEW TIPS:
• Mention ASGI vs WSGI differences
• Highlight automatic documentation generation
• Discuss type safety and validation benefits
• Compare performance characteristics
""")

def demonstrate_fastapi_architecture():
    """Demonstrate FastAPI architecture and components"""
    print("FastAPI Architecture Components Demo:")
    
    print("\n1. Basic FastAPI Application Structure:")
    print("-" * 42)
    print("""
# main.py - Entry point
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

from routers import users, items
from dependencies import get_database
from models import database
from config import settings

# Initialize FastAPI app
app = FastAPI(
    title="My API",
    description="A sample API with FastAPI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Include routers
app.include_router(users.router, prefix="/api/v1", tags=["users"])
app.include_router(items.router, prefix="/api/v1", tags=["items"])

# Startup/shutdown events
@app.on_event("startup")
async def startup_event():
    await database.connect()

@app.on_event("shutdown")
async def shutdown_event():
    await database.disconnect()

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
    """)
    
    print("\n2. Project Structure (Large FastAPI Application):")
    print("-" * 51)
    print("""
fastapi_project/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app instance
│   ├── config.py              # Configuration settings
│   ├── dependencies.py        # Dependency injection
│   │
│   ├── api/                   # API routes
│   │   ├── __init__.py
│   │   ├── api_v1/
│   │   │   ├── __init__.py
│   │   │   ├── api.py         # API router
│   │   │   └── endpoints/
│   │   │       ├── __init__.py
│   │   │       ├── users.py
│   │   │       ├── items.py
│   │   │       └── auth.py
│   │   └── deps.py            # API dependencies
│   │
│   ├── core/                  # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py          # Settings management
│   │   ├── security.py        # Authentication/authorization
│   │   └── database.py        # Database connection
│   │
│   ├── models/                # Data models
│   │   ├── __init__.py
│   │   ├── user.py           # User model
│   │   ├── item.py           # Item model
│   │   └── base.py           # Base model
│   │
│   ├── schemas/               # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── user.py           # User schemas
│   │   ├── item.py           # Item schemas
│   │   └── token.py          # Token schemas
│   │
│   ├── services/              # Business logic
│   │   ├── __init__.py
│   │   ├── user_service.py
│   │   ├── item_service.py
│   │   └── auth_service.py
│   │
│   ├── utils/                 # Utility functions
│   │   ├── __init__.py
│   │   ├── email.py
│   │   └── helpers.py
│   │
│   └── tests/                 # Test files
│       ├── __init__.py
│       ├── conftest.py
│       ├── test_users.py
│       └── test_items.py
│
├── alembic/                   # Database migrations
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
    """)
    
    print("\n3. FastAPI Components Explained:")
    print("-" * 37)
    print("""
CORE COMPONENTS:

1. ROUTERS (api/endpoints/):
   • Group related endpoints
   • Organize code by feature/domain
   • Enable modular development
   • Support dependency injection

2. SCHEMAS (schemas/):
   • Pydantic models for request/response
   • Automatic validation and serialization
   • Type safety and IDE support
   • Generate OpenAPI documentation

3. MODELS (models/):
   • Database ORM models (SQLAlchemy)
   • Define database structure
   • Handle relationships
   • Database operations

4. SERVICES (services/):
   • Business logic layer
   • Reusable business operations
   • Database interactions
   • External API calls

5. DEPENDENCIES (dependencies.py):
   • Dependency injection system
   • Database connections
   • Authentication
   • Common validations

6. MIDDLEWARE:
   • Request/response processing
   • CORS handling
   • Authentication
   • Rate limiting
   • Logging
    """)

demonstrate_fastapi_architecture()

# Question 2: FastAPI Best Practices
print("\n2. What are FastAPI best practices and design patterns?")
print("-" * 59)
print("""
FASTAPI BEST PRACTICES:

🎯 PROJECT STRUCTURE:
✅ Separate concerns (routers, services, models, schemas)
✅ Use dependency injection for database connections
✅ Implement proper error handling
✅ Create reusable components
✅ Follow RESTful API design principles

🔒 SECURITY BEST PRACTICES:
✅ Use OAuth2 with JWT tokens
✅ Implement proper CORS policies
✅ Validate all input data with Pydantic
✅ Use HTTPS in production
✅ Implement rate limiting
✅ Sanitize user inputs

⚡ PERFORMANCE OPTIMIZATION:
✅ Use async/await for I/O operations
✅ Implement database connection pooling
✅ Add response caching where appropriate
✅ Use background tasks for heavy operations
✅ Optimize database queries
✅ Implement proper pagination

📊 DEVELOPMENT PRACTICES:
✅ Use type hints everywhere
✅ Write comprehensive tests
✅ Document APIs with docstrings
✅ Use environment variables for configuration
✅ Implement proper logging
✅ Use code formatting (black, isort)

DESIGN PATTERNS:
1. Repository Pattern: Database abstraction
2. Service Layer Pattern: Business logic separation
3. Dependency Injection: Loose coupling
4. Factory Pattern: Object creation
5. Observer Pattern: Event handling
""")

def demonstrate_fastapi_best_practices():
    """Demonstrate FastAPI best practices and patterns"""
    print("FastAPI Best Practices Implementation:")
    
    print("\n1. Dependency Injection Pattern:")
    print("-" * 37)
    print("""
# dependencies.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from database import SessionLocal
from auth import verify_token

security = HTTPBearer()

def get_db() -> Session:
    '''Database dependency'''
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    '''Get current authenticated user'''
    token = credentials.credentials
    user = await verify_token(token, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return user

# Usage in endpoints
from fastapi import APIRouter, Depends
from dependencies import get_db, get_current_user

router = APIRouter()

@router.get("/users/me")
async def read_users_me(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return current_user
    """)
    
    print("\n2. Service Layer Pattern:")
    print("-" * 31)
    print("""
# services/user_service.py
from sqlalchemy.orm import Session
from models.user import User
from schemas.user import UserCreate, UserUpdate
from core.security import get_password_hash, verify_password

class UserService:
    def __init__(self, db: Session):
        self.db = db
    
    async def create_user(self, user_data: UserCreate) -> User:
        '''Create a new user'''
        hashed_password = get_password_hash(user_data.password)
        db_user = User(
            email=user_data.email,
            username=user_data.username,
            hashed_password=hashed_password
        )
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        return db_user
    
    async def authenticate_user(self, username: str, password: str) -> User:
        '''Authenticate user credentials'''
        user = self.db.query(User).filter(
            User.username == username
        ).first()
        
        if not user or not verify_password(password, user.hashed_password):
            return None
        return user
    
    async def get_user_by_id(self, user_id: int) -> User:
        '''Get user by ID'''
        return self.db.query(User).filter(User.id == user_id).first()
    
    async def update_user(self, user_id: int, user_data: UserUpdate) -> User:
        '''Update user information'''
        user = await self.get_user_by_id(user_id)
        if not user:
            return None
        
        for field, value in user_data.dict(exclude_unset=True).items():
            setattr(user, field, value)
        
        self.db.commit()
        self.db.refresh(user)
        return user

# Usage in endpoints
@router.post("/users/", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    service = UserService(db)
    user = await service.create_user(user_data)
    return user
    """)
    
    print("\n3. Error Handling and Validation:")
    print("-" * 38)
    print("""
# exceptions.py
from fastapi import HTTPException, status

class UserNotFoundError(HTTPException):
    def __init__(self, user_id: int):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id {user_id} not found"
        )

class DuplicateEmailError(HTTPException):
    def __init__(self, email: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User with email {email} already exists"
        )

# Global exception handler
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(UserNotFoundError)
async def user_not_found_handler(request: Request, exc: UserNotFoundError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail, "type": "USER_NOT_FOUND"}
    )

# Pydantic validation schemas
from pydantic import BaseModel, validator, EmailStr
from typing import Optional

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    
    @validator('username')
    def username_must_be_valid(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v
    
    @validator('password')
    def password_must_be_strong(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool
    
    class Config:
        orm_mode = True
    """)

demonstrate_fastapi_best_practices()

# ============================================================================
# SECTION 2: FLASK ARCHITECTURE AND PATTERNS
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 2: FLASK ARCHITECTURE AND PATTERNS")
print("=" * 60)

# Question 3: Flask Architecture Overview
print("\n3. How does Flask architecture work and what are its patterns?")
print("-" * 64)
print("""
FLASK ARCHITECTURE OVERVIEW:

🏗️ FLASK PHILOSOPHY:
Flask follows the "micro-framework" philosophy - minimal core with 
extensions for additional functionality.

CORE COMPONENTS:
┌─────────────────────────────────────────┐
│            Application Layer            │  ← Business Logic
├─────────────────────────────────────────┤
│             Blueprint Layer             │  ← Route Organization
├─────────────────────────────────────────┤
│              Flask Core                 │  ← Request/Response
├─────────────────────────────────────────┤
│             Werkzeug WSGI               │  ← HTTP Handling
└─────────────────────────────────────────┘

FLASK vs FASTAPI vs DJANGO:
Feature          | Flask         | FastAPI       | Django
-----------------|---------------|---------------|----------------
Philosophy       | Minimal       | Modern/Fast   | Batteries included
Learning Curve   | Medium        | Easy          | Steep
Performance      | ⭐⭐⭐          | ⭐⭐⭐⭐⭐        | ⭐⭐
Async Support    | Limited       | Native        | Limited
Documentation    | Good          | Excellent     | Excellent
Flexibility      | High          | High          | Medium
Built-in ORM     | None          | None          | Yes (Django ORM)

FLASK ADVANTAGES:
✅ Flexible and lightweight
✅ Easy to learn and get started
✅ Large ecosystem of extensions
✅ Fine-grained control over components
✅ Great for small to medium applications
✅ Excellent documentation
✅ Strong community support

FLASK DISADVANTAGES:
❌ Requires more setup for complex apps
❌ No built-in async support
❌ Manual dependency management
❌ No automatic API documentation
❌ Less opinionated (can lead to inconsistency)

WHEN TO USE FLASK:
🎯 Small to medium web applications
🎯 Prototyping and MVP development
🎯 Learning web development concepts
🎯 Custom requirements needing flexibility
🎯 Integration with existing Python codebases
🎯 When you want full control over components
""")

def demonstrate_flask_architecture():
    """Demonstrate Flask architecture and patterns"""
    print("Flask Architecture and Patterns Demo:")
    
    print("\n1. Flask Application Factory Pattern:")
    print("-" * 40)
    print("""
# app/__init__.py - Application Factory
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_mail import Mail
from config import Config

db = SQLAlchemy()
migrate = Migrate()
login = LoginManager()
mail = Mail()

def create_app(config_class=Config):
    '''Application factory function'''
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login.init_app(app)
    mail.init_app(app)
    
    # Configure login
    login.login_view = 'auth.login'
    login.login_message = 'Please log in to access this page.'
    
    # Register blueprints
    from app.main import bp as main_bp
    app.register_blueprint(main_bp)
    
    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')
    
    from app.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Error handlers
    from app.errors import bp as errors_bp
    app.register_blueprint(errors_bp)
    
    return app

# config.py - Configuration management
import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \\
        'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAIL_SERVER = os.environ.get('MAIL_SERVER')
    MAIL_PORT = int(os.environ.get('MAIL_PORT') or 587)
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in \\
        ['true', 'on', '1']
    ADMINS = ['your-email@example.com']

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
    """)
    
    print("\n2. Flask Blueprint Organization:")
    print("-" * 36)
    print("""
flask_project/
├── app/
│   ├── __init__.py           # Application factory
│   ├── models.py             # Database models
│   │
│   ├── main/                 # Main blueprint
│   │   ├── __init__.py
│   │   ├── routes.py         # Main routes
│   │   └── forms.py          # WTForms
│   │
│   ├── auth/                 # Authentication blueprint
│   │   ├── __init__.py
│   │   ├── routes.py         # Auth routes
│   │   ├── forms.py          # Auth forms
│   │   └── email.py          # Email utilities
│   │
│   ├── api/                  # API blueprint
│   │   ├── __init__.py
│   │   ├── users.py          # User API
│   │   ├── posts.py          # Posts API
│   │   └── auth.py           # API authentication
│   │
│   ├── errors/               # Error handling
│   │   ├── __init__.py
│   │   └── handlers.py       # Error handlers
│   │
│   ├── static/               # Static files
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   │
│   └── templates/            # Jinja2 templates
│       ├── base.html
│       ├── index.html
│       └── auth/
│           ├── login.html
│           └── register.html
│
├── migrations/               # Database migrations
├── tests/                    # Test files
├── config.py                 # Configuration
├── requirements.txt
└── run.py                    # Application entry point

# app/main/__init__.py - Blueprint definition
from flask import Blueprint

bp = Blueprint('main', __name__)

from app.main import routes

# app/main/routes.py - Route definitions
from flask import render_template, request, current_app
from app.main import bp
from app.models import User, Post

@bp.route('/')
@bp.route('/index')
def index():
    posts = Post.query.order_by(Post.timestamp.desc()).all()
    return render_template('index.html', title='Home', posts=posts)

@bp.route('/user/<username>')
def user(username):
    user = User.query.filter_by(username=username).first_or_404()
    posts = user.posts.order_by(Post.timestamp.desc()).all()
    return render_template('user.html', user=user, posts=posts)
    """)
    
    print("\n3. Flask Model and Service Patterns:")
    print("-" * 38)
    print("""
# app/models.py - Database models
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db, login

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    posts = db.relationship('Post', backref='author', lazy='dynamic')
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    body = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    
    def __repr__(self):
        return f'<Post {self.title}>'

@login.user_loader
def load_user(id):
    return User.query.get(int(id))

# app/services/user_service.py - Service layer
from app import db
from app.models import User
from flask import current_app

class UserService:
    @staticmethod
    def create_user(username, email, password):
        '''Create a new user'''
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        return user
    
    @staticmethod
    def authenticate_user(username, password):
        '''Authenticate user credentials'''
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            return user
        return None
    
    @staticmethod
    def get_user_by_email(email):
        '''Get user by email'''
        return User.query.filter_by(email=email).first()
    
    @staticmethod
    def update_user_profile(user, **kwargs):
        '''Update user profile'''
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
        db.session.commit()
        return user
    """)

demonstrate_flask_architecture()

# ============================================================================
# SECTION 3: DJANGO AND DJANGO REST FRAMEWORK
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 3: DJANGO AND DJANGO REST FRAMEWORK")
print("=" * 60)

# Question 4: Django Architecture
print("\n4. How does Django architecture work and what is Django REST Framework?")
print("-" * 76)
print("""
DJANGO ARCHITECTURE OVERVIEW:

🏗️ DJANGO MVT PATTERN:
Django follows Model-View-Template (MVT) architecture pattern.

MVT ARCHITECTURE:
┌─────────────────────────────────────────┐
│              Templates                  │  ← Presentation Layer
├─────────────────────────────────────────┤
│               Views                     │  ← Business Logic
├─────────────────────────────────────────┤
│               Models                    │  ← Data Layer
├─────────────────────────────────────────┤
│              Database                   │  ← Persistence
└─────────────────────────────────────────┘

DJANGO COMPONENTS:
🎯 Models: Database abstraction (ORM)
🎯 Views: Business logic and request handling
🎯 Templates: HTML generation with template engine
🎯 URLs: URL routing and mapping
🎯 Forms: Form handling and validation
🎯 Admin: Automatic admin interface
🎯 Middleware: Request/response processing

DJANGO ADVANTAGES:
✅ "Batteries included" philosophy
✅ Powerful ORM with migrations
✅ Built-in admin interface
✅ Robust security features
✅ Excellent documentation
✅ Large ecosystem and community
✅ Scalable architecture
✅ Built-in user authentication

DJANGO REST FRAMEWORK (DRF):
Powerful toolkit for building Web APIs in Django.

DRF FEATURES:
🔧 Serializers: Data conversion and validation
🔧 ViewSets: Organized API views
🔧 Routers: Automatic URL routing
🔧 Authentication: Multiple auth schemes
🔧 Permissions: Fine-grained access control
🔧 Browsable API: Interactive API documentation
🔧 Pagination: Built-in pagination support
🔧 Filtering: Advanced filtering capabilities

WHEN TO USE DJANGO:
🎯 Large, complex web applications
🎯 Content management systems
🎯 E-commerce platforms
🎯 Admin-heavy applications
🎯 Rapid development requirements
🎯 Team projects with multiple developers
""")

def demonstrate_django_architecture():
    """Demonstrate Django and DRF architecture"""
    print("Django and Django REST Framework Architecture:")
    
    print("\n1. Django Project Structure:")
    print("-" * 33)
    print("""
django_project/
├── manage.py                 # Django management script
├── requirements.txt
├── .env                      # Environment variables
│
├── config/                   # Project configuration
│   ├── __init__.py
│   ├── settings/             # Split settings
│   │   ├── __init__.py
│   │   ├── base.py          # Base settings
│   │   ├── development.py    # Dev settings
│   │   ├── production.py     # Prod settings
│   │   └── testing.py        # Test settings
│   ├── urls.py              # Root URL configuration
│   ├── wsgi.py              # WSGI configuration
│   └── asgi.py              # ASGI configuration
│
├── apps/                     # Django applications
│   ├── users/               # User management app
│   │   ├── __init__.py
│   │   ├── admin.py         # Admin configuration
│   │   ├── apps.py          # App configuration
│   │   ├── models.py        # Data models
│   │   ├── views.py         # View functions/classes
│   │   ├── urls.py          # URL patterns
│   │   ├── forms.py         # Forms
│   │   ├── serializers.py   # DRF serializers
│   │   ├── permissions.py   # Custom permissions
│   │   ├── tests.py         # Tests
│   │   └── migrations/      # Database migrations
│   │
│   ├── blog/                # Blog app
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── serializers.py
│   │   └── ...
│   │
│   └── api/                 # API app
│       ├── v1/              # API versioning
│       │   ├── urls.py
│       │   └── views.py
│       └── ...
│
├── static/                   # Static files
├── media/                    # User uploaded files
├── templates/                # HTML templates
├── locale/                   # Internationalization
└── tests/                    # Project-wide tests
    """)
    
    print("\n2. Django Models and ORM:")
    print("-" * 31)
    print("""
# apps/users/models.py
from django.contrib.auth.models import AbstractUser
from django.db import models
from django.urls import reverse

class User(AbstractUser):
    email = models.EmailField(unique=True)
    bio = models.TextField(max_length=500, blank=True)
    location = models.CharField(max_length=30, blank=True)
    birth_date = models.DateField(null=True, blank=True)
    avatar = models.ImageField(upload_to='avatars/', blank=True)
    
    def get_absolute_url(self):
        return reverse('user-detail', kwargs={'pk': self.pk})
    
    def __str__(self):
        return self.username

# apps/blog/models.py
from django.db import models
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.utils import timezone

User = get_user_model()

class Category(models.Model):
    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(unique=True)
    description = models.TextField(blank=True)
    
    class Meta:
        verbose_name_plural = "categories"
    
    def __str__(self):
        return self.name

class Post(models.Model):
    DRAFT = 'draft'
    PUBLISHED = 'published'
    STATUS_CHOICES = [
        (DRAFT, 'Draft'),
        (PUBLISHED, 'Published'),
    ]
    
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.TextField()
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default=DRAFT)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    published_at = models.DateTimeField(null=True, blank=True)
    tags = models.ManyToManyField('Tag', blank=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status', 'published_at']),
        ]
    
    def save(self, *args, **kwargs):
        if self.status == self.PUBLISHED and not self.published_at:
            self.published_at = timezone.now()
        super().save(*args, **kwargs)
    
    def get_absolute_url(self):
        return reverse('post-detail', kwargs={'slug': self.slug})
    
    def __str__(self):
        return self.title

class Tag(models.Model):
    name = models.CharField(max_length=50, unique=True)
    
    def __str__(self):
        return self.name
    """)
    
    print("\n3. Django REST Framework Implementation:")
    print("-" * 44)
    print("""
# apps/blog/serializers.py
from rest_framework import serializers
from .models import Post, Category, Tag
from apps.users.serializers import UserSerializer

class TagSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tag
        fields = ['id', 'name']

class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = ['id', 'name', 'slug', 'description']

class PostSerializer(serializers.ModelSerializer):
    author = UserSerializer(read_only=True)
    category = CategorySerializer(read_only=True)
    tags = TagSerializer(many=True, read_only=True)
    
    class Meta:
        model = Post
        fields = [
            'id', 'title', 'slug', 'author', 'content',
            'category', 'status', 'created_at', 'updated_at',
            'published_at', 'tags'
        ]
        read_only_fields = ['created_at', 'updated_at', 'published_at']
    
    def create(self, validated_data):
        validated_data['author'] = self.context['request'].user
        return super().create(validated_data)

class PostCreateSerializer(serializers.ModelSerializer):
    category_id = serializers.IntegerField(write_only=True)
    tag_ids = serializers.ListField(
        child=serializers.IntegerField(),
        write_only=True,
        required=False
    )
    
    class Meta:
        model = Post
        fields = [
            'title', 'slug', 'content', 'status',
            'category_id', 'tag_ids'
        ]
    
    def create(self, validated_data):
        tag_ids = validated_data.pop('tag_ids', [])
        category_id = validated_data.pop('category_id')
        
        # Set author and category
        validated_data['author'] = self.context['request'].user
        validated_data['category_id'] = category_id
        
        post = Post.objects.create(**validated_data)
        
        # Add tags
        if tag_ids:
            post.tags.set(tag_ids)
        
        return post

# apps/blog/views.py
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter
from .models import Post, Category, Tag
from .serializers import (
    PostSerializer, PostCreateSerializer,
    CategorySerializer, TagSerializer
)
from .permissions import IsAuthorOrReadOnly

class PostViewSet(viewsets.ModelViewSet):
    queryset = Post.objects.select_related('author', 'category').prefetch_related('tags')
    permission_classes = [permissions.IsAuthenticatedOrReadOnly, IsAuthorOrReadOnly]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['status', 'category', 'author']
    search_fields = ['title', 'content']
    ordering_fields = ['created_at', 'updated_at']
    ordering = ['-created_at']
    
    def get_serializer_class(self):
        if self.action in ['create', 'update', 'partial_update']:
            return PostCreateSerializer
        return PostSerializer
    
    def get_queryset(self):
        queryset = super().get_queryset()
        if not self.request.user.is_authenticated:
            # Only show published posts to anonymous users
            queryset = queryset.filter(status=Post.PUBLISHED)
        return queryset
    
    @action(detail=True, methods=['post'])
    def publish(self, request, pk=None):
        post = self.get_object()
        post.status = Post.PUBLISHED
        post.save()
        return Response({'status': 'published'})
    
    @action(detail=False)
    def published(self, request):
        published_posts = self.get_queryset().filter(status=Post.PUBLISHED)
        serializer = self.get_serializer(published_posts, many=True)
        return Response(serializer.data)

# apps/blog/permissions.py
from rest_framework import permissions

class IsAuthorOrReadOnly(permissions.BasePermission):
    '''
    Custom permission to only allow authors to edit their own posts.
    '''
    
    def has_object_permission(self, request, view, obj):
        # Read permissions for any request
        if request.method in permissions.SAFE_METHODS:
            return True
        
        # Write permissions only to the author
        return obj.author == request.user

# apps/api/v1/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from apps.blog.views import PostViewSet
from apps.users.views import UserViewSet

router = DefaultRouter()
router.register(r'posts', PostViewSet)
router.register(r'users', UserViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('auth/', include('rest_framework.urls')),
]
    """)

demonstrate_django_architecture()

# ============================================================================
# SECTION 4: FRAMEWORK COMPARISON AND INTERVIEW QUESTIONS
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 4: FRAMEWORK COMPARISON AND INTERVIEW QUESTIONS")
print("=" * 60)

# Question 5: Framework Comparison
print("\n5. How do you choose between FastAPI, Flask, and Django?")
print("-" * 58)
print("""
FRAMEWORK COMPARISON MATRIX:

PERFORMANCE COMPARISON:
Framework    | Requests/sec | Latency (ms) | Memory Usage
-------------|--------------|--------------|-------------
FastAPI      | 20,000+      | 1-5          | Low
Flask        | 5,000-8,000  | 10-50        | Medium
Django       | 3,000-5,000  | 20-100       | High

PROJECT SIZE SUITABILITY:
Size         | FastAPI      | Flask        | Django
-------------|--------------|--------------|-------------
Small        | ✅ Excellent | ✅ Perfect   | ⚠️ Overkill
Medium       | ✅ Excellent | ✅ Good      | ✅ Good
Large        | ✅ Good      | ⚠️ Complex   | ✅ Excellent
Enterprise   | ✅ Good      | ❌ Limited   | ✅ Perfect

DEVELOPMENT SPEED:
Use Case              | FastAPI | Flask | Django
----------------------|---------|-------|--------
API Development       | ⭐⭐⭐⭐⭐ | ⭐⭐⭐   | ⭐⭐⭐⭐
Web Applications      | ⭐⭐⭐   | ⭐⭐⭐⭐  | ⭐⭐⭐⭐⭐
Admin Interfaces      | ⭐⭐     | ⭐⭐    | ⭐⭐⭐⭐⭐
Prototyping          | ⭐⭐⭐⭐  | ⭐⭐⭐⭐⭐ | ⭐⭐⭐
Complex Business Logic| ⭐⭐⭐   | ⭐⭐⭐   | ⭐⭐⭐⭐⭐

DECISION MATRIX:
Choose FastAPI when:
✅ Building high-performance APIs
✅ Need automatic documentation
✅ Working with modern Python (3.7+)
✅ Async/await requirements
✅ ML/AI model serving
✅ Microservices architecture

Choose Flask when:
✅ Learning web development
✅ Small to medium applications
✅ Need maximum flexibility
✅ Custom requirements
✅ Gradual migration from other frameworks
✅ Integration with existing code

Choose Django when:
✅ Large, complex applications
✅ Content management needs
✅ Admin interface requirements
✅ Team development
✅ Rapid prototyping
✅ Database-heavy applications
""")

def demonstrate_framework_comparison():
    """Demonstrate practical framework comparison"""
    print("Framework Comparison - Practical Examples:")
    
    print("\n1. API Endpoint Comparison:")
    print("-" * 35)
    print("""
# FASTAPI IMPLEMENTATION
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

class UserResponse(BaseModel):
    id: int
    username: str
    email: str

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, db: Session = Depends(get_db)):
    user = await UserService.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# FLASK IMPLEMENTATION
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
db = SQLAlchemy(app)

@app.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = User.query.get_or_404(user_id)
    return jsonify({
        'id': user.id,
        'username': user.username,
        'email': user.email
    })

# DJANGO REST FRAMEWORK IMPLEMENTATION
from rest_framework import viewsets
from rest_framework.response import Response
from .models import User
from .serializers import UserSerializer

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    
    def retrieve(self, request, pk=None):
        user = self.get_object()
        serializer = self.get_serializer(user)
        return Response(serializer.data)
    """)
    
    print("\n2. Project Setup Complexity:")
    print("-" * 35)
    print("""
SETUP COMPLEXITY COMPARISON:

FastAPI (Simple API):
- Files needed: 3-5
- Setup time: 15 minutes
- Dependencies: 2-3
- Code lines: ~50

Flask (Simple Web App):
- Files needed: 5-8
- Setup time: 30 minutes
- Dependencies: 3-5
- Code lines: ~100

Django (Full Application):
- Files needed: 15-20
- Setup time: 60 minutes
- Dependencies: 5-10
- Code lines: ~200

LEARNING CURVE:
Framework | Time to Basic App | Time to Production
----------|-------------------|-------------------
FastAPI   | 2-3 hours        | 1-2 weeks
Flask     | 4-6 hours        | 2-4 weeks
Django    | 8-12 hours       | 4-8 weeks
    """)

demonstrate_framework_comparison()

print("\n" + "=" * 80)
print("WEB FRAMEWORK INTERVIEW QUESTIONS")
print("=" * 80)

print("""
TOP INTERVIEW QUESTIONS & ANSWERS:

Q1: "What's the difference between FastAPI and Flask?"
A: "FastAPI is built on ASGI with native async support and automatic 
   documentation. Flask is WSGI-based, more flexible but requires 
   more setup. FastAPI is better for APIs, Flask for web apps."

Q2: "When would you choose Django over FastAPI?"
A: "Choose Django for complex web applications needing admin interface,
   ORM, user management, and rapid development. FastAPI is better 
   for high-performance APIs and microservices."

Q3: "Explain Django's MVT pattern."
A: "Model-View-Template: Models define data structure, Views handle 
   business logic and requests, Templates handle presentation. 
   URLs route requests to appropriate views."

Q4: "What is Django REST Framework and why use it?"
A: "DRF is a toolkit for building APIs in Django. Provides serializers,
   viewsets, authentication, permissions, and browsable API. 
   Reduces API development time significantly."

Q5: "How do you structure a large Flask application?"
A: "Use Application Factory pattern, organize with Blueprints,
   separate concerns (models, views, services), use configuration
   classes, and implement proper dependency injection."

Q6: "What are FastAPI's main advantages for API development?"
A: "Automatic documentation, type safety, high performance, async
   support, built-in validation, modern Python features, and
   excellent developer experience with IDE support."

Q7: "How does async/await work in FastAPI vs Flask?"
A: "FastAPI has native async support with ASGI. Flask requires
   additional setup and doesn't handle async as efficiently.
   FastAPI can handle more concurrent requests."

Q8: "Explain dependency injection in FastAPI."
A: "FastAPI's Depends() system allows injecting dependencies like
   database connections, authentication, etc. Dependencies are
   resolved automatically and can be cached or scoped."

PROJECT STRUCTURE BEST PRACTICES:

FASTAPI Project Structure:
✅ Separate routers, models, schemas, services
✅ Use dependency injection for database/auth
✅ Implement proper error handling
✅ Add comprehensive testing
✅ Use environment-based configuration

FLASK Project Structure:
✅ Application Factory pattern
✅ Blueprint organization by feature
✅ Service layer for business logic
✅ Proper configuration management
✅ Extension initialization

DJANGO Project Structure:
✅ App-based organization
✅ Split settings by environment
✅ Use Django best practices
✅ Implement custom managers/querysets
✅ Proper URL organization

KEY TAKEAWAYS:
✅ Choose framework based on project requirements
✅ FastAPI: High-performance APIs, automatic docs
✅ Flask: Flexibility, learning, custom requirements  
✅ Django: Complex apps, rapid development, admin needs
✅ Understand architecture patterns for each framework
✅ Structure projects for maintainability and scalability

NEXT STEPS:
🔗 Build sample projects with each framework
🔗 Practice implementing authentication
🔗 Study performance optimization techniques
🔗 Learn deployment strategies
🔗 Understand testing approaches for each framework
""")

print("\n" + "=" * 80)
print("END OF WEB FRAMEWORK ARCHITECTURE GUIDE")
print("Master all three frameworks for complete full-stack expertise!")
print("=" * 80)
