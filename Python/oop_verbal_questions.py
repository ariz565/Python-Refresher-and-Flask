# ===============================================================================
# PYTHON OOP VERBAL INTERVIEW QUESTIONS
# Complete Guide with Short Code Examples for Experienced Developers
# ===============================================================================

"""
COMPREHENSIVE OOP VERBAL QUESTIONS COVERAGE:
==========================================
1. What is OOP? (Object-Oriented Programming)
2. Structural vs Functional vs Object-Oriented Programming
3. Four Pillars of OOP
4. Access Modifiers in Python
5. Types of Inheritance
6. Class Methods vs Static Methods vs Instance Methods
7. Special Methods (Magic Methods)
8. Property Decorators
9. Abstract Classes and Interfaces
10. Design Patterns
11. Memory Management
12. Advanced OOP Concepts
"""

print("=" * 100)
print("PYTHON OOP VERBAL INTERVIEW QUESTIONS WITH CODE EXAMPLES")
print("=" * 100)

# ===============================================================================
# 1. WHAT IS OOP? (OBJECT-ORIENTED PROGRAMMING)
# ===============================================================================

print("\n" + "=" * 80)
print("1. WHAT IS OOP?")
print("=" * 80)

print("""
Q: What is Object-Oriented Programming (OOP)?
A: OOP is a programming paradigm based on the concept of "objects" which contain:
   - Data (attributes/properties)
   - Code (methods/functions)
   
Key Benefits:
- Code Reusability
- Modularity
- Maintainability
- Data Security (Encapsulation)
- Real-world modeling
""")

# Basic OOP Example
class Car:
    """Simple Car class demonstrating OOP basics"""
    def __init__(self, brand, model):
        self.brand = brand      # Data/Attribute
        self.model = model      # Data/Attribute
        self.speed = 0
    
    def accelerate(self):       # Method/Behavior
        self.speed += 10
        return f"{self.brand} {self.model} accelerating to {self.speed} mph"

# Creating objects (instances)
car1 = Car("Toyota", "Camry")
car2 = Car("Honda", "Civic")

print("Basic OOP Example:")
print(car1.accelerate())
print(car2.accelerate())

# ===============================================================================
# 2. PROGRAMMING PARADIGMS COMPARISON
# ===============================================================================

print("\n" + "=" * 80)
print("2. PROGRAMMING PARADIGMS")
print("=" * 80)

print("""
Q: Difference between Structural, Functional, and Object-Oriented Programming?

1. STRUCTURAL PROGRAMMING:
   - Linear, top-down approach
   - Uses functions and procedures
   - Data and functions are separate
   
2. FUNCTIONAL PROGRAMMING:
   - Based on mathematical functions
   - Immutable data
   - No side effects
   - Functions are first-class citizens
   
3. OBJECT-ORIENTED PROGRAMMING:
   - Based on objects and classes
   - Encapsulation of data and methods
   - Inheritance and polymorphism
   - Real-world modeling
""")

# Structural Programming Example
def calculate_area_structural(length, width):
    """Structural approach - just a function"""
    return length * width

print("Structural Programming:")
area = calculate_area_structural(10, 5)
print(f"Area: {area}")

# Functional Programming Example
from functools import reduce

def calculate_area_functional(dimensions):
    """Functional approach - pure function"""
    return reduce(lambda x, y: x * y, dimensions)

print("\nFunctional Programming:")
area = calculate_area_functional([10, 5])
print(f"Area: {area}")

# Object-Oriented Programming Example
class Rectangle:
    """OOP approach - encapsulated data and methods"""
    def __init__(self, length, width):
        self.length = length
        self.width = width
    
    def calculate_area(self):
        return self.length * self.width

print("\nObject-Oriented Programming:")
rect = Rectangle(10, 5)
print(f"Area: {rect.calculate_area()}")

# ===============================================================================
# 3. FOUR PILLARS OF OOP
# ===============================================================================

print("\n" + "=" * 80)
print("3. FOUR PILLARS OF OOP")
print("=" * 80)

print("""
Q: What are the four pillars of OOP?

1. ENCAPSULATION - Bundling data and methods together, hiding internal details
2. INHERITANCE - Creating new classes based on existing classes
3. POLYMORPHISM - Same interface, different implementations
4. ABSTRACTION - Hiding complex implementation details
""")

# 1. ENCAPSULATION Example
class BankAccount:
    """Encapsulation: Private data with controlled access"""
    def __init__(self, balance):
        self.__balance = balance  # Private attribute
    
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
    
    def get_balance(self):
        return self.__balance

print("1. ENCAPSULATION:")
account = BankAccount(1000)
account.deposit(500)
print(f"Balance: {account.get_balance()}")
# print(account.__balance)  # This would raise AttributeError

# 2. INHERITANCE Example
class Animal:
    """Base class"""
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        pass

class Dog(Animal):  # Inheritance
    """Derived class"""
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):  # Inheritance
    """Derived class"""
    def speak(self):
        return f"{self.name} says Meow!"

print("\n2. INHERITANCE:")
dog = Dog("Buddy")
cat = Cat("Whiskers")
print(dog.speak())
print(cat.speak())

# 3. POLYMORPHISM Example
def animal_sound(animal):
    """Polymorphism: Same interface, different behavior"""
    return animal.speak()

print("\n3. POLYMORPHISM:")
animals = [Dog("Rex"), Cat("Fluffy")]
for animal in animals:
    print(animal_sound(animal))

# 4. ABSTRACTION Example
from abc import ABC, abstractmethod

class Vehicle(ABC):
    """Abstract class"""
    @abstractmethod
    def start_engine(self):
        pass
    
    @abstractmethod
    def stop_engine(self):
        pass

class Car_Abstract(Vehicle):
    """Concrete implementation"""
    def start_engine(self):
        return "Car engine started"
    
    def stop_engine(self):
        return "Car engine stopped"

print("\n4. ABSTRACTION:")
car = Car_Abstract()
print(car.start_engine())

# ===============================================================================
# 4. ACCESS MODIFIERS IN PYTHON
# ===============================================================================

print("\n" + "=" * 80)
print("4. ACCESS MODIFIERS IN PYTHON")
print("=" * 80)

print("""
Q: What are access modifiers in Python?

Python uses naming conventions for access control:

1. PUBLIC - No underscore prefix (accessible everywhere)
2. PROTECTED - Single underscore prefix _ (internal use, accessible but not recommended)
3. PRIVATE - Double underscore prefix __ (name mangled, not directly accessible)

Note: Python doesn't have true access modifiers like Java/C++
""")

class AccessModifierExample:
    """Demonstrating access modifiers"""
    
    def __init__(self):
        self.public_var = "I'm public"           # PUBLIC
        self._protected_var = "I'm protected"    # PROTECTED  
        self.__private_var = "I'm private"       # PRIVATE
    
    def public_method(self):
        """Public method - accessible everywhere"""
        return "Public method called"
    
    def _protected_method(self):
        """Protected method - internal use"""
        return "Protected method called"
    
    def __private_method(self):
        """Private method - name mangled"""
        return "Private method called"
    
    def access_all_methods(self):
        """Method to access all types"""
        return {
            'public': self.public_method(),
            'protected': self._protected_method(),
            'private': self.__private_method()
        }

print("Access Modifiers Example:")
obj = AccessModifierExample()

# Public access
print(f"Public variable: {obj.public_var}")
print(f"Public method: {obj.public_method()}")

# Protected access (accessible but not recommended)
print(f"Protected variable: {obj._protected_var}")
print(f"Protected method: {obj._protected_method()}")

# Private access (name mangled)
try:
    print(obj.__private_var)  # This will raise AttributeError
except AttributeError:
    print("Cannot access private variable directly")

# Accessing private through name mangling
print(f"Private via name mangling: {obj._AccessModifierExample__private_var}")

print(f"All methods: {obj.access_all_methods()}")

# ===============================================================================
# 5. TYPES OF INHERITANCE
# ===============================================================================

print("\n" + "=" * 80)
print("5. TYPES OF INHERITANCE IN PYTHON")
print("=" * 80)

print("""
Q: What are the types of inheritance in Python?

1. SINGLE INHERITANCE - One child class inherits from one parent class
2. MULTIPLE INHERITANCE - One child class inherits from multiple parent classes
3. MULTILEVEL INHERITANCE - Chain of inheritance (A -> B -> C)
4. HIERARCHICAL INHERITANCE - Multiple child classes inherit from one parent
5. HYBRID INHERITANCE - Combination of multiple inheritance types
""")

# 1. SINGLE INHERITANCE
class Parent:
    def parent_method(self):
        return "Parent method"

class Child(Parent):  # Single inheritance
    def child_method(self):
        return "Child method"

print("1. SINGLE INHERITANCE:")
child = Child()
print(child.parent_method())
print(child.child_method())

# 2. MULTIPLE INHERITANCE
class Father:
    def father_trait(self):
        return "Father's trait"

class Mother:
    def mother_trait(self):
        return "Mother's trait"

class Child_Multiple(Father, Mother):  # Multiple inheritance
    def child_trait(self):
        return "Child's trait"

print("\n2. MULTIPLE INHERITANCE:")
child_mult = Child_Multiple()
print(child_mult.father_trait())
print(child_mult.mother_trait())
print(child_mult.child_trait())

# 3. MULTILEVEL INHERITANCE
class GrandParent:
    def grandparent_method(self):
        return "GrandParent method"

class Parent_Multi(GrandParent):  # Level 1
    def parent_method(self):
        return "Parent method"

class Child_Multi(Parent_Multi):  # Level 2
    def child_method(self):
        return "Child method"

print("\n3. MULTILEVEL INHERITANCE:")
child_multi = Child_Multi()
print(child_multi.grandparent_method())
print(child_multi.parent_method())
print(child_multi.child_method())

# 4. HIERARCHICAL INHERITANCE
class Animal_Hier:
    def animal_method(self):
        return "Animal method"

class Dog_Hier(Animal_Hier):  # Child 1
    def dog_method(self):
        return "Dog method"

class Cat_Hier(Animal_Hier):  # Child 2
    def cat_method(self):
        return "Cat method"

print("\n4. HIERARCHICAL INHERITANCE:")
dog_hier = Dog_Hier()
cat_hier = Cat_Hier()
print(f"Dog: {dog_hier.animal_method()}, {dog_hier.dog_method()}")
print(f"Cat: {cat_hier.animal_method()}, {cat_hier.cat_method()}")

# 5. HYBRID INHERITANCE (Multiple + Multilevel)
class A:
    def method_a(self):
        return "Method A"

class B(A):  # Multilevel
    def method_b(self):
        return "Method B"

class C(A):  # Hierarchical from A
    def method_c(self):
        return "Method C"

class D(B, C):  # Multiple inheritance (Hybrid)
    def method_d(self):
        return "Method D"

print("\n5. HYBRID INHERITANCE:")
d = D()
print(f"MRO: {[cls.__name__ for cls in D.__mro__]}")
print(f"Methods: {d.method_a()}, {d.method_b()}, {d.method_c()}, {d.method_d()}")

# ===============================================================================
# 6. CLASS METHODS vs STATIC METHODS vs INSTANCE METHODS
# ===============================================================================

print("\n" + "=" * 80)
print("6. METHODS TYPES")
print("=" * 80)

print("""
Q: Difference between @classmethod, @staticmethod, and instance methods?

1. INSTANCE METHOD:
   - First parameter is 'self'
   - Can access instance and class variables
   - Called on instance: obj.method()

2. CLASS METHOD (@classmethod):
   - First parameter is 'cls'
   - Can access class variables, not instance variables
   - Called on class: Class.method() or obj.method()
   - Used for alternative constructors

3. STATIC METHOD (@staticmethod):
   - No special first parameter
   - Cannot access instance or class variables
   - Called on class: Class.method() or obj.method()
   - Utility functions related to the class
""")

class MethodExample:
    class_variable = "I'm a class variable"
    
    def __init__(self, name):
        self.name = name  # Instance variable
    
    # INSTANCE METHOD
    def instance_method(self):
        """Can access both instance and class variables"""
        return f"Instance method: {self.name}, {self.class_variable}"
    
    # CLASS METHOD
    @classmethod
    def class_method(cls):
        """Can access class variables, not instance variables"""
        return f"Class method: {cls.class_variable}"
    
    @classmethod
    def alternative_constructor(cls, name_with_title):
        """Alternative constructor using class method"""
        name = name_with_title.split()[1]  # Extract name from "Mr. John"
        return cls(name)
    
    # STATIC METHOD
    @staticmethod
    def static_method(x, y):
        """Utility function, no access to class or instance"""
        return f"Static method: {x + y}"

print("Methods Example:")
obj = MethodExample("John")

# Instance method (needs instance)
print(obj.instance_method())

# Class method (can be called on class or instance)
print(MethodExample.class_method())
print(obj.class_method())

# Alternative constructor using class method
obj2 = MethodExample.alternative_constructor("Mr. Smith")
print(obj2.instance_method())

# Static method (can be called on class or instance)
print(MethodExample.static_method(5, 3))
print(obj.static_method(10, 7))

# ===============================================================================
# 7. SPECIAL METHODS (MAGIC METHODS)
# ===============================================================================

print("\n" + "=" * 80)
print("7. SPECIAL METHODS (MAGIC METHODS)")
print("=" * 80)

print("""
Q: What are magic methods (dunder methods) in Python?

Magic methods are special methods with double underscores that define
how objects behave with built-in functions and operators.

Common magic methods:
- __init__: Constructor
- __str__: String representation for users
- __repr__: String representation for developers
- __len__: Length of object
- __getitem__: Index access obj[key]
- __setitem__: Item assignment obj[key] = value
- __add__: Addition operator +
- __eq__: Equality operator ==
- __lt__: Less than operator <
""")

class MagicMethodExample:
    """Class demonstrating magic methods"""
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.data = {}
    
    def __str__(self):
        """User-friendly string representation"""
        return f"Person: {self.name}, Age: {self.age}"
    
    def __repr__(self):
        """Developer-friendly string representation"""
        return f"MagicMethodExample('{self.name}', {self.age})"
    
    def __len__(self):
        """Length of the object"""
        return len(self.data)
    
    def __getitem__(self, key):
        """Index access: obj[key]"""
        return self.data[key]
    
    def __setitem__(self, key, value):
        """Item assignment: obj[key] = value"""
        self.data[key] = value
    
    def __add__(self, other):
        """Addition operator: obj1 + obj2"""
        if isinstance(other, MagicMethodExample):
            combined_name = f"{self.name}+{other.name}"
            combined_age = (self.age + other.age) // 2
            return MagicMethodExample(combined_name, combined_age)
        return NotImplemented
    
    def __eq__(self, other):
        """Equality operator: obj1 == obj2"""
        if isinstance(other, MagicMethodExample):
            return self.name == other.name and self.age == other.age
        return False
    
    def __lt__(self, other):
        """Less than operator: obj1 < obj2"""
        if isinstance(other, MagicMethodExample):
            return self.age < other.age
        return NotImplemented

print("Magic Methods Example:")
person1 = MagicMethodExample("Alice", 25)
person2 = MagicMethodExample("Bob", 30)

# __str__ and __repr__
print(f"str(): {str(person1)}")
print(f"repr(): {repr(person1)}")

# __setitem__ and __getitem__
person1["hobby"] = "reading"
person1["city"] = "New York"
print(f"person1['hobby']: {person1['hobby']}")

# __len__
print(f"len(person1): {len(person1)}")

# __add__
person3 = person1 + person2
print(f"Combined: {person3}")

# __eq__ and __lt__
print(f"person1 == person2: {person1 == person2}")
print(f"person1 < person2: {person1 < person2}")

# ===============================================================================
# 8. PROPERTY DECORATORS
# ===============================================================================

print("\n" + "=" * 80)
print("8. PROPERTY DECORATORS")
print("=" * 80)

print("""
Q: What are property decorators in Python?

Property decorators allow you to define methods that can be accessed like attributes,
providing controlled access to class attributes.

Types:
- @property: Getter method
- @attribute.setter: Setter method  
- @attribute.deleter: Deleter method

Benefits:
- Data validation
- Computed properties
- Backward compatibility
- Encapsulation
""")

class Temperature:
    """Class demonstrating property decorators"""
    
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        """Getter for celsius"""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        """Setter with validation"""
        if value < -273.15:
            raise ValueError("Temperature cannot be below absolute zero")
        self._celsius = value
    
    @celsius.deleter
    def celsius(self):
        """Deleter for celsius"""
        print("Deleting temperature")
        self._celsius = 0
    
    @property
    def fahrenheit(self):
        """Computed property - calculated from celsius"""
        return (self._celsius * 9/5) + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        """Set fahrenheit, convert to celsius"""
        self._celsius = (value - 32) * 5/9
    
    @property
    def kelvin(self):
        """Computed property - calculated from celsius"""
        return self._celsius + 273.15

print("Property Decorators Example:")
temp = Temperature(25)

# Using properties like attributes
print(f"Celsius: {temp.celsius}")
print(f"Fahrenheit: {temp.fahrenheit}")
print(f"Kelvin: {temp.kelvin}")

# Setting through property
temp.fahrenheit = 100
print(f"After setting Fahrenheit to 100:")
print(f"Celsius: {temp.celsius:.2f}")

# Validation
try:
    temp.celsius = -300  # This will raise ValueError
except ValueError as e:
    print(f"Validation error: {e}")

# ===============================================================================
# 9. ABSTRACT CLASSES AND INTERFACES
# ===============================================================================

print("\n" + "=" * 80)
print("9. ABSTRACT CLASSES AND INTERFACES")
print("=" * 80)

print("""
Q: What are abstract classes and how do they differ from interfaces?

ABSTRACT CLASS:
- Cannot be instantiated
- May contain abstract and concrete methods
- Used with ABC (Abstract Base Class) module
- Provides partial implementation

INTERFACE (Protocol in Python):
- Defines method signatures only
- No implementation
- Multiple inheritance possible
- Used for type checking
""")

from abc import ABC, abstractmethod
from typing import Protocol

# ABSTRACT CLASS Example
class Shape(ABC):
    """Abstract class with abstract and concrete methods"""
    
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def area(self):
        """Abstract method - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def perimeter(self):
        """Abstract method - must be implemented by subclasses"""
        pass
    
    def description(self):
        """Concrete method - can be used by subclasses"""
        return f"This is a {self.name}"

class Rectangle_Abstract(Shape):
    """Concrete implementation of abstract class"""
    
    def __init__(self, width, height):
        super().__init__("Rectangle")
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

# INTERFACE (Protocol) Example
class Drawable(Protocol):
    """Interface using Protocol"""
    
    def draw(self) -> str:
        """Method that must be implemented"""
        ...

class Circle:
    """Class implementing Drawable protocol"""
    
    def __init__(self, radius):
        self.radius = radius
    
    def draw(self):
        return f"Drawing a circle with radius {self.radius}"

class Square:
    """Class implementing Drawable protocol"""
    
    def __init__(self, side):
        self.side = side
    
    def draw(self):
        return f"Drawing a square with side {self.side}"

def draw_shape(shape: Drawable):
    """Function accepting any object implementing Drawable protocol"""
    return shape.draw()

print("Abstract Classes and Interfaces Example:")

# Abstract class usage
rect = Rectangle_Abstract(5, 3)
print(f"Rectangle area: {rect.area()}")
print(f"Rectangle perimeter: {rect.perimeter()}")
print(rect.description())

# Cannot instantiate abstract class
try:
    shape = Shape("Generic")  # This will raise TypeError
except TypeError as e:
    print(f"Cannot instantiate abstract class: {e}")

# Protocol/Interface usage
circle = Circle(5)
square = Square(4)

print(f"Circle: {draw_shape(circle)}")
print(f"Square: {draw_shape(square)}")

# ===============================================================================
# 10. ADVANCED OOP CONCEPTS
# ===============================================================================

print("\n" + "=" * 80)
print("10. ADVANCED OOP CONCEPTS")
print("=" * 80)

print("""
Q: Explain advanced OOP concepts in Python:

1. METACLASSES: Classes that create classes
2. DESCRIPTORS: Objects that define attribute access
3. COMPOSITION vs AGGREGATION: 
   - Composition: Strong "has-a" relationship (owner destroys parts)
   - Aggregation: Weak "has-a" relationship (parts exist independently)
4. DIAMOND PROBLEM: Multiple inheritance ambiguity
5. METHOD OVERLOADING: Multiple methods with same name, different parameters
6. OPERATOR OVERLOADING: Defining custom behavior for operators
""")

# METACLASS Example
class SingletonMeta(type):
    """Metaclass that creates singleton instances"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class DatabaseConnection_Meta(metaclass=SingletonMeta):
    """Class using singleton metaclass"""
    def __init__(self, host):
        self.host = host

print("Metaclass Example:")
db1 = DatabaseConnection_Meta("localhost")
db2 = DatabaseConnection_Meta("remote")
print(f"Same instance? {db1 is db2}")  # True - singleton

# DESCRIPTOR Example
class PositiveNumber:
    """Descriptor that ensures positive numbers"""
    
    def __init__(self, name):
        self.name = name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name, 0)
    
    def __set__(self, obj, value):
        if value < 0:
            raise ValueError(f"{self.name} must be positive")
        obj.__dict__[self.name] = value

class Product:
    """Class using descriptor"""
    price = PositiveNumber("price")
    quantity = PositiveNumber("quantity")
    
    def __init__(self, name, price, quantity):
        self.name = name
        self.price = price
        self.quantity = quantity

print("\nDescriptor Example:")
product = Product("Laptop", 1000, 5)
print(f"Product: {product.name}, Price: {product.price}, Quantity: {product.quantity}")

try:
    product.price = -100  # This will raise ValueError
except ValueError as e:
    print(f"Descriptor validation: {e}")

# COMPOSITION vs AGGREGATION Example
class Engine_Comp:
    """Engine for composition example"""
    def __init__(self, horsepower):
        self.horsepower = horsepower

class Car_Composition:
    """Car with composition - owns the engine"""
    def __init__(self, brand):
        self.brand = brand
        self.engine = Engine_Comp(200)  # Composition - car creates engine

class Wheel:
    """Wheel for aggregation example"""
    def __init__(self, size):
        self.size = size

class Car_Aggregation:
    """Car with aggregation - uses existing wheels"""
    def __init__(self, brand, wheels):
        self.brand = brand
        self.wheels = wheels  # Aggregation - wheels exist independently

print("\nComposition vs Aggregation:")
# Composition
comp_car = Car_Composition("Toyota")
print(f"Composition - Car engine: {comp_car.engine.horsepower}HP")

# Aggregation
wheels = [Wheel(16) for _ in range(4)]
agg_car = Car_Aggregation("Honda", wheels)
print(f"Aggregation - Car wheels: {len(agg_car.wheels)} wheels of size {agg_car.wheels[0].size}")

print("\n" + "=" * 80)
print("OOP VERBAL QUESTIONS COMPLETE!")
print("=" * 80)
print("""
🎯 COMPREHENSIVE COVERAGE ACHIEVED:
✅ What is OOP and its benefits
✅ Programming paradigms comparison
✅ Four pillars of OOP with examples
✅ Access modifiers in Python
✅ All types of inheritance
✅ Method types (instance, class, static)
✅ Magic methods (dunder methods)
✅ Property decorators
✅ Abstract classes and interfaces
✅ Advanced concepts (metaclasses, descriptors, etc.)

📝 KEY INTERVIEW POINTS COVERED:
- Real-world examples for each concept
- When to use @classmethod vs @staticmethod
- Access modifier naming conventions
- Inheritance types with code examples
- Magic methods for operator overloading
- Property decorators for data validation
- Abstract classes vs Protocols
- Advanced OOP patterns

🚀 READY FOR VERBAL OOP INTERVIEWS!
""")
