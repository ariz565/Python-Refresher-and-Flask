"""
Complete Python Object-Oriented Programming Interview Questions
Covering OOP concepts, classes, inheritance, and advanced OOP topics
Based on comprehensive Python interview preparation material
"""

print("=" * 80)
print("COMPLETE PYTHON OBJECT-ORIENTED PROGRAMMING QUESTIONS")
print("=" * 80)

# ============================================================================
# SECTION 1: CLASSES AND OBJECTS FUNDAMENTALS
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 1: CLASSES AND OBJECTS FUNDAMENTALS")
print("=" * 50)

# Question 1: Classes and Objects
print("\n1. How do you create a class in Python?")
print("-" * 38)
print("""
A class is a blueprint for creating objects. It defines attributes and methods
that objects of the class will have.

SYNTAX:
class ClassName:
    def __init__(self, parameters):
        # Constructor method
        self.attribute = value
    
    def method_name(self):
        # Instance method
        pass

KEY CONCEPTS:
• self: Refers to the instance of the class
• __init__(): Constructor method (automatically called)
• Instance variables: Unique to each object
• Instance methods: Functions defined in the class
""")

# Example
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def introduce(self):
        return f"Hi, I'm {self.name} and I'm {self.age} years old."
    
    def have_birthday(self):
        self.age += 1
        return f"Happy birthday! {self.name} is now {self.age}."

# Create objects
person1 = Person("Alice", 25)
person2 = Person("Bob", 30)

print("Example:")
print(f"Person1: {person1.introduce()}")
print(f"Person2: {person2.introduce()}")
print(f"After birthday: {person1.have_birthday()}")

# Question 2: The __init__ method
print("\n2. What is the __init__ method in Python?")
print("-" * 42)
print("""
The __init__ method is a special method called a constructor:

PURPOSE:
• Automatically called when creating a new object
• Initializes object attributes
• Sets up the initial state of the object
• Can accept parameters to customize initialization

CHARACTERISTICS:
• First parameter is always 'self'
• Can have default parameters
• Not required but commonly used
• Different from __new__ method
""")

class Book:
    def __init__(self, title, author, pages=100):
        self.title = title
        self.author = author
        self.pages = pages
        self.is_read = False
        print(f"Created book: '{title}' by {author}")
    
    def mark_as_read(self):
        self.is_read = True
        return f"Marked '{self.title}' as read."

print("Example:")
book1 = Book("Python Programming", "John Doe")
book2 = Book("Data Science", "Jane Smith", 250)
print(book1.mark_as_read())

# Question 3: Instance vs Class Variables
print("\n3. What's the difference between instance and class variables?")
print("-" * 63)
print("""
INSTANCE VARIABLES:
• Unique to each object/instance
• Defined inside __init__ or instance methods
• Use self.variable_name
• Each object has its own copy

CLASS VARIABLES:
• Shared by all instances of the class
• Defined directly in the class (not in methods)
• Can be accessed via class name or instance
• One copy shared by all objects
""")

class Student:
    # Class variable - shared by all instances
    school = "DataCamp University"
    student_count = 0
    
    def __init__(self, name, grade):
        # Instance variables - unique to each instance
        self.name = name
        self.grade = grade
        Student.student_count += 1
    
    def display_info(self):
        return f"{self.name}, Grade: {self.grade}, School: {self.school}"
    
    @classmethod
    def get_student_count(cls):
        return f"Total students: {cls.student_count}"

print("Example:")
student1 = Student("Alice", "A")
student2 = Student("Bob", "B")

print(f"Student1: {student1.display_info()}")
print(f"Student2: {student2.display_info()}")
print(f"Class variable access: {Student.get_student_count()}")

# Modifying class variable
Student.school = "Python Academy"
print(f"After changing school: {student1.display_info()}")

# ============================================================================
# SECTION 2: INHERITANCE AND POLYMORPHISM
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 2: INHERITANCE AND POLYMORPHISM")
print("=" * 50)

# Question 4: Inheritance in Python
print("\n4. How does inheritance work in Python?")
print("-" * 41)
print("""
Inheritance allows a class to inherit attributes and methods from another class:

TYPES OF INHERITANCE:
• Single Inheritance: Child inherits from one parent
• Multiple Inheritance: Child inherits from multiple parents
• Multilevel Inheritance: Chain of inheritance
• Hierarchical Inheritance: Multiple children from one parent

BENEFITS:
• Code reusability
• Establishes relationships between classes
• Enables polymorphism
• Reduces code duplication
""")

# Single Inheritance Example
class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def speak(self):
        return f"{self.name} makes a sound."
    
    def info(self):
        return f"{self.name} is a {self.species}."

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Dog")  # Call parent constructor
        self.breed = breed
    
    def speak(self):  # Override parent method
        return f"{self.name} barks!"
    
    def fetch(self):
        return f"{self.name} fetches the ball!"

class Cat(Animal):
    def __init__(self, name, color):
        super().__init__(name, "Cat")
        self.color = color
    
    def speak(self):
        return f"{self.name} meows!"
    
    def climb(self):
        return f"{self.name} climbs the tree!"

print("Single Inheritance Example:")
dog = Dog("Buddy", "Golden Retriever")
cat = Cat("Whiskers", "Orange")

print(f"Dog info: {dog.info()}")
print(f"Dog speak: {dog.speak()}")
print(f"Dog specific: {dog.fetch()}")

print(f"Cat info: {cat.info()}")
print(f"Cat speak: {cat.speak()}")
print(f"Cat specific: {cat.climb()}")

# Question 5: Multiple Inheritance
print("\n5. What is multiple inheritance in Python?")
print("-" * 44)
print("""
Multiple inheritance allows a class to inherit from multiple parent classes:

SYNTAX:
class Child(Parent1, Parent2, Parent3):
    pass

METHOD RESOLUTION ORDER (MRO):
• Python uses C3 linearization algorithm
• Determines which method to call when conflicts exist
• Use ClassName.__mro__ to see the order
• Use super() to call parent methods properly
""")

# Multiple Inheritance Example
class Flyable:
    def fly(self):
        return "Flying high!"

class Swimmable:
    def swim(self):
        return "Swimming gracefully!"

class Duck(Animal, Flyable, Swimmable):
    def __init__(self, name):
        super().__init__(name, "Duck")
    
    def speak(self):
        return f"{self.name} quacks!"

print("Multiple Inheritance Example:")
duck = Duck("Donald")
print(f"Duck info: {duck.info()}")
print(f"Duck speak: {duck.speak()}")
print(f"Duck fly: {duck.fly()}")
print(f"Duck swim: {duck.swim()}")
print(f"MRO: {Duck.__mro__}")

# Question 6: Polymorphism
print("\n6. What is polymorphism in Python?")
print("-" * 37)
print("""
Polymorphism allows different classes to have methods with the same name
but different implementations:

TYPES:
• Method Overriding: Child class provides specific implementation
• Duck Typing: If it behaves like a duck, it's a duck
• Operator Overloading: Customize operator behavior

BENEFITS:
• Write flexible, reusable code
• Same interface, different behaviors
• Easier to extend and maintain code
""")

# Polymorphism Example
def animal_sound(animal):
    """Function that works with any animal object"""
    return animal.speak()

def animal_info(animals):
    """Process a list of different animals"""
    for animal in animals:
        print(f"- {animal.speak()}")

print("Polymorphism Example:")
animals = [dog, cat, duck]
print("All animals speaking:")
animal_info(animals)

# Question 7: Method Overriding
print("\n7. What is method overriding in Python?")
print("-" * 41)
print("""
Method overriding occurs when a child class provides a specific implementation
of a method that exists in the parent class:

CHARACTERISTICS:
• Child method has same name as parent method
• Child method replaces parent method behavior
• Use super() to call parent method if needed
• Enables polymorphism
""")

class Shape:
    def __init__(self, name):
        self.name = name
    
    def area(self):
        return "Area calculation not implemented"
    
    def perimeter(self):
        return "Perimeter calculation not implemented"

class Rectangle(Shape):
    def __init__(self, width, height):
        super().__init__("Rectangle")
        self.width = width
        self.height = height
    
    def area(self):  # Override parent method
        return self.width * self.height
    
    def perimeter(self):  # Override parent method
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius):
        super().__init__("Circle")
        self.radius = radius
    
    def area(self):  # Override parent method
        return 3.14159 * self.radius ** 2
    
    def perimeter(self):  # Override parent method
        return 2 * 3.14159 * self.radius

print("Method Overriding Example:")
rectangle = Rectangle(5, 3)
circle = Circle(4)

shapes = [rectangle, circle]
for shape in shapes:
    print(f"{shape.name} - Area: {shape.area():.2f}, Perimeter: {shape.perimeter():.2f}")

# ============================================================================
# SECTION 3: ENCAPSULATION AND ABSTRACTION
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 3: ENCAPSULATION AND ABSTRACTION")
print("=" * 50)

# Question 8: Encapsulation
print("\n8. What is encapsulation in Python?")
print("-" * 37)
print("""
Encapsulation is the bundling of data and methods that operate on that data
within a single unit (class), and restricting access to some components:

ACCESS MODIFIERS:
• Public: No underscore prefix (accessible everywhere)
• Protected: Single underscore prefix _ (internal use)
• Private: Double underscore prefix __ (name mangling)

BENEFITS:
• Data hiding and security
• Control access to object internals
• Prevent accidental modification
• Maintain object state integrity
""")

class BankAccount:
    def __init__(self, account_number, initial_balance=0):
        self.account_number = account_number  # Public
        self._balance = initial_balance        # Protected
        self.__pin = "1234"                   # Private
    
    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
            return f"Deposited ${amount}. New balance: ${self._balance}"
        return "Invalid deposit amount"
    
    def withdraw(self, amount, pin):
        if pin != self.__pin:
            return "Invalid PIN"
        if amount > self._balance:
            return "Insufficient funds"
        if amount > 0:
            self._balance -= amount
            return f"Withdrew ${amount}. New balance: ${self._balance}"
        return "Invalid withdrawal amount"
    
    def get_balance(self):
        return self._balance
    
    def _internal_method(self):  # Protected method
        return "This is an internal method"
    
    def __private_method(self):  # Private method
        return "This is a private method"

print("Encapsulation Example:")
account = BankAccount("12345", 1000)
print(f"Account: {account.account_number}")
print(account.deposit(500))
print(account.withdraw(200, "1234"))
print(f"Balance: ${account.get_balance()}")

# Accessing protected member (possible but not recommended)
print(f"Protected balance: ${account._balance}")

# Accessing private member (name mangling)
try:
    print(account.__pin)
except AttributeError:
    print("Cannot access private member directly")

# Question 9: Abstraction
print("\n9. What is abstraction in Python?")
print("-" * 35)
print("""
Abstraction hides complex implementation details and shows only essential features:

IMPLEMENTATION:
• Abstract Base Classes (ABC) module
• @abstractmethod decorator
• Cannot instantiate abstract classes
• Child classes must implement abstract methods

BENEFITS:
• Hide complex implementation
• Define interface contracts
• Ensure consistent behavior
• Focus on what, not how
""")

from abc import ABC, abstractmethod

class Vehicle(ABC):
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model
    
    @abstractmethod
    def start_engine(self):
        pass
    
    @abstractmethod
    def stop_engine(self):
        pass
    
    def get_info(self):  # Concrete method
        return f"{self.brand} {self.model}"

class Car(Vehicle):
    def start_engine(self):
        return f"{self.get_info()} engine started with key"
    
    def stop_engine(self):
        return f"{self.get_info()} engine stopped"

class Motorcycle(Vehicle):
    def start_engine(self):
        return f"{self.get_info()} engine started with kick"
    
    def stop_engine(self):
        return f"{self.get_info()} engine stopped"

print("Abstraction Example:")
car = Car("Toyota", "Camry")
motorcycle = Motorcycle("Honda", "CBR")

vehicles = [car, motorcycle]
for vehicle in vehicles:
    print(f"Start: {vehicle.start_engine()}")
    print(f"Stop: {vehicle.stop_engine()}")

# Cannot instantiate abstract class
try:
    # vehicle = Vehicle("Generic", "Model")  # This would raise TypeError
    pass
except TypeError as e:
    print(f"Cannot instantiate abstract class: {e}")

# ============================================================================
# SECTION 4: SPECIAL METHODS AND OPERATOR OVERLOADING
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 4: SPECIAL METHODS AND OPERATOR OVERLOADING")
print("=" * 50)

# Question 10: Magic Methods
print("\n10. What are magic methods (dunder methods) in Python?")
print("-" * 56)
print("""
Magic methods are special methods with double underscores (dunder):

COMMON MAGIC METHODS:
• __init__: Constructor
• __str__: String representation for users
• __repr__: String representation for developers
• __len__: Length of object
• __eq__: Equality comparison
• __lt__, __gt__: Less than, greater than
• __add__, __sub__: Addition, subtraction
• __getitem__, __setitem__: Index access

BENEFITS:
• Make objects behave like built-in types
• Enable operator overloading
• Integrate with Python's built-in functions
""")

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __repr__(self):
        return f"Vector(x={self.x}, y={self.y})"
    
    def __eq__(self, other):
        if isinstance(other, Vector):
            return self.x == other.x and self.y == other.y
        return False
    
    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        raise TypeError("Can only add Vector to Vector")
    
    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y)
        raise TypeError("Can only subtract Vector from Vector")
    
    def __len__(self):
        return int((self.x**2 + self.y**2)**0.5)
    
    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError("Vector index out of range")

print("Magic Methods Example:")
v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(f"v1: {v1}")
print(f"v2: {v2}")
print(f"v1 + v2: {v1 + v2}")
print(f"v1 - v2: {v1 - v2}")
print(f"v1 == v2: {v1 == v2}")
print(f"len(v1): {len(v1)}")
print(f"v1[0]: {v1[0]}, v1[1]: {v1[1]}")

# Question 11: Property Decorators
print("\n11. What are property decorators in Python?")
print("-" * 44)
print("""
Property decorators allow you to define methods that can be accessed like attributes:

DECORATORS:
• @property: Creates a getter method
• @method_name.setter: Creates a setter method
• @method_name.deleter: Creates a deleter method

BENEFITS:
• Control access to attributes
• Add validation to attribute setting
• Compute attributes dynamically
• Maintain backward compatibility
""")

class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature cannot be below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        return (self._celsius * 9/5) + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9
    
    @property
    def kelvin(self):
        return self._celsius + 273.15
    
    @kelvin.setter
    def kelvin(self, value):
        self.celsius = value - 273.15

print("Property Decorators Example:")
temp = Temperature(25)
print(f"Celsius: {temp.celsius}°C")
print(f"Fahrenheit: {temp.fahrenheit}°F")
print(f"Kelvin: {temp.kelvin}K")

# Setting temperature via different units
temp.fahrenheit = 100
print(f"After setting to 100°F: {temp.celsius}°C")

temp.kelvin = 300
print(f"After setting to 300K: {temp.celsius}°C")

# ============================================================================
# SECTION 5: CLASS METHODS AND STATIC METHODS
# ============================================================================

print("\n" + "=" * 50)
print("SECTION 5: CLASS METHODS AND STATIC METHODS")
print("=" * 50)

# Question 12: Class Methods vs Static Methods
print("\n12. What's the difference between @classmethod and @staticmethod?")
print("-" * 65)
print("""
@CLASSMETHOD:
• First parameter is 'cls' (reference to class)
• Can access and modify class variables
• Can create alternative constructors
• Called on class or instance

@STATICMETHOD:
• No special first parameter
• Cannot access class or instance variables
• Utility functions related to the class
• Called on class or instance

INSTANCE METHOD:
• First parameter is 'self' (reference to instance)
• Can access instance and class variables
• Called on instance only
""")

class MathUtils:
    pi = 3.14159
    calculations_count = 0
    
    def __init__(self, name):
        self.name = name
    
    # Instance method
    def instance_info(self):
        return f"Calculator: {self.name}"
    
    # Class method
    @classmethod
    def increment_count(cls):
        cls.calculations_count += 1
        return cls.calculations_count
    
    @classmethod
    def from_string(cls, calc_string):
        """Alternative constructor"""
        name = calc_string.split('-')[1]
        return cls(name)
    
    # Static method
    @staticmethod
    def add(a, b):
        MathUtils.increment_count()
        return a + b
    
    @staticmethod
    def multiply(a, b):
        MathUtils.increment_count()
        return a * b
    
    @staticmethod
    def circle_area(radius):
        MathUtils.increment_count()
        return MathUtils.pi * radius ** 2

print("Class and Static Methods Example:")

# Creating instances
calc1 = MathUtils("Scientific")
calc2 = MathUtils.from_string("calc-Basic")  # Using class method constructor

print(f"Calc1: {calc1.instance_info()}")
print(f"Calc2: {calc2.instance_info()}")

# Using static methods
print(f"Add: {MathUtils.add(5, 3)}")
print(f"Multiply: {calc1.multiply(4, 6)}")  # Can call on instance too
print(f"Circle area: {MathUtils.circle_area(5)}")

print(f"Total calculations: {MathUtils.calculations_count}")

# Question 13: Composition vs Inheritance
print("\n13. What's the difference between composition and inheritance?")
print("-" * 63)
print("""
INHERITANCE (IS-A relationship):
• Child class extends parent class
• Inherits all parent methods and attributes
• Tight coupling between classes
• Use when there's a clear hierarchical relationship

COMPOSITION (HAS-A relationship):
• Object contains other objects
• Delegates functionality to contained objects
• Loose coupling between classes
• More flexible and maintainable
""")

# Inheritance Example
class Engine:
    def __init__(self, horsepower):
        self.horsepower = horsepower
    
    def start(self):
        return f"Engine with {self.horsepower}HP started"
    
    def stop(self):
        return "Engine stopped"

class Wheels:
    def __init__(self, count):
        self.count = count
    
    def rotate(self):
        return f"{self.count} wheels rotating"

# Composition Example - Car HAS-A Engine and Wheels
class Car:
    def __init__(self, brand, model, horsepower, wheel_count=4):
        self.brand = brand
        self.model = model
        self.engine = Engine(horsepower)  # Composition
        self.wheels = Wheels(wheel_count)  # Composition
        self.is_running = False
    
    def start(self):
        if not self.is_running:
            self.is_running = True
            return f"{self.brand} {self.model}: {self.engine.start()}"
        return "Car is already running"
    
    def drive(self):
        if self.is_running:
            return f"{self.brand} {self.model}: {self.wheels.rotate()}"
        return "Start the car first"
    
    def stop(self):
        if self.is_running:
            self.is_running = False
            return f"{self.brand} {self.model}: {self.engine.stop()}"
        return "Car is already stopped"

print("Composition Example:")
my_car = Car("Toyota", "Camry", 200)
print(my_car.start())
print(my_car.drive())
print(my_car.stop())

print("\n" + "=" * 80)
print("END OF OBJECT-ORIENTED PROGRAMMING SECTION")
print("Continue with advanced Python topics in the next file...")
print("=" * 80)
