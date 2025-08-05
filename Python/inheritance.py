# ===============================================================================
# PYTHON OOP PILLAR 1: INHERITANCE
# Real-Life Examples & Complete Mastery Guide
# ===============================================================================

"""
COMPREHENSIVE INHERITANCE COVERAGE:
==================================
1. Basic Inheritance Concepts
2. Single Inheritance
3. Multiple Inheritance & MRO
4. Method Overriding & super()
5. Abstract Base Classes (ABC)
6. Mixins & Composition vs Inheritance
7. Diamond Problem Resolution
8. Advanced Inheritance Patterns
9. Real-World Applications
10. Best Practices & Common Pitfalls
"""

import abc
from abc import ABC, abstractmethod
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
import logging

# ===============================================================================
# 1. BASIC INHERITANCE CONCEPTS
# ===============================================================================

print("=" * 80)
print("1. BASIC INHERITANCE CONCEPTS")
print("=" * 80)

print("\n--- Real-Life Example: Animal Kingdom Hierarchy ---")

class Animal:
    """Base class representing any animal"""
    
    def __init__(self, name: str, species: str, age: int):
        self.name = name
        self.species = species
        self.age = age
        self.energy = 100
        self.is_sleeping = False
        self.location = "Unknown"
    
    def eat(self, food: str):
        """Basic eating behavior"""
        self.energy = min(100, self.energy + 20)
        print(f"{self.name} the {self.species} eats {food}. Energy: {self.energy}")
    
    def sleep(self):
        """Basic sleeping behavior"""
        self.is_sleeping = True
        self.energy = 100
        print(f"{self.name} is sleeping and restored energy to {self.energy}")
    
    def wake_up(self):
        """Wake up from sleep"""
        self.is_sleeping = False
        print(f"{self.name} wakes up!")
    
    def move(self, new_location: str):
        """Basic movement"""
        if self.is_sleeping:
            print(f"{self.name} is sleeping and cannot move")
            return
        
        self.energy -= 10
        self.location = new_location
        print(f"{self.name} moves to {new_location}. Energy: {self.energy}")
    
    def make_sound(self):
        """Basic sound - to be overridden by subclasses"""
        print(f"{self.name} makes a generic animal sound")
    
    def get_info(self):
        """Get animal information"""
        return {
            'name': self.name,
            'species': self.species,
            'age': self.age,
            'energy': self.energy,
            'is_sleeping': self.is_sleeping,
            'location': self.location
        }

# Child classes inherit from Animal
class Dog(Animal):
    """Dog class inheriting from Animal"""
    
    def __init__(self, name: str, breed: str, age: int):
        # Call parent constructor
        super().__init__(name, "Dog", age)
        self.breed = breed
        self.is_trained = False
        self.tricks = []
    
    def make_sound(self):
        """Override parent method"""
        print(f"{self.name} barks: Woof! Woof!")
    
    def fetch(self, item: str):
        """Dog-specific behavior"""
        if self.is_sleeping:
            print(f"{self.name} is sleeping and cannot fetch")
            return
        
        self.energy -= 15
        print(f"{self.name} fetches the {item}! Energy: {self.energy}")
    
    def learn_trick(self, trick: str):
        """Learn a new trick"""
        if trick not in self.tricks:
            self.tricks.append(trick)
            print(f"{self.name} learned a new trick: {trick}")
            self.is_trained = True
        else:
            print(f"{self.name} already knows {trick}")
    
    def perform_trick(self, trick: str):
        """Perform a learned trick"""
        if trick in self.tricks:
            self.energy -= 5
            print(f"{self.name} performs {trick}! Energy: {self.energy}")
        else:
            print(f"{self.name} doesn't know how to {trick}")

class Cat(Animal):
    """Cat class inheriting from Animal"""
    
    def __init__(self, name: str, breed: str, age: int):
        super().__init__(name, "Cat", age)
        self.breed = breed
        self.lives_remaining = 9
        self.is_purring = False
    
    def make_sound(self):
        """Override parent method"""
        print(f"{self.name} meows: Meow! Meow!")
    
    def purr(self):
        """Cat-specific behavior"""
        self.is_purring = True
        print(f"{self.name} is purring contentedly")
    
    def climb(self, height: str):
        """Cat-specific climbing behavior"""
        if self.is_sleeping:
            print(f"{self.name} is sleeping and cannot climb")
            return
        
        self.energy -= 20
        print(f"{self.name} climbs to {height}. Energy: {self.energy}")
    
    def hunt(self, prey: str):
        """Cat-specific hunting behavior"""
        if self.energy < 30:
            print(f"{self.name} is too tired to hunt")
            return
        
        self.energy -= 25
        print(f"{self.name} hunts {prey}. Energy: {self.energy}")

# Example usage
print("Creating animals:")
generic_animal = Animal("Rex", "Unknown", 5)
dog = Dog("Buddy", "Golden Retriever", 3)
cat = Cat("Whiskers", "Persian", 2)

print(f"Dog info: {dog.get_info()}")

# Demonstrate inheritance - all can use parent methods
generic_animal.eat("food")
dog.eat("dog treats")
cat.eat("fish")

# Demonstrate method overriding
generic_animal.make_sound()
dog.make_sound()
cat.make_sound()

# Demonstrate child-specific methods
dog.fetch("ball")
dog.learn_trick("sit")
dog.learn_trick("roll over")
dog.perform_trick("sit")

cat.purr()
cat.climb("tree")
cat.hunt("mouse")

# ===============================================================================
# 2. SINGLE INHERITANCE CHAIN
# ===============================================================================

print("\n" + "=" * 80)
print("2. SINGLE INHERITANCE CHAIN")
print("=" * 80)

print("\n--- Real-Life Example: Vehicle Hierarchy ---")

class Vehicle:
    """Base vehicle class"""
    
    def __init__(self, make: str, model: str, year: int):
        self.make = make
        self.model = model
        self.year = year
        self.fuel_level = 100
        self.is_running = False
        self.mileage = 0
        self.maintenance_due = False
    
    def start_engine(self):
        """Start the vehicle"""
        if not self.is_running:
            self.is_running = True
            print(f"{self.make} {self.model} engine started")
    
    def stop_engine(self):
        """Stop the vehicle"""
        if self.is_running:
            self.is_running = False
            print(f"{self.make} {self.model} engine stopped")
    
    def refuel(self):
        """Refuel the vehicle"""
        self.fuel_level = 100
        print(f"{self.make} {self.model} refueled to 100%")
    
    def drive(self, distance: float):
        """Basic driving method"""
        if not self.is_running:
            print("Cannot drive - engine not running")
            return
        
        fuel_consumption = distance * 0.1
        if fuel_consumption > self.fuel_level:
            print("Not enough fuel!")
            return
        
        self.mileage += distance
        self.fuel_level -= fuel_consumption
        print(f"Drove {distance} miles. Total mileage: {self.mileage}")
    
    def get_vehicle_info(self):
        """Get basic vehicle information"""
        return {
            'make': self.make,
            'model': self.model,
            'year': self.year,
            'fuel_level': self.fuel_level,
            'mileage': self.mileage,
            'is_running': self.is_running
        }

class MotorVehicle(Vehicle):
    """Motor vehicle - inherits from Vehicle"""
    
    def __init__(self, make: str, model: str, year: int, engine_type: str):
        super().__init__(make, model, year)
        self.engine_type = engine_type
        self.engine_temperature = 70  # Fahrenheit
        self.oil_level = 100
    
    def check_engine(self):
        """Check engine status"""
        if self.oil_level < 20:
            self.maintenance_due = True
            print(f"Warning: {self.make} {self.model} needs oil change!")
        
        if self.engine_temperature > 220:
            print(f"Warning: {self.make} {self.model} engine overheating!")
    
    def change_oil(self):
        """Change engine oil"""
        self.oil_level = 100
        self.maintenance_due = False
        print(f"{self.make} {self.model} oil changed")
    
    def drive(self, distance: float):
        """Override parent drive method with engine checks"""
        # Call parent method first
        super().drive(distance)
        
        # Add motor vehicle specific behavior
        if self.is_running and distance > 0:
            self.engine_temperature += distance * 0.5
            self.oil_level -= distance * 0.05
            self.check_engine()

class Car(MotorVehicle):
    """Car - inherits from MotorVehicle"""
    
    def __init__(self, make: str, model: str, year: int, engine_type: str, doors: int):
        super().__init__(make, model, year, engine_type)
        self.doors = doors
        self.passengers = 0
        self.max_passengers = doors + 1  # Rough estimate
        self.air_conditioning = False
        self.radio_on = False
    
    def add_passenger(self, count: int = 1):
        """Add passengers to the car"""
        if self.passengers + count <= self.max_passengers:
            self.passengers += count
            print(f"Added {count} passenger(s). Total: {self.passengers}")
        else:
            print(f"Cannot add {count} passenger(s). Max capacity: {self.max_passengers}")
    
    def remove_passenger(self, count: int = 1):
        """Remove passengers from the car"""
        self.passengers = max(0, self.passengers - count)
        print(f"Removed {count} passenger(s). Total: {self.passengers}")
    
    def turn_on_ac(self):
        """Turn on air conditioning"""
        if self.is_running:
            self.air_conditioning = True
            print(f"{self.make} {self.model} AC turned on")
        else:
            print("Cannot turn on AC - engine not running")
    
    def turn_on_radio(self):
        """Turn on radio"""
        self.radio_on = True
        print(f"{self.make} {self.model} radio turned on")
    
    def drive(self, distance: float):
        """Override with car-specific behavior"""
        # Call parent method
        super().drive(distance)
        
        # Add car-specific behavior
        if self.air_conditioning and self.is_running:
            # AC uses more fuel
            self.fuel_level -= distance * 0.02
    
    def get_car_info(self):
        """Get comprehensive car information"""
        info = self.get_vehicle_info()
        info.update({
            'engine_type': self.engine_type,
            'doors': self.doors,
            'passengers': self.passengers,
            'max_passengers': self.max_passengers,
            'air_conditioning': self.air_conditioning,
            'radio_on': self.radio_on,
            'engine_temperature': self.engine_temperature,
            'oil_level': self.oil_level
        })
        return info

class SportsCar(Car):
    """Sports car - inherits from Car"""
    
    def __init__(self, make: str, model: str, year: int, engine_type: str, horsepower: int):
        super().__init__(make, model, year, engine_type, 2)  # Sports cars typically have 2 doors
        self.horsepower = horsepower
        self.turbo_mode = False
        self.top_speed = horsepower / 5  # Simplified calculation
        self.max_passengers = 2  # Override for sports car
    
    def activate_turbo(self):
        """Activate turbo mode"""
        if self.is_running:
            self.turbo_mode = True
            print(f"{self.make} {self.model} TURBO ACTIVATED!")
        else:
            print("Cannot activate turbo - engine not running")
    
    def deactivate_turbo(self):
        """Deactivate turbo mode"""
        self.turbo_mode = False
        print(f"{self.make} {self.model} turbo deactivated")
    
    def drive(self, distance: float):
        """Override with sports car specific behavior"""
        # Call parent method
        super().drive(distance)
        
        # Sports car specific behavior
        if self.turbo_mode and self.is_running:
            # Turbo mode uses more fuel and increases temperature
            self.fuel_level -= distance * 0.05
            self.engine_temperature += distance * 0.8
            print(f"TURBO MODE: Extra performance at cost of fuel and heat!")
    
    def get_sports_car_info(self):
        """Get comprehensive sports car information"""
        info = self.get_car_info()
        info.update({
            'horsepower': self.horsepower,
            'turbo_mode': self.turbo_mode,
            'top_speed': self.top_speed
        })
        return info

# Example usage - Inheritance Chain
print("Creating vehicles in inheritance chain:")

# Base vehicle
basic_vehicle = Vehicle("Generic", "Vehicle", 2020)
basic_vehicle.start_engine()
basic_vehicle.drive(50)

# Motor vehicle
motor_vehicle = MotorVehicle("Ford", "Engine", 2021, "V6")
motor_vehicle.start_engine()
motor_vehicle.drive(100)

# Car
car = Car("Toyota", "Camry", 2022, "4-cylinder", 4)
car.start_engine()
car.add_passenger(3)
car.turn_on_ac()
car.drive(75)
print(f"Car info: {car.get_car_info()}")

# Sports car
sports_car = SportsCar("Ferrari", "F8", 2023, "V8 Twin-Turbo", 710)
sports_car.start_engine()
sports_car.add_passenger(1)
sports_car.activate_turbo()
sports_car.drive(50)
print(f"Sports car info: {sports_car.get_sports_car_info()}")

# Demonstrate method resolution - each level adds functionality
print(f"\nInheritance chain for SportsCar:")
print(f"SportsCar -> Car -> MotorVehicle -> Vehicle -> object")
print(f"MRO: {SportsCar.__mro__}")

# ===============================================================================
# 3. MULTIPLE INHERITANCE & MRO (Method Resolution Order)
# ===============================================================================

print("\n" + "=" * 80)
print("3. MULTIPLE INHERITANCE & MRO")
print("=" * 80)

print("\n--- Real-Life Example: Smart Device Ecosystem ---")

class Device:
    """Base device class"""
    
    def __init__(self, name: str, brand: str):
        self.name = name
        self.brand = brand
        self.is_on = False
        self.battery_level = 100
    
    def power_on(self):
        """Turn on the device"""
        self.is_on = True
        print(f"{self.name} is now ON")
    
    def power_off(self):
        """Turn off the device"""
        self.is_on = False
        print(f"{self.name} is now OFF")
    
    def check_battery(self):
        """Check battery level"""
        print(f"{self.name} battery: {self.battery_level}%")
        return self.battery_level

class NetworkCapable:
    """Mixin for network capabilities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_connected = False
        self.network_name = None
        self.ip_address = None
    
    def connect_to_network(self, network_name: str):
        """Connect to a network"""
        if hasattr(self, 'is_on') and not self.is_on:
            print("Cannot connect - device is off")
            return
        
        self.is_connected = True
        self.network_name = network_name
        self.ip_address = f"192.168.1.{hash(network_name) % 255}"
        print(f"Connected to {network_name} with IP {self.ip_address}")
    
    def disconnect_from_network(self):
        """Disconnect from network"""
        if self.is_connected:
            print(f"Disconnected from {self.network_name}")
            self.is_connected = False
            self.network_name = None
            self.ip_address = None

class AudioCapable:
    """Mixin for audio capabilities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.volume = 50
        self.is_muted = False
    
    def play_audio(self, audio_file: str):
        """Play audio"""
        if hasattr(self, 'is_on') and not self.is_on:
            print("Cannot play audio - device is off")
            return
        
        if self.is_muted:
            print(f"Playing {audio_file} (MUTED)")
        else:
            print(f"Playing {audio_file} at volume {self.volume}")
    
    def set_volume(self, volume: int):
        """Set volume level"""
        self.volume = max(0, min(100, volume))
        print(f"Volume set to {self.volume}")
    
    def mute(self):
        """Mute audio"""
        self.is_muted = True
        print("Audio muted")
    
    def unmute(self):
        """Unmute audio"""
        self.is_muted = False
        print("Audio unmuted")

class VideoCapable:
    """Mixin for video capabilities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolution = "1080p"
        self.brightness = 50
    
    def play_video(self, video_file: str):
        """Play video"""
        if hasattr(self, 'is_on') and not self.is_on:
            print("Cannot play video - device is off")
            return
        
        print(f"Playing {video_file} at {self.resolution} resolution")
    
    def set_resolution(self, resolution: str):
        """Set video resolution"""
        self.resolution = resolution
        print(f"Resolution set to {resolution}")
    
    def adjust_brightness(self, brightness: int):
        """Adjust screen brightness"""
        self.brightness = max(0, min(100, brightness))
        print(f"Brightness adjusted to {self.brightness}")

class VoiceAssistant:
    """Mixin for voice assistant capabilities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wake_word = "Hey Assistant"
        self.listening = False
    
    def listen_for_commands(self):
        """Start listening for voice commands"""
        if hasattr(self, 'is_on') and not self.is_on:
            print("Cannot listen - device is off")
            return
        
        self.listening = True
        print(f"Listening for '{self.wake_word}'...")
    
    def process_command(self, command: str):
        """Process voice command"""
        if not self.listening:
            print("Not listening for commands")
            return
        
        print(f"Processing command: '{command}'")
        
        # Simple command processing
        if "volume up" in command.lower():
            if hasattr(self, 'set_volume'):
                self.set_volume(self.volume + 10)
        elif "volume down" in command.lower():
            if hasattr(self, 'set_volume'):
                self.set_volume(self.volume - 10)
        elif "mute" in command.lower():
            if hasattr(self, 'mute'):
                self.mute()
        else:
            print(f"Command '{command}' not recognized")

# Multiple inheritance examples
class SmartSpeaker(Device, NetworkCapable, AudioCapable, VoiceAssistant):
    """Smart speaker with multiple capabilities"""
    
    def __init__(self, name: str, brand: str):
        super().__init__(name, brand)
        self.playlist = []
        print(f"Smart Speaker {name} initialized")
    
    def add_to_playlist(self, song: str):
        """Add song to playlist"""
        self.playlist.append(song)
        print(f"Added '{song}' to playlist")
    
    def play_playlist(self):
        """Play the entire playlist"""
        if not self.playlist:
            print("Playlist is empty")
            return
        
        for song in self.playlist:
            self.play_audio(song)

class SmartTV(Device, NetworkCapable, AudioCapable, VideoCapable, VoiceAssistant):
    """Smart TV with multiple capabilities"""
    
    def __init__(self, name: str, brand: str, screen_size: int):
        super().__init__(name, brand)
        self.screen_size = screen_size
        self.current_channel = 1
        self.apps = ["Netflix", "YouTube", "Prime Video"]
        print(f"Smart TV {name} ({screen_size}') initialized")
    
    def change_channel(self, channel: int):
        """Change TV channel"""
        if not self.is_on:
            print("Cannot change channel - TV is off")
            return
        
        self.current_channel = channel
        print(f"Changed to channel {channel}")
    
    def open_app(self, app_name: str):
        """Open streaming app"""
        if not self.is_on:
            print("Cannot open app - TV is off")
            return
        
        if app_name in self.apps:
            print(f"Opening {app_name}")
        else:
            print(f"App '{app_name}' not available")

class Smartphone(Device, NetworkCapable, AudioCapable, VideoCapable, VoiceAssistant):
    """Smartphone with all capabilities"""
    
    def __init__(self, name: str, brand: str, storage_gb: int):
        super().__init__(name, brand)
        self.storage_gb = storage_gb
        self.contacts = []
        self.apps_installed = ["Phone", "Messages", "Camera", "Music", "Video"]
        print(f"Smartphone {name} ({storage_gb}GB) initialized")
    
    def make_call(self, contact: str):
        """Make a phone call"""
        if not self.is_on:
            print("Cannot make call - phone is off")
            return
        
        if not self.is_connected:
            print("Cannot make call - no network connection")
            return
        
        print(f"Calling {contact}...")
    
    def install_app(self, app_name: str):
        """Install a new app"""
        if app_name not in self.apps_installed:
            self.apps_installed.append(app_name)
            print(f"Installed {app_name}")
        else:
            print(f"{app_name} is already installed")

# Example usage - Multiple Inheritance
print("Creating smart devices with multiple inheritance:")

# Smart Speaker
speaker = SmartSpeaker("Echo Dot", "Amazon")
speaker.power_on()
speaker.connect_to_network("HomeWiFi")
speaker.add_to_playlist("Song 1")
speaker.add_to_playlist("Song 2")
speaker.listen_for_commands()
speaker.process_command("volume up")
speaker.play_playlist()

print(f"\nSmart Speaker MRO: {SmartSpeaker.__mro__}")

# Smart TV
tv = SmartTV("Smart TV", "Samsung", 55)
tv.power_on()
tv.connect_to_network("HomeWiFi")
tv.set_volume(30)
tv.set_resolution("4K")
tv.change_channel(5)
tv.open_app("Netflix")

print(f"\nSmart TV MRO: {SmartTV.__mro__}")

# Smartphone
phone = Smartphone("iPhone", "Apple", 256)
phone.power_on()
phone.connect_to_network("CellularNetwork")
phone.install_app("Instagram")
phone.make_call("John Doe")
phone.listen_for_commands()
phone.process_command("mute")

print(f"\nSmartphone MRO: {Smartphone.__mro__}")

# ===============================================================================
# 4. METHOD OVERRIDING & super()
# ===============================================================================

print("\n" + "=" * 80)
print("4. METHOD OVERRIDING & super()")
print("=" * 80)

print("\n--- Real-Life Example: Payment Processing System ---")

class PaymentProcessor:
    """Base payment processor"""
    
    def __init__(self, merchant_id: str):
        self.merchant_id = merchant_id
        self.transaction_fee_rate = 0.03  # 3%
        self.transactions = []
    
    def validate_payment(self, amount: float, currency: str = "USD"):
        """Validate payment details"""
        if amount <= 0:
            raise ValueError("Amount must be positive")
        
        if currency not in ["USD", "EUR", "GBP"]:
            raise ValueError("Unsupported currency")
        
        print(f"Payment validation passed for {amount} {currency}")
        return True
    
    def calculate_fee(self, amount: float):
        """Calculate transaction fee"""
        return amount * self.transaction_fee_rate
    
    def process_payment(self, amount: float, currency: str = "USD", **kwargs):
        """Process payment - base implementation"""
        # Validate payment
        self.validate_payment(amount, currency)
        
        # Calculate fee
        fee = self.calculate_fee(amount)
        net_amount = amount - fee
        
        # Create transaction record
        transaction = {
            'amount': amount,
            'currency': currency,
            'fee': fee,
            'net_amount': net_amount,
            'timestamp': datetime.now(),
            'status': 'pending',
            'processor': 'base'
        }
        
        self.transactions.append(transaction)
        
        print(f"Base payment processing: {amount} {currency}")
        print(f"Fee: {fee:.2f}, Net: {net_amount:.2f}")
        
        return transaction
    
    def get_transaction_history(self):
        """Get transaction history"""
        return self.transactions.copy()

class CreditCardProcessor(PaymentProcessor):
    """Credit card payment processor"""
    
    def __init__(self, merchant_id: str):
        super().__init__(merchant_id)
        self.transaction_fee_rate = 0.025  # 2.5% for credit cards
        self.supported_cards = ["Visa", "MasterCard", "American Express"]
    
    def validate_payment(self, amount: float, currency: str = "USD", **kwargs):
        """Override validation with credit card specific checks"""
        # Call parent validation first
        super().validate_payment(amount, currency)
        
        # Credit card specific validation
        card_number = kwargs.get('card_number', '')
        cvv = kwargs.get('cvv', '')
        expiry = kwargs.get('expiry', '')
        
        if not card_number or len(card_number) < 13:
            raise ValueError("Invalid card number")
        
        if not cvv or len(cvv) != 3:
            raise ValueError("Invalid CVV")
        
        if not expiry:
            raise ValueError("Expiry date required")
        
        print("Credit card validation passed")
        return True
    
    def check_card_type(self, card_number: str):
        """Determine card type"""
        if card_number.startswith('4'):
            return "Visa"
        elif card_number.startswith('5'):
            return "MasterCard"
        elif card_number.startswith('3'):
            return "American Express"
        else:
            return "Unknown"
    
    def process_payment(self, amount: float, currency: str = "USD", **kwargs):
        """Override payment processing with credit card specific logic"""
        
        # Additional credit card validation
        card_number = kwargs.get('card_number', '')
        card_type = self.check_card_type(card_number)
        
        if card_type not in self.supported_cards:
            raise ValueError(f"Unsupported card type: {card_type}")
        
        # Call parent method to handle basic processing
        transaction = super().process_payment(amount, currency, **kwargs)
        
        # Add credit card specific information
        transaction.update({
            'processor': 'credit_card',
            'card_type': card_type,
            'last_four': card_number[-4:],
            'status': 'completed'
        })
        
        print(f"Credit card payment processed via {card_type}")
        return transaction

class PayPalProcessor(PaymentProcessor):
    """PayPal payment processor"""
    
    def __init__(self, merchant_id: str):
        super().__init__(merchant_id)
        self.transaction_fee_rate = 0.029  # 2.9% for PayPal
    
    def validate_payment(self, amount: float, currency: str = "USD", **kwargs):
        """Override validation with PayPal specific checks"""
        # Call parent validation
        super().validate_payment(amount, currency)
        
        # PayPal specific validation
        email = kwargs.get('paypal_email', '')
        if not email or '@' not in email:
            raise ValueError("Valid PayPal email required")
        
        print("PayPal validation passed")
        return True
    
    def process_payment(self, amount: float, currency: str = "USD", **kwargs):
        """Override payment processing with PayPal specific logic"""
        
        # Call parent method for basic processing
        transaction = super().process_payment(amount, currency, **kwargs)
        
        # Add PayPal specific information
        transaction.update({
            'processor': 'paypal',
            'paypal_email': kwargs.get('paypal_email', ''),
            'status': 'completed'
        })
        
        print(f"PayPal payment processed for {kwargs.get('paypal_email', '')}")
        return transaction

class CryptoProcessor(PaymentProcessor):
    """Cryptocurrency payment processor"""
    
    def __init__(self, merchant_id: str):
        super().__init__(merchant_id)
        self.transaction_fee_rate = 0.01  # 1% for crypto
        self.supported_cryptos = ["BTC", "ETH", "LTC"]
    
    def validate_payment(self, amount: float, currency: str = "BTC", **kwargs):
        """Override validation with crypto specific checks"""
        # Don't call parent validation since crypto doesn't use traditional currencies
        if amount <= 0:
            raise ValueError("Amount must be positive")
        
        if currency not in self.supported_cryptos:
            raise ValueError(f"Unsupported cryptocurrency: {currency}")
        
        wallet_address = kwargs.get('wallet_address', '')
        if not wallet_address or len(wallet_address) < 26:
            raise ValueError("Valid wallet address required")
        
        print(f"Cryptocurrency validation passed for {amount} {currency}")
        return True
    
    def calculate_fee(self, amount: float):
        """Override fee calculation for crypto"""
        # Crypto has fixed minimum fee
        calculated_fee = super().calculate_fee(amount)
        minimum_fee = 0.001  # Minimum fee in crypto
        return max(calculated_fee, minimum_fee)
    
    def process_payment(self, amount: float, currency: str = "BTC", **kwargs):
        """Override payment processing with crypto specific logic"""
        
        # Validate crypto payment
        self.validate_payment(amount, currency, **kwargs)
        
        # Calculate crypto fee
        fee = self.calculate_fee(amount)
        net_amount = amount - fee
        
        # Create crypto transaction
        transaction = {
            'amount': amount,
            'currency': currency,
            'fee': fee,
            'net_amount': net_amount,
            'timestamp': datetime.now(),
            'status': 'pending',  # Crypto needs confirmation
            'processor': 'cryptocurrency',
            'wallet_address': kwargs.get('wallet_address', ''),
            'confirmations': 0
        }
        
        self.transactions.append(transaction)
        
        print(f"Cryptocurrency payment processing: {amount} {currency}")
        print(f"Fee: {fee:.6f}, Net: {net_amount:.6f}")
        print("Waiting for blockchain confirmation...")
        
        return transaction
    
    def confirm_transaction(self, transaction_index: int):
        """Confirm crypto transaction"""
        if 0 <= transaction_index < len(self.transactions):
            transaction = self.transactions[transaction_index]
            if transaction['processor'] == 'cryptocurrency':
                transaction['confirmations'] += 1
                if transaction['confirmations'] >= 3:
                    transaction['status'] = 'completed'
                    print(f"Crypto transaction confirmed!")
                else:
                    print(f"Confirmation {transaction['confirmations']}/3")

# Example usage - Method Overriding
print("Creating payment processors:")

# Base processor
base_processor = PaymentProcessor("MERCHANT_001")
base_transaction = base_processor.process_payment(100.0)

# Credit card processor
cc_processor = CreditCardProcessor("MERCHANT_002")
cc_transaction = cc_processor.process_payment(
    150.0, 
    card_number="4111111111111111",
    cvv="123",
    expiry="12/25"
)

# PayPal processor
paypal_processor = PayPalProcessor("MERCHANT_003")
paypal_transaction = paypal_processor.process_payment(
    200.0,
    paypal_email="user@example.com"
)

# Crypto processor
crypto_processor = CryptoProcessor("MERCHANT_004")
crypto_transaction = crypto_processor.process_payment(
    0.005,
    currency="BTC",
    wallet_address="1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
)

# Demonstrate super() usage at multiple levels
print(f"\nTransaction histories:")
print(f"Credit card transactions: {len(cc_processor.get_transaction_history())}")
print(f"PayPal transactions: {len(paypal_processor.get_transaction_history())}")
print(f"Crypto transactions: {len(crypto_processor.get_transaction_history())}")

# Confirm crypto transaction
crypto_processor.confirm_transaction(0)
crypto_processor.confirm_transaction(0)
crypto_processor.confirm_transaction(0)

print("\n" + "=" * 80)
print("INHERITANCE PART 1 COMPLETE!")
print("=" * 80)
print("""
🎯 CONCEPTS COVERED:
✅ Basic Inheritance Concepts
✅ Single Inheritance Chain
✅ Multiple Inheritance & MRO
✅ Method Overriding & super()

📝 REAL-LIFE EXAMPLES:
- Animal Kingdom Hierarchy
- Vehicle Inheritance Chain
- Smart Device Ecosystem (Multiple Inheritance)
- Payment Processing System (Method Overriding)

🚀 NEXT: Continue with inheritance_advanced.py for ABC, Mixins, and Diamond Problem
""")
