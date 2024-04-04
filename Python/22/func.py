def hello():
    print("Hello!")


hello()

# -- Defining vs. calling --
# It's still all sequential!


def user_age_in_seconds():
    user_age = int(input("Enter your age: "))
    age_seconds = user_age * 365 * 24 * 60 * 60
    print(f"Your age in seconds is {age_seconds}.")


print("Welcome to the age in seconds program!")
user_age_in_seconds()

print("Goodbye!")

# -- Don't reuse names --


def print():
    print("Hello, world!")  # Error!


# -- Don't reuse names, it's generally confusing! --
friends = ["Rolf", "Bob"]


def add_friend():
    friend_name = input("Enter your friend name: ")
    friends = friends + [friend_name]  # Another way of adding to a list!


add_friend()
print(friends)  # Always ['Rolf', 'Bob']

# -- Can't call a function before defining it --

# say_hello()


def say_hello():
    print("Hello!")


# -- Remember function body only runs when the function is called --


def add_friend():
    friends.append("Rolf")


friends = []
add_friend()

print(friends)  # [Rolf]

#When you define a variable inside a function with the same name as a global variable in Python, the function creates a new local variable that shadows the global variable. This local variable only exists within the scope of the function.