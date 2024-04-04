def add(x, y):
    print(x + y)


add(5, 8)
result = add(5, 8)
print(result)  # None

# If we want to get something back from the function, it must return a value.
# All functions return _something_. By default, it's None.

# -- Returning values --


def add(x, y):
    return x + y


add(1, 2)  # Nothing printed out anymore.
result = add(2, 3)
print(result)  # 5

# -- Returning terminates the function --


def add(x, y):
    return
    print(x + y)
    return x + y


result = add(5, 8)  # Nothing printed out
print(result)  # None, as is the first return

# -- Returning with conditionals --


def divide(dividend, divisor):
    if divisor != 0:
        return dividend / divisor
    else:
        return "You fool!"


result = divide(15, 3)
print(result)  # 5

another = divide(15, 0)
print(another)  # You fool!





a = 10
 
def my_function(param_1=a):
    print(param_1)
 
a = 20
my_function()
#In Python, if a function contains multiple return statements, Python will return the value specified in the first return statement it encounters when running the function. The function execution is terminated at that point.
#In Python, if you change the value of a variable that was used as a default value for a parameter in a function definition, the default value of the parameter remains the same. The default value is determined when the function is defined, not when it is called.

def my_function():
    print('Bob')
 
result = my_function()







