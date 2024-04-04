number = 7 
user_input = input("Enter 'y' if you would like to play: ")


# if user_input == 'y':
if user_input in ('y', 'Y'):
    user_number = int(input("Guess our number: "))
    if user_number == number:
        print("You guessed correctly!")
    elif abs(number - user_number) == 1:
        print("You were off by one.")
    else:
        print("Sorry, it's wrong!")
        
# We could also do a transformation instead of checking multiple options.

number = 7
user_input = input("Enter 'y' if you would like to play: ")

if user_input.lower() == "y":
    user_number = int(input("Guess our number: "))
    if user_number == number:
        print("You guessed correctly!")
    elif abs(number - user_number) == 1:
        print("You were off by 1.")
    else:
        print("Sorry, it's wrong!")
        
# the `abs` function returns the absolute value of `-1` which is `1`. Since 1 is present in the list, the expresion will be evaluate to `True`.