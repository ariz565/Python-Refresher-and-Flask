grades = [35, 67, 98, 100, 100]
total = 0
amount = len(grades) # length of the list
for grade in grades:
    total += grade
print(total / amount)