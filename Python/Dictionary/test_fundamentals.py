# Testing the enhanced Part 1 - Dictionary Fundamentals
print("Testing Enhanced Dictionary Fundamentals")
print("=" * 50)

# 1. Creation Methods Demo
print("\n1. CREATION METHODS:")
print("-" * 30)

# Various creation methods
dict1 = {"name": "John", "age": 25}
dict2 = dict(name="Alice", age=30)
dict3 = dict([("a", 1), ("b", 2)])
keys = ["x", "y", "z"]
values = [1, 2, 3]
dict4 = dict(zip(keys, values))
dict5 = {x: x**2 for x in range(4)}

print(f"Literal: {dict1}")
print(f"Constructor: {dict2}")
print(f"From tuples: {dict3}")
print(f"From zip: {dict4}")
print(f"Comprehension: {dict5}")

# 2. Data Access Methods
print("\n2. DATA ACCESS:")
print("-" * 30)

student = {
    "name": "Alice Johnson",
    "age": 22,
    "grades": [85, 92, 78],
    "address": {"city": "Boston", "zip": "02101"}
}

print(f"Name: {student['name']}")
print(f"Age with get(): {student.get('age')}")
print(f"Phone (missing): {student.get('phone', 'Not provided')}")
print(f"City (nested): {student['address']['city']}")

# 3. Dictionary Methods Demo
print("\n3. DICTIONARY METHODS:")
print("-" * 30)

sample = {"a": 1, "b": 2, "c": 3}
print(f"Original: {sample}")

# Adding data
sample.setdefault("d", 4)
sample.update({"e": 5, "f": 6})
print(f"After additions: {sample}")

# Views
print(f"Keys: {list(sample.keys())}")
print(f"Values: {list(sample.values())}")
print(f"Items: {list(sample.items())}")

# 4. Sorting Demo
print("\n4. SORTING:")
print("-" * 30)

grades = {"Alice": 85, "Bob": 92, "Charlie": 78, "Diana": 96}
print(f"Original: {grades}")

# Sort by keys
sorted_keys = dict(sorted(grades.items()))
print(f"By keys: {sorted_keys}")

# Sort by values
sorted_values = dict(sorted(grades.items(), key=lambda x: x[1], reverse=True))
print(f"By values (desc): {sorted_values}")

# Top 2 students
top_2 = dict(sorted(grades.items(), key=lambda x: x[1], reverse=True)[:2])
print(f"Top 2: {top_2}")

# 5. Dictionary Merging
print("\n5. MERGING:")
print("-" * 30)

dict_a = {"a": 1, "b": 2}
dict_b = {"c": 3, "d": 4}
dict_c = {"b": 20, "e": 5}

# Different merge methods
merged1 = {**dict_a, **dict_b}
print(f"Unpacking merge: {merged1}")

merged2 = {**dict_a, **dict_b, **dict_c}
print(f"Multiple merge: {merged2}")  # Note: 'b' gets overwritten

# 6. Iteration Patterns
print("\n6. ITERATION:")
print("-" * 30)

demo = {"apple": 5, "banana": 3, "cherry": 8}

print("Key-value iteration:")
for key, value in demo.items():
    print(f"  {key}: {value}")

print("Conditional iteration (value > 4):")
for key, value in demo.items():
    if value > 4:
        print(f"  {key}: {value}")

# 7. Advanced Comprehensions
print("\n7. ADVANCED COMPREHENSIONS:")
print("-" * 30)

# Filter and transform
prices = {"apple": 1.00, "banana": 0.50, "cherry": 2.00}
expensive = {item: price for item, price in prices.items() if price > 0.75}
print(f"Expensive items: {expensive}")

# Discounted prices
discounted = {item: round(price * 0.8, 2) for item, price in prices.items()}
print(f"20% discount: {discounted}")

# Swap keys and values
swapped = {value: key for key, value in demo.items()}
print(f"Swapped: {swapped}")

print("\n" + "=" * 50)
print("Dictionary Fundamentals Demo Complete!")
