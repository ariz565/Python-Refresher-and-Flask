"""
=============================================================================
COMPREHENSIVE NUMPY INTERVIEW QUESTIONS
=============================================================================
Created: August 2025
Total Questions: 60
Coverage: Fundamentals, Arrays, Operations, Broadcasting, Performance
=============================================================================
"""

import numpy as np
import time
import matplotlib.pyplot as plt

print("="*70)
print("NUMPY INTERVIEW QUESTIONS - COMPREHENSIVE COLLECTION")
print("="*70)

# =============================================================================
# SECTION 1: NUMPY FUNDAMENTALS (Questions 1-20)
# =============================================================================

print("\n" + "="*50)
print("SECTION 1: NUMPY FUNDAMENTALS (Questions 1-20)")
print("="*50)

# Question 1: What is NumPy?
"""
Q1. What is NumPy and why is it important?

Answer:
NumPy (Numerical Python) is a fundamental library for scientific computing in Python.
It provides:
- N-dimensional array objects (ndarray)
- Mathematical functions for arrays
- Tools for integrating with C/C++ and Fortran
- Broadcasting functionality
- Linear algebra, Fourier transform, and random number capabilities

Key advantages:
- 50x faster than Python lists for numerical operations
- Memory efficient
- Vectorized operations
- Foundation for other libraries (Pandas, Scikit-learn, etc.)
"""

# Performance comparison
print("Q1. NumPy vs Python Lists Performance:")
python_list = list(range(1000000))
numpy_array = np.arange(1000000)

# Python list operation
start_time = time.time()
python_result = [x * 2 for x in python_list]
python_time = time.time() - start_time

# NumPy operation
start_time = time.time()
numpy_result = numpy_array * 2
numpy_time = time.time() - start_time

print(f"Python list time: {python_time:.6f} seconds")
print(f"NumPy array time: {numpy_time:.6f} seconds")
print(f"NumPy is {python_time/numpy_time:.1f}x faster!")

# Question 2: How to create NumPy arrays?
"""
Q2. What are different ways to create NumPy arrays?

Answer:
1. From Python lists/tuples
2. Using built-in functions (zeros, ones, empty, etc.)
3. Using ranges (arange, linspace)
4. From existing data (asarray, copy)
5. Random arrays
6. Special arrays (identity, diagonal, etc.)
"""

print("\nQ2. Different ways to create NumPy arrays:")

# From lists
arr_from_list = np.array([1, 2, 3, 4, 5])
print(f"From list: {arr_from_list}")

# Zeros and ones
zeros_arr = np.zeros((3, 4))
ones_arr = np.ones((2, 3))
print(f"Zeros array shape {zeros_arr.shape}:\n{zeros_arr}")
print(f"Ones array shape {ones_arr.shape}:\n{ones_arr}")

# Range functions
arange_arr = np.arange(0, 10, 2)
linspace_arr = np.linspace(0, 1, 5)
print(f"Arange (0,10,2): {arange_arr}")
print(f"Linspace (0,1,5): {linspace_arr}")

# Random arrays
random_arr = np.random.random((2, 3))
random_int = np.random.randint(1, 10, (2, 3))
print(f"Random floats:\n{random_arr}")
print(f"Random integers:\n{random_int}")

# Identity and diagonal
identity_arr = np.eye(3)
diagonal_arr = np.diag([1, 2, 3, 4])
print(f"Identity matrix:\n{identity_arr}")
print(f"Diagonal matrix:\n{diagonal_arr}")

# Question 3: NumPy array attributes
"""
Q3. What are important NumPy array attributes?

Answer:
- shape: Tuple of array dimensions
- size: Total number of elements
- ndim: Number of dimensions
- dtype: Data type of elements
- itemsize: Size of each element in bytes
- nbytes: Total bytes consumed
- strides: Bytes to step in each dimension
"""

print("\nQ3. NumPy array attributes:")
sample_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
print(f"Array:\n{sample_array}")
print(f"Shape: {sample_array.shape}")
print(f"Size: {sample_array.size}")
print(f"Dimensions: {sample_array.ndim}")
print(f"Data type: {sample_array.dtype}")
print(f"Item size: {sample_array.itemsize} bytes")
print(f"Total bytes: {sample_array.nbytes} bytes")
print(f"Strides: {sample_array.strides}")

# Question 4: NumPy data types
"""
Q4. What are NumPy data types?

Answer:
NumPy supports various data types:
- Integer: int8, int16, int32, int64, uint8, uint16, uint32, uint64
- Float: float16, float32, float64, float128
- Complex: complex64, complex128, complex256
- Boolean: bool
- String: string_, unicode_
- Object: object (Python objects)
"""

print("\nQ4. NumPy data types:")
# Integer types
int_array = np.array([1, 2, 3], dtype=np.int32)
print(f"Int32 array: {int_array} (dtype: {int_array.dtype})")

# Float types
float_array = np.array([1.5, 2.7, 3.14], dtype=np.float32)
print(f"Float32 array: {float_array} (dtype: {float_array.dtype})")

# Boolean
bool_array = np.array([True, False, True])
print(f"Boolean array: {bool_array} (dtype: {bool_array.dtype})")

# Type conversion
converted_array = int_array.astype(np.float64)
print(f"Converted to float64: {converted_array} (dtype: {converted_array.dtype})")

# Question 5: Array indexing and slicing
"""
Q5. How does array indexing and slicing work in NumPy?

Answer:
- Basic indexing: arr[index]
- Negative indexing: arr[-1]
- Slicing: arr[start:stop:step]
- Multi-dimensional: arr[row, col]
- Boolean indexing: arr[condition]
- Fancy indexing: arr[[indices]]
"""

print("\nQ5. Array indexing and slicing:")
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

print(f"Original array:\n{arr_2d}")
print(f"Element at [1,2]: {arr_2d[1, 2]}")
print(f"First row: {arr_2d[0, :]}")
print(f"Last column: {arr_2d[:, -1]}")
print(f"Subarray [0:2, 1:3]:\n{arr_2d[0:2, 1:3]}")

# Boolean indexing
condition = arr_2d > 6
print(f"Elements > 6: {arr_2d[condition]}")

# Fancy indexing
fancy_indices = np.array([0, 2])
print(f"Rows 0 and 2:\n{arr_2d[fancy_indices]}")

# =============================================================================
# SECTION 2: ARRAY OPERATIONS (Questions 21-40)
# =============================================================================

print("\n" + "="*50)
print("SECTION 2: ARRAY OPERATIONS (Questions 21-40)")
print("="*50)

# Question 21: Broadcasting
"""
Q21. What is broadcasting in NumPy?

Answer:
Broadcasting allows NumPy to perform operations on arrays with different shapes.
Rules:
1. Start from trailing dimensions
2. Dimensions are compatible if they are equal or one of them is 1
3. Missing dimensions are assumed to be 1

Example: (3,4) + (4,) → (3,4) + (1,4) → (3,4)
"""

print("\nQ21. Broadcasting examples:")
arr_a = np.array([[1, 2, 3],
                  [4, 5, 6]])
arr_b = np.array([10, 20, 30])

print(f"Array A shape {arr_a.shape}:\n{arr_a}")
print(f"Array B shape {arr_b.shape}: {arr_b}")
result = arr_a + arr_b
print(f"A + B (broadcasting):\n{result}")

# Broadcasting with scalar
scalar_result = arr_a * 2
print(f"A * 2 (scalar broadcasting):\n{scalar_result}")

# Question 22: Mathematical operations
"""
Q22. What are common mathematical operations in NumPy?

Answer:
- Arithmetic: +, -, *, /, //, **, %
- Trigonometric: sin, cos, tan, arcsin, etc.
- Exponential: exp, log, log10, log2
- Statistical: mean, median, std, var, min, max
- Linear algebra: dot, matmul, linalg functions
"""

print("\nQ22. Mathematical operations:")
data = np.array([[1, 2, 3], [4, 5, 6]])

# Arithmetic operations
print(f"Original data:\n{data}")
print(f"Square: \n{data ** 2}")
print(f"Square root: \n{np.sqrt(data)}")
print(f"Exponential: \n{np.exp(data)}")

# Statistical operations
print(f"Mean: {np.mean(data)}")
print(f"Mean along axis 0: {np.mean(data, axis=0)}")
print(f"Mean along axis 1: {np.mean(data, axis=1)}")
print(f"Standard deviation: {np.std(data)}")
print(f"Min: {np.min(data)}, Max: {np.max(data)}")

# Question 23: Array manipulation
"""
Q23. How to manipulate array shapes and structures?

Answer:
- Reshaping: reshape, resize
- Flattening: flatten, ravel
- Stacking: vstack, hstack, stack, concatenate
- Splitting: split, hsplit, vsplit
- Transposing: transpose, T
- Rotating: rot90, flip
"""

print("\nQ23. Array manipulation:")
original = np.arange(12)
print(f"Original array: {original}")

# Reshaping
reshaped = original.reshape(3, 4)
print(f"Reshaped (3,4):\n{reshaped}")

# Flattening
flattened = reshaped.flatten()
print(f"Flattened: {flattened}")

# Transposing
transposed = reshaped.T
print(f"Transposed:\n{transposed}")

# Stacking
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
vstacked = np.vstack((arr1, arr2))
hstacked = np.hstack((arr1, arr2))
print(f"Vertical stack:\n{vstacked}")
print(f"Horizontal stack: {hstacked}")

# Question 24: Linear algebra operations
"""
Q24. What are linear algebra operations in NumPy?

Answer:
NumPy provides comprehensive linear algebra functions through numpy.linalg:
- Matrix multiplication: dot, matmul, @
- Decompositions: svd, qr, cholesky
- Eigenvalues: eig, eigvals
- Matrix properties: det, rank, norm
- Solving systems: solve, lstsq
"""

print("\nQ24. Linear algebra operations:")
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])

print(f"Matrix A:\n{matrix_a}")
print(f"Matrix B:\n{matrix_b}")

# Matrix multiplication
dot_product = np.dot(matrix_a, matrix_b)
matmul_product = matrix_a @ matrix_b
print(f"Dot product (A·B):\n{dot_product}")
print(f"Matrix multiplication (A@B):\n{matmul_product}")

# Matrix properties
determinant = np.linalg.det(matrix_a)
eigenvalues, eigenvectors = np.linalg.eig(matrix_a)
print(f"Determinant of A: {determinant:.2f}")
print(f"Eigenvalues of A: {eigenvalues}")

# Question 25: Boolean operations and filtering
"""
Q25. How to perform boolean operations and filtering in NumPy?

Answer:
- Comparison operators: <, >, <=, >=, ==, !=
- Logical operators: logical_and, logical_or, logical_not
- Boolean indexing: arr[condition]
- Functions: where, select, choose
- All/any: all, any
"""

print("\nQ25. Boolean operations:")
data = np.random.randint(1, 10, (3, 4))
print(f"Random data:\n{data}")

# Boolean conditions
condition1 = data > 5
condition2 = data % 2 == 0
print(f"Values > 5:\n{condition1}")
print(f"Even values:\n{condition2}")

# Combining conditions
combined = np.logical_and(condition1, condition2)
print(f"Values > 5 AND even:\n{combined}")

# Boolean indexing
filtered_data = data[combined]
print(f"Filtered values: {filtered_data}")

# Using where
result = np.where(data > 5, data, 0)
print(f"Replace values ≤5 with 0:\n{result}")

# =============================================================================
# SECTION 3: ADVANCED NUMPY (Questions 41-60)
# =============================================================================

print("\n" + "="*50)
print("SECTION 3: ADVANCED NUMPY (Questions 41-60)")
print("="*50)

# Question 41: Memory layout and views vs copies
"""
Q41. What is the difference between views and copies in NumPy?

Answer:
View: Shares data with original array (same memory)
- Created by: slicing, reshape (sometimes), transpose
- Changes affect original array

Copy: Independent array with own memory
- Created by: copy(), fancy indexing, boolean indexing
- Changes don't affect original array
"""

print("\nQ41. Views vs Copies:")
original = np.arange(12).reshape(3, 4)
print(f"Original array:\n{original}")

# Creating a view
view = original[1:3, 1:3]
print(f"View (slice):\n{view}")
print(f"View shares data: {np.shares_memory(original, view)}")

# Modifying view affects original
view[0, 0] = 999
print(f"After modifying view:\n{original}")

# Creating a copy
copy_arr = original.copy()
print(f"Copy shares data: {np.shares_memory(original, copy_arr)}")
copy_arr[0, 0] = 111
print(f"Original after modifying copy:\n{original}")

# Question 42: Random number generation
"""
Q42. How does random number generation work in NumPy?

Answer:
NumPy provides numpy.random module with:
- Random sampling: random, uniform, normal, binomial
- Random integers: randint, choice
- Shuffling: shuffle, permutation
- Seeding: seed, RandomState, default_rng
"""

print("\nQ42. Random number generation:")
# Set seed for reproducibility
np.random.seed(42)

# Different random distributions
uniform_samples = np.random.uniform(0, 1, 5)
normal_samples = np.random.normal(0, 1, 5)
integer_samples = np.random.randint(1, 10, 5)

print(f"Uniform [0,1]: {uniform_samples}")
print(f"Normal (μ=0, σ=1): {normal_samples}")
print(f"Random integers [1,10): {integer_samples}")

# Random choice
choices = np.random.choice(['A', 'B', 'C'], size=5, p=[0.5, 0.3, 0.2])
print(f"Random choice with probabilities: {choices}")

# Shuffle
arr_to_shuffle = np.arange(10)
np.random.shuffle(arr_to_shuffle)
print(f"Shuffled array: {arr_to_shuffle}")

# Question 43: Structured arrays
"""
Q43. What are structured arrays in NumPy?

Answer:
Structured arrays allow different data types in a single array.
They're useful for representing heterogeneous data (like database records).
"""

print("\nQ43. Structured arrays:")
# Define dtype for structured array
dtype = np.dtype([
    ('name', 'U20'),      # Unicode string, max 20 chars
    ('age', 'i4'),        # 32-bit integer
    ('salary', 'f8')      # 64-bit float
])

# Create structured array
employees = np.array([
    ('Alice', 25, 50000.0),
    ('Bob', 30, 60000.0),
    ('Charlie', 35, 70000.0)
], dtype=dtype)

print(f"Structured array:\n{employees}")
print(f"Names: {employees['name']}")
print(f"Ages: {employees['age']}")
print(f"Average salary: {np.mean(employees['salary'])}")

# Question 44: Memory optimization
"""
Q44. How to optimize memory usage in NumPy?

Answer:
1. Choose appropriate data types (int8 vs int64)
2. Use views instead of copies when possible
3. Delete unused arrays
4. Use memory mapping for large files
5. Understand C vs Fortran order
"""

print("\nQ44. Memory optimization:")
# Data type optimization
large_array_64 = np.ones(1000000, dtype=np.int64)
large_array_8 = np.ones(1000000, dtype=np.int8)

print(f"int64 array memory: {large_array_64.nbytes / 1024:.1f} KB")
print(f"int8 array memory: {large_array_8.nbytes / 1024:.1f} KB")
print(f"Memory savings: {large_array_64.nbytes / large_array_8.nbytes:.0f}x")

# C vs Fortran order
c_order = np.ones((1000, 1000), order='C')  # Row-major
f_order = np.ones((1000, 1000), order='F')  # Column-major

print(f"C-order strides: {c_order.strides}")
print(f"F-order strides: {f_order.strides}")

# Question 45: Universal functions (ufuncs)
"""
Q45. What are universal functions (ufuncs) in NumPy?

Answer:
Ufuncs are functions that operate element-wise on arrays.
They support:
- Broadcasting
- Type casting
- Reduction operations
- Accumulation operations
- Custom ufuncs
"""

print("\nQ45. Universal functions:")
arr = np.array([1, 2, 3, 4, 5])

# Built-in ufuncs
print(f"Original: {arr}")
print(f"Square: {np.square(arr)}")
print(f"Sqrt: {np.sqrt(arr)}")
print(f"Sin: {np.sin(arr)}")

# Ufunc methods
print(f"Sum (reduce): {np.add.reduce(arr)}")
print(f"Cumulative sum (accumulate): {np.add.accumulate(arr)}")

# Multiple arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(f"Add arrays: {np.add(arr1, arr2)}")
print(f"Maximum: {np.maximum(arr1, arr2)}")

# =============================================================================
# PRACTICAL QUESTIONS AND EXAMPLES
# =============================================================================

print("\n" + "="*50)
print("PRACTICAL NUMPY QUESTIONS AND SOLUTIONS")
print("="*50)

# Question 46: Performance optimization
"""
Q46. How to optimize NumPy code for performance?

Answer:
1. Vectorize operations instead of loops
2. Use built-in functions
3. Minimize array copies
4. Use appropriate data types
5. Leverage broadcasting
6. Use compiled code (Numba, Cython) for complex operations
"""

print("\nQ46. Performance optimization example:")

# Inefficient approach
def slow_distance(points1, points2):
    distances = []
    for p1, p2 in zip(points1, points2):
        dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        distances.append(dist)
    return np.array(distances)

# Efficient approach
def fast_distance(points1, points2):
    diff = points1 - points2
    return np.sqrt(np.sum(diff**2, axis=1))

# Test performance
np.random.seed(42)
points1 = np.random.random((1000, 2))
points2 = np.random.random((1000, 2))

start_time = time.time()
slow_result = slow_distance(points1, points2)
slow_time = time.time() - start_time

start_time = time.time()
fast_result = fast_distance(points1, points2)
fast_time = time.time() - start_time

print(f"Slow approach: {slow_time:.6f} seconds")
print(f"Fast approach: {fast_time:.6f} seconds")
print(f"Speedup: {slow_time/fast_time:.1f}x")

# Question 47: Working with missing data
"""
Q47. How to handle missing data in NumPy?

Answer:
1. Use np.nan for missing floating point values
2. Use np.ma (masked arrays) for general missing data
3. Functions: isnan, nanmean, nanstd, etc.
4. Interpolation for filling missing values
"""

print("\nQ47. Handling missing data:")
# Array with missing values
data_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan])
print(f"Data with NaN: {data_with_nan}")

# Check for NaN
nan_mask = np.isnan(data_with_nan)
print(f"NaN mask: {nan_mask}")

# Statistics ignoring NaN
print(f"Mean (ignoring NaN): {np.nanmean(data_with_nan):.2f}")
print(f"Standard deviation (ignoring NaN): {np.nanstd(data_with_nan):.2f}")

# Masked arrays
masked_data = np.ma.masked_invalid(data_with_nan)
print(f"Masked array: {masked_data}")
print(f"Masked mean: {np.ma.mean(masked_data):.2f}")

# Question 48: Array comparison and set operations
"""
Q48. How to perform array comparisons and set operations?

Answer:
- Element-wise comparison: ==, !=, <, >, etc.
- Array equality: array_equal, allclose
- Set operations: unique, intersect1d, union1d, setdiff1d
- Membership: in1d, isin
"""

print("\nQ48. Array comparisons and set operations:")
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([3, 4, 5, 6, 7])

print(f"Array 1: {arr1}")
print(f"Array 2: {arr2}")

# Set operations
intersection = np.intersect1d(arr1, arr2)
union = np.union1d(arr1, arr2)
difference = np.setdiff1d(arr1, arr2)

print(f"Intersection: {intersection}")
print(f"Union: {union}")
print(f"Difference (arr1 - arr2): {difference}")

# Membership testing
membership = np.isin(arr1, arr2)
print(f"Elements of arr1 in arr2: {membership}")

# Question 49: Sorting and searching
"""
Q49. How to sort and search arrays in NumPy?

Answer:
- Sorting: sort, argsort, lexsort, partition
- Searching: searchsorted, where, argmax, argmin
- Conditions: nonzero, flatnonzero
"""

print("\nQ49. Sorting and searching:")
unsorted_arr = np.array([64, 34, 25, 12, 22, 11, 90])
print(f"Unsorted: {unsorted_arr}")

# Sorting
sorted_arr = np.sort(unsorted_arr)
sort_indices = np.argsort(unsorted_arr)
print(f"Sorted: {sorted_arr}")
print(f"Sort indices: {sort_indices}")

# Searching
search_value = 25
position = np.searchsorted(sorted_arr, search_value)
print(f"Position to insert {search_value}: {position}")

# Finding extremes
max_index = np.argmax(unsorted_arr)
min_index = np.argmin(unsorted_arr)
print(f"Max value at index {max_index}: {unsorted_arr[max_index]}")
print(f"Min value at index {min_index}: {unsorted_arr[min_index]}")

# Question 50: Advanced indexing techniques
"""
Q50. What are advanced indexing techniques in NumPy?

Answer:
1. Fancy indexing with integer arrays
2. Boolean indexing with conditions
3. Multi-dimensional indexing
4. ix_ for open mesh indexing
5. take and put for flexible indexing
"""

print("\nQ50. Advanced indexing:")
data = np.arange(20).reshape(4, 5)
print(f"Original data:\n{data}")

# Fancy indexing
row_indices = [0, 2, 3]
col_indices = [1, 3, 4]
fancy_result = data[row_indices][:, col_indices]
print(f"Fancy indexing result:\n{fancy_result}")

# Advanced boolean indexing
condition = (data > 5) & (data < 15)
filtered = data[condition]
print(f"Values between 5 and 15: {filtered}")

# ix_ for mesh indexing
mesh_result = data[np.ix_(row_indices, col_indices)]
print(f"Mesh indexing result:\n{mesh_result}")

print("\n" + "="*70)
print("SUMMARY: NUMPY INTERVIEW PREPARATION CHECKLIST")
print("="*70)

checklist = [
    "✓ Understand NumPy fundamentals and advantages over Python lists",
    "✓ Master array creation methods and data types",
    "✓ Know array attributes and memory layout",
    "✓ Practice indexing, slicing, and boolean operations",
    "✓ Understand broadcasting rules and applications",
    "✓ Master mathematical and statistical operations",
    "✓ Know array manipulation functions (reshape, stack, split)",
    "✓ Understand linear algebra operations",
    "✓ Practice advanced indexing techniques",
    "✓ Know performance optimization strategies",
    "✓ Understand views vs copies concept",
    "✓ Master random number generation",
    "✓ Know how to handle missing data",
    "✓ Practice sorting and searching operations",
    "✓ Understand universal functions (ufuncs)",
    "✓ Know memory optimization techniques"
]

for item in checklist:
    print(item)

print("\n" + "="*70)
print("END OF NUMPY INTERVIEW QUESTIONS")
print("Total Questions: 50+ | Practical Examples: 20+ | Performance Tips: 10+")
print("="*70)
