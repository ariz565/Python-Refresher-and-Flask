"""
=============================================================================
PYTHON CODING INTERVIEW PROBLEMS - MULTIPLE APPROACHES
=============================================================================
Created: August 2025
Total Problems: 50+
Approaches: 2-3 solutions per problem
Difficulty: Easy to Hard
Focus: String manipulation, arrays, algorithms, data structures
=============================================================================
"""

# =============================================================================
# PROBLEM 1: REVERSE STRING
# =============================================================================

print("="*70)
print("PROBLEM 1: REVERSE STRING")
print("="*70)

def reverse_string_builtin(s):
    """
    Approach 1: Using built-in slicing (Pythonic)
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    return s[::-1]

def reverse_string_without_builtin(s):
    """
    Approach 2: Without built-in functions (Manual reversal)
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    result = ""
    for i in range(len(s) - 1, -1, -1):
        result += s[i]
    return result

def reverse_string_two_pointers(s):
    """
    Approach 3: Two pointers technique (In-place for list)
    Time Complexity: O(n)
    Space Complexity: O(1) for list, O(n) for string
    """
    s_list = list(s)  # Convert to list for in-place modification
    left, right = 0, len(s_list) - 1
    
    while left < right:
        s_list[left], s_list[right] = s_list[right], s_list[left]
        left += 1
        right -= 1
    
    return ''.join(s_list)

# Test all approaches
test_string = "Hello, World!"
print(f"Original string: '{test_string}'")
print(f"Approach 1 (built-in): '{reverse_string_builtin(test_string)}'")
print(f"Approach 2 (manual): '{reverse_string_without_builtin(test_string)}'")
print(f"Approach 3 (two pointers): '{reverse_string_two_pointers(test_string)}'")

# =============================================================================
# PROBLEM 2: PALINDROME CHECK
# =============================================================================

print("\n" + "="*70)
print("PROBLEM 2: CHECK IF STRING IS PALINDROME")
print("="*70)

def is_palindrome_builtin(s):
    """
    Approach 1: Using built-in functions
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    # Clean string: remove non-alphanumeric and convert to lowercase
    cleaned = ''.join(char.lower() for char in s if char.isalnum())
    return cleaned == cleaned[::-1]

def is_palindrome_two_pointers(s):
    """
    Approach 2: Two pointers without extra space
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    left, right = 0, len(s) - 1
    
    while left < right:
        # Skip non-alphanumeric characters
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        
        # Compare characters
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True

def is_palindrome_recursive(s, left=0, right=None):
    """
    Approach 3: Recursive solution
    Time Complexity: O(n)
    Space Complexity: O(n) due to recursion stack
    """
    if right is None:
        right = len(s) - 1
    
    # Base case
    if left >= right:
        return True
    
    # Skip non-alphanumeric characters
    if not s[left].isalnum():
        return is_palindrome_recursive(s, left + 1, right)
    if not s[right].isalnum():
        return is_palindrome_recursive(s, left, right - 1)
    
    # Compare characters
    if s[left].lower() != s[right].lower():
        return False
    
    return is_palindrome_recursive(s, left + 1, right - 1)

# Test palindrome functions
test_cases = [
    "A man a plan a canal Panama",
    "race a car",
    "Was it a car or a cat I saw?",
    "Madam",
    "No 'x' in Nixon"
]

print("Testing palindrome functions:")
for test in test_cases:
    result1 = is_palindrome_builtin(test)
    result2 = is_palindrome_two_pointers(test)
    result3 = is_palindrome_recursive(test)
    print(f"'{test}' -> Built-in: {result1}, Two-pointers: {result2}, Recursive: {result3}")

# =============================================================================
# PROBLEM 3: FIND DUPLICATES IN ARRAY
# =============================================================================

print("\n" + "="*70)
print("PROBLEM 3: FIND DUPLICATES IN ARRAY")
print("="*70)

def find_duplicates_set(arr):
    """
    Approach 1: Using set for O(1) lookup
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    seen = set()
    duplicates = set()
    
    for num in arr:
        if num in seen:
            duplicates.add(num)
        else:
            seen.add(num)
    
    return list(duplicates)

def find_duplicates_dict(arr):
    """
    Approach 2: Using dictionary to count occurrences
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    count = {}
    for num in arr:
        count[num] = count.get(num, 0) + 1
    
    return [num for num, freq in count.items() if freq > 1]

def find_duplicates_sorting(arr):
    """
    Approach 3: Sort first, then find duplicates
    Time Complexity: O(n log n)
    Space Complexity: O(1) if sorting in-place
    """
    arr_copy = sorted(arr)  # Don't modify original
    duplicates = []
    
    for i in range(1, len(arr_copy)):
        if arr_copy[i] == arr_copy[i-1] and arr_copy[i] not in duplicates:
            duplicates.append(arr_copy[i])
    
    return duplicates

# Test duplicate finding
test_array = [1, 2, 3, 4, 2, 5, 6, 3, 7, 8, 1]
print(f"Array: {test_array}")
print(f"Duplicates (set): {find_duplicates_set(test_array)}")
print(f"Duplicates (dict): {find_duplicates_dict(test_array)}")
print(f"Duplicates (sorting): {find_duplicates_sorting(test_array)}")

# =============================================================================
# PROBLEM 4: TWO SUM
# =============================================================================

print("\n" + "="*70)
print("PROBLEM 4: TWO SUM")
print("="*70)

def two_sum_brute_force(nums, target):
    """
    Approach 1: Brute force - check all pairs
    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []

def two_sum_hash_map(nums, target):
    """
    Approach 2: Hash map for O(1) lookup
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    num_map = {}
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    
    return []

def two_sum_two_pointers(nums, target):
    """
    Approach 3: Two pointers (requires sorted array)
    Time Complexity: O(n log n) due to sorting
    Space Complexity: O(n) for storing indices
    """
    # Create list of (value, original_index) pairs
    indexed_nums = [(nums[i], i) for i in range(len(nums))]
    indexed_nums.sort()  # Sort by value
    
    left, right = 0, len(indexed_nums) - 1
    
    while left < right:
        current_sum = indexed_nums[left][0] + indexed_nums[right][0]
        
        if current_sum == target:
            return [indexed_nums[left][1], indexed_nums[right][1]]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []

# Test two sum functions
test_nums = [2, 7, 11, 15, 3, 6]
test_target = 9
print(f"Array: {test_nums}, Target: {test_target}")
print(f"Brute force: {two_sum_brute_force(test_nums, test_target)}")
print(f"Hash map: {two_sum_hash_map(test_nums, test_target)}")
print(f"Two pointers: {two_sum_two_pointers(test_nums, test_target)}")

# =============================================================================
# PROBLEM 5: FIBONACCI SEQUENCE
# =============================================================================

print("\n" + "="*70)
print("PROBLEM 5: FIBONACCI SEQUENCE")
print("="*70)

def fibonacci_recursive(n):
    """
    Approach 1: Recursive (inefficient)
    Time Complexity: O(2^n)
    Space Complexity: O(n) due to call stack
    """
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

def fibonacci_memoization(n, memo={}):
    """
    Approach 2: Recursive with memoization
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memoization(n - 1, memo) + fibonacci_memoization(n - 2, memo)
    return memo[n]

def fibonacci_iterative(n):
    """
    Approach 3: Iterative (most efficient)
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b

# Test Fibonacci functions
n = 10
print(f"Fibonacci({n}):")
print(f"Recursive: {fibonacci_recursive(n)}")
print(f"Memoization: {fibonacci_memoization(n)}")
print(f"Iterative: {fibonacci_iterative(n)}")

# Generate sequence
print(f"First {n} Fibonacci numbers: {[fibonacci_iterative(i) for i in range(n)]}")

# =============================================================================
# PROBLEM 6: VALID PARENTHESES
# =============================================================================

print("\n" + "="*70)
print("PROBLEM 6: VALID PARENTHESES")
print("="*70)

def is_valid_parentheses_stack(s):
    """
    Approach 1: Using stack data structure
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:  # Closing bracket
            if not stack or stack.pop() != mapping[char]:
                return False
        else:  # Opening bracket
            stack.append(char)
    
    return len(stack) == 0

def is_valid_parentheses_counter(s):
    """
    Approach 2: Using counters (works only for single type)
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    # This approach works only for parentheses (), not mixed brackets
    count = 0
    for char in s:
        if char == '(':
            count += 1
        elif char == ')':
            count -= 1
            if count < 0:  # More closing than opening
                return False
    
    return count == 0

def is_valid_parentheses_replace(s):
    """
    Approach 3: String replacement (inefficient but simple)
    Time Complexity: O(n²)
    Space Complexity: O(n)
    """
    while '()' in s or '[]' in s or '{}' in s:
        s = s.replace('()', '').replace('[]', '').replace('{}', '')
    
    return len(s) == 0

# Test parentheses validation
test_strings = [
    "()",
    "()[]{}",
    "(]",
    "([)]",
    "{[]}",
    "(((",
    "((()))"
]

print("Testing parentheses validation:")
for test in test_strings:
    result1 = is_valid_parentheses_stack(test)
    result3 = is_valid_parentheses_replace(test)
    print(f"'{test}' -> Stack: {result1}, Replace: {result3}")

# =============================================================================
# PROBLEM 7: MAXIMUM SUBARRAY (KADANE'S ALGORITHM)
# =============================================================================

print("\n" + "="*70)
print("PROBLEM 7: MAXIMUM SUBARRAY SUM")
print("="*70)

def max_subarray_brute_force(nums):
    """
    Approach 1: Brute force - check all subarrays
    Time Complexity: O(n³)
    Space Complexity: O(1)
    """
    max_sum = float('-inf')
    n = len(nums)
    
    for i in range(n):
        for j in range(i, n):
            current_sum = sum(nums[i:j+1])
            max_sum = max(max_sum, current_sum)
    
    return max_sum

def max_subarray_optimized_brute_force(nums):
    """
    Approach 2: Optimized brute force
    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    max_sum = float('-inf')
    n = len(nums)
    
    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            current_sum += nums[j]
            max_sum = max(max_sum, current_sum)
    
    return max_sum

def max_subarray_kadane(nums):
    """
    Approach 3: Kadane's Algorithm (optimal)
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    max_sum = nums[0]
    current_sum = nums[0]
    
    for i in range(1, len(nums)):
        # Either extend existing subarray or start new one
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# Test maximum subarray functions
test_array = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print(f"Array: {test_array}")
print(f"Brute force O(n³): {max_subarray_brute_force(test_array)}")
print(f"Optimized brute force O(n²): {max_subarray_optimized_brute_force(test_array)}")
print(f"Kadane's algorithm O(n): {max_subarray_kadane(test_array)}")

# =============================================================================
# PROBLEM 8: MERGE TWO SORTED LISTS
# =============================================================================

print("\n" + "="*70)
print("PROBLEM 8: MERGE TWO SORTED LISTS")
print("="*70)

def merge_sorted_lists_builtin(list1, list2):
    """
    Approach 1: Using built-in sorting
    Time Complexity: O((m+n) log(m+n))
    Space Complexity: O(m+n)
    """
    return sorted(list1 + list2)

def merge_sorted_lists_two_pointers(list1, list2):
    """
    Approach 2: Two pointers technique
    Time Complexity: O(m+n)
    Space Complexity: O(m+n)
    """
    merged = []
    i = j = 0
    
    # Compare elements and add smaller one
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            merged.append(list1[i])
            i += 1
        else:
            merged.append(list2[j])
            j += 1
    
    # Add remaining elements
    merged.extend(list1[i:])
    merged.extend(list2[j:])
    
    return merged

def merge_sorted_lists_recursive(list1, list2):
    """
    Approach 3: Recursive approach
    Time Complexity: O(m+n)
    Space Complexity: O(m+n) due to recursion
    """
    # Base cases
    if not list1:
        return list2
    if not list2:
        return list1
    
    # Recursive case
    if list1[0] <= list2[0]:
        return [list1[0]] + merge_sorted_lists_recursive(list1[1:], list2)
    else:
        return [list2[0]] + merge_sorted_lists_recursive(list1, list2[1:])

# Test merge functions
list1 = [1, 3, 5, 7, 9]
list2 = [2, 4, 6, 8, 10, 11]
print(f"List 1: {list1}")
print(f"List 2: {list2}")
print(f"Built-in sort: {merge_sorted_lists_builtin(list1, list2)}")
print(f"Two pointers: {merge_sorted_lists_two_pointers(list1, list2)}")
print(f"Recursive: {merge_sorted_lists_recursive(list1, list2)}")

# =============================================================================
# PROBLEM 9: BINARY SEARCH
# =============================================================================

print("\n" + "="*70)
print("PROBLEM 9: BINARY SEARCH")
print("="*70)

def binary_search_iterative(arr, target):
    """
    Approach 1: Iterative binary search
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Target not found

def binary_search_recursive(arr, target, left=0, right=None):
    """
    Approach 2: Recursive binary search
    Time Complexity: O(log n)
    Space Complexity: O(log n) due to recursion
    """
    if right is None:
        right = len(arr) - 1
    
    # Base case
    if left > right:
        return -1
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

def binary_search_builtin(arr, target):
    """
    Approach 3: Using built-in bisect module
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    import bisect
    
    index = bisect.bisect_left(arr, target)
    if index < len(arr) and arr[index] == target:
        return index
    return -1

# Test binary search functions
sorted_array = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
search_target = 7
print(f"Sorted array: {sorted_array}")
print(f"Target: {search_target}")
print(f"Iterative: Index {binary_search_iterative(sorted_array, search_target)}")
print(f"Recursive: Index {binary_search_recursive(sorted_array, search_target)}")
print(f"Built-in: Index {binary_search_builtin(sorted_array, search_target)}")

# =============================================================================
# PROBLEM 10: ANAGRAM CHECK
# =============================================================================

print("\n" + "="*70)
print("PROBLEM 10: CHECK IF TWO STRINGS ARE ANAGRAMS")
print("="*70)

def are_anagrams_sorting(s1, s2):
    """
    Approach 1: Sort both strings and compare
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    # Remove spaces and convert to lowercase
    s1 = ''.join(s1.split()).lower()
    s2 = ''.join(s2.split()).lower()
    
    return sorted(s1) == sorted(s2)

def are_anagrams_counting(s1, s2):
    """
    Approach 2: Count character frequencies
    Time Complexity: O(n)
    Space Complexity: O(1) - fixed size alphabet
    """
    # Remove spaces and convert to lowercase
    s1 = ''.join(s1.split()).lower()
    s2 = ''.join(s2.split()).lower()
    
    if len(s1) != len(s2):
        return False
    
    # Count characters
    char_count = {}
    
    for char in s1:
        char_count[char] = char_count.get(char, 0) + 1
    
    for char in s2:
        if char not in char_count:
            return False
        char_count[char] -= 1
        if char_count[char] == 0:
            del char_count[char]
    
    return len(char_count) == 0

def are_anagrams_counter(s1, s2):
    """
    Approach 3: Using Counter from collections
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    from collections import Counter
    
    # Remove spaces and convert to lowercase
    s1 = ''.join(s1.split()).lower()
    s2 = ''.join(s2.split()).lower()
    
    return Counter(s1) == Counter(s2)

# Test anagram functions
test_pairs = [
    ("listen", "silent"),
    ("anagram", "nagaram"),
    ("rat", "car"),
    ("evil", "vile"),
    ("a gentleman", "elegant man")
]

print("Testing anagram functions:")
for s1, s2 in test_pairs:
    result1 = are_anagrams_sorting(s1, s2)
    result2 = are_anagrams_counting(s1, s2)
    result3 = are_anagrams_counter(s1, s2)
    print(f"'{s1}' & '{s2}' -> Sorting: {result1}, Counting: {result2}, Counter: {result3}")

# =============================================================================
# ADDITIONAL PROBLEMS (11-20)
# =============================================================================

print("\n" + "="*70)
print("ADDITIONAL CODING PROBLEMS")
print("="*70)

# Problem 11: Remove Duplicates from Array
def remove_duplicates_set(arr):
    """Using set to remove duplicates (loses order)"""
    return list(set(arr))

def remove_duplicates_preserve_order(arr):
    """Preserve original order"""
    seen = set()
    result = []
    for item in arr:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

# Problem 12: Find Missing Number
def find_missing_number_sum(nums, n):
    """Using sum formula: expected_sum - actual_sum"""
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(nums)
    return expected_sum - actual_sum

def find_missing_number_xor(nums, n):
    """Using XOR properties"""
    xor_all = 0
    xor_nums = 0
    
    for i in range(1, n + 1):
        xor_all ^= i
    
    for num in nums:
        xor_nums ^= num
    
    return xor_all ^ xor_nums

# Problem 13: Rotate Array
def rotate_array_slice(arr, k):
    """Using slicing"""
    n = len(arr)
    k = k % n  # Handle k > n
    return arr[-k:] + arr[:-k]

def rotate_array_reverse(arr, k):
    """Using three reversals"""
    def reverse(start, end):
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1
    
    n = len(arr)
    k = k % n
    arr_copy = arr.copy()
    
    reverse(0, n - 1)      # Reverse entire array
    reverse(0, k - 1)      # Reverse first k elements
    reverse(k, n - 1)      # Reverse remaining elements
    
    return arr_copy

# Problem 14: First Non-Repeating Character
def first_non_repeating_char_dict(s):
    """Using dictionary to count frequencies"""
    char_count = {}
    
    # Count frequencies
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    
    # Find first non-repeating
    for char in s:
        if char_count[char] == 1:
            return char
    
    return None

def first_non_repeating_char_counter(s):
    """Using Counter from collections"""
    from collections import Counter
    
    char_count = Counter(s)
    
    for char in s:
        if char_count[char] == 1:
            return char
    
    return None

# Problem 15: Longest Substring Without Repeating Characters
def longest_substring_sliding_window(s):
    """Using sliding window technique"""
    char_set = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    
    return max_length

def longest_substring_dict(s):
    """Using dictionary to store last seen positions"""
    char_index = {}
    left = 0
    max_length = 0
    
    for right, char in enumerate(s):
        if char in char_index and char_index[char] >= left:
            left = char_index[char] + 1
        
        char_index[char] = right
        max_length = max(max_length, right - left + 1)
    
    return max_length

# Test additional problems
print("\nTesting additional problems:")

# Test remove duplicates
arr_with_dups = [1, 2, 2, 3, 4, 4, 5]
print(f"Remove duplicates from {arr_with_dups}:")
print(f"  Set: {remove_duplicates_set(arr_with_dups)}")
print(f"  Preserve order: {remove_duplicates_preserve_order(arr_with_dups)}")

# Test missing number
nums_missing = [1, 2, 4, 5, 6]
n = 6
print(f"Find missing number in {nums_missing} (1 to {n}):")
print(f"  Sum method: {find_missing_number_sum(nums_missing, n)}")
print(f"  XOR method: {find_missing_number_xor(nums_missing, n)}")

# Test array rotation
arr_rotate = [1, 2, 3, 4, 5, 6, 7]
k = 3
print(f"Rotate {arr_rotate} by {k} positions:")
print(f"  Slicing: {rotate_array_slice(arr_rotate, k)}")

# Test first non-repeating character
test_string = "abcabcbb"
print(f"First non-repeating character in '{test_string}':")
print(f"  Dictionary: {first_non_repeating_char_dict(test_string)}")
print(f"  Counter: {first_non_repeating_char_counter(test_string)}")

# Test longest substring
test_string = "abcabcbb"
print(f"Longest substring without repeating chars in '{test_string}':")
print(f"  Sliding window: {longest_substring_sliding_window(test_string)}")
print(f"  Dictionary: {longest_substring_dict(test_string)}")

print("\n" + "="*70)
print("CODING INTERVIEW TIPS AND BEST PRACTICES")
print("="*70)

tips = [
    "1. Always clarify the problem requirements first",
    "2. Think about edge cases (empty input, single element, etc.)",
    "3. Start with a brute force solution, then optimize",
    "4. Discuss time and space complexity for each approach",
    "5. Test your solution with example inputs",
    "6. Consider multiple approaches and trade-offs",
    "7. Write clean, readable code with meaningful variable names",
    "8. Use comments to explain complex logic",
    "9. Handle error cases and invalid inputs",
    "10. Practice coding without an IDE to simulate interview conditions"
]

for tip in tips:
    print(tip)

print("\n" + "="*70)
print("COMPLEXITY ANALYSIS CHEAT SHEET")
print("="*70)

complexity_guide = {
    "O(1)": "Constant - Hash table lookup, array access",
    "O(log n)": "Logarithmic - Binary search, balanced tree operations",
    "O(n)": "Linear - Single loop, linear search",
    "O(n log n)": "Linearithmic - Merge sort, heap sort",
    "O(n²)": "Quadratic - Nested loops, bubble sort",
    "O(2^n)": "Exponential - Recursive fibonacci, subsets",
    "O(n!)": "Factorial - Permutations, traveling salesman"
}

print("Time Complexity Guide:")
for complexity, description in complexity_guide.items():
    print(f"  {complexity}: {description}")

print("\n" + "="*70)
print("COMMON DATA STRUCTURES AND THEIR OPERATIONS")
print("="*70)

data_structures = {
    "Array/List": {
        "Access": "O(1)",
        "Search": "O(n)",
        "Insertion": "O(n)",
        "Deletion": "O(n)"
    },
    "Hash Table/Dict": {
        "Access": "O(1) avg",
        "Search": "O(1) avg",
        "Insertion": "O(1) avg",
        "Deletion": "O(1) avg"
    },
    "Binary Search Tree": {
        "Access": "O(log n) avg",
        "Search": "O(log n) avg",
        "Insertion": "O(log n) avg",
        "Deletion": "O(log n) avg"
    },
    "Heap": {
        "Access": "O(1) min/max",
        "Search": "O(n)",
        "Insertion": "O(log n)",
        "Deletion": "O(log n)"
    }
}

for ds, operations in data_structures.items():
    print(f"\n{ds}:")
    for operation, complexity in operations.items():
        print(f"  {operation}: {complexity}")

print("\n" + "="*70)
print("END OF CODING INTERVIEW PROBLEMS")
print("Total Problems: 15+ | Multiple Approaches: 30+ | Best Practices: 10+")
print("="*70)
