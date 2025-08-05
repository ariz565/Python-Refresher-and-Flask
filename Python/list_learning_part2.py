# ===============================================================================
# COMPREHENSIVE PYTHON LISTS LEARNING GUIDE - PART 2
# Continuing from list_learning.py
# ===============================================================================

"""
CONTINUATION OF SECTIONS:
=======================
5. Slicing & Indexing Mastery
6. Real-World Applications
7. Interview Problems & Solutions
8. Advanced Concepts for Experienced Developers
9. System Design with Lists
10. Threading & Concurrency Considerations
"""

import time
import sys
from collections import defaultdict, deque
import threading
import operator
import functools
import itertools
import bisect
import heapq
import copy
import concurrent.futures
import queue

# ===============================================================================
# 5. SLICING & INDEXING MASTERY
# ===============================================================================

print("=" * 80)
print("5. SLICING & INDEXING MASTERY")
print("=" * 80)

print("\n--- Basic Indexing ---")

# Positive and negative indexing
sample_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
print(f"Sample list: {sample_list}")

print(f"First element [0]: {sample_list[0]}")
print(f"Last element [-1]: {sample_list[-1]}")
print(f"Second last [-2]: {sample_list[-2]}")
print(f"Middle element [5]: {sample_list[5]}")

# Index bounds checking
try:
    print(f"Out of bounds [100]: {sample_list[100]}")
except IndexError as e:
    print(f"IndexError: {e}")

print("\n--- Basic Slicing ---")

# Basic slice syntax: [start:stop:step]
print(f"First 3 elements [0:3]: {sample_list[0:3]}")
print(f"Elements 2-5 [2:6]: {sample_list[2:6]}")
print(f"Last 3 elements [-3:]: {sample_list[-3:]}")
print(f"All except last 2 [:-2]: {sample_list[:-2]}")

# Default values
print(f"From index 3 to end [3:]: {sample_list[3:]}")
print(f"From start to index 5 [:5]: {sample_list[:5]}")
print(f"Entire list [:]: {sample_list[:]}")

print("\n--- Advanced Slicing with Step ---")

# Step parameter
print(f"Every 2nd element [::2]: {sample_list[::2]}")
print(f"Every 3rd element [1::3]: {sample_list[1::3]}")
print(f"Reverse order [::-1]: {sample_list[::-1]}")
print(f"Every 2nd element reversed [::-2]: {sample_list[::-2]}")

# Complex slicing
print(f"Middle elements with step [2:8:2]: {sample_list[2:8:2]}")
print(f"Reverse slice [8:2:-1]: {sample_list[8:2:-1]}")

print("\n--- Slice Assignment ---")

# Replacing elements with slice assignment
demo_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(f"Original: {demo_list}")

# Replace middle elements
demo_list[3:6] = [40, 50, 60]
print(f"After [3:6] = [40, 50, 60]: {demo_list}")

# Insert elements (replace with longer sequence)
demo_list[2:4] = [30, 35, 40, 45]
print(f"After [2:4] = [30, 35, 40, 45]: {demo_list}")

# Delete elements (replace with empty list)
demo_list[5:8] = []
print(f"After [5:8] = []: {demo_list}")

# Insert at specific position
demo_list[2:2] = [25, 27]
print(f"After [2:2] = [25, 27]: {demo_list}")

print("\n--- Advanced Slice Techniques ---")

# Using slice objects
slice_obj = slice(1, 8, 2)
print(f"Using slice(1, 8, 2): {sample_list[slice_obj]}")

# Dynamic slicing
def get_middle_elements(lst, percentage=0.5):
    """Get middle percentage of elements"""
    length = len(lst)
    start = int(length * (1 - percentage) / 2)
    end = int(length * (1 + percentage) / 2)
    return lst[start:end]

middle_50 = get_middle_elements(sample_list, 0.5)
print(f"Middle 50%: {middle_50}")

# Sliding window with slicing
def sliding_window(lst, window_size):
    """Generate sliding windows"""
    return [lst[i:i+window_size] for i in range(len(lst)-window_size+1)]

windows = sliding_window([1, 2, 3, 4, 5, 6], 3)
print(f"Sliding windows of size 3: {windows}")

print("\n--- Slice Performance ---")

# Slice vs loop performance
large_list = list(range(1000000))

# Slicing
start = time.time()
slice_result = large_list[::2]
slice_time = time.time() - start

# List comprehension
start = time.time()
comp_result = [large_list[i] for i in range(0, len(large_list), 2)]
comp_time = time.time() - start

print(f"Slice [::2] time: {slice_time:.6f}s")
print(f"Comprehension time: {comp_time:.6f}s")
print(f"Slice is {comp_time/slice_time:.2f}x faster")

# ===============================================================================
# 6. REAL-WORLD APPLICATIONS
# ===============================================================================

print("\n" + "=" * 80)
print("6. REAL-WORLD APPLICATIONS")
print("=" * 80)

print("\n--- Application 1: Data Processing Pipeline ---")

class DataProcessor:
    def __init__(self):
        self.pipeline = []
    
    def add_step(self, func):
        """Add processing step to pipeline"""
        self.pipeline.append(func)
    
    def process(self, data):
        """Process data through pipeline"""
        result = data[:]  # Start with copy of data
        
        for step in self.pipeline:
            result = step(result)
        
        return result
    
    def process_batches(self, data, batch_size=1000):
        """Process data in batches"""
        results = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            processed_batch = self.process(batch)
            results.extend(processed_batch)
        
        return results

# Example pipeline
processor = DataProcessor()

# Add processing steps
processor.add_step(lambda data: [x for x in data if x > 0])  # Filter positive
processor.add_step(lambda data: [x * 2 for x in data])       # Double values
processor.add_step(lambda data: sorted(data))                # Sort

# Process data
raw_data = [-5, 3, -2, 8, 1, -7, 4, 6, -1, 9]
processed = processor.process(raw_data)
print(f"Raw data: {raw_data}")
print(f"Processed: {processed}")

print("\n--- Application 2: Time Series Analysis ---")

class TimeSeriesAnalyzer:
    def __init__(self, data, timestamps=None):
        self.data = data[:]  # Copy to avoid mutations
        self.timestamps = timestamps or list(range(len(data)))
    
    def moving_average(self, window_size):
        """Calculate moving average"""
        if window_size > len(self.data):
            return []
        
        averages = []
        for i in range(len(self.data) - window_size + 1):
            window = self.data[i:i+window_size]
            averages.append(sum(window) / window_size)
        
        return averages
    
    def detect_outliers(self, threshold=2.0):
        """Detect outliers using standard deviation"""
        mean = sum(self.data) / len(self.data)
        variance = sum((x - mean) ** 2 for x in self.data) / len(self.data)
        std_dev = variance ** 0.5
        
        outliers = []
        for i, value in enumerate(self.data):
            if abs(value - mean) > threshold * std_dev:
                outliers.append((i, value))
        
        return outliers
    
    def find_peaks(self, min_height=None):
        """Find local peaks in the data"""
        if len(self.data) < 3:
            return []
        
        peaks = []
        for i in range(1, len(self.data) - 1):
            if (self.data[i] > self.data[i-1] and 
                self.data[i] > self.data[i+1]):
                if min_height is None or self.data[i] >= min_height:
                    peaks.append((i, self.data[i]))
        
        return peaks
    
    def trend_analysis(self):
        """Simple trend analysis"""
        if len(self.data) < 2:
            return "insufficient_data"
        
        increases = sum(1 for i in range(1, len(self.data)) 
                       if self.data[i] > self.data[i-1])
        decreases = sum(1 for i in range(1, len(self.data)) 
                       if self.data[i] < self.data[i-1])
        
        if increases > decreases * 1.5:
            return "increasing"
        elif decreases > increases * 1.5:
            return "decreasing"
        else:
            return "stable"

# Example usage
time_series_data = [10, 12, 11, 15, 18, 16, 14, 17, 20, 22, 19, 21, 25, 23]
analyzer = TimeSeriesAnalyzer(time_series_data)

moving_avg = analyzer.moving_average(3)
outliers = analyzer.detect_outliers(1.5)
peaks = analyzer.find_peaks()
trend = analyzer.trend_analysis()

print(f"Time series: {time_series_data}")
print(f"Moving average (window=3): {moving_avg}")
print(f"Outliers: {outliers}")
print(f"Peaks: {peaks}")
print(f"Trend: {trend}")

print("\n--- Application 3: Shopping Cart System ---")

class ShoppingCart:
    def __init__(self):
        self.items = []  # List of (product_id, name, price, quantity) tuples
    
    def add_item(self, product_id, name, price, quantity=1):
        """Add item to cart or update quantity if exists"""
        for i, (pid, pname, pprice, pqty) in enumerate(self.items):
            if pid == product_id:
                self.items[i] = (pid, pname, pprice, pqty + quantity)
                return
        
        self.items.append((product_id, name, price, quantity))
    
    def remove_item(self, product_id):
        """Remove item from cart"""
        self.items = [item for item in self.items if item[0] != product_id]
    
    def update_quantity(self, product_id, new_quantity):
        """Update item quantity"""
        if new_quantity <= 0:
            self.remove_item(product_id)
            return
        
        for i, (pid, pname, pprice, pqty) in enumerate(self.items):
            if pid == product_id:
                self.items[i] = (pid, pname, pprice, new_quantity)
                break
    
    def get_total(self):
        """Calculate total price"""
        return sum(price * quantity for _, _, price, quantity in self.items)
    
    def get_item_count(self):
        """Get total number of items"""
        return sum(quantity for _, _, _, quantity in self.items)
    
    def apply_discount(self, discount_percent):
        """Apply discount to all items"""
        discount_factor = (100 - discount_percent) / 100
        self.items = [
            (pid, name, price * discount_factor, qty)
            for pid, name, price, qty in self.items
        ]
    
    def get_most_expensive(self):
        """Get most expensive item by total price"""
        if not self.items:
            return None
        
        return max(self.items, key=lambda item: item[2] * item[3])
    
    def sort_by_price(self, reverse=False):
        """Sort items by unit price"""
        self.items.sort(key=lambda item: item[2], reverse=reverse)

# Example usage
cart = ShoppingCart()
cart.add_item(1, "Laptop", 999.99, 1)
cart.add_item(2, "Mouse", 29.99, 2)
cart.add_item(3, "Keyboard", 79.99, 1)
cart.add_item(2, "Mouse", 29.99, 1)  # Add another mouse

print(f"Cart items: {cart.items}")
print(f"Total: ${cart.get_total():.2f}")
print(f"Item count: {cart.get_item_count()}")

most_expensive = cart.get_most_expensive()
print(f"Most expensive: {most_expensive}")

cart.apply_discount(10)  # 10% discount
print(f"After 10% discount: ${cart.get_total():.2f}")

print("\n--- Application 4: Log File Analysis ---")

class LogAnalyzer:
    def __init__(self, log_entries):
        self.logs = log_entries[:]
    
    def filter_by_level(self, level):
        """Filter logs by level (ERROR, WARN, INFO, DEBUG)"""
        return [log for log in self.logs if level.upper() in log.upper()]
    
    def find_errors_in_timeframe(self, start_time, end_time):
        """Find errors within time frame"""
        errors = []
        for log in self.logs:
            if 'ERROR' in log.upper():
                # Simple time extraction (assumes timestamp at start)
                parts = log.split()
                if parts and start_time <= parts[0] <= end_time:
                    errors.append(log)
        return errors
    
    def get_error_summary(self):
        """Get summary of different error types"""
        error_counts = defaultdict(int)
        
        for log in self.logs:
            if 'ERROR' in log.upper():
                # Extract error type (simplified)
                if 'connection' in log.lower():
                    error_counts['Connection Error'] += 1
                elif 'timeout' in log.lower():
                    error_counts['Timeout Error'] += 1
                elif 'database' in log.lower():
                    error_counts['Database Error'] += 1
                else:
                    error_counts['Other Error'] += 1
        
        return dict(error_counts)
    
    def find_patterns(self, pattern):
        """Find logs matching pattern"""
        return [log for log in self.logs if pattern.lower() in log.lower()]

# Example log data
log_data = [
    "2025-01-01 10:00:00 INFO User login successful",
    "2025-01-01 10:05:00 ERROR Database connection failed",
    "2025-01-01 10:10:00 WARN High memory usage detected",
    "2025-01-01 10:15:00 ERROR Timeout while connecting to server",
    "2025-01-01 10:20:00 INFO User logout",
    "2025-01-01 10:25:00 ERROR Database connection failed",
    "2025-01-01 10:30:00 DEBUG Cache miss for user data"
]

analyzer = LogAnalyzer(log_data)

errors = analyzer.filter_by_level('ERROR')
error_summary = analyzer.get_error_summary()
database_issues = analyzer.find_patterns('database')

print(f"Error logs: {len(errors)}")
for error in errors:
    print(f"  {error}")

print(f"Error summary: {error_summary}")
print(f"Database related logs: {len(database_issues)}")

# ===============================================================================
# 7. INTERVIEW PROBLEMS & SOLUTIONS
# ===============================================================================

print("\n" + "=" * 80)
print("7. INTERVIEW PROBLEMS & SOLUTIONS")
print("=" * 80)

print("\n--- Problem 1: Two Sum ---")

def two_sum(nums, target):
    """Find indices of two numbers that add up to target"""
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# Example
nums = [2, 7, 11, 15]
target = 9
result = two_sum(nums, target)
print(f"Two sum: nums={nums}, target={target}")
print(f"Result: indices {result}, values {[nums[i] for i in result]}")

print("\n--- Problem 2: Maximum Subarray (Kadane's Algorithm) ---")

def max_subarray(nums):
    """Find maximum sum of contiguous subarray"""
    if not nums:
        return 0
    
    max_sum = current_sum = nums[0]
    start = end = temp_start = 0
    
    for i in range(1, len(nums)):
        if current_sum < 0:
            current_sum = nums[i]
            temp_start = i
        else:
            current_sum += nums[i]
        
        if current_sum > max_sum:
            max_sum = current_sum
            start = temp_start
            end = i
    
    return max_sum, start, end

# Example
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
max_sum, start, end = max_subarray(nums)
print(f"Array: {nums}")
print(f"Maximum subarray sum: {max_sum}")
print(f"Subarray: {nums[start:end+1]} (indices {start} to {end})")

print("\n--- Problem 3: Merge Intervals ---")

def merge_intervals(intervals):
    """Merge overlapping intervals"""
    if not intervals:
        return []
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last_merged = merged[-1]
        
        # If current overlaps with last merged
        if current[0] <= last_merged[1]:
            # Merge intervals
            merged[-1] = [last_merged[0], max(last_merged[1], current[1])]
        else:
            # No overlap, add current
            merged.append(current)
    
    return merged

# Example
intervals = [[1,3],[2,6],[8,10],[15,18]]
merged = merge_intervals(intervals)
print(f"Original intervals: {intervals}")
print(f"Merged intervals: {merged}")

print("\n--- Problem 4: Product of Array Except Self ---")

def product_except_self(nums):
    """Calculate product of all elements except self without division"""
    n = len(nums)
    result = [1] * n
    
    # Left pass
    for i in range(1, n):
        result[i] = result[i-1] * nums[i-1]
    
    # Right pass
    right_product = 1
    for i in range(n-1, -1, -1):
        result[i] *= right_product
        right_product *= nums[i]
    
    return result

# Example
nums = [1, 2, 3, 4]
result = product_except_self(nums)
print(f"Array: {nums}")
print(f"Product except self: {result}")

print("\n--- Problem 5: Rotate Array ---")

def rotate_array(nums, k):
    """Rotate array to the right by k steps"""
    n = len(nums)
    k = k % n  # Handle k > n
    
    # Method 1: Using extra space
    result = [0] * n
    for i in range(n):
        result[(i + k) % n] = nums[i]
    
    return result

def rotate_array_in_place(nums, k):
    """Rotate array in place"""
    n = len(nums)
    k = k % n
    
    # Reverse entire array
    nums.reverse()
    # Reverse first k elements
    nums[:k] = nums[:k][::-1]
    # Reverse remaining elements
    nums[k:] = nums[k:][::-1]
    
    return nums

# Example
nums = [1, 2, 3, 4, 5, 6, 7]
k = 3
rotated = rotate_array(nums[:], k)  # Use copy to preserve original
print(f"Original: {nums}")
print(f"Rotated by {k}: {rotated}")

print("\n--- Problem 6: Find Minimum in Rotated Sorted Array ---")

def find_min_rotated(nums):
    """Find minimum in rotated sorted array"""
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = (left + right) // 2
        
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    
    return nums[left]

# Example
rotated_array = [4, 5, 6, 7, 0, 1, 2]
min_val = find_min_rotated(rotated_array)
print(f"Rotated array: {rotated_array}")
print(f"Minimum value: {min_val}")

print("\n--- Problem 7: Container With Most Water ---")

def max_area(heights):
    """Find container with most water"""
    left, right = 0, len(heights) - 1
    max_water = 0
    
    while left < right:
        # Calculate water area
        width = right - left
        height = min(heights[left], heights[right])
        water = width * height
        max_water = max(max_water, water)
        
        # Move pointer with smaller height
        if heights[left] < heights[right]:
            left += 1
        else:
            right -= 1
    
    return max_water

# Example
heights = [1, 8, 6, 2, 5, 4, 8, 3, 7]
max_water = max_area(heights)
print(f"Heights: {heights}")
print(f"Maximum water area: {max_water}")

print("\n--- Problem 8: Sliding Window Maximum ---")

def sliding_window_maximum(nums, k):
    """Find maximum in each sliding window of size k"""
    if not nums or k == 0:
        return []
    
    from collections import deque
    
    result = []
    window = deque()  # Store indices
    
    for i in range(len(nums)):
        # Remove indices outside current window
        while window and window[0] <= i - k:
            window.popleft()
        
        # Remove indices with smaller values than current
        while window and nums[window[-1]] <= nums[i]:
            window.pop()
        
        window.append(i)
        
        # Add maximum to result when window is full
        if i >= k - 1:
            result.append(nums[window[0]])
    
    return result

# Example
nums = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3
result = sliding_window_maximum(nums, k)
print(f"Array: {nums}")
print(f"Sliding window max (k={k}): {result}")

print("\n--- Problem 9: Longest Increasing Subsequence ---")

def longest_increasing_subsequence(nums):
    """Find length of longest increasing subsequence"""
    if not nums:
        return 0
    
    # dp[i] = length of LIS ending at index i
    dp = [1] * len(nums)
    
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

def lis_with_path(nums):
    """Find LIS and reconstruct the actual subsequence"""
    if not nums:
        return []
    
    dp = [1] * len(nums)
    parent = [-1] * len(nums)
    
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
    
    # Find the index with maximum LIS length
    max_length = max(dp)
    max_index = dp.index(max_length)
    
    # Reconstruct the LIS
    lis = []
    current = max_index
    while current != -1:
        lis.append(nums[current])
        current = parent[current]
    
    return list(reversed(lis))

# Example
nums = [10, 9, 2, 5, 3, 7, 101, 18]
lis_length = longest_increasing_subsequence(nums)
lis_sequence = lis_with_path(nums)
print(f"Array: {nums}")
print(f"LIS length: {lis_length}")
print(f"LIS sequence: {lis_sequence}")

print("\n--- Problem 10: Trapping Rain Water ---")

def trap_rain_water(heights):
    """Calculate trapped rain water"""
    if not heights or len(heights) < 3:
        return 0
    
    left, right = 0, len(heights) - 1
    left_max = right_max = water = 0
    
    while left < right:
        if heights[left] < heights[right]:
            if heights[left] >= left_max:
                left_max = heights[left]
            else:
                water += left_max - heights[left]
            left += 1
        else:
            if heights[right] >= right_max:
                right_max = heights[right]
            else:
                water += right_max - heights[right]
            right -= 1
    
    return water

# Example
heights = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
trapped_water = trap_rain_water(heights)
print(f"Heights: {heights}")
print(f"Trapped water: {trapped_water}")

print("\n" + "=" * 80)
print("LIST LEARNING GUIDE PART 2 COMPLETE")
print("Continue with list_learning_part3.py for remaining sections...")
print("=" * 80)
