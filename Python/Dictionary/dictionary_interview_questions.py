"""
PYTHON DICTIONARY INTERVIEW QUESTIONS
=====================================

This file contains comprehensive interview questions about dictionaries
organized by difficulty level and company patterns.

Categories:
- Easy (0-1 year experience)
- Medium (1-2 years experience) 
- Hard (2+ years experience)
- System Design (Senior level)

Author: Interview Preparation Guide
Date: August 2025
"""

# =============================================================================
# EASY LEVEL INTERVIEW QUESTIONS (0-1 YEAR EXPERIENCE)
# =============================================================================

print("=" * 70)
print("EASY LEVEL INTERVIEW QUESTIONS")
print("=" * 70)

# Question 1: Two Sum
print("\n1. TWO SUM")
print("Given an array of integers and a target, return indices of two numbers that add up to target")
print("-" * 80)

def two_sum(nums, target):
    """
    Time: O(n), Space: O(n)
    Companies: Amazon, Google, Facebook, Microsoft
    """
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# Test
nums = [2, 7, 11, 15]
target = 9
result = two_sum(nums, target)
print(f"Input: {nums}, Target: {target}")
print(f"Output: {result}")
print(f"Explanation: nums[{result[0]}] + nums[{result[1]}] = {nums[result[0]]} + {nums[result[1]]} = {target}")

# Question 2: Valid Anagram
print("\n2. VALID ANAGRAM")
print("Given two strings, determine if they are anagrams of each other")
print("-" * 80)

def is_anagram(s, t):
    """
    Time: O(n), Space: O(1) - limited to 26 characters
    Companies: Facebook, Amazon, Google
    """
    if len(s) != len(t):
        return False
    
    char_count = {}
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    
    for char in t:
        if char not in char_count:
            return False
        char_count[char] -= 1
        if char_count[char] == 0:
            del char_count[char]
    
    return len(char_count) == 0

# Test
s1, t1 = "anagram", "nagaram"
s2, t2 = "rat", "car"
print(f"'{s1}' and '{t1}': {is_anagram(s1, t1)}")
print(f"'{s2}' and '{t2}': {is_anagram(s2, t2)}")

# Question 3: First Non-Repeating Character
print("\n3. FIRST NON-REPEATING CHARACTER")
print("Find the first non-repeating character in a string")
print("-" * 80)

def first_unique_char(s):
    """
    Time: O(n), Space: O(1) - limited to 26 characters
    Companies: Amazon, Microsoft, Apple
    """
    char_count = {}
    
    # Count frequencies
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    
    # Find first unique
    for i, char in enumerate(s):
        if char_count[char] == 1:
            return i
    
    return -1

# Test
test_string = "leetcode"
index = first_unique_char(test_string)
print(f"String: '{test_string}'")
print(f"First unique character index: {index}")
if index != -1:
    print(f"Character: '{test_string[index]}'")

# Question 4: Contains Duplicate
print("\n4. CONTAINS DUPLICATE")
print("Check if array contains any duplicates")
print("-" * 80)

def contains_duplicate(nums):
    """
    Time: O(n), Space: O(n)
    Companies: Google, Facebook, Amazon
    """
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False

# Alternative using dictionary
def contains_duplicate_dict(nums):
    """Using dictionary approach"""
    count = {}
    for num in nums:
        if num in count:
            return True
        count[num] = 1
    return False

# Test
test_arrays = [[1, 2, 3, 1], [1, 2, 3, 4], [1, 1, 1, 3, 3, 4, 3, 2, 4, 2]]
for arr in test_arrays:
    print(f"Array: {arr}, Contains duplicate: {contains_duplicate(arr)}")

# Question 5: Group Anagrams
print("\n5. GROUP ANAGRAMS")
print("Group strings that are anagrams of each other")
print("-" * 80)

def group_anagrams(strs):
    """
    Time: O(n * k log k), Space: O(n * k)
    where n = number of strings, k = max length of string
    Companies: Amazon, Facebook, Google, Uber
    """
    groups = {}
    
    for s in strs:
        # Sort characters to create key
        key = ''.join(sorted(s))
        if key not in groups:
            groups[key] = []
        groups[key].append(s)
    
    return list(groups.values())

# Alternative using character count as key
def group_anagrams_v2(strs):
    """Using character count as key"""
    from collections import defaultdict
    
    groups = defaultdict(list)
    
    for s in strs:
        # Create character count signature
        count = [0] * 26
        for char in s:
            count[ord(char) - ord('a')] += 1
        groups[tuple(count)].append(s)
    
    return list(groups.values())

# Test
words = ["eat", "tea", "tan", "ate", "nat", "bat"]
grouped = group_anagrams(words)
print(f"Words: {words}")
print(f"Grouped: {grouped}")

# =============================================================================
# MEDIUM LEVEL INTERVIEW QUESTIONS (1-2 YEARS EXPERIENCE)
# =============================================================================

print("\n" + "=" * 70)
print("MEDIUM LEVEL INTERVIEW QUESTIONS")
print("=" * 70)

# Question 6: Top K Frequent Elements
print("\n6. TOP K FREQUENT ELEMENTS")
print("Find k most frequent elements in array")
print("-" * 80)

def top_k_frequent(nums, k):
    """
    Time: O(n log k), Space: O(n)
    Companies: Amazon, Facebook, Google, LinkedIn
    """
    from collections import Counter
    import heapq
    
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)

# Alternative using bucket sort
def top_k_frequent_bucket(nums, k):
    """
    Time: O(n), Space: O(n)
    Bucket sort approach - more efficient
    """
    from collections import Counter
    
    count = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    
    # Place elements in buckets by frequency
    for num, freq in count.items():
        buckets[freq].append(num)
    
    # Collect top k elements
    result = []
    for i in range(len(buckets) - 1, 0, -1):
        for num in buckets[i]:
            result.append(num)
            if len(result) == k:
                return result
    
    return result

# Test
nums = [1, 1, 1, 2, 2, 3]
k = 2
result = top_k_frequent(nums, k)
print(f"Array: {nums}, K: {k}")
print(f"Top {k} frequent: {result}")

# Question 7: Subarray Sum Equals K
print("\n7. SUBARRAY SUM EQUALS K")
print("Count number of continuous subarrays whose sum equals k")
print("-" * 80)

def subarray_sum(nums, k):
    """
    Time: O(n), Space: O(n)
    Companies: Facebook, Google, Amazon
    """
    count = 0
    cumsum = 0
    sum_count = {0: 1}  # Handle case where subarray starts from index 0
    
    for num in nums:
        cumsum += num
        
        # If (cumsum - k) exists, we found valid subarrays
        if cumsum - k in sum_count:
            count += sum_count[cumsum - k]
        
        # Update count of current cumsum
        sum_count[cumsum] = sum_count.get(cumsum, 0) + 1
    
    return count

# Test
test_cases = [
    ([1, 1, 1], 2),
    ([1, 2, 3], 3),
    ([1, -1, 0], 0)
]

for nums, k in test_cases:
    result = subarray_sum(nums, k)
    print(f"Array: {nums}, K: {k}, Count: {result}")

# Question 8: Longest Substring Without Repeating Characters
print("\n8. LONGEST SUBSTRING WITHOUT REPEATING CHARACTERS")
print("Find length of longest substring without repeating characters")
print("-" * 80)

def length_of_longest_substring(s):
    """
    Time: O(n), Space: O(min(m,n)) where m is charset size
    Companies: Amazon, Facebook, Google, Microsoft, Apple
    """
    char_index = {}
    max_length = 0
    start = 0
    
    for end, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        
        char_index[char] = end
        max_length = max(max_length, end - start + 1)
    
    return max_length

# Test
test_strings = ["abcabcbb", "bbbbb", "pwwkew", ""]
for s in test_strings:
    length = length_of_longest_substring(s)
    print(f"String: '{s}', Longest substring length: {length}")

# Question 9: 4Sum II
print("\n9. 4SUM II")
print("Count tuples (i,j,k,l) such that A[i] + B[j] + C[k] + D[l] = 0")
print("-" * 80)

def four_sum_count(A, B, C, D):
    """
    Time: O(n²), Space: O(n²)
    Companies: Facebook, Amazon
    """
    sum_count = {}
    
    # Count all possible sums of A[i] + B[j]
    for a in A:
        for b in B:
            sum_ab = a + b
            sum_count[sum_ab] = sum_count.get(sum_ab, 0) + 1
    
    count = 0
    # For each C[k] + D[l], check if -(C[k] + D[l]) exists in sum_count
    for c in C:
        for d in D:
            target = -(c + d)
            if target in sum_count:
                count += sum_count[target]
    
    return count

# Test
A = [1, 2]
B = [-2, -1]
C = [-1, 2]
D = [0, 2]
result = four_sum_count(A, B, C, D)
print(f"A: {A}, B: {B}, C: {C}, D: {D}")
print(f"Number of tuples: {result}")

# Question 10: Word Pattern
print("\n10. WORD PATTERN")
print("Check if string follows the same pattern")
print("-" * 80)

def word_pattern(pattern, s):
    """
    Time: O(n), Space: O(n)
    Companies: Google, Facebook
    """
    words = s.split()
    
    if len(pattern) != len(words):
        return False
    
    char_to_word = {}
    word_to_char = {}
    
    for char, word in zip(pattern, words):
        if char in char_to_word:
            if char_to_word[char] != word:
                return False
        else:
            char_to_word[char] = word
        
        if word in word_to_char:
            if word_to_char[word] != char:
                return False
        else:
            word_to_char[word] = char
    
    return True

# Test
test_cases = [
    ("abba", "dog cat cat dog"),
    ("abba", "dog cat cat fish"),
    ("aaaa", "dog cat cat dog")
]

for pattern, s in test_cases:
    result = word_pattern(pattern, s)
    print(f"Pattern: '{pattern}', String: '{s}', Matches: {result}")

# =============================================================================
# HARD LEVEL INTERVIEW QUESTIONS (2+ YEARS EXPERIENCE)
# =============================================================================

print("\n" + "=" * 70)
print("HARD LEVEL INTERVIEW QUESTIONS")
print("=" * 70)

# Question 11: Minimum Window Substring
print("\n11. MINIMUM WINDOW SUBSTRING")
print("Find minimum window in S that contains all characters in T")
print("-" * 80)

def min_window(s, t):
    """
    Time: O(|s| + |t|), Space: O(|s| + |t|)
    Companies: Facebook, Amazon, Google, Microsoft, Uber
    """
    if not s or not t:
        return ""
    
    # Dictionary to keep count of all unique characters in t
    dict_t = {}
    for char in t:
        dict_t[char] = dict_t.get(char, 0) + 1
    
    required = len(dict_t)
    formed = 0
    
    # Dictionary to keep count of current window
    window_counts = {}
    
    # Left and right pointers
    l, r = 0, 0
    
    # Answer tuple (window length, left, right)
    ans = float("inf"), None, None
    
    while r < len(s):
        # Add character from right to window
        character = s[r]
        window_counts[character] = window_counts.get(character, 0) + 1
        
        # Check if frequency of current character equals desired count in t
        if character in dict_t and window_counts[character] == dict_t[character]:
            formed += 1
        
        # Try to contract window until it ceases to be 'desirable'
        while l <= r and formed == required:
            character = s[l]
            
            # Save smallest window
            if r - l + 1 < ans[0]:
                ans = (r - l + 1, l, r)
            
            # Remove character from left
            window_counts[character] -= 1
            if character in dict_t and window_counts[character] < dict_t[character]:
                formed -= 1
            
            l += 1
        
        r += 1
    
    return "" if ans[0] == float("inf") else s[ans[1]:ans[2] + 1]

# Test
s = "ADOBECODEBANC"
t = "ABC"
result = min_window(s, t)
print(f"S: '{s}', T: '{t}'")
print(f"Minimum window: '{result}'")

# Question 12: Alien Dictionary
print("\n12. ALIEN DICTIONARY")
print("Find order of characters in alien language from sorted words")
print("-" * 80)

def alien_order(words):
    """
    Time: O(C) where C is total content of words, Space: O(1) - limited alphabet
    Companies: Google, Facebook, Amazon, Airbnb
    """
    from collections import defaultdict, deque
    
    # Build graph
    graph = defaultdict(set)
    in_degree = defaultdict(int)
    
    # Initialize all characters
    for word in words:
        for char in word:
            in_degree[char] = 0
    
    # Build edges
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        min_len = min(len(word1), len(word2))
        
        # Check for invalid case
        if len(word1) > len(word2) and word1[:min_len] == word2[:min_len]:
            return ""
        
        # Find first different character
        for j in range(min_len):
            if word1[j] != word2[j]:
                if word2[j] not in graph[word1[j]]:
                    graph[word1[j]].add(word2[j])
                    in_degree[word2[j]] += 1
                break
    
    # Topological sort
    queue = deque([char for char in in_degree if in_degree[char] == 0])
    result = []
    
    while queue:
        char = queue.popleft()
        result.append(char)
        
        for neighbor in graph[char]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return ''.join(result) if len(result) == len(in_degree) else ""

# Test
words = ["wrt", "wrf", "er", "ett", "rftt"]
order = alien_order(words)
print(f"Words: {words}")
print(f"Alien order: '{order}'")

# Question 13: Number of Islands II
print("\n13. NUMBER OF ISLANDS II (DYNAMIC)")
print("Count islands as land is added dynamically")
print("-" * 80)

def num_islands_2(m, n, positions):
    """
    Time: O(k * α(mn)) where k is number of positions, Space: O(mn)
    Companies: Google, Facebook
    """
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False
    
    parent = {}
    result = []
    islands = 0
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    for r, c in positions:
        if (r, c) in parent:
            result.append(islands)
            continue
        
        parent[(r, c)] = (r, c)
        islands += 1
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and (nr, nc) in parent:
                if union((r, c), (nr, nc)):
                    islands -= 1
        
        result.append(islands)
    
    return result

# Test
m, n = 3, 3
positions = [[0,0], [0,1], [1,2], [2,1]]
result = num_islands_2(m, n, positions)
print(f"Grid: {m}x{n}, Positions: {positions}")
print(f"Islands count after each addition: {result}")

# =============================================================================
# SYSTEM DESIGN LEVEL QUESTIONS (SENIOR LEVEL)
# =============================================================================

print("\n" + "=" * 70)
print("SYSTEM DESIGN LEVEL QUESTIONS")
print("=" * 70)

# Question 14: LRU Cache
print("\n14. LRU CACHE IMPLEMENTATION")
print("Design and implement Least Recently Used cache")
print("-" * 80)

class LRUCache:
    """
    All operations in O(1) time
    Companies: Facebook, Amazon, Google, Microsoft, Apple
    """
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        
        # Create dummy head and tail
        self.head = self.Node(0, 0)
        self.tail = self.Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head
    
    class Node:
        def __init__(self, key, val):
            self.key = key
            self.val = val
            self.prev = None
            self.next = None
    
    def get(self, key):
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add_to_head(node)
            return node.val
        return -1
    
    def put(self, key, value):
        if key in self.cache:
            node = self.cache[key]
            node.val = value
            self._remove(node)
            self._add_to_head(node)
        else:
            if len(self.cache) >= self.capacity:
                # Remove LRU
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]
            
            new_node = self.Node(key, value)
            self.cache[key] = new_node
            self._add_to_head(new_node)
    
    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _add_to_head(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

# Test LRU Cache
print("Testing LRU Cache:")
lru = LRUCache(2)
lru.put(1, 1)
lru.put(2, 2)
print(f"get(1): {lru.get(1)}")  # returns 1
lru.put(3, 3)  # evicts key 2
print(f"get(2): {lru.get(2)}")  # returns -1 (not found)
lru.put(4, 4)  # evicts key 1
print(f"get(1): {lru.get(1)}")  # returns -1 (not found)
print(f"get(3): {lru.get(3)}")  # returns 3
print(f"get(4): {lru.get(4)}")  # returns 4

# Question 15: Design Twitter
print("\n15. DESIGN TWITTER")
print("Design a simplified version of Twitter")
print("-" * 80)

class Twitter:
    """
    Companies: Twitter, Facebook, Amazon
    """
    
    def __init__(self):
        self.tweets = {}  # user_id -> list of (time, tweet_id)
        self.following = {}  # user_id -> set of followee_ids
        self.time = 0
    
    def postTweet(self, userId, tweetId):
        if userId not in self.tweets:
            self.tweets[userId] = []
        self.tweets[userId].append((self.time, tweetId))
        self.time += 1
    
    def getNewsFeed(self, userId):
        import heapq
        
        # Get all relevant users (self + following)
        users = self.following.get(userId, set()) | {userId}
        
        # Use min heap to get top 10 most recent tweets
        heap = []
        
        for user in users:
            if user in self.tweets:
                tweets = self.tweets[user]
                for i in range(len(tweets) - 1, max(-1, len(tweets) - 11), -1):
                    time, tweet_id = tweets[i]
                    if len(heap) < 10:
                        heapq.heappush(heap, (time, tweet_id))
                    elif time > heap[0][0]:
                        heapq.heapreplace(heap, (time, tweet_id))
        
        # Extract tweets and sort by time (most recent first)
        result = []
        while heap:
            time, tweet_id = heapq.heappop(heap)
            result.append(tweet_id)
        
        return result[::-1][:10]
    
    def follow(self, followerId, followeeId):
        if followerId not in self.following:
            self.following[followerId] = set()
        self.following[followerId].add(followeeId)
    
    def unfollow(self, followerId, followeeId):
        if followerId in self.following:
            self.following[followerId].discard(followeeId)

# Test Twitter
print("Testing Twitter Design:")
twitter = Twitter()
twitter.postTweet(1, 5)
print(f"User 1 news feed: {twitter.getNewsFeed(1)}")
twitter.follow(1, 2)
twitter.postTweet(2, 6)
print(f"User 1 news feed after following 2: {twitter.getNewsFeed(1)}")

# =============================================================================
# VERBAL/CONCEPTUAL INTERVIEW QUESTIONS
# =============================================================================

print("\n" + "=" * 70)
print("VERBAL/CONCEPTUAL INTERVIEW QUESTIONS")
print("=" * 70)

print("""
These are the theoretical questions commonly asked about dictionaries in interviews.
Be prepared to explain concepts clearly and provide examples.
""")

# =============================================================================
# BASIC CONCEPTS (ALL LEVELS)
# =============================================================================

print("\n" + "=" * 50)
print("BASIC CONCEPTS QUESTIONS")
print("=" * 50)

basic_questions = """
1. WHAT IS A DICTIONARY IN PYTHON?
   Answer: A dictionary is a mutable, unordered collection of key-value pairs.
   - Keys must be immutable (hashable) objects
   - Values can be any data type
   - Provides O(1) average access time
   - Also called hash map or associative array

2. HOW ARE DICTIONARIES IMPLEMENTED INTERNALLY?
   Answer: Python dictionaries use hash tables:
   - Hash function converts keys to indices
   - Open addressing with random probing (Python 3.6+)
   - Compact representation saves memory
   - Load factor maintained at ~2/3 for performance

3. WHAT MAKES A GOOD HASH FUNCTION?
   Answer: A good hash function should:
   - Distribute keys uniformly across hash table
   - Be deterministic (same input = same output)
   - Be fast to compute
   - Minimize collisions
   - Example: Python uses SipHash for strings

4. WHAT IS A HASH COLLISION?
   Answer: When two different keys hash to the same index:
   - Inevitable due to pigeonhole principle
   - Python uses open addressing to resolve
   - Can degrade performance to O(n) in worst case
   - Good hash functions minimize collisions

5. WHY CAN'T LISTS BE DICTIONARY KEYS?
   Answer: Lists are mutable (not hashable):
   - Hash value could change if list is modified
   - Would break dictionary integrity
   - Only immutable objects can be keys
   - Example: tuples, strings, numbers are valid keys

6. WHAT'S THE DIFFERENCE BETWEEN DICT AND SET?
   Answer: 
   - Dict: key-value pairs, access by key
   - Set: unique values only, membership testing
   - Both use hash tables internally
   - Set is like dict with only keys, no values
"""

print(basic_questions)

# =============================================================================
# PERFORMANCE AND COMPLEXITY QUESTIONS
# =============================================================================

print("\n" + "=" * 50)
print("PERFORMANCE & COMPLEXITY QUESTIONS")
print("=" * 50)

performance_questions = """
7. WHAT IS THE TIME COMPLEXITY OF DICTIONARY OPERATIONS?
   Answer:
   - Access/Insert/Delete: O(1) average, O(n) worst case
   - Iteration: O(n)
   - The worst case happens when many hash collisions occur
   - In practice, almost always O(1) due to good hash function

8. WHY IS DICTIONARY ACCESS O(1)?
   Answer: Hash table implementation:
   - Hash function maps key directly to index
   - No need to search through elements
   - Direct array access after hashing
   - Example: dict[key] → hash(key) → array[index]

9. WHEN WOULD DICTIONARY OPERATIONS BE O(N)?
   Answer: In worst-case scenarios:
   - All keys hash to same bucket (hash collision)
   - Pathological input designed to cause collisions
   - Very rare in practice with good hash functions
   - Python's hash function is robust against attacks

10. HOW MUCH MEMORY DO DICTIONARIES USE?
    Answer: Memory overhead considerations:
    - ~240 bytes base overhead + 24 bytes per key-value pair
    - Hash table typically 1.5x-3x the number of elements
    - More memory efficient than lists for sparse data
    - Compact dict representation (Python 3.6+) saves ~20% memory

11. WHEN TO USE DICT VS LIST VS SET?
    Answer: Choose based on use case:
    - Dict: key-value mapping, O(1) lookup by key
    - List: ordered collection, access by index, duplicates allowed
    - Set: unique elements, O(1) membership testing
    - Example: phonebook (dict), shopping list (list), unique IDs (set)
"""

print(performance_questions)

# =============================================================================
# DESIGN AND ARCHITECTURE QUESTIONS
# =============================================================================

print("\n" + "=" * 50)
print("DESIGN & ARCHITECTURE QUESTIONS")
print("=" * 50)

design_questions = """
12. HOW WOULD YOU IMPLEMENT A DICTIONARY FROM SCRATCH?
    Answer: Key components needed:
    - Hash function to convert keys to indices
    - Array/bucket storage for key-value pairs
    - Collision resolution strategy (chaining/open addressing)
    - Dynamic resizing when load factor gets high
    - Methods: get, put, delete, iterate

13. WHAT IS A LOAD FACTOR AND WHY IS IT IMPORTANT?
    Answer: Load factor = number of elements / table size
    - Measures how full the hash table is
    - Python maintains ~66% load factor
    - Too high → more collisions, slower operations
    - Too low → wasted memory
    - Triggers resize when threshold exceeded

14. HOW DOES DICTIONARY RESIZING WORK?
    Answer: Dynamic resizing process:
    - Create new larger hash table (usually 2x size)
    - Rehash all existing keys to new positions
    - Copy key-value pairs to new table
    - Delete old table
    - Expensive O(n) operation but amortized over time

15. EXPLAIN DIFFERENT COLLISION RESOLUTION STRATEGIES?
    Answer: Two main approaches:
    - Chaining: Each bucket stores linked list of colliding items
    - Open Addressing: Find next available slot (linear/quadratic probing)
    - Python uses open addressing with random probing
    - Trade-offs: memory vs. cache performance

16. WHAT ARE THE ADVANTAGES OF DICTIONARIES OVER OTHER DATA STRUCTURES?
    Answer: Key advantages:
    - O(1) average lookup time vs O(log n) for trees
    - Natural key-value mapping unlike arrays
    - Dynamic sizing unlike fixed arrays
    - Memory efficient for sparse data
    - Built-in to most languages
"""

print(design_questions)

# =============================================================================
# PYTHON-SPECIFIC QUESTIONS
# =============================================================================

print("\n" + "=" * 50)
print("PYTHON-SPECIFIC QUESTIONS")
print("=" * 50)

python_questions = """
17. HOW DID DICTIONARY ORDERING CHANGE IN PYTHON 3.7+?
    Answer: Major change in behavior:
    - Before 3.7: Order not guaranteed (implementation detail)
    - Python 3.7+: Insertion order preserved (language guarantee)
    - Made possible by compact dict implementation
    - OrderedDict still useful for explicit ordering operations

18. WHAT'S THE DIFFERENCE BETWEEN dict() AND {}?
    Answer: Both create dictionaries but:
    - {} is literal syntax, faster for empty/simple dicts
    - dict() is constructor call, slower but more flexible
    - dict() can take iterables, keyword args
    - {} only for literal key-value pairs
    - Performance: {} is ~3x faster for empty dict

19. EXPLAIN get() vs [] FOR ACCESSING DICTIONARY VALUES?
    Answer: Safety vs. performance trade-off:
    - dict[key]: Faster but raises KeyError if key missing
    - dict.get(key): Slower but returns None/default if missing
    - Use [] when you're sure key exists
    - Use get() for safe access with unknown keys

20. WHAT ARE DICTIONARY VIEWS AND WHY ARE THEY USEFUL?
    Answer: Dynamic views of dict data:
    - keys(), values(), items() return view objects
    - Views update automatically when dict changes
    - Memory efficient (don't create copies)
    - Support set-like operations (intersection, union)
    - Use list() to get snapshot if needed

21. WHEN WOULD YOU USE defaultdict vs REGULAR DICT?
    Answer: defaultdict for automatic initialization:
    - Regular dict: Need to check if key exists first
    - defaultdict: Automatically creates missing keys
    - Cleaner code for grouping, counting operations
    - Performance benefit (no key existence checks)
    - Example: grouping items by category

22. EXPLAIN DICTIONARY COMPREHENSIONS VS TRADITIONAL LOOPS?
    Answer: Functional vs imperative approach:
    - Comprehensions: More Pythonic, often faster
    - Traditional loops: More readable for complex logic
    - Comprehensions use optimized C code internally
    - Memory efficient (generator-like behavior)
    - Example: {k: v for k, v in items if condition}
"""

print(python_questions)

# =============================================================================
# ADVANCED CONCEPTS QUESTIONS
# =============================================================================

print("\n" + "=" * 50)
print("ADVANCED CONCEPTS QUESTIONS")
print("=" * 50)

advanced_questions = """
23. WHAT IS THE DIFFERENCE BETWEEN SHALLOW AND DEEP COPY FOR DICTIONARIES?
    Answer: Copying behavior with nested objects:
    - Shallow copy: Copies references to nested objects
    - Deep copy: Recursively copies nested objects
    - dict.copy() and dict() create shallow copies
    - Use copy.deepcopy() for true independent copy
    - Matters when values are mutable objects

24. HOW DO YOU HANDLE THREAD SAFETY WITH DICTIONARIES?
    Answer: Concurrency considerations:
    - Python dicts are not thread-safe
    - Use threading.Lock for synchronization
    - Consider queue.Queue for producer-consumer
    - multiprocessing.Manager() for cross-process
    - Read operations generally safe, writes need locking

25. EXPLAIN WEAK REFERENCES IN CONTEXT OF DICTIONARIES?
    Answer: Memory management pattern:
    - weakref.WeakKeyDictionary/WeakValueDictionary
    - Don't prevent garbage collection of referenced objects
    - Useful for caches, callbacks, observer patterns
    - Automatically remove entries when objects deleted
    - Prevents memory leaks in circular references

26. WHAT ARE SOME MEMORY OPTIMIZATION TECHNIQUES FOR LARGE DICTIONARIES?
    Answer: Optimization strategies:
    - Use __slots__ in custom objects to reduce overhead
    - Consider array.array for numeric data
    - Use sys.getsizeof() to measure memory usage
    - Implement custom hash methods for complex keys
    - Consider alternative data structures (tries, bloom filters)

27. HOW WOULD YOU IMPLEMENT AN LRU CACHE USING DICTIONARIES?
    Answer: Combine dict + doubly linked list:
    - Dict provides O(1) access to nodes
    - Linked list maintains order (recent to old)
    - Move accessed items to front
    - Remove from tail when capacity exceeded
    - OrderedDict can simplify implementation

28. EXPLAIN DICTIONARY SERIALIZATION/PERSISTENCE OPTIONS?
    Answer: Storage and transmission:
    - JSON: Human readable, language interoperable
    - Pickle: Python-specific, preserves object types
    - msgpack: Binary, compact, cross-language
    - Database storage: SQL/NoSQL for large datasets
    - Consider security implications of deserialization
"""

print(advanced_questions)

# =============================================================================
# REAL-WORLD APPLICATION QUESTIONS
# =============================================================================

print("\n" + "=" * 50)
print("REAL-WORLD APPLICATION QUESTIONS")
print("=" * 50)

application_questions = """
29. HOW ARE DICTIONARIES USED IN WEB DEVELOPMENT?
    Answer: Common web development uses:
    - HTTP headers storage (key-value pairs)
    - JSON data parsing and generation
    - Session storage and user state
    - Configuration management
    - Database query results mapping
    - API response formatting

30. EXPLAIN CACHING STRATEGIES USING DICTIONARIES?
    Answer: Caching implementation patterns:
    - Simple cache: dict with size limit
    - LRU cache: OrderedDict or custom implementation
    - TTL cache: timestamps for expiration
    - Write-through vs write-back strategies
    - Cache invalidation patterns

31. HOW DO DICTIONARIES HELP IN ALGORITHM OPTIMIZATION?
    Answer: Performance improvement techniques:
    - Replace O(n) searches with O(1) lookups
    - Memoization for dynamic programming
    - Frequency counting for analysis
    - Grouping and categorization
    - Index mapping for fast access

32. DESCRIBE DATABASE INDEXING RELATIONSHIP TO HASH TABLES?
    Answer: Similar concepts in databases:
    - Hash indexes use same principle as dictionaries
    - B-tree indexes for range queries
    - Trade-offs: hash (equality) vs tree (range)
    - Memory vs disk storage considerations
    - Query optimization using indexes

33. HOW ARE DICTIONARIES USED IN DISTRIBUTED SYSTEMS?
    Answer: Distributed computing applications:
    - Consistent hashing for load balancing
    - Distributed hash tables (DHT) in P2P systems
    - Sharding strategies using hash functions
    - Cache distribution (Redis, Memcached)
    - Configuration management across services
"""

print(application_questions)

# =============================================================================
# TROUBLESHOOTING AND DEBUGGING QUESTIONS
# =============================================================================

print("\n" + "=" * 50)
print("TROUBLESHOOTING & DEBUGGING QUESTIONS")
print("=" * 50)

debugging_questions = """
34. HOW WOULD YOU DEBUG PERFORMANCE ISSUES WITH DICTIONARIES?
    Answer: Diagnostic approaches:
    - Profile using cProfile, timeit modules
    - Check for hash collision patterns
    - Measure memory usage with sys.getsizeof()
    - Analyze key distribution and types
    - Consider alternative data structures

35. WHAT CAUSES KeyError AND HOW TO PREVENT IT?
    Answer: Common error handling:
    - KeyError when accessing non-existent key
    - Use get() method with default values
    - Check 'key in dict' before access
    - Use try/except for error handling
    - setdefault() for conditional initialization

36. HOW TO HANDLE DICTIONARY MUTATION DURING ITERATION?
    Answer: RuntimeError prevention:
    - Never modify dict size during iteration
    - Create copy of keys/items if modification needed
    - Use list(dict.keys()) for safe iteration
    - Consider itertools.islice() for large dicts
    - Document mutation assumptions clearly

37. EXPLAIN MEMORY LEAKS WITH DICTIONARIES?
    Answer: Common leak patterns:
    - Circular references preventing garbage collection
    - Growing caches without size limits
    - Event handlers not properly removed
    - Use weak references when appropriate
    - Monitor memory usage in production

38. HOW TO OPTIMIZE DICTIONARY LOOKUPS IN HOT CODE PATHS?
    Answer: Performance optimization:
    - Minimize hash computation overhead
    - Use simple, consistent key types
    - Pre-compute frequently used keys
    - Consider dict subclassing for specialized use
    - Profile to identify actual bottlenecks
"""

print(debugging_questions)

# =============================================================================
# INTERVIEW TIPS FOR VERBAL QUESTIONS
# =============================================================================

print("\n" + "=" * 50)
print("INTERVIEW TIPS FOR VERBAL QUESTIONS")
print("=" * 50)

interview_tips = """
HOW TO ANSWER VERBAL DICTIONARY QUESTIONS:
==========================================

1. STRUCTURE YOUR ANSWERS:
   - Start with a clear definition
   - Provide examples when possible
   - Explain trade-offs and alternatives
   - Mention real-world applications

2. SHOW DEPTH OF KNOWLEDGE:
   - Connect concepts to other data structures
   - Discuss performance implications
   - Mention Python-specific features
   - Reference common design patterns

3. BE PREPARED TO ELABORATE:
   - Interviewers often ask follow-up questions
   - Know the 'why' behind the 'what'
   - Understand implementation details
   - Can you implement it from scratch?

4. COMMON FOLLOW-UP PATTERNS:
   - "Can you give an example?"
   - "How would you implement this?"
   - "What are the trade-offs?"
   - "When would you not use a dictionary?"

5. CONNECT TO PRACTICAL EXPERIENCE:
   - Share real projects where you used dictionaries
   - Discuss optimization challenges you've faced
   - Mention debugging experiences
   - Show understanding of production considerations

SAMPLE INTERVIEW DIALOGUE:
=========================

Interviewer: "Explain how Python dictionaries work internally?"

Good Answer: "Python dictionaries are implemented using hash tables. When you 
store a key-value pair, Python computes a hash of the key using a hash function, 
which maps the key to an index in an underlying array. This allows for O(1) 
average-case access time.

Python uses open addressing with random probing to handle hash collisions. 
Since Python 3.6, dictionaries also maintain insertion order as a language 
guarantee, which was made possible by a compact representation that stores 
keys and values in separate arrays.

The hash table maintains a load factor of about 2/3, and when this threshold 
is exceeded, the table is resized and all elements are rehashed. This resizing 
operation is O(n) but amortized over many operations.

Would you like me to elaborate on any particular aspect, such as the collision 
resolution strategy or the memory optimization techniques used?"

This answer shows:
✓ Technical accuracy
✓ Understanding of implementation details  
✓ Knowledge of recent Python changes
✓ Awareness of performance characteristics
✓ Invitation for follow-up questions
"""

print(interview_tips)

print("\n" + "=" * 70)
print("VERBAL/CONCEPTUAL QUESTIONS SECTION COMPLETE!")
print("Study these concepts to handle any theoretical dictionary questions!")
print("=" * 70)

# =============================================================================
# INTERVIEW TIPS AND PATTERNS
# =============================================================================

print("\n" + "=" * 70)
print("INTERVIEW TIPS AND COMMON PATTERNS")
print("=" * 70)

tips = """
DICTIONARY INTERVIEW PATTERNS:
==============================

1. FREQUENCY COUNTING:
   - Character/word frequency problems
   - Use Counter from collections or manual dict
   - Examples: Valid Anagram, Group Anagrams

2. TWO SUM PATTERN:
   - Store complements in dictionary
   - O(n) solution instead of O(n²)
   - Examples: Two Sum, 4Sum II

3. SLIDING WINDOW + HASHMAP:
   - Track characters in current window
   - Examples: Longest Substring Without Repeating

4. PREFIX SUM + HASHMAP:
   - Store cumulative sums
   - Examples: Subarray Sum Equals K

5. GRAPH REPRESENTATION:
   - Use dict for adjacency lists
   - Examples: Alien Dictionary, Course Schedule

6. CACHING/MEMOIZATION:
   - Store computed results
   - Examples: LRU Cache, Dynamic Programming

COMMON MISTAKES TO AVOID:
========================
• Forgetting to handle edge cases (empty inputs)
• Not checking if key exists before accessing
• Modifying dictionary while iterating
• Using mutable objects as keys
• Not considering time/space complexity

OPTIMIZATION TECHNIQUES:
========================
• Use get() with default instead of checking 'in'
• Use setdefault() for conditional initialization
• Use defaultdict for automatic initialization
• Consider Counter for frequency problems
• Use OrderedDict if order matters (Python < 3.7)

TIME COMPLEXITY REVIEW:
======================
• Access/Insert/Delete: O(1) average, O(n) worst
• Iteration: O(n)
• Sorting dict items: O(n log n)
"""

print(tips)

print("\n" + "=" * 70)
print("DICTIONARY INTERVIEW QUESTIONS COMPLETE!")
print("Master these patterns and you'll excel in interviews!")
print("=" * 70)
