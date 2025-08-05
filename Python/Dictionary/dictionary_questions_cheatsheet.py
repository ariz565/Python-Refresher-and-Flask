"""
DICTIONARY INTERVIEW QUESTIONS - QUICK REFERENCE
================================================

Quick reference guide for practicing dictionary-based interview questions.
Contains problem statements and solution patterns.

Format: Problem -> Pattern -> Solution Approach
"""

# =============================================================================
# QUICK REFERENCE TABLE
# =============================================================================

INTERVIEW_QUESTIONS = {
    "EASY": {
        "Two Sum": {
            "pattern": "Complement lookup",
            "time": "O(n)",
            "space": "O(n)",
            "companies": ["Amazon", "Google", "Facebook", "Microsoft"],
            "solution": "Store complements in hash map"
        },
        
        "Valid Anagram": {
            "pattern": "Frequency counting",
            "time": "O(n)",
            "space": "O(1)",
            "companies": ["Facebook", "Amazon", "Google"],
            "solution": "Count characters, compare frequencies"
        },
        
        "First Non-Repeating Character": {
            "pattern": "Frequency counting",
            "time": "O(n)",
            "space": "O(1)",
            "companies": ["Amazon", "Microsoft", "Apple"],
            "solution": "Count frequencies, find first with count=1"
        },
        
        "Contains Duplicate": {
            "pattern": "Existence check",
            "time": "O(n)",
            "space": "O(n)",
            "companies": ["Google", "Facebook", "Amazon"],
            "solution": "Use set/dict to track seen elements"
        },
        
        "Group Anagrams": {
            "pattern": "Grouping by key",
            "time": "O(n*k log k)",
            "space": "O(n*k)",
            "companies": ["Amazon", "Facebook", "Google", "Uber"],
            "solution": "Sort characters as key, group strings"
        }
    },
    
    "MEDIUM": {
        "Top K Frequent Elements": {
            "pattern": "Frequency + Priority Queue",
            "time": "O(n log k)",
            "space": "O(n)",
            "companies": ["Amazon", "Facebook", "Google", "LinkedIn"],
            "solution": "Count frequencies, use heap for top K"
        },
        
        "Subarray Sum Equals K": {
            "pattern": "Prefix sum + HashMap",
            "time": "O(n)",
            "space": "O(n)",
            "companies": ["Facebook", "Google", "Amazon"],
            "solution": "Store cumulative sums, check (sum-k)"
        },
        
        "Longest Substring Without Repeating": {
            "pattern": "Sliding window + HashMap",
            "time": "O(n)",
            "space": "O(min(m,n))",
            "companies": ["Amazon", "Facebook", "Google", "Microsoft"],
            "solution": "Track char positions, adjust window"
        },
        
        "4Sum II": {
            "pattern": "Two-pointer reduction",
            "time": "O(n²)",
            "space": "O(n²)",
            "companies": ["Facebook", "Amazon"],
            "solution": "Split into two 2-sum problems"
        },
        
        "Word Pattern": {
            "pattern": "Bidirectional mapping",
            "time": "O(n)",
            "space": "O(n)",
            "companies": ["Google", "Facebook"],
            "solution": "Map char->word and word->char"
        }
    },
    
    "HARD": {
        "Minimum Window Substring": {
            "pattern": "Sliding window + Frequency map",
            "time": "O(|s| + |t|)",
            "space": "O(|s| + |t|)",
            "companies": ["Facebook", "Amazon", "Google", "Microsoft"],
            "solution": "Expand/contract window, track requirements"
        },
        
        "Alien Dictionary": {
            "pattern": "Topological sort",
            "time": "O(C)",
            "space": "O(1)",
            "companies": ["Google", "Facebook", "Amazon", "Airbnb"],
            "solution": "Build graph from word order, topo sort"
        },
        
        "Number of Islands II": {
            "pattern": "Union Find + Dynamic updates",
            "time": "O(k * α(mn))",
            "space": "O(mn)",
            "companies": ["Google", "Facebook"],
            "solution": "Union-Find with dynamic land addition"
        }
    },
    
    "SYSTEM_DESIGN": {
        "LRU Cache": {
            "pattern": "HashMap + Doubly Linked List",
            "time": "O(1) all operations",
            "space": "O(capacity)",
            "companies": ["Facebook", "Amazon", "Google", "Microsoft"],
            "solution": "Dict for O(1) access, DLL for O(1) updates"
        },
        
        "Design Twitter": {
            "pattern": "Timeline + Following system",
            "time": "Various",
            "space": "O(users + tweets)",
            "companies": ["Twitter", "Facebook", "Amazon"],
            "solution": "Hash maps for users, tweets, relationships"
        }
    }
}

# =============================================================================
# COMMON PATTERNS CHEAT SHEET
# =============================================================================

PATTERNS = {
    "FREQUENCY_COUNTING": {
        "when_to_use": "Count occurrences of elements",
        "template": """
        count = {}
        for item in items:
            count[item] = count.get(item, 0) + 1
        """,
        "examples": ["Valid Anagram", "Group Anagrams", "Top K Frequent"]
    },
    
    "COMPLEMENT_LOOKUP": {
        "when_to_use": "Find pairs that sum to target",
        "template": """
        seen = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in seen:
                return [seen[complement], i]
            seen[num] = i
        """,
        "examples": ["Two Sum", "3Sum", "4Sum II"]
    },
    
    "SLIDING_WINDOW": {
        "when_to_use": "Subarray/substring problems",
        "template": """
        window = {}
        left = 0
        for right in range(len(arr)):
            # Add right element to window
            # Shrink window if invalid
            while invalid_condition:
                # Remove left element
                left += 1
            # Update result
        """,
        "examples": ["Longest Substring", "Minimum Window"]
    },
    
    "PREFIX_SUM": {
        "when_to_use": "Subarray sum problems",
        "template": """
        prefix_sum = 0
        sum_count = {0: 1}
        for num in nums:
            prefix_sum += num
            if prefix_sum - target in sum_count:
                # Found valid subarray
            sum_count[prefix_sum] = sum_count.get(prefix_sum, 0) + 1
        """,
        "examples": ["Subarray Sum Equals K", "Maximum Size Subarray"]
    },
    
    "GRAPH_ADJACENCY": {
        "when_to_use": "Graph problems, dependencies",
        "template": """
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        # Build graph
        for u, v in edges:
            graph[u].append(v)
            in_degree[v] += 1
        
        # Process (DFS/BFS/Topological sort)
        """,
        "examples": ["Alien Dictionary", "Course Schedule"]
    }
}

# =============================================================================
# INTERVIEW PREPARATION CHECKLIST
# =============================================================================

PREPARATION_CHECKLIST = """
DICTIONARY INTERVIEW PREPARATION CHECKLIST:
===========================================

□ BASIC OPERATIONS:
  □ Creation methods (literal, constructor, comprehension)
  □ Access patterns ([], get(), setdefault())
  □ Modification (assignment, update(), pop(), del)
  □ Iteration (keys(), values(), items())

□ CORE PATTERNS:
  □ Frequency counting with Counter/dict
  □ Two Sum and variants (complement lookup)
  □ Sliding window with character tracking
  □ Prefix sum with cumulative counts
  □ Graph representation with adjacency lists

□ COMMON PROBLEMS:
  □ Two Sum (Easy) - Master this first!
  □ Valid Anagram (Easy) - Frequency pattern
  □ Group Anagrams (Easy-Medium) - Grouping pattern
  □ Top K Frequent (Medium) - Heap + frequency
  □ Longest Substring (Medium) - Sliding window
  □ Subarray Sum K (Medium) - Prefix sum
  □ Minimum Window (Hard) - Advanced sliding window

□ ADVANCED CONCEPTS:
  □ LRU Cache implementation
  □ System design with dictionaries
  □ Time/space complexity analysis
  □ Edge case handling

□ PRACTICE STRATEGY:
  □ Start with easy problems (Two Sum pattern)
  □ Master frequency counting problems
  □ Practice sliding window variations
  □ Understand when to use each pattern
  □ Time yourself (aim for 15-20 min per medium problem)

COMMON INTERVIEW MISTAKES:
=========================
❌ Not handling empty inputs
❌ Forgetting to check key existence
❌ Using mutable objects as keys
❌ Not optimizing nested loops to O(n)
❌ Incorrect time/space complexity analysis

SUCCESS TIPS:
============
✅ Always discuss approach before coding
✅ Start with brute force, then optimize
✅ Use meaningful variable names
✅ Handle edge cases explicitly
✅ Test with examples during coding
✅ Analyze time/space complexity at the end
"""

# =============================================================================
# COMPANY-SPECIFIC PATTERNS
# =============================================================================

COMPANY_PATTERNS = {
    "GOOGLE": [
        "Focus on optimization and edge cases",
        "Expect follow-up questions about scaling",
        "Common: Sliding window, graph problems"
    ],
    
    "FACEBOOK": [
        "Emphasis on system design thinking",
        "Real-world application scenarios",
        "Common: LRU Cache, social network problems"
    ],
    
    "AMAZON": [
        "Behavioral questions + coding",
        "Leadership principles alignment",
        "Common: Two Sum variants, frequency problems"
    ],
    
    "MICROSOFT": [
        "Focus on clean, readable code",
        "Discussion of alternative approaches",
        "Common: String problems, basic algorithms"
    ],
    
    "APPLE": [
        "Attention to detail and efficiency",
        "Memory-conscious solutions",
        "Common: Array/string manipulation"
    ]
}

def print_cheat_sheet():
    """Print the complete cheat sheet"""
    print("=" * 60)
    print("DICTIONARY INTERVIEW QUESTIONS - CHEAT SHEET")
    print("=" * 60)
    
    for difficulty, questions in INTERVIEW_QUESTIONS.items():
        print(f"\n{difficulty} LEVEL:")
        print("-" * 40)
        for problem, details in questions.items():
            print(f"{problem}:")
            print(f"  Pattern: {details['pattern']}")
            print(f"  Time: {details['time']}, Space: {details['space']}")
            print(f"  Companies: {', '.join(details['companies'][:3])}")
            print(f"  Approach: {details['solution']}")
            print()
    
    print("\nKEY PATTERNS:")
    print("-" * 40)
    for pattern, details in PATTERNS.items():
        print(f"{pattern}: {details['when_to_use']}")
    
    print(PREPARATION_CHECKLIST)

if __name__ == "__main__":
    print_cheat_sheet()
