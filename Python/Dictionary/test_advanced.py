# Test some of the advanced interview questions
from collections import defaultdict, deque
import time
import hashlib

print("Testing Advanced Dictionary Interview Questions")
print("=" * 50)

# Test LRU Cache
print("\n1. LRU CACHE TEST:")
print("-" * 30)

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
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
            self._add_to_front(node)
            return node.val
        return -1
    
    def put(self, key, value):
        if key in self.cache:
            node = self.cache[key]
            node.val = value
            self._remove(node)
            self._add_to_front(node)
        else:
            if len(self.cache) >= self.capacity:
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]
            
            new_node = self.Node(key, value)
            self.cache[key] = new_node
            self._add_to_front(new_node)
    
    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _add_to_front(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

lru = LRUCache(2)
lru.put(1, 1)
lru.put(2, 2)
print(f"Get 1: {lru.get(1)}")  # Should return 1
lru.put(3, 3)  # Evicts key 2
print(f"Get 2: {lru.get(2)}")  # Should return -1
print(f"Get 3: {lru.get(3)}")  # Should return 3

# Test Hit Counter
print("\n2. HIT COUNTER TEST:")
print("-" * 30)

class HitCounter:
    def __init__(self):
        self.hits = deque()
    
    def hit(self, timestamp):
        if self.hits and self.hits[-1][0] == timestamp:
            self.hits[-1] = (timestamp, self.hits[-1][1] + 1)
        else:
            self.hits.append((timestamp, 1))
        self._cleanup(timestamp)
    
    def getHits(self, timestamp):
        self._cleanup(timestamp)
        return sum(count for _, count in self.hits)
    
    def _cleanup(self, timestamp):
        while self.hits and self.hits[0][0] <= timestamp - 300:
            self.hits.popleft()

hc = HitCounter()
hc.hit(1)
hc.hit(2)
hc.hit(3)
print(f"Hits at timestamp 4: {hc.getHits(4)}")  # Should be 3
hc.hit(300)
print(f"Hits at timestamp 300: {hc.getHits(300)}")  # Should be 4
print(f"Hits at timestamp 301: {hc.getHits(301)}")  # Should be 3

# Test Alien Dictionary
print("\n3. ALIEN DICTIONARY TEST:")
print("-" * 30)

def alien_order(words):
    graph = defaultdict(set)
    in_degree = defaultdict(int)
    
    for word in words:
        for char in word:
            in_degree[char] = 0
    
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        min_len = min(len(word1), len(word2))
        
        if len(word1) > len(word2) and word1[:min_len] == word2[:min_len]:
            return ""
        
        for j in range(min_len):
            if word1[j] != word2[j]:
                if word2[j] not in graph[word1[j]]:
                    graph[word1[j]].add(word2[j])
                    in_degree[word2[j]] += 1
                break
    
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

words = ["wrt", "wrf", "er", "ett", "rftt"]
print(f"Words: {words}")
print(f"Alien order: '{alien_order(words)}'")

print("\n" + "=" * 50)
print("ADVANCED CONCEPTS DEMONSTRATION COMPLETE!")
print("These are the types of problems asked to experienced developers")
print("=" * 50)
