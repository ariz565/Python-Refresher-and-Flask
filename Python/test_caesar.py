def decipher(ciphertext, known_word):
    def decrypt(text, shift):
        result = ""
        for char in text:
            if char.isupper():
                result += chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
            elif char.islower():
                result += chr((ord(char) - ord('a') - shift) % 26 + ord('a'))
            else:
                result += char
        return result
    
    for shift in range(26):
        dec = decrypt(ciphertext, shift)
        if known_word in dec:
            return dec
    return "Invalid"

# Test cases from the problem
test_cases = [
    ("Eqfkpi vguvu ctg hwp!", "tests", "Coding tests are fun!"),
    ("cdeb nqxg", "love", "abcz love"),
    ("Uifsf jt op tvdi xpse", "hello", "Invalid")
]

print("Testing Caesar cipher solution...")
print("=" * 50)

for i, (ciphertext, known_word, expected) in enumerate(test_cases, 1):
    result = decipher(ciphertext, known_word)
    status = "✅ PASS" if result == expected else "❌ FAIL"
    
    print(f"Test Case {i}: {status}")
    print(f"Input: '{ciphertext}' with known word '{known_word}'")
    print(f"Expected: '{expected}'")
    print(f"Got:      '{result}'")
    
    if result != expected:
        print(f"❌ Mismatch detected!")
    
    print("-" * 50)

# Let's also show the shift values for successful decryptions
print("\nDetailed analysis:")
print("=" * 50)

for i, (ciphertext, known_word, expected) in enumerate(test_cases, 1):
    print(f"\nTest Case {i}: '{ciphertext}' with known word '{known_word}'")
    
    def decrypt(text, shift):
        result = ""
        for char in text:
            if char.isupper():
                result += chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
            elif char.islower():
                result += chr((ord(char) - ord('a') - shift) % 26 + ord('a'))
            else:
                result += char
        return result
    
    found = False
    for shift in range(26):
        dec = decrypt(ciphertext, shift)
        if known_word in dec:
            print(f"  Shift {shift}: '{dec}' (contains '{known_word}')")
            found = True
    
    if not found:
        print(f"  No shift produces text containing '{known_word}'")
