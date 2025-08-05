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

# The following lines are for input and output, do not modify
ciphertext = input().strip()
known_word = input().strip()
result = decipher(ciphertext, known_word)
print(result)