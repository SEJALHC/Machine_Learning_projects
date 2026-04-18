import hashlib

def check(file):
    with open(file, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

print(check("C:/Users/HP/OneDrive/Desktop/duplicates/FILE.txt"))
print(check("C:/Users/HP/OneDrive/Desktop/duplicates/FILE - Copy.txt"))
print(check("C:/Users/HP/OneDrive/Desktop/duplicates/FILE - Copy (2).txt"))