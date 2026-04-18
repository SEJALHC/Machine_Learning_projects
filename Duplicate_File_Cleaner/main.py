from scanner import start_scan
from deleter import delete_duplicates

path = input("Enter directory: ")

file_map = start_scan(path)
deleted = delete_duplicates(file_map)

print("Deleted files:", len(deleted))