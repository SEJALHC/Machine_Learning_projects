import os
from send2trash import send2trash

def delete_duplicates(file_map):
    deleted = []

    for files in file_map.values():
        if len(files) > 1:
            for file in files[1:]:
                try:
                    clean_path = os.path.normpath(file)  # 🔥 FIX HERE
                    send2trash(clean_path)
                    deleted.append(clean_path)
                except Exception as e:
                    print("Error deleting:", file, e)

    return deleted