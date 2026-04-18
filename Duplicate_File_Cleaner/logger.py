import os
import time

def create_log(deleted_files):
    if not os.path.exists("logs"):
        os.mkdir("logs")

    filename = f"logs/log_{int(time.time())}.txt"

    with open(filename, "w") as f:
        f.write("Deleted Files Log\n")
        f.write("="*40 + "\n")

        for file in deleted_files:
            f.write(file + "\n")

    return filename