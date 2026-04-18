import tkinter as tk
from tkinter import filedialog, messagebox
from scanner import start_scan
from deleter import delete_duplicates
from logger import create_log


def browse():
    path = filedialog.askdirectory()
    entry.delete(0, tk.END)
    entry.insert(0, path)


def run():
    path = entry.get()

    if not path:
        messagebox.showerror("Error", "Please select a folder first.")
        return

    # Step 1: Scan files
    file_map = start_scan(path)

    # Step 2: Calculate stats
    total_files = sum(len(v) for v in file_map.values())
    duplicate_groups = [v for v in file_map.values() if len(v) > 1]

    # Step 3: Delete duplicates
    deleted = delete_duplicates(file_map)

    # Step 4: Create log (only if something deleted)
    if deleted:
        log_file = create_log(deleted)
    else:
        log_file = None

    # Step 5: Show proper result
    if len(deleted) == 0:
        messagebox.showinfo(
            "Scan Result",
            f"No duplicate files found.\n\n"
            f"Total Files Scanned: {total_files}"
        )
    else:
        messagebox.showinfo(
            "Scan Result",
            f"Total Files Scanned: {total_files}\n"
            f"Duplicate Groups Found: {len(duplicate_groups)}\n"
            f"Files Deleted: {len(deleted)}\n\n"
            f"Log File Created: {log_file}"
        )


# GUI setup
root = tk.Tk()
root.title("Duplicate File Manager")
root.geometry("500x250")

# Input field
entry = tk.Entry(root, width=60)
entry.pack(pady=15)

# Buttons
tk.Button(root, text="Browse Folder", command=browse).pack(pady=5)
tk.Button(root, text="Scan & Clean", command=run).pack(pady=10)

# Run app
root.mainloop()