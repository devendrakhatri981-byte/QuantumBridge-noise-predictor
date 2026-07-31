import os

# ── CREATE nested directories ──────────────────────────
os.makedirs("college/cse/aiml", exist_ok=True)   # nested folders
os.makedirs("college/cse/ds",   exist_ok=True)

# Create some files inside
with open("college/cse/aiml/notes.txt", "w") as f:
    f.write("AI/ML Notes")
with open("college/cse/ds/data.txt", "w") as f:
    f.write("Data Science Notes")

print("Directories and files created.\n")

# ── LIST using os.listdir() ────────────────────────────
print("Contents of 'college':", os.listdir("college"))
print("Contents of 'college/cse':", os.listdir("college/cse"))
print("Contents of 'college/cse/aiml':", os.listdir("college/cse/aiml"))

# ── WALK entire tree ───────────────────────────────────
print("\nFull directory tree:")
for root, dirs, files in os.walk("college"):
    print(f"  Folder: {root}")
    for d in dirs:
        print(f"    Dir : {d}")
    for f in files:
        print(f"    File: {f}")

# ── REMOVE nested directories ──────────────────────────
import shutil
shutil.rmtree("college")   # removes entire folder tree
print("\nDeleted 'college' directory and all contents.")
print("Exists?", os.path.exists("college"))   # False