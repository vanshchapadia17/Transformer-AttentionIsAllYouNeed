import os
from pathlib import Path

# Define the project structure
list_of_files = [
    ".github/workflows/.gitkeep",
    
    "components/__init__.py",
    "components/attention.py",
    "components/feedforward.py",
    "components/block.py",
    "components/embeddings.py"
    
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Create directory if it doesn't exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        print(f"Creating directory: {filedir} for the file {filename}")

    # Create empty file if it doesn't exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        print(f"Creating empty file: {filepath}")
    else:
        print(f"{filename} already exists")