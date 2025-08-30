import os
import subprocess

# Define the repository and directory
REPO_URL = "https://github.com/vaastav/Fantasy-Premier-League"
DIR_NAME = "Fantasy-Premier-League"

# Check if directory exists
if not os.path.exists(DIR_NAME):
    subprocess.check_call(["git", "clone", REPO_URL])
else:
    os.chdir(DIR_NAME)
    subprocess.check_call(["git", "reset", "--hard", "HEAD"])  # Reset local changes
    subprocess.check_call(["git", "pull", "origin", "master"])
