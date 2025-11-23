from pathlib import Path

# -----------------------------------------------------------------------------
# WHY THIS FILE EXISTS:
# This file defines "Constants" - values that never change during the project.
# 
# Instead of typing "config/config.yaml" in 10 different files (and risking typos),
# we define it here ONCE. Everyone else imports 'CONFIG_FILE_PATH' from here.
# -----------------------------------------------------------------------------

"""
WHAT: Defines the paths to our key configuration files.

WHY Path():
- Windows uses backslashes (\) and Mac/Linux use forward slashes (/).
- 'Path' from pathlib handles this automatically. It makes our code work on ANY OS.
"""

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")