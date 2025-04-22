"""
Open Interpreter profile that reads configuration from a TOML file.

Learn about all the available settings - https://docs.openinterpreter.com/settings/all-settings
"""

# Import required libraries
from interpreter import interpreter
from datetime import date
import tomllib
import os

# Set variables
today = date.today()

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read configuration from TOML file
config_path = os.path.join(script_dir, "config.toml")
with open(config_path, "rb") as f:
    config = tomllib.load(f)

# Configure LLM settings
for key, value in config["llm"].items():
    setattr(interpreter.llm, key, value)

# Configure interpreter settings
for key, value in config["interpreter"].items():
    if key == "import_computer_api":
        interpreter.computer.import_computer_api = value
    else:
        setattr(interpreter, key, value)

# Format custom instructions with current date
custom_instructions = config["custom"]["instructions"].format(today=today)
interpreter.custom_instructions = custom_instructions

# Start the interpreter
interpreter.chat()
