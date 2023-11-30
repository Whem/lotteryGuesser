import os
import importlib

def list_processor_files():
    processor_files = []
    directory = "processors"  # Path to your processors folder

    for filename in os.listdir(directory):
        if filename.endswith(".py") and not filename.startswith("__"):
            processor_files.append(filename[:-3])  # Remove '.py' from the filename

    return processor_files

def call_get_numbers_dynamically(lottery_type):
    processor_files = list_processor_files()
    results = {}

    for file in processor_files:
        # Dynamically import the module
        module = importlib.import_module(f"processors.{file}")

        if hasattr(module, 'get_numbers'):
            # Call the get_numbers function if it exists in the module
            results[file] = module.get_numbers(lottery_type)

    return results