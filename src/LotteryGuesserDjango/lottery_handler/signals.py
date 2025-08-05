import os
import importlib
import numpy as np

from django.utils import timezone
from django.http import JsonResponse

from lottery_handler.models import lg_generated_lottery_draw, lg_algorithm_score
from algorithms.models import lg_lottery_type, lg_lottery_winner_number


def list_processor_files():
    processor_files = []
    directory = "processors"  # Path to your processors folder

    for filename in os.listdir(directory):
        if filename.endswith(".py") and not filename.startswith("__"):
            processor_files.append(filename[:-3])  # Remove '.py' from the filename

    return processor_files


def get_top_algorithms():
    # Get the top 5 performing algorithms based on their scores
    return [score.algorithm_name for score in lg_algorithm_score.objects.all().order_by('-current_score')[:5]]


def convert_numpy_to_python_types(obj):
    """Convert NumPy types to Python native types to ensure JSON serialization works"""
    # Handle lists and tuples recursively
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python_types(item) for item in obj]
    
    # Handle dictionaries recursively
    if isinstance(obj, dict):
        return {key: convert_numpy_to_python_types(value) for key, value in obj.items()}
    
    # Convert NumPy integer types to Python int
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    
    # Convert NumPy float types to Python float
    if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    
    # Convert NumPy boolean to Python bool
    if isinstance(obj, np.bool_):
        return bool(obj)
    
    # Return original object if no conversion needed
    return obj


def call_get_numbers_dynamically(lottery_type_id):
    processor_files = list_processor_files()
    results = []

    lottery_type = lg_lottery_type.objects.filter(id=lottery_type_id.id).first()

    if lottery_type is None:
        return JsonResponse({"error": "Item not found"}, status=404)

    # Get the top 10 performing algorithms
    top_algorithms = get_top_algorithms()

    # Filter processor_files to only include top performing algorithms
    processor_files = [file for file in processor_files if file in top_algorithms]

    for file in processor_files:
        try:
            print(f"Calling get_numbers function in {file} module...")

            module = importlib.import_module(f"processors.{file}")

            if hasattr(module, 'get_numbers'):
                # Call the get_numbers function if it exists in the module
                lottery_numbers = module.get_numbers(lottery_type)
                
                # Handle tuple return (main_numbers, additional_numbers)
                if isinstance(lottery_numbers, tuple) and len(lottery_numbers) == 2:
                    main_numbers, additional_numbers = lottery_numbers
                    # Convert NumPy types to Python native types
                    main_numbers = convert_numpy_to_python_types(main_numbers)
                    additional_numbers = convert_numpy_to_python_types(additional_numbers)
                    
                    # Combine numbers for storage or use only main numbers
                    final_numbers = main_numbers
                    if additional_numbers:
                        final_numbers = {"main": main_numbers, "additional": additional_numbers}
                else:
                    # Handle legacy format or single list
                    final_numbers = convert_numpy_to_python_types(lottery_numbers)

                result = lg_generated_lottery_draw.objects.create(
                    lottery_type=lottery_type,
                    lottery_type_number=final_numbers,
                    lottery_type_number_year=timezone.now().year,
                    lottery_type_number_week=timezone.now().isocalendar()[1],
                    lottery_algorithm=file,
                    created_at=timezone.now())
                results.append(result)
        except Exception as e:
            print(f"Error calling get_numbers function in {file} module: {e}")
            import traceback
            traceback.print_exc()
            # Don't return error, just skip this algorithm and continue with others
            continue

    return results