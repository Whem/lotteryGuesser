import os
import importlib
import numpy as np
import statistics
import logging

from django.utils import timezone
from django.http import JsonResponse

from lottery_handler.models import lg_generated_lottery_draw, lg_algorithm_score
from algorithms.models import lg_lottery_type, lg_lottery_winner_number

logger = logging.getLogger(__name__)


def list_processor_files():
    processor_files = []
    directory = "processors"  # Path to your processors folder

    for filename in os.listdir(directory):
        if filename.endswith(".py") and not filename.startswith("__"):
            processor_files.append(filename[:-3])  # Remove '.py' from the filename

    return processor_files


def get_top_algorithms(limit=10):
    # Get the top performing algorithms based on their scores
    return [score.algorithm_name for score in lg_algorithm_score.objects.all().order_by('-current_score')[:limit]]


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
    import time
    import random
    
    processor_files = list_processor_files()
    results = []

    lottery_type = lg_lottery_type.objects.filter(id=lottery_type_id.id).first()

    if lottery_type is None:
        return JsonResponse({"error": "Item not found"}, status=404)

    # Get the top 15 performing algorithms (increased from 5)
    top_algorithms = get_top_algorithms(limit=15)

    # Filter processor_files to only include top performing algorithms
    processor_files = [file for file in processor_files if file in top_algorithms]

    # Randomize algorithm order for variety
    random.shuffle(processor_files)

    for file in processor_files:
        try:
            # Set random seed with current timestamp for variety
            current_time = int(time.time() * 1000000)  # Microsecond precision
            random.seed(current_time)
            
            # Also seed numpy if the algorithm uses it
            try:
                import numpy as np
                np.random.seed(current_time % (2**32))  # NumPy seed must be < 2^32
            except ImportError:
                pass
            
            logger.info(f"EXECUTING: {file} (seed: {current_time})")
            print(f"Executing algorithm: {file} with seed: {current_time}")

            # Small delay to ensure different seeds between algorithms
            time.sleep(0.001)  # 1 millisecond delay

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

                # Create result with enhanced data
                current_time = timezone.now()
                
                # Calculate statistics only for main numbers (list format)
                main_nums = main_numbers if isinstance(lottery_numbers, tuple) else final_numbers
                if isinstance(main_nums, list) and all(isinstance(n, (int, float)) for n in main_nums):
                    sum_val = sum(main_nums)
                    average_val = statistics.mean(main_nums)
                    median_val = statistics.median(main_nums)
                    try:
                        mode_val = statistics.mode(main_nums)
                    except statistics.StatisticsError:
                        mode_val = main_nums[0] if main_nums else 0
                    std_val = statistics.stdev(main_nums) if len(main_nums) > 1 else 0.0
                else:
                    sum_val = average_val = median_val = mode_val = std_val = 0

                result = lg_generated_lottery_draw.objects.create(
                    lottery_type=lottery_type,
                    lottery_type_number=final_numbers,
                    lottery_type_number_year=current_time.year,
                    lottery_type_number_week=current_time.isocalendar()[1],
                    sum=sum_val,
                    average=average_val,
                    median=median_val,
                    mode=mode_val,
                    standard_deviation=std_val,
                    lottery_algorithm=file,
                    created_at=current_time)
                results.append(result)
        except Exception as e:
            logger.error(f"Hiba a {file} algoritmus futtatásakor: {e}")
            # Hibák naplózása teljesítményfigyeléshez
            try:
                import traceback
                logger.debug(f"Teljes stacktrace: {traceback.format_exc()}")
            except:
                pass
            # Folytatás a következő algoritmussal
            continue

    return results