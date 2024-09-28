import os
import importlib

from django.utils import timezone

from lottery_handler.models import lg_generated_lottery_draw
from algorithms.models import lg_lottery_type, lg_lottery_winner_number

def list_processor_files():
    processor_files = []
    directory = "processors"  # Path to your processors folder

    for filename in os.listdir(directory):
        if filename.endswith(".py") and not filename.startswith("__"):
            processor_files.append(filename[:-3])  # Remove '.py' from the filename

    return processor_files

def call_get_numbers_dynamically(lottery_type_id):
    processor_files = list_processor_files()
    results = []


    lottery_type = lg_lottery_type.objects.filter(id=lottery_type_id.id).first()

    if lottery_type is None:
        return JsonResponse({"error": "Item not found"}, status=404)


    for file in processor_files:

        try:
            print(f"Calling get_numbers function in {file} module...")

            module = importlib.import_module(f"processors.{file}")

            if hasattr(module, 'get_numbers'):

                # Call the get_numbers function if it exists in the module
                lottery_numbers = module.get_numbers(lottery_type)


                result = lg_generated_lottery_draw.objects.create(
                    lottery_type=lottery_type,
                    lottery_type_number=lottery_numbers,
                    lottery_type_number_year=timezone.now().year,
                    lottery_type_number_week=timezone.now().isocalendar()[1],
                    lottery_algorithm=file,
                    created_at=timezone.now())
                results.append(result)
        except Exception as e:
            print(f"Error calling get_numbers function in {file} module: {e}")
            return JsonResponse({"error": str(e)}, status=500)

    return results