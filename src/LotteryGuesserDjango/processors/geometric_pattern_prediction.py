import random


def get_numbers(lottery_type_instance):
    # This function needs a more complex implementation to analyze geometric patterns
    # For simplicity, it randomly selects numbers for now
    selected_numbers = set()
    while len(selected_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        selected_numbers.add(random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number))

    return sorted(selected_numbers)