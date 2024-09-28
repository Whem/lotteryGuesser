import cmath
from collections import Counter
from typing import List
import random
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)
    complex_map = Counter()

    for draw in past_draws:
        for i, number in enumerate(draw):
            # Map the number to a point on the complex plane using both real and imaginary parts
            angle = 2 * cmath.pi * i / len(draw)  # Distribute numbers evenly around the unit circle
            complex_number = cmath.rect(number, angle)
            complex_map[complex_number] += 1

    # Calculate the centroid of all mapped points
    if complex_map:
        centroid = sum(num * count for num, count in complex_map.items()) / sum(complex_map.values())
    else:
        # If no past draws, use a random point
        centroid = cmath.rect(random.uniform(lottery_type_instance.min_number, lottery_type_instance.max_number),
                              random.uniform(0, 2 * cmath.pi))

    predicted_numbers = []
    for _ in range(lottery_type_instance.pieces_of_draw_numbers):
        # Find the closest mapped number to the centroid
        closest_complex = min(complex_map, key=lambda x: abs(x - centroid))

        # Convert back to an integer within the valid range
        predicted_number = max(lottery_type_instance.min_number,
                               min(lottery_type_instance.max_number,
                                   int(abs(closest_complex))))

        predicted_numbers.append(predicted_number)

        # Remove the used number from the map to avoid repetition
        del complex_map[closest_complex]

        # If we've used all mapped numbers, break the loop
        if not complex_map:
            break

    # If we don't have enough numbers, fill with random ones
    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        new_number = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
        if new_number not in predicted_numbers:
            predicted_numbers.append(new_number)

    return sorted(predicted_numbers)