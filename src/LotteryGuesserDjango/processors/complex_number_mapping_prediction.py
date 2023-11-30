import cmath
from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    complex_map = Counter()

    for draw in past_draws:
        for number in draw:
            # Map the number to a point on the complex plane, e.g., as a complex number with real part 0
            complex_number = complex(0, number)
            complex_map[complex_number] += 1

    # Select the number whose corresponding complex number has the shortest distance from the origin
    predicted_complex_number = min(complex_map, key=lambda x: abs(x))
    predicted_number = int(predicted_complex_number.imag)

    return [predicted_number for _ in range(lottery_type_instance.pieces_of_draw_numbers)]
