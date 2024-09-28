from collections import Counter

from algorithms.models import lg_lottery_winner_number
import random

def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)

    # Számok gyakorisága
    number_frequency = Counter()
    for draw in past_draws:
        number_frequency.update(draw)

    # Leggyakoribb számok kiválasztása
    most_common_numbers = [num for num, _ in
                           number_frequency.most_common(lottery_type_instance.pieces_of_draw_numbers * 2)]

    # Véletlenszerű választás a leggyakoribb számok közül
    selected_numbers = set()
    while len(selected_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        if most_common_numbers:
            selected_numbers.add(random.choice(most_common_numbers))
        else:
            # Ha elfogytak a gyakori számok, válasszunk a teljes tartományból
            selected_numbers.add(random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number))

    return sorted(selected_numbers)