import random
from collections import Counter

import numpy as np

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    cumulative_products = [np.prod(draw) for draw in past_draws]
    common_cumulative_product = Counter(cumulative_products).most_common(1)[0][0]

    predicted_numbers = set()
    while np.prod(list(predicted_numbers)) != common_cumulative_product and len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        predicted_numbers.add(random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number))
        if np.prod(list(predicted_numbers)) > common_cumulative_product:
            predicted_numbers.clear()

    return sorted(predicted_numbers)
