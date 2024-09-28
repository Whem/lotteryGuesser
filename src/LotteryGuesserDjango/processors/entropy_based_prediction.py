import numpy as np
from scipy.stats import entropy
from typing import List
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:100].values_list('lottery_type_number', flat=True))

    if len(past_draws) < 20:
        return sorted(np.random.choice(range(lottery_type_instance.min_number,
                                             lottery_type_instance.max_number + 1),
                                       lottery_type_instance.pieces_of_draw_numbers,
                                       replace=False).tolist())

    flat_past_draws = [num for draw in past_draws for num in draw]

    # Calculate probabilities
    unique, counts = np.unique(flat_past_draws, return_counts=True)
    probs = counts.astype(float) / len(flat_past_draws)

    # Calculate entropy
    draw_entropy = float(entropy(probs))

    # Generate numbers based on entropy
    if draw_entropy > 0.9:  # High entropy, more random
        predicted_numbers = np.random.choice(range(lottery_type_instance.min_number,
                                                   lottery_type_instance.max_number + 1),
                                             lottery_type_instance.pieces_of_draw_numbers,
                                             replace=False)
    else:  # Lower entropy, use probabilities
        predicted_numbers = np.random.choice(unique,
                                             size=lottery_type_instance.pieces_of_draw_numbers,
                                             p=probs,
                                             replace=False)

    # Convert to list of integers
    predicted_numbers = [int(num) for num in predicted_numbers]

    return sorted(predicted_numbers)