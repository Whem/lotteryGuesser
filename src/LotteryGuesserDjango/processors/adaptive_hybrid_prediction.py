import numpy as np
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

    # Simulating different prediction methods
    methods = [
        lambda: np.random.choice(range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1),
                                 lottery_type_instance.pieces_of_draw_numbers, replace=False).tolist(),
        lambda: sorted(past_draws[0])[:lottery_type_instance.pieces_of_draw_numbers],
        lambda: np.random.choice(np.unique(past_draws), lottery_type_instance.pieces_of_draw_numbers, replace=False).tolist()
    ]

    # Adaptive weights (initially equal)
    weights = [1.0 / len(methods)] * len(methods)

    # Generate predictions
    predictions = [method() for method in methods]

    # Evaluate recent performance (simplified)
    recent_draw = past_draws[0]
    performances = [len(set(pred) & set(recent_draw)) for pred in predictions]

    # Update weights
    total_performance = sum(performances)
    if total_performance > 0:
        weights = [p / total_performance for p in performances]
    else:
        weights = [1.0 / len(methods)] * len(methods)

    # Generate final prediction
    final_prediction = set()
    while len(final_prediction) < lottery_type_instance.pieces_of_draw_numbers:
        method_index = int(np.random.choice(len(methods), p=weights))
        number = int(np.random.choice(predictions[method_index]))
        if lottery_type_instance.min_number <= number <= lottery_type_instance.max_number:
            final_prediction.add(number)

    return sorted(list(final_prediction))