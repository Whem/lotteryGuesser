import random
from collections import Counter

from algorithms.models import lg_lottery_winner_number


def weighted_random_selection(weights):
    total = sum(weights.values())
    r = random.uniform(0, total)
    upto = 0
    for number, weight in weights.items():
        if upto + weight >= r:
            return number
        upto += weight
    assert False, "Shouldn't get here"

def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number',
                                                                                              flat=True)
    frequency = Counter()
    recency = {}
    combined_scores = {}

    # Adjustments based on lg_lottery_type instance
    min_number = lottery_type_instance.min_number
    max_number = lottery_type_instance.max_number
    num_draws = lottery_type_instance.pieces_of_draw_numbers

    # Frequency Analysis
    for draw in past_draws:
        for number in draw:
            frequency[number] += 1

    # Recency Analysis
    for i, draw in enumerate(past_draws[::-1]):
        for number in draw:
            if number not in recency:
                recency[number] = i + 1

    # Combine Scores
    for number in range(min_number, max_number + 1):
        combined_scores[number] = frequency.get(number, 0) * recency.get(number, 0)

    # Weighted Random Selection
    selected_numbers = set()
    while len(selected_numbers) < num_draws:
        selected_number = weighted_random_selection(combined_scores)
        selected_numbers.add(selected_number)

    return sorted(selected_numbers)