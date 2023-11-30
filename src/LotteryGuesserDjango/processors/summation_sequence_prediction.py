import random
from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    sum_sequences = Counter()

    for draw in past_draws:
        sum_sequences[sum(draw)] += 1

    target_sum = sum_sequences.most_common(1)[0][0]
    predicted_numbers = []

    while sum(predicted_numbers) != target_sum:
        predicted_numbers.append(random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number))
        if sum(predicted_numbers) > target_sum:
            predicted_numbers.pop()

    return predicted_numbers
