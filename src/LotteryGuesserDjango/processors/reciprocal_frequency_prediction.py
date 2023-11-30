from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    frequency = Counter()

    for draw in past_draws:
        frequency.update(draw)

    # Assign a weight based on the reciprocal of frequency
    weights = {number: 1.0 / count for number, count in frequency.items()}
    predicted_numbers = [max(weights, key=weights.get) for _ in range(lottery_type_instance.pieces_of_draw_numbers)]

    return predicted_numbers
