from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number',
                                                                                              flat=True)
    min_number = lottery_type_instance.min_number
    max_number = lottery_type_instance.max_number
    num_draws = lottery_type_instance.pieces_of_draw_numbers

    # Track the change in frequency over time
    frequency_change = {number: 0 for number in range(min_number, max_number + 1)}
    previous_frequency = Counter()

    for draw in past_draws:
        current_frequency = Counter(draw)
        for number in range(min_number, max_number + 1):
            frequency_change[number] += current_frequency[number] - previous_frequency[number]
        previous_frequency = current_frequency

    # Select numbers based on the highest positive change in frequency
    trending_numbers = sorted(frequency_change, key=frequency_change.get, reverse=True)[:num_draws]

    return sorted(trending_numbers)