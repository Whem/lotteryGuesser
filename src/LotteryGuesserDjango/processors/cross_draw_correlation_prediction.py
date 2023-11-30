from collections import Counter

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    cross_correlation_counter = Counter()

    for i in range(len(past_draws) - 2):
        for number in past_draws[i]:
            for next_number in past_draws[i + 2]:
                cross_correlation_counter[(number, next_number)] += 1

    most_common_correlation = cross_correlation_counter.most_common(1)[0][0]
    return sorted(most_common_correlation)
