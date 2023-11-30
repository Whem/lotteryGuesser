from collections import Counter

from scipy.stats import bayes_mvs

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    number_frequencies = Counter()

    for draw in past_draws:
        number_frequencies.update(draw)

    # Convert frequencies to probabilities
    total_draws = sum(number_frequencies.values())
    probabilities = {num: freq / total_draws for num, freq in number_frequencies.items()}

    # Apply Bayesian inference to update probabilities
    # ...

    # Predict numbers with highest updated probabilities
    predicted_numbers = sorted(probabilities, key=probabilities.get, reverse=True)[:lottery_type_instance.pieces_of_draw_numbers]

    return predicted_numbers
