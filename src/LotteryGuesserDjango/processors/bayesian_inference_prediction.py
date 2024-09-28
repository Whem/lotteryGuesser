from collections import Counter



from typing import List, Dict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    number_frequencies = Counter()

    for draw in past_draws:
        number_frequencies.update(draw)

    # Convert frequencies to probabilities
    total_draws = sum(number_frequencies.values())
    probabilities = {num: freq / total_draws for num, freq in number_frequencies.items()}

    # Apply Bayesian inference to update probabilities
    updated_probabilities = bayesian_update(probabilities, lottery_type_instance)

    # Predict numbers with highest updated probabilities
    all_numbers = list(range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1))
    predicted_numbers = sorted(all_numbers, key=lambda x: updated_probabilities.get(x, 0), reverse=True)[:lottery_type_instance.pieces_of_draw_numbers]

    return predicted_numbers

def bayesian_update(prior_probabilities: Dict[int, float], lottery_type_instance: lg_lottery_type) -> Dict[int, float]:
    # Define a simple likelihood function
    def likelihood(number: int) -> float:
        # Example: numbers closer to the middle of the range are slightly more likely
        mid = (lottery_type_instance.min_number + lottery_type_instance.max_number) / 2
        return 1 - abs(number - mid) / (lottery_type_instance.max_number - lottery_type_instance.min_number)

    # Calculate posterior probabilities
    posterior_probabilities = {}
    normalization_constant = 0

    for number in range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1):
        prior = prior_probabilities.get(number, 1 / (lottery_type_instance.max_number - lottery_type_instance.min_number + 1))
        likelihood_value = likelihood(number)
        posterior = prior * likelihood_value
        posterior_probabilities[number] = posterior
        normalization_constant += posterior

    # Normalize probabilities
    for number in posterior_probabilities:
        posterior_probabilities[number] /= normalization_constant

    return posterior_probabilities
