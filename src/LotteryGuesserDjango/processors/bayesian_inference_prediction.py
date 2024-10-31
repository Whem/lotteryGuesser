#bayesian_inference_prediction.py
from collections import Counter
from typing import List, Dict, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate main and additional numbers using Bayesian inference.
    Returns a tuple of (main_numbers, additional_numbers).
    """
    past_draws = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True)

    # Calculate frequency of each number in past draws
    number_frequencies = Counter()
    for draw in past_draws:
        number_frequencies.update(draw)

    # Convert frequencies to probabilities for main numbers
    total_draws = sum(number_frequencies.values())
    probabilities = {num: freq / total_draws for num, freq in number_frequencies.items()}

    # Apply Bayesian inference to update probabilities for main numbers
    updated_main_probabilities = bayesian_update(probabilities, lottery_type_instance.min_number,
                                                 lottery_type_instance.max_number)

    # Predict main numbers with the highest updated probabilities
    all_main_numbers = list(range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1))
    main_numbers = sorted(all_main_numbers, key=lambda x: updated_main_probabilities.get(x, 0), reverse=True)[
                   :lottery_type_instance.pieces_of_draw_numbers]

    # Generate additional numbers if required
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_probabilities = {num: probabilities.get(num, 1 / (
                    lottery_type_instance.additional_max_number - lottery_type_instance.additional_min_number + 1))
                                    for num in range(lottery_type_instance.additional_min_number,
                                                     lottery_type_instance.additional_max_number + 1)}

        # Apply Bayesian inference for additional numbers
        updated_additional_probabilities = bayesian_update(additional_probabilities,
                                                           lottery_type_instance.additional_min_number,
                                                           lottery_type_instance.additional_max_number)

        # Predict additional numbers with the highest updated probabilities
        all_additional_numbers = list(
            range(lottery_type_instance.additional_min_number, lottery_type_instance.additional_max_number + 1))
        additional_numbers = sorted(all_additional_numbers, key=lambda x: updated_additional_probabilities.get(x, 0),
                                    reverse=True)[:lottery_type_instance.additional_numbers_count]

    return sorted(main_numbers), sorted(additional_numbers)


def bayesian_update(prior_probabilities: Dict[int, float], min_num: int, max_num: int) -> Dict[int, float]:
    """
    Perform a Bayesian update on the prior probabilities for a given range of numbers.
    """

    def likelihood(number: int) -> float:
        mid = (min_num + max_num) / 2
        return 1 - abs(number - mid) / (max_num - min_num)

    posterior_probabilities = {}
    normalization_constant = 0

    for number in range(min_num, max_num + 1):
        prior = prior_probabilities.get(number, 1 / (max_num - min_num + 1))
        likelihood_value = likelihood(number)
        posterior = prior * likelihood_value
        posterior_probabilities[number] = posterior
        normalization_constant += posterior

    # Normalize probabilities
    for number in posterior_probabilities:
        posterior_probabilities[number] /= normalization_constant

    return posterior_probabilities
