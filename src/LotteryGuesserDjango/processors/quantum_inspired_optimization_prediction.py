# quantum_inspired_optimization_prediction.py

import numpy as np
from algorithms.models import lg_lottery_winner_number

def quantum_inspired_random(min_val, max_val, n, prob_dist):
    """
    Generate quantum-inspired random numbers.
    """
    phase = np.random.random(max_val - min_val + 1) * 2 * np.pi
    amplitude = np.sqrt(prob_dist)
    quantum_state = amplitude * np.exp(1j * phase)
    prob = np.abs(quantum_state) ** 2
    prob /= np.sum(prob)  # Normalize probabilities
    return np.random.choice(range(min_val, max_val + 1), size=n, p=prob)

def get_numbers(lottery_type_instance):
    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    total_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Retrieve past winning numbers
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:50].values_list('lottery_type_number', flat=True))

    # Calculate the frequency of each number
    number_frequency = {i: 1 for i in range(min_num, max_num + 1)}  # Initialize with 1 to avoid zero probabilities
    for draw in past_draws:
        for num in draw:
            number_frequency[num] += 1

    # Create probability distribution
    prob_distribution = [number_frequency[i] for i in range(min_num, max_num + 1)]
    prob_distribution = np.array(prob_distribution) / sum(prob_distribution)  # Normalize

    # Generate quantum-inspired numbers based on the probability distribution
    quantum_numbers = quantum_inspired_random(min_num, max_num, total_numbers * 2, prob_distribution)

    # Select unique numbers
    selected_numbers = []
    for num in quantum_numbers:
        if num not in selected_numbers:
            selected_numbers.append(int(num))  # Convert to Python int
        if len(selected_numbers) == total_numbers:
            break

    # If not enough numbers, fill with random selection
    if len(selected_numbers) < total_numbers:
        remaining = set(range(min_num, max_num + 1)) - set(selected_numbers)
        selected_numbers += [int(num) for num in np.random.choice(list(remaining), total_numbers - len(selected_numbers), replace=False)]

    return sorted(selected_numbers)