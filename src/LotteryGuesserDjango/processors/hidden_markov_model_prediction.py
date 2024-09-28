# hidden_markov_model_prediction.py

import random
import numpy as np
from hmmlearn import hmm
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers using a Hidden Markov Model.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Parameters
    min_num = lottery_type_instance.min_number  # Minimum number in the lottery
    max_num = lottery_type_instance.max_number  # Maximum number in the lottery
    total_numbers = lottery_type_instance.pieces_of_draw_numbers  # Numbers to draw
    n_symbols = max_num - min_num + 1  # Total number of possible symbols

    # Retrieve past winning numbers
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list('lottery_type_number', flat=True)

    # Filter and prepare past draws
    past_draws = [
        draw for draw in past_draws_queryset
        if isinstance(draw, list) and len(draw) == total_numbers
    ]

    if len(past_draws) < 10:
        # Not enough data to train the model
        selected_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
        selected_numbers.sort()
        return selected_numbers

    # Map lottery numbers to a range starting from 0
    num_to_idx = {num: idx for idx, num in enumerate(range(min_num, max_num + 1))}
    idx_to_num = {idx: num for num, idx in num_to_idx.items()}

    # Prepare the data
    sequences = []
    lengths = []
    for draw in past_draws:
        # Convert draw numbers to indices
        indices = [num_to_idx[num] for num in draw]
        sequences.extend(indices)
        lengths.append(len(indices))

    sequences = np.array(sequences).reshape(-1, 1)

    # Train the HMM using CategoricalHMM
    model = hmm.CategoricalHMM(n_components=5, n_iter=100, random_state=42)
    model.fit(sequences, lengths)

    # Generate predictions
    _, sampled_indices = model.sample(total_numbers * 2, random_state=42)
    sampled_indices = sampled_indices.flatten().tolist()

    # Map indices back to numbers
    predicted_numbers = [idx_to_num[idx] for idx in sampled_indices]

    # Ensure numbers are within valid range and unique
    predicted_numbers = [num for num in predicted_numbers if min_num <= num <= max_num]
    predicted_numbers = list(set(predicted_numbers))

    # If not enough numbers, fill with random numbers
    if len(predicted_numbers) < total_numbers:
        all_numbers = set(range(min_num, max_num + 1))
        remaining_numbers = list(all_numbers - set(predicted_numbers))
        random.shuffle(remaining_numbers)
        predicted_numbers.extend(remaining_numbers[:total_numbers - len(predicted_numbers)])
    else:
        predicted_numbers = predicted_numbers[:total_numbers]

    # Sort and return the numbers
    predicted_numbers.sort()
    return predicted_numbers
