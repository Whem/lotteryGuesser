# hidden_markov_model_prediction.py
import random
import numpy as np
from hmmlearn import hmm
from typing import List, Tuple, Dict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """HMM predictor for combined lottery types."""
    main_numbers = generate_number_set(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        True
    )

    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_number_set(
            lottery_type_instance,
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count,
            False
        )

    return main_numbers, additional_numbers


def generate_number_set(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        required_numbers: int,
        is_main: bool
) -> List[int]:
    """Generate numbers using HMM analysis."""
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if len(past_draws) < 10:
        return random_selection(min_num, max_num, required_numbers)

    # Setup HMM
    n_symbols = max_num - min_num + 1
    num_mapping = create_number_mapping(min_num, max_num)

    # Prepare sequences
    sequences, lengths = prepare_sequences(past_draws, num_mapping)

    # Train and generate predictions
    predicted_numbers = generate_predictions(
        sequences,
        lengths,
        n_symbols,
        num_mapping,
        min_num,
        max_num,
        required_numbers
    )

    return predicted_numbers


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get historical lottery data."""
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id'))

    if is_main:
        draws = [draw.lottery_type_number for draw in past_draws
                 if isinstance(draw.lottery_type_number, list)]
    else:
        draws = [draw.additional_numbers for draw in past_draws
                 if hasattr(draw, 'additional_numbers') and
                 isinstance(draw.additional_numbers, list)]

    return draws


def create_number_mapping(min_num: int, max_num: int) -> Dict[str, Dict[int, int]]:
    """Create mappings between numbers and indices."""
    num_to_idx = {num: idx for idx, num in enumerate(range(min_num, max_num + 1))}
    idx_to_num = {idx: num for num, idx in num_to_idx.items()}
    return {'num_to_idx': num_to_idx, 'idx_to_num': idx_to_num}


def prepare_sequences(past_draws: List[List[int]], num_mapping: Dict) -> Tuple[np.ndarray, List[int]]:
    """Prepare sequences for HMM training."""
    sequences = []
    lengths = []
    num_to_idx = num_mapping['num_to_idx']

    for draw in past_draws:
        indices = [num_to_idx[num] for num in draw]
        sequences.extend(indices)
        lengths.append(len(indices))

    return np.array(sequences).reshape(-1, 1), lengths


def generate_predictions(
        sequences: np.ndarray,
        lengths: List[int],
        n_symbols: int,
        num_mapping: Dict,
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate predictions using trained HMM."""
    try:
        # Train HMM
        model = hmm.CategoricalHMM(
            n_components=5,
            n_iter=100,
            random_state=42
        )
        model.fit(sequences, lengths)

        # Generate samples
        _, sampled_indices = model.sample(required_numbers * 2)
        sampled_indices = sampled_indices.flatten()

        # Convert to numbers
        idx_to_num = num_mapping['idx_to_num']
        predicted_numbers = [
            idx_to_num[idx] for idx in sampled_indices
            if min_num <= idx_to_num[idx] <= max_num
        ]
        predicted_numbers = list(set(predicted_numbers))

        # Handle insufficient predictions
        if len(predicted_numbers) < required_numbers:
            remaining = set(range(min_num, max_num + 1)) - set(predicted_numbers)
            additional = random.sample(list(remaining),
                                       required_numbers - len(predicted_numbers))
            predicted_numbers.extend(additional)
        else:
            predicted_numbers = predicted_numbers[:required_numbers]

        return sorted(predicted_numbers)

    except Exception as e:
        print(f"HMM prediction error: {str(e)}")
        return random_selection(min_num, max_num, required_numbers)


def random_selection(min_num: int, max_num: int, count: int) -> List[int]:
    """Generate random number selection."""
    return sorted(random.sample(range(min_num, max_num + 1), count))