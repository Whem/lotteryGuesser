# harmonic_oscillation_lottery_predictor.py
import numpy as np
import random  # Make sure to import random
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Harmonic oscillation predictor for combined lottery types."""
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

    # Ensure all numbers are standard Python ints
    main_numbers = [int(n) for n in main_numbers]
    additional_numbers = [int(n) for n in additional_numbers]

    return main_numbers, additional_numbers


def generate_number_set(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        required_numbers: int,
        is_main: bool
) -> List[int]:
    """Generate numbers using harmonic oscillation analysis."""
    past_draws = get_historical_data(lottery_type_instance, is_main)
    flat_sequence = [num for draw in past_draws for num in draw]

    if len(flat_sequence) < required_numbers:
        return random_selection(min_num, max_num, required_numbers)

    # Fourier analysis
    fft_result = fourier_transform(flat_sequence)
    frequencies, amplitudes, phases = analyze_frequencies(fft_result)

    # Generate predictions
    harmonic_sequence = generate_harmonic_sequence(
        frequencies, amplitudes, phases, required_numbers * 2
    )

    # Process predictions
    predicted_numbers = process_predictions(
        harmonic_sequence,
        min_num,
        max_num,
        required_numbers
    )

    return predicted_numbers


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get historical lottery data."""
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:100])

    if is_main:
        return [draw.lottery_type_number for draw in past_draws
                if isinstance(draw.lottery_type_number, list)]
    else:
        return [draw.additional_numbers for draw in past_draws
                if hasattr(draw, 'additional_numbers') and
                isinstance(draw.additional_numbers, list)]


def fourier_transform(data: List[int]) -> np.ndarray:
    """Compute Fourier transform of input data."""
    return np.fft.fft(data)


def analyze_frequencies(fft_result: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analyze dominant frequencies from FFT result."""
    frequencies = np.fft.fftfreq(len(fft_result))
    amplitudes = np.abs(fft_result)

    top_indices = np.argsort(amplitudes)[-5:]
    top_frequencies = frequencies[top_indices]
    top_amplitudes = amplitudes[top_indices]
    top_phases = np.angle(fft_result[top_indices])

    return top_frequencies, top_amplitudes, top_phases


def generate_harmonic_sequence(
        frequencies: np.ndarray,
        amplitudes: np.ndarray,
        phases: np.ndarray,
        length: int
) -> np.ndarray:
    """Generate sequence from harmonic oscillations."""
    t = np.arange(length)
    sequence = np.zeros(length)

    for freq, amp, phase in zip(frequencies, amplitudes, phases):
        if amp > 0:
            sequence += amp * np.sin(2 * np.pi * freq * t + phase)

    return sequence


def process_predictions(
        harmonic_sequence: np.ndarray,
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Process harmonic sequence into valid predictions."""
    if np.all(harmonic_sequence == 0):
        return random_selection(min_num, max_num, required_numbers)

    # Scale sequence
    min_seq, max_seq = np.min(harmonic_sequence), np.max(harmonic_sequence)
    if min_seq == max_seq:
        scaled_sequence = np.full_like(harmonic_sequence, (min_num + max_num) / 2)
    else:
        scaled_sequence = (harmonic_sequence - min_seq) / (max_seq - min_seq)
        scaled_sequence = scaled_sequence * (max_num - min_num) + min_num

    # Generate predictions
    predicted_numbers = []
    for num in scaled_sequence:
        rounded_num = int(round(float(num)))
        rounded_num = max(min_num, min(rounded_num, max_num))
        if rounded_num not in predicted_numbers:
            predicted_numbers.append(rounded_num)
        if len(predicted_numbers) == required_numbers:
            break

    # Fill if needed
    if len(predicted_numbers) < required_numbers:
        remaining = set(range(min_num, max_num + 1)) - set(predicted_numbers)
        additional = random_selection(
            min_num,
            max_num,
            required_numbers - len(predicted_numbers),
            list(remaining)
        )
        predicted_numbers.extend(additional)

    # Ensure all numbers are standard Python ints
    predicted_numbers = [int(n) for n in predicted_numbers]

    return sorted(predicted_numbers)


def random_selection(
        min_num: int,
        max_num: int,
        count: int,
        available: List[int] = None
) -> List[int]:
    """Generate random number selection."""
    if available is None:
        available = list(range(min_num, max_num + 1))
    selection = np.random.choice(available, count, replace=False)
    return sorted([int(n) for n in selection])
