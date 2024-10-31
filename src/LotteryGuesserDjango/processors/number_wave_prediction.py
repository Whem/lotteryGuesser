# number_wave_prediction.py
from collections import defaultdict
import random
import numpy as np
from typing import List, Dict, Tuple, Set
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate both main and additional numbers using wave prediction.
    Returns a tuple of (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_number_set(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        is_main=True
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_number_set(
            lottery_type_instance,
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count,
            is_main=False
        )

    return main_numbers, additional_numbers


def generate_number_set(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        required_numbers: int,
        is_main: bool
) -> List[int]:
    """Generate a set of numbers using wave prediction."""
    # Get past draws
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:50])  # Analyze more draws for better wave patterns

    # Extract appropriate number sets from past draws
    if is_main:
        past_numbers = [draw.lottery_type_number for draw in past_draws
                        if isinstance(draw.lottery_type_number, list)]
    else:
        past_numbers = [draw.additional_numbers for draw in past_draws
                        if hasattr(draw, 'additional_numbers') and
                        isinstance(draw.additional_numbers, list)]

    if not past_numbers:
        return generate_random_numbers(min_num, max_num, required_numbers)

    wave_patterns = analyze_wave_patterns(past_numbers, min_num, max_num)
    selected_numbers = select_numbers_by_waves(
        wave_patterns,
        min_num,
        max_num,
        required_numbers
    )

    return sorted(selected_numbers)


def analyze_wave_patterns(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int
) -> Dict[int, Dict[str, float]]:
    """
    Analyze wave patterns in number frequencies over time.
    Returns comprehensive wave statistics for each number.
    """
    wave_stats = {num: {
        'frequency_wave': [],
        'amplitude': 0,
        'trend': 0,
        'cycle_length': 0,
        'current_phase': 0
    } for num in range(min_num, max_num + 1)}

    if not past_draws:
        return wave_stats

    # Build frequency waves
    window_size = 5  # Size of rolling window for wave analysis
    for num in range(min_num, max_num + 1):
        frequencies = []
        for i in range(len(past_draws) - window_size + 1):
            window = past_draws[i:i + window_size]
            freq = sum(1 for draw in window if num in draw)
            frequencies.append(freq)

        if frequencies:
            stats = wave_stats[num]
            stats['frequency_wave'] = frequencies

            # Calculate wave characteristics
            wave = np.array(frequencies)

            # Amplitude (peak-to-peak difference)
            stats['amplitude'] = np.ptp(wave)

            # Trend (linear regression slope)
            if len(wave) > 1:
                x = np.arange(len(wave))
                slope, _ = np.polyfit(x, wave, 1)
                stats['trend'] = slope

            # Estimate cycle length using autocorrelation
            if len(wave) > 4:
                autocorr = np.correlate(wave - np.mean(wave), wave - np.mean(wave), mode='full')
                peaks = find_peaks(autocorr[len(autocorr) // 2:])
                if peaks:
                    stats['cycle_length'] = peaks[0]

            # Current phase (position in the wave cycle)
            if len(wave) >= 2:
                if wave[-1] > wave[-2]:
                    stats['current_phase'] = 1  # Rising
                else:
                    stats['current_phase'] = -1  # Falling

    return wave_stats


def find_peaks(arr: np.ndarray) -> List[int]:
    """Find indices of local maxima in an array."""
    peaks = []
    for i in range(1, len(arr) - 1):
        if arr[i - 1] < arr[i] > arr[i + 1]:
            peaks.append(i)
    return peaks


def select_numbers_by_waves(
        wave_stats: Dict[int, Dict[str, float]],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Select numbers based on wave pattern analysis."""
    selection_scores = {}

    # Calculate selection scores based on wave characteristics
    for num in range(min_num, max_num + 1):
        stats = wave_stats[num]

        # Skip numbers with no wave data
        if not stats['frequency_wave']:
            continue

        # Calculate composite wave score
        wave_score = (
                stats['amplitude'] * 0.3 +  # Higher amplitude suggests more active number
                stats['trend'] * 0.3 +  # Positive trend suggests increasing frequency
                (1 / (stats['cycle_length'] + 1)) * 0.2 +  # Shorter cycles preferred
                stats['current_phase'] * 0.2  # Prefer numbers in rising phase
        )
        selection_scores[num] = wave_score

    # Select numbers with highest wave scores
    sorted_numbers = sorted(selection_scores.items(), key=lambda x: x[1], reverse=True)
    selected_numbers = [num for num, _ in sorted_numbers[:required_numbers]]

    # If we don't have enough numbers, add random ones
    while len(selected_numbers) < required_numbers:
        num = random.randint(min_num, max_num)
        if num not in selected_numbers:
            selected_numbers.append(num)

    return selected_numbers


def generate_random_numbers(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate random numbers when no historical data is available."""
    numbers = set()
    while len(numbers) < required_numbers:
        numbers.add(random.randint(min_num, max_num))
    return sorted(list(numbers))