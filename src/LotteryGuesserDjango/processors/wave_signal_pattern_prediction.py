# wave_signal_pattern_prediction.py
# Advanced signal processing and wave analysis for lottery prediction
# Combines digital signal processing, wavelet analysis, and frequency domain patterns

import random
import math
from typing import List, Set, Dict, Tuple
from collections import defaultdict
import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
from scipy.signal import find_peaks, savgol_filter
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import statistics


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers based on wave and signal pattern prediction.

    This function generates both main numbers and additional numbers (if applicable) by analyzing
    historical lottery draws using advanced signal processing techniques, including wave pattern analysis,
    frequency domain analysis, and signal peak detection. It prioritizes numbers based on identified
    patterns and fills any remaining slots with weighted random selections.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A tuple containing two lists:
        - main_numbers: A sorted list of predicted main lottery numbers.
        - additional_numbers: A sorted list of predicted additional lottery numbers (if applicable).
    """
    try:
        # Generate main numbers
        main_numbers = generate_numbers(
            lottery_type_instance=lottery_type_instance,
            number_field='lottery_type_number',
            min_num=int(lottery_type_instance.min_number),
            max_num=int(lottery_type_instance.max_number),
            total_numbers=int(lottery_type_instance.pieces_of_draw_numbers)
        )

        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            # Generate additional numbers
            additional_numbers = generate_numbers(
                lottery_type_instance=lottery_type_instance,
                number_field='additional_numbers',
                min_num=int(lottery_type_instance.additional_min_number),
                max_num=int(lottery_type_instance.additional_max_number),
                total_numbers=int(lottery_type_instance.additional_numbers_count)
            )

        return main_numbers, additional_numbers

    except Exception as e:
        # Log the error (consider using a proper logging system)
        print(f"Error in get_numbers: {str(e)}")
        # Fall back to random number generation
        min_num = int(lottery_type_instance.min_number)
        max_num = int(lottery_type_instance.max_number)
        total_numbers = int(lottery_type_instance.pieces_of_draw_numbers)
        main_numbers = generate_random_numbers(min_num, max_num, total_numbers)

        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            additional_min_num = int(lottery_type_instance.additional_min_number)
            additional_max_num = int(lottery_type_instance.additional_max_number)
            additional_total_numbers = int(lottery_type_instance.additional_numbers_count)
            additional_numbers = generate_random_numbers(additional_min_num, additional_max_num, additional_total_numbers)

        return main_numbers, additional_numbers


def generate_numbers(
    lottery_type_instance: lg_lottery_type,
    number_field: str,
    min_num: int,
    max_num: int,
    total_numbers: int
) -> List[int]:
    """
    Generates a list of lottery numbers using wave and signal pattern prediction.

    This helper function encapsulates the logic for generating numbers, allowing reuse for both
    main and additional numbers. It performs the following steps:
    1. Retrieves and preprocesses historical lottery data.
    2. Analyzes wave patterns to identify potential number candidates.
    3. Analyzes the frequency domain to identify dominant number trends.
    4. Detects significant signal peaks to identify number patterns.
    5. Fills any remaining slots with weighted random selections based on wave-based probabilities.

    Parameters:
    - lottery_type_instance: The lottery type instance.
    - number_field: The field name in lg_lottery_winner_number to retrieve past numbers
                    ('lottery_type_number' or 'additional_numbers').
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Number of numbers to generate.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    try:
        # Retrieve past winning numbers
        past_draws_queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('-id')[:200].values_list(number_field, flat=True)

        past_draws = [
            draw for draw in past_draws_queryset
            if isinstance(draw, list)
        ]

        if not past_draws:
            # If no past draws are available, generate random numbers
            return generate_random_numbers(min_num, max_num, total_numbers)

        required_numbers = total_numbers
        candidates = set()

        # 1. Wave Pattern Analysis
        wave_numbers = analyze_wave_patterns(
            past_draws=past_draws,
            min_num=min_num,
            max_num=max_num
        )
        candidates.update(wave_numbers[:required_numbers // 3])

        # 2. Frequency Domain Analysis
        frequency_numbers = analyze_frequency_domain(
            past_draws=past_draws,
            min_num=min_num,
            max_num=max_num
        )
        candidates.update(frequency_numbers[:required_numbers // 3])

        # 3. Signal Peak Detection
        peak_numbers = detect_signal_peaks(
            past_draws=past_draws,
            min_num=min_num,
            max_num=max_num
        )
        candidates.update(peak_numbers[:required_numbers // 3])

        # 4. Fill remaining slots using wave-based probability
        while len(candidates) < required_numbers:
            weights = calculate_wave_probabilities(
                past_draws=past_draws,
                min_num=min_num,
                max_num=max_num,
                excluded_numbers=candidates
            )

            available_numbers = set(range(min_num, max_num + 1)) - candidates

            if available_numbers:
                number_weights = [weights.get(num, 1.0) for num in available_numbers]
                selected = random.choices(list(available_numbers), weights=number_weights, k=1)[0]
                candidates.add(selected)
            else:
                break  # No more available numbers to select

        # Ensure uniqueness and correct count
        selected_numbers = sorted(list(candidates))[:required_numbers]

        # If still not enough, fill with random numbers
        if len(selected_numbers) < required_numbers:
            remaining_slots = required_numbers - len(selected_numbers)
            remaining_numbers = list(set(range(min_num, max_num + 1)) - set(selected_numbers))
            selected_numbers.extend(random.sample(remaining_numbers, remaining_slots))

        return sorted(selected_numbers)

    except Exception as e:
        # Log the error (consider using a proper logging system)
        print(f"Error in generate_numbers: {str(e)}")
        # Fall back to random number generation
        return generate_random_numbers(min_num, max_num, total_numbers)


def analyze_wave_patterns(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """
    Analyze wave patterns in the number sequences to identify potential number candidates.

    Parameters:
    - past_draws: A list of past lottery number draws.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.

    Returns:
    - A list of numbers sorted by their wave pattern scores in descending order.
    """
    wave_scores = defaultdict(float)

    # Convert draws to continuous signals per position
    signals = []
    for i in range(len(past_draws[0])):  # For each number position
        position_signal = []
        for draw in past_draws:
            if i < len(draw):
                position_signal.append(draw[i])
        signals.append(position_signal)

    for signal_data in signals:
        if len(signal_data) < 4:  # Need minimum data points
            continue

        # Apply various window functions
        windows = {
            'hamming': np.hamming(len(signal_data)),
            'hanning': np.hanning(len(signal_data)),
            'blackman': np.blackman(len(signal_data))
        }

        for window_name, window in windows.items():
            try:
                # Apply window function
                windowed_signal = np.array(signal_data) * window

                # Find local maxima and minima
                peaks, _ = find_peaks(windowed_signal)
                valleys, _ = find_peaks(-windowed_signal)

                if len(peaks) >= 2 and len(valleys) >= 2:
                    # Calculate wave periods
                    peak_period = np.mean(np.diff(peaks))
                    valley_period = np.mean(np.diff(valleys))

                    # Predict next peak and valley
                    next_peak_pos = int(peaks[-1] + peak_period)
                    next_valley_pos = int(valleys[-1] + valley_period)

                    # Ensure the predicted positions are within bounds
                    if next_peak_pos < len(signal_data):
                        next_peak = windowed_signal[next_peak_pos]
                        if min_num <= next_peak <= max_num:
                            wave_scores[int(next_peak)] += 1.0

                    if next_valley_pos < len(signal_data):
                        next_valley = windowed_signal[next_valley_pos]
                        if min_num <= next_valley <= max_num:
                            wave_scores[int(next_valley)] += 0.8

                # Analyze wave envelope using Hilbert transform
                envelope = np.abs(signal.hilbert(windowed_signal))
                if len(envelope) > 0:
                    trend = np.polyfit(np.arange(len(envelope)), envelope, 2)
                    next_value = np.polyval(trend, len(envelope))
                    if min_num <= next_value <= max_num:
                        wave_scores[int(next_value)] += 1.2

            except Exception as e:
                print(f"Wave analysis error in window '{window_name}': {e}")
                continue

    # Sort numbers by their wave scores in descending order
    sorted_wave_numbers = sorted(wave_scores.keys(), key=lambda x: wave_scores[x], reverse=True)
    return sorted_wave_numbers


def analyze_frequency_domain(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """
    Analyze frequency domain characteristics of the number sequences to identify potential number candidates.

    Parameters:
    - past_draws: A list of past lottery number draws.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.

    Returns:
    - A list of numbers sorted by their frequency domain scores in descending order.
    """
    frequency_scores = defaultdict(float)

    # Analyze frequency domain for each position
    for position in range(max(len(draw) for draw in past_draws)):
        position_data = []
        for draw in past_draws:
            if position < len(draw):
                position_data.append(draw[position])

        if len(position_data) < 4:  # Need minimum data points
            continue

        try:
            # Apply FFT
            fft_result = fft(position_data)
            frequencies = np.abs(fft_result)

            # Find dominant frequencies (top 3 excluding DC component)
            dominant_freq_indices = np.argsort(frequencies[1:])[-3:] + 1  # Exclude DC
            dominant_frequencies = frequencies[dominant_freq_indices]

            # Predict next values based on dominant frequencies
            for idx, amp in zip(dominant_freq_indices, dominant_frequencies):
                # Project next value based on frequency component
                t = len(position_data)
                predicted = amp * math.cos(2 * math.pi * idx * t / len(frequencies))

                if min_num <= predicted <= max_num:
                    frequency_scores[int(predicted)] += amp / sum(frequencies)

            # Inverse FFT for time domain prediction
            filtered_fft = np.zeros_like(fft_result)
            filtered_fft[dominant_freq_indices] = fft_result[dominant_freq_indices]
            reconstructed = ifft(filtered_fft).real

            # Use last reconstructed value as prediction
            if min_num <= reconstructed[-1] <= max_num:
                frequency_scores[int(reconstructed[-1])] += 1.0

        except Exception as e:
            print(f"Frequency analysis error at position {position}: {e}")
            continue

    # Sort numbers by their frequency domain scores in descending order
    sorted_frequency_numbers = sorted(frequency_scores.keys(), key=lambda x: frequency_scores[x], reverse=True)
    return sorted_frequency_numbers


def detect_signal_peaks(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """
    Detect significant peaks and patterns in the number sequences to identify potential number candidates.

    Parameters:
    - past_draws: A list of past lottery number draws.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.

    Returns:
    - A list of numbers sorted by their peak detection scores in descending order.
    """
    peak_scores = defaultdict(float)

    # Analyze multiple signal representations with different window sizes
    for window_size in [5, 10, 20]:
        for position in range(max(len(draw) for draw in past_draws)):
            signal_data = []
            for draw in past_draws:
                if position < len(draw):
                    signal_data.append(draw[position])

            if len(signal_data) < window_size:
                continue

            try:
                # Apply smoothing with Savitzky-Golay filter
                smoothed = savgol_filter(signal_data, window_length=window_size, polyorder=3)

                # Find peaks
                peaks, properties = find_peaks(
                    smoothed,
                    height=None,
                    threshold=None,
                    distance=2,
                    prominence=1
                )

                if len(peaks) >= 2:
                    # Calculate average peak distance
                    peak_distances = np.diff(peaks)
                    avg_distance = np.mean(peak_distances)

                    # Predict next peak position
                    next_peak_pos = peaks[-1] + int(avg_distance)
                    if next_peak_pos < len(smoothed):
                        next_peak = smoothed[next_peak_pos]
                        if min_num <= next_peak <= max_num:
                            peak_scores[int(next_peak)] += 1.0

                    # Analyze peak height trend
                    peak_heights = properties.get('peak_heights', [])
                    if len(peak_heights) >= 2:
                        height_trend = np.polyfit(peaks[:len(peak_heights)], peak_heights, 1)
                        next_height = np.polyval(height_trend, next_peak_pos)
                        if min_num <= next_height <= max_num:
                            peak_scores[int(next_height)] += 0.8

                # Detect valleys by inverting the signal
                valleys, valley_props = find_peaks(-smoothed)
                if len(valleys) >= 2:
                    valley_distances = np.diff(valleys)
                    avg_valley_distance = np.mean(valley_distances)

                    # Predict next valley position
                    next_valley_pos = valleys[-1] + int(avg_valley_distance)
                    if next_valley_pos < len(smoothed):
                        next_valley = smoothed[next_valley_pos]
                        if min_num <= next_valley <= max_num:
                            peak_scores[int(next_valley)] += 0.7

            except Exception as e:
                print(f"Peak detection error with window size {window_size} at position {position}: {e}")
                continue

    # Sort numbers by their peak detection scores in descending order
    sorted_peak_numbers = sorted(peak_scores.keys(), key=lambda x: peak_scores[x], reverse=True)
    return sorted_peak_numbers


def calculate_wave_probabilities(
    past_draws: List[List[int]],
    min_num: int,
    max_num: int,
    excluded_numbers: Set[int]
) -> Dict[int, float]:
    """
    Calculate probabilities based on wave characteristics for weighted random selection.

    This function assigns weights to each number based on how closely it aligns with the
    statistical properties (mean and standard deviation) of historical data. Numbers not
    yet selected are given higher weights if they are more statistically probable.

    Parameters:
    - past_draws: List of past lottery number draws.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - excluded_numbers: Set of numbers to exclude from selection.

    Returns:
    - A dictionary mapping each number to its calculated weight.
    """
    weights = defaultdict(float)

    if not past_draws:
        return weights

    try:
        # Calculate overall mean and standard deviation from past draws
        all_numbers = [num for draw in past_draws for num in draw]
        overall_mean = statistics.mean(all_numbers)
        overall_stdev = statistics.stdev(all_numbers) if len(all_numbers) > 1 else 1.0

        for num in range(min_num, max_num + 1):
            if num in excluded_numbers:
                continue

            # Calculate z-score for the number
            z_score = abs((num - overall_mean) / overall_stdev) if overall_stdev != 0 else 0.0

            # Assign higher weight to numbers closer to the mean
            weight = max(0, 1 - z_score)
            weights[num] = weight

    except Exception as e:
        print(f"Weight calculation error: {e}")
        # Fallback to uniform weights
        for num in range(min_num, max_num + 1):
            if num not in excluded_numbers:
                weights[num] = 1.0

    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        for num in weights:
            weights[num] /= total_weight

    return weights


def weighted_random_choice(weights: Dict[int, float], available_numbers: Set[int]) -> int:
    """
    Selects a random number based on weighted probabilities.

    Parameters:
    - weights: A dictionary mapping numbers to their weights.
    - available_numbers: A set of numbers available for selection.

    Returns:
    - A single selected number.
    """
    try:
        numbers = list(available_numbers)
        number_weights = [weights.get(num, 1.0) for num in numbers]
        total = sum(number_weights)
        if total == 0:
            return random.choice(numbers)
        probabilities = [w / total for w in number_weights]
        selected = random.choices(numbers, weights=probabilities, k=1)[0]
        return selected
    except Exception as e:
        print(f"Weighted random choice error: {e}")
        return random.choice(list(available_numbers)) if available_numbers else None


def generate_random_numbers(min_num: int, max_num: int, total_numbers: int) -> List[int]:
    """
    Generates a sorted list of unique random numbers within the specified range.

    Parameters:
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Number of numbers to generate.

    Returns:
    - A sorted list of randomly generated lottery numbers.
    """
    try:
        numbers = set()
        while len(numbers) < total_numbers:
            num = random.randint(min_num, max_num)
            numbers.add(num)
        return sorted(list(numbers))
    except Exception as e:
        print(f"Error in generate_random_numbers: {str(e)}")
        # As a last resort, return a sequential list
        return list(range(min_num, min_num + total_numbers))
