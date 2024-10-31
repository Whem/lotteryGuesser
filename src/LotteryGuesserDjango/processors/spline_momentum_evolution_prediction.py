# spline_momentum_evolution_prediction.py
# Advanced spline interpolation with adaptive momentum and wave mechanics
# Combines spline analysis with evolutionary momentum patterns

import numpy as np
import statistics
import random
from typing import List, Tuple, Set, Dict
from collections import Counter, defaultdict
from scipy.interpolate import CubicSpline
from scipy import signal
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using Spline Interpolation with Adaptive Momentum
    and Wave Mechanics Analysis.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A tuple containing two lists:
        - main_numbers: A sorted list of predicted main lottery numbers.
        - additional_numbers: A sorted list of predicted additional lottery numbers (if applicable).
    """
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


def generate_numbers(
    lottery_type_instance: lg_lottery_type,
    number_field: str,
    min_num: int,
    max_num: int,
    total_numbers: int
) -> List[int]:
    """
    Generates a list of lottery numbers using Spline Interpolation with Adaptive Momentum
    and Wave Mechanics Analysis.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.
    - number_field: The field name in lg_lottery_winner_number to retrieve past numbers
                    ('lottery_type_number' or 'additional_numbers').
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Total numbers to generate.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Retrieve past winning numbers
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list(number_field, flat=True).order_by('-id')[:200])

    # Filter valid past draws
    past_draws = [draw for draw in past_draws if isinstance(draw, list) and len(draw) == total_numbers]
    if not past_draws:
        return generate_random_numbers(lottery_type_instance, min_num, max_num, total_numbers)

    candidates = set()

    # 1. Spline Interpolation Analysis
    spline_numbers = analyze_spline_patterns(
        past_draws,
        min_num,
        max_num
    )
    candidates.update(spline_numbers[:max(total_numbers // 3, 1)])

    # 2. Momentum Evolution Analysis
    momentum_numbers = analyze_momentum_evolution(
        past_draws,
        min_num,
        max_num
    )
    candidates.update(momentum_numbers[:max(total_numbers // 3, 1)])

    # 3. Wave Resonance Analysis
    resonance_numbers = analyze_wave_resonance(
        past_draws,
        min_num,
        max_num
    )
    candidates.update(resonance_numbers[:max(total_numbers // 3, 1)])

    # Fill remaining slots using adaptive field theory
    while len(candidates) < total_numbers:
        weights = calculate_field_weights(
            past_draws,
            min_num,
            max_num,
            candidates
        )

        available_numbers = set(range(min_num, max_num + 1)) - candidates
        if not available_numbers:
            break  # No more numbers to select

        number_weights = [weights.get(num, 1.0) for num in available_numbers]
        selected = random.choices(list(available_numbers), weights=number_weights, k=1)[0]
        candidates.add(selected)

    return sorted(list(candidates))[:total_numbers]


def analyze_spline_patterns(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """
    Analyze patterns using spline interpolation.

    Parameters:
    - past_draws: List of past lottery number draws.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.

    Returns:
    - A list of predicted numbers based on spline analysis.
    """
    spline_scores = defaultdict(float)

    # Convert draws to positional time series
    position_series = defaultdict(list)
    for draw in past_draws:
        sorted_nums = sorted(draw)
        for pos, num in enumerate(sorted_nums):
            position_series[pos].append(num)

    # Analyze each position with splines
    for pos, numbers in position_series.items():
        if len(numbers) < 4:  # Need minimum points for cubic spline
            continue

        try:
            # Create time points
            t = np.arange(len(numbers))

            # Fit cubic spline
            cs = CubicSpline(t, numbers)

            # Predict next values
            next_t = len(numbers)
            predicted = cs(next_t)

            if min_num <= predicted <= max_num:
                spline_scores[int(predicted)] += 1

            # Analyze spline derivatives
            derivatives = cs.derivative()(t)
            trend = np.mean(derivatives[-3:])  # Use recent trend

            projected = numbers[-1] + trend
            if min_num <= projected <= max_num:
                spline_scores[int(projected)] += 0.8

            # Look for spline inflection points
            second_deriv = cs.derivative(2)(t)
            inflections = np.where(np.diff(np.signbit(second_deriv)))[0]

            for infl in inflections[-2:]:  # Focus on recent inflections
                infl_value = cs(infl)
                if min_num <= infl_value <= max_num:
                    spline_scores[int(infl_value)] += 0.5

        except Exception as e:
            print(f"Spline analysis error at position {pos}: {e}")
            continue

    return sorted(spline_scores.keys(), key=spline_scores.get, reverse=True)


def analyze_momentum_evolution(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """
    Analyze momentum patterns and their evolution.

    Parameters:
    - past_draws: List of past lottery number draws.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.

    Returns:
    - A list of predicted numbers based on momentum evolution analysis.
    """
    momentum_scores = defaultdict(float)

    # Calculate momentum at different time scales
    time_scales = [3, 5, 8, 13]  # Fibonacci-like scales

    for scale in time_scales:
        if len(past_draws) < scale:
            continue

        # Calculate momentum vectors
        momentum_vectors = []
        for i in range(len(past_draws) - scale):
            start_nums = set(past_draws[i])
            end_nums = set(past_draws[i + scale])

            # Calculate various momentum metrics
            mean_shift = statistics.mean(list(end_nums)) - statistics.mean(list(start_nums))
            common_numbers = start_nums & end_nums
            momentum_strength = len(common_numbers) / len(start_nums) if start_nums else 0

            momentum_vectors.append({
                'shift': mean_shift,
                'strength': momentum_strength,
                'scale': scale
            })

        if momentum_vectors:
            # Analyze momentum evolution
            recent_momentum = momentum_vectors[-1]

            # Project based on momentum
            last_numbers = past_draws[0]
            for base_num in last_numbers:
                projected = base_num + int(recent_momentum['shift'] * recent_momentum['strength'])
                if min_num <= projected <= max_num:
                    momentum_scores[projected] += 1 / scale

            # Look for momentum patterns
            if len(momentum_vectors) >= 3:
                momentum_trend = np.mean([v['shift'] for v in momentum_vectors[-3:]])
                strength_trend = np.mean([v['strength'] for v in momentum_vectors[-3:]])

                for base_num in last_numbers:
                    projected = base_num + int(momentum_trend * strength_trend)
                    if min_num <= projected <= max_num:
                        momentum_scores[projected] += 0.5 / scale

    return sorted(momentum_scores.keys(), key=momentum_scores.get, reverse=True)


def analyze_wave_resonance(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """
    Analyze wave resonance patterns in the number sequence.

    Parameters:
    - past_draws: List of past lottery number draws.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.

    Returns:
    - A list of predicted numbers based on wave resonance analysis.
    """
    resonance_scores = defaultdict(float)

    # Convert number sequence to wave form
    waves = []
    for draw in past_draws:
        sorted_nums = sorted(draw)
        # Create wave representation
        wave = np.array(sorted_nums) / max_num  # Normalize to [0,1]
        waves.append(wave)

    if not waves:
        return []

    try:
        # Stack waves for analysis
        wave_matrix = np.vstack(waves)

        # Find dominant frequencies
        frequencies = np.fft.fft2(wave_matrix)
        dominant_freqs = np.sort(np.abs(frequencies).flatten())[-5:]  # Top 5 frequencies

        # Analyze resonance patterns
        for freq in dominant_freqs:
            # Project resonant numbers
            for base in past_draws[0]:
                phase = 2 * np.pi * base / max_num
                resonant = int(base + freq * np.sin(phase))

                if min_num <= resonant <= max_num:
                    resonance_scores[resonant] += 1

        # Analyze wave interference
        if len(waves) >= 2:
            interference = waves[0] + waves[1]  # Recent wave interference
            peaks, _ = signal.find_peaks(interference)

            for peak_idx in peaks:
                if peak_idx < len(past_draws[0]):
                    base = past_draws[0][peak_idx]
                    projected = int(base * (1 + interference[peak_idx]))

                    if min_num <= projected <= max_num:
                        resonance_scores[projected] += 0.8

        # Look for standing wave patterns
        if len(waves) >= 3:
            wave_diff = np.diff(wave_matrix, axis=0)
            stable_points = np.where(np.abs(wave_diff).mean(axis=0) < 0.1)[0]

            for point in stable_points:
                if point < len(past_draws[0]):
                    stable_value = int(past_draws[0][point] * (1 + np.random.normal(0, 0.1)))
                    if min_num <= stable_value <= max_num:
                        resonance_scores[stable_value] += 0.5

    except Exception as e:
        print(f"Wave analysis error: {e}")

    return sorted(resonance_scores.keys(), key=resonance_scores.get, reverse=True)


def calculate_field_weights(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        excluded_numbers: Set[int]
) -> Dict[int, float]:
    """
    Calculate weights using field theory principles.

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
        # Create potential field
        field = np.zeros(max_num - min_num + 1)

        # Add contributions from each number
        for draw in past_draws:
            for num in draw:
                # Gaussian field contribution
                idx = num - min_num
                width = (max_num - min_num) / 20  # Field width
                x = np.arange(len(field))
                field += np.exp(-(x - idx) ** 2 / (2 * width ** 2))

        # Normalize field
        if np.max(field) != 0:
            field = field / np.max(field)
        else:
            field = np.ones_like(field) / len(field)  # Avoid division by zero

        # Calculate weights from field
        for num in range(min_num, max_num + 1):
            if num in excluded_numbers:
                continue

            idx = num - min_num

            # Base field weight
            field_weight = field[idx]

            # Gradient weight (prefer stable points)
            gradient = np.gradient(field)
            gradient_weight = 1 / (1 + abs(gradient[idx]))

            # Local maxima weight
            peaks, _ = signal.find_peaks(field)
            if len(peaks) == 0:
                peak_weight = 1.0
            else:
                peak_distance = min(abs(idx - peak) for peak in peaks)
                peak_weight = 1 / (1 + peak_distance)

            # Combine weights
            weights[num] = field_weight * gradient_weight * peak_weight

    except Exception as e:
        print(f"Field calculation error: {e}")
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


def generate_random_numbers(lottery_type_instance: lg_lottery_type, min_num: int, max_num: int, total_numbers: int) -> List[int]:
    """
    Generate random numbers using field theory as a fallback mechanism.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Total numbers to generate.

    Returns:
    - A sorted list of randomly generated lottery numbers.
    """
    numbers = set()
    required_numbers = total_numbers

    # Create potential field
    field = np.ones(max_num - min_num + 1)
    field = field / np.sum(field)  # Normalize

    while len(numbers) < required_numbers:
        # Sample from field distribution
        idx = np.random.choice(len(field), p=field)
        num = idx + min_num

        if num not in numbers:
            numbers.add(num)

            # Update field (reduce probability nearby)
            x = np.arange(len(field))
            width = len(field) / 20
            field *= (1 - 0.5 * np.exp(-(x - idx) ** 2 / (2 * width ** 2)))
            if np.sum(field) > 0:
                field = field / np.sum(field)  # Renormalize
            else:
                # Reset field if all probabilities are zero
                field = np.ones(len(field)) / len(field)

    return sorted(list(numbers))
