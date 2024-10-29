# spline_momentum_evolution_prediction.py
# Advanced spline interpolation with adaptive momentum and wave mechanics
# Combines spline analysis with evolutionary momentum patterns

import numpy as np
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import statistics
from scipy.interpolate import CubicSpline
from scipy import signal
from sklearn.preprocessing import StandardScaler
import random
from itertools import combinations


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    """
    Spline interpolation based prediction with adaptive momentum
    and wave mechanics analysis.
    """
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True).order_by('-id')[:200])

    past_draws = [draw for draw in past_draws if isinstance(draw, list)]
    if not past_draws:
        return generate_random_numbers(lottery_type_instance)

    required_numbers = lottery_type_instance.pieces_of_draw_numbers
    candidates = set()

    # 1. Spline Interpolation Analysis
    spline_numbers = analyze_spline_patterns(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    candidates.update(spline_numbers[:required_numbers // 3])

    # 2. Momentum Evolution Analysis
    momentum_numbers = analyze_momentum_evolution(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    candidates.update(momentum_numbers[:required_numbers // 3])

    # 3. Wave Resonance Analysis
    resonance_numbers = analyze_wave_resonance(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    candidates.update(resonance_numbers[:required_numbers // 3])

    # Fill remaining slots using adaptive field theory
    while len(candidates) < required_numbers:
        weights = calculate_field_weights(
            past_draws,
            lottery_type_instance.min_number,
            lottery_type_instance.max_number,
            candidates
        )

        available_numbers = set(range(
            lottery_type_instance.min_number,
            lottery_type_instance.max_number + 1
        )) - candidates

        if available_numbers:
            number_weights = [weights.get(num, 1.0) for num in available_numbers]
            selected = random.choices(list(available_numbers), weights=number_weights, k=1)[0]
            candidates.add(selected)

    return sorted(list(candidates))[:required_numbers]


def analyze_spline_patterns(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze patterns using spline interpolation."""
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
            print(f"Spline analysis error: {e}")
            continue

    return sorted(spline_scores.keys(), key=spline_scores.get, reverse=True)


def analyze_momentum_evolution(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze momentum patterns and their evolution."""
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
    """Analyze wave resonance patterns in the number sequence."""
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
    """Calculate weights using field theory principles."""
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
        field = field / np.max(field)

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
            peak_distance = min(abs(idx - peak) for peak in peaks) if peaks.size > 0 else len(field)
            peak_weight = 1 / (1 + peak_distance)

            # Combine weights
            weights[num] = field_weight * gradient_weight * peak_weight

    except Exception as e:
        print(f"Field calculation error: {e}")
        return {num: 1.0 for num in range(min_num, max_num + 1) if num not in excluded_numbers}

    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        for num in weights:
            weights[num] /= total_weight

    return weights


def generate_random_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    """Generate random numbers using field theory."""
    numbers = set()
    required_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Create potential field
    field = np.ones(lottery_type_instance.max_number - lottery_type_instance.min_number + 1)
    field = field / np.sum(field)  # Normalize

    while len(numbers) < required_numbers:
        # Sample from field distribution
        idx = np.random.choice(len(field), p=field)
        num = idx + lottery_type_instance.min_number

        if num not in numbers:
            numbers.add(num)

            # Update field (reduce probability nearby)
            x = np.arange(len(field))
            width = len(field) / 20
            field *= (1 - 0.5 * np.exp(-(x - idx) ** 2 / (2 * width ** 2)))
            field = field / np.sum(field)  # Renormalize

    return sorted(list(numbers))