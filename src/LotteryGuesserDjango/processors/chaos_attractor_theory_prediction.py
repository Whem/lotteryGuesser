# chaos_attractor_theory_prediction.py
# Combines chaos theory with attractor point analysis and basin stability
# Uses Lyapunov exponents and attractor dynamics for prediction

import numpy as np
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import statistics
from scipy.stats import entropy
from sklearn.cluster import DBSCAN
import random
from itertools import combinations


def convert_to_native_types(value):
    """Convert numpy types to native Python types."""
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    elif isinstance(value, (np.float32, np.float64)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return [convert_to_native_types(x) for x in value]
    return value


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Chaos theory-based prediction using attractor point analysis and basin stability measurements.
    Returns a tuple of (main_numbers, additional_numbers).
    """
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True).order_by('-id')[:200])

    past_draws = [draw for draw in past_draws if isinstance(draw, list)]
    if not past_draws:
        return generate_random_numbers(lottery_type_instance), generate_random_additional_numbers(lottery_type_instance)

    required_numbers = lottery_type_instance.pieces_of_draw_numbers
    main_candidates = set()

    # 1. Attractor Point Analysis for Main Numbers
    attractor_numbers = analyze_attractor_points(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    main_candidates.update(attractor_numbers[:required_numbers // 3])

    # 2. Chaos Basin Analysis for Main Numbers
    basin_numbers = analyze_chaos_basins(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    main_candidates.update(basin_numbers[:required_numbers // 3])

    # 3. Lyapunov Stability Analysis for Main Numbers
    stability_numbers = analyze_lyapunov_stability(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    main_candidates.update(stability_numbers[:required_numbers // 3])

    # Fill remaining slots for main numbers
    main_numbers = fill_remaining_slots(main_candidates, past_draws, required_numbers, lottery_type_instance.min_number, lottery_type_instance.max_number)

    # Generate additional numbers if required
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_min = lottery_type_instance.additional_min_number
        additional_max = lottery_type_instance.additional_max_number
        additional_required = lottery_type_instance.additional_numbers_count

        # Chaos Theory predictions for additional numbers
        additional_attractor_numbers = analyze_attractor_points(past_draws, additional_min, additional_max)
        additional_basin_numbers = analyze_chaos_basins(past_draws, additional_min, additional_max)
        additional_stability_numbers = analyze_lyapunov_stability(past_draws, additional_min, additional_max)

        additional_candidates = set()
        additional_candidates.update(additional_attractor_numbers[:additional_required // 3])
        additional_candidates.update(additional_basin_numbers[:additional_required // 3])
        additional_candidates.update(additional_stability_numbers[:additional_required // 3])

        # Fill remaining slots for additional numbers
        additional_numbers = fill_remaining_slots(additional_candidates, past_draws, additional_required, additional_min, additional_max)

    return sorted(main_numbers), sorted(additional_numbers)


def fill_remaining_slots(candidates: set, past_draws: List[List[int]], required_numbers: int, min_num: int, max_num: int) -> List[int]:
    """
    Fill the remaining slots with the highest probability numbers based on dynamic basin probabilities.
    """
    weights = calculate_basin_weights(past_draws, min_num, max_num, candidates)
    available_numbers = set(range(min_num, max_num + 1)) - candidates

    while len(candidates) < required_numbers:
        number_weights = [weights.get(num, 1.0) for num in available_numbers]
        selected = random.choices(list(available_numbers), weights=number_weights, k=1)[0]
        candidates.add(selected)
        available_numbers.remove(selected)

    return list(candidates)[:required_numbers]


def generate_random_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    """Generate random numbers for main set based on attractor principles."""
    return generate_random_set(lottery_type_instance.min_number, lottery_type_instance.max_number, lottery_type_instance.pieces_of_draw_numbers)


def generate_random_additional_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    """Generate random numbers for additional set based on attractor principles."""
    return generate_random_set(lottery_type_instance.additional_min_number, lottery_type_instance.additional_max_number, lottery_type_instance.additional_numbers_count)


def generate_random_set(min_num: int, max_num: int, count: int) -> List[int]:
    """Generate random numbers using chaos theory principles within a specified range and count."""
    numbers = set()
    attractors = [min_num, (min_num + max_num) // 2, max_num]
    while len(numbers) < count:
        attractor = random.choice(attractors)
        noise = random.gauss(0, (max_num - min_num) / 10)
        num = int(attractor + noise)
        num = max(min_num, min(max_num, num))
        numbers.add(num)
    return sorted(numbers)


def analyze_attractor_points(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze attractor points in the system dynamics."""
    attractor_scores = defaultdict(float)

    # Convert sequence to phase space
    phase_points = []
    for i in range(len(past_draws) - 1):
        current = np.array(sorted(past_draws[i]))
        next_draw = np.array(sorted(past_draws[i + 1]))
        if len(current) == len(next_draw):
            phase_points.append(np.concatenate([current, next_draw]))

    if not phase_points:
        return []

    try:
        # Cluster phase space points to find attractors
        clustering = DBSCAN(eps=3, min_samples=2).fit(phase_points)
        labels = clustering.labels_

        # Analyze each cluster
        for label in set(labels):
            if label == -1:  # Skip noise points
                continue

            cluster_points = [p for i, p in enumerate(phase_points) if labels[i] == label]

            # Calculate cluster center
            center = np.mean(cluster_points, axis=0)

            # Calculate cluster stability
            distances = [np.linalg.norm(p - center) for p in cluster_points]
            stability = 1 / (1 + np.std(distances))

            # Project attractor influence
            dimension = len(center) // 2
            for i in range(dimension):
                # Current point influence
                current_pos = center[i]
                if min_num <= current_pos <= max_num:
                    attractor_scores[int(current_pos)] += stability

                # Next point influence
                next_pos = center[i + dimension]
                if min_num <= next_pos <= max_num:
                    attractor_scores[int(next_pos)] += stability * 1.5  # Higher weight for prediction

        # Analyze periodic attractors
        for period in range(2, 6):
            if len(past_draws) >= period:
                for i in range(len(past_draws) - period):
                    sequence = past_draws[i:i + period]
                    if all(len(draw) == len(sequence[0]) for draw in sequence):
                        similarity = calculate_sequence_similarity(sequence)
                        if similarity > 0.7:  # High similarity threshold
                            next_predicted = predict_next_in_sequence(sequence)
                            for num in next_predicted:
                                if min_num <= num <= max_num:
                                    attractor_scores[num] += similarity

    except Exception as e:
        print(f"Attractor analysis error: {e}")

    return [convert_to_native_types(num) for num in
            sorted(attractor_scores.keys(), key=attractor_scores.get, reverse=True)]


def analyze_chaos_basins(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze chaos basins and their stability."""
    basin_scores = defaultdict(float)

    # Create phase space trajectories
    trajectories = []
    for i in range(len(past_draws) - 2):
        traj = [sorted(draw) for draw in past_draws[i:i + 3]]
        if all(len(t) == len(traj[0]) for t in traj):
            trajectories.append(np.array(traj))

    if not trajectories:
        return []

    try:
        # Calculate basin entropy for each number
        for num in range(min_num, max_num + 1):
            # Find trajectories containing this number
            containing_trajectories = [t for t in trajectories if num in t[0]]
            if not containing_trajectories:
                continue

            # Calculate trajectory stability
            stabilities = []
            for traj in containing_trajectories:
                # Calculate Lyapunov-like stability metric
                diffs = np.diff(traj, axis=0)
                stability = 1 / (1 + np.mean(np.abs(diffs)))
                stabilities.append(stability)

            # Score based on stability and frequency
            basin_scores[num] = np.mean(stabilities) * len(containing_trajectories)

            # Analyze basin transitions
            for traj in containing_trajectories:
                for i in range(len(traj) - 1):
                    if num in traj[i]:
                        pos = list(traj[i]).index(num)
                        if pos < len(traj[i + 1]):
                            next_num = traj[i + 1][pos]
                            basin_scores[next_num] += 0.5

        # Analyze basin interconnections
        for num1, num2 in combinations(range(min_num, max_num + 1), 2):
            connections = sum(1 for t in trajectories if num1 in t[0] and num2 in t[-1])
            if connections > 0:
                connection_strength = connections / len(trajectories)
                basin_scores[num1] += connection_strength
                basin_scores[num2] += connection_strength

    except Exception as e:
        print(f"Basin analysis error: {e}")

    return [convert_to_native_types(num) for num in
            sorted(basin_scores.keys(), key=basin_scores.get, reverse=True)]


def analyze_lyapunov_stability(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze system stability using Lyapunov-like exponents."""
    stability_scores = defaultdict(float)

    # Create time series for each position
    position_series = defaultdict(list)
    for draw in past_draws:
        sorted_nums = sorted(draw)
        for pos, num in enumerate(sorted_nums):
            position_series[pos].append(num)

    try:
        # Calculate position-based stability
        for pos, series in position_series.items():
            if len(series) < 3:  # Need minimum points
                continue

            # Calculate local Lyapunov exponents
            for i in range(len(series) - 2):
                delta_initial = abs(series[i + 1] - series[i])
                delta_final = abs(series[i + 2] - series[i + 1])

                if delta_initial > 0:
                    lyap = np.log(delta_final / delta_initial)

                    # Predict based on Lyapunov stability
                    prediction = series[-1] + delta_final * np.exp(lyap)
                    if min_num <= prediction <= max_num:
                        stability_scores[int(prediction)] += 1 / (1 + abs(lyap))

            # Analyze local stability regions
            for window_size in [3, 5, 8]:
                if len(series) >= window_size:
                    for i in range(len(series) - window_size + 1):
                        window = series[i:i + window_size]
                        stability = 1 / (1 + np.std(window))
                        trend = np.polyfit(range(window_size), window, 1)[0]

                        # Project stable trends
                        prediction = int(window[-1] + trend)
                        if min_num <= prediction <= max_num:
                            stability_scores[prediction] += stability

    except Exception as e:
        print(f"Stability analysis error: {e}")

    return [convert_to_native_types(num) for num in
            sorted(stability_scores.keys(), key=stability_scores.get, reverse=True)]


def calculate_basin_weights(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        excluded_numbers: Set[int]
) -> Dict[int, float]:
    """Calculate weights based on basin of attraction theory."""
    weights = defaultdict(float)

    if not past_draws:
        return weights

    try:
        # Calculate basic frequency distribution
        frequency = Counter(num for draw in past_draws for num in draw)

        # Create transition matrix
        transitions = defaultdict(lambda: defaultdict(int))
        for i in range(len(past_draws) - 1):
            current = set(past_draws[i])
            next_draw = set(past_draws[i + 1])

            for num1 in current:
                for num2 in next_draw:
                    transitions[num1][num2] += 1

        for num in range(min_num, max_num + 1):
            if num in excluded_numbers:
                continue

            # Base frequency weight
            weights[num] = frequency.get(num, 0) / len(past_draws)

            # Basin stability
            in_transitions = sum(transitions[other][num] for other in range(min_num, max_num + 1))
            out_transitions = sum(transitions[num].values())
            if out_transitions > 0:
                basin_stability = in_transitions / out_transitions
                weights[num] *= (1 + basin_stability)

            # Local attractor strength
            local_numbers = set(range(max(min_num, num - 5), min(max_num + 1, num + 6)))
            attractor_strength = sum(transitions[num][other] for other in local_numbers)
            weights[num] *= (1 + attractor_strength / (len(local_numbers) + 1))

    except Exception as e:
        print(f"Weight calculation error: {e}")
        return {num: 1.0 for num in range(min_num, max_num + 1) if num not in excluded_numbers}

    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        for num in weights:
            weights[num] /= total_weight

    return {convert_to_native_types(k): convert_to_native_types(v)
            for k, v in weights.items()}


def calculate_sequence_similarity(sequence: List[List[int]]) -> float:
    """Calculate similarity measure for a sequence of draws."""
    if not sequence or not all(len(s) == len(sequence[0]) for s in sequence):
        return 0.0

    similarities = []
    for i in range(len(sequence) - 1):
        current = np.array(sorted(sequence[i]))
        next_draw = np.array(sorted(sequence[i + 1]))
        similarity = 1 / (1 + np.mean(np.abs(next_draw - current)))
        similarities.append(similarity)

    return np.mean(similarities) if similarities else 0.0


def predict_next_in_sequence(sequence: List[List[int]]) -> List[int]:
    """Predict next numbers in a periodic sequence."""
    if not sequence or not all(len(s) == len(sequence[0]) for s in sequence):
        return []

    # Convert to numpy arrays for easier manipulation
    arrays = [np.array(sorted(draw)) for draw in sequence]

    # Calculate differences
    diffs = [arrays[i + 1] - arrays[i] for i in range(len(arrays) - 1)]
    avg_diff = np.mean(diffs, axis=0)

    # Predict next values
    predicted = arrays[-1] + avg_diff

    return [int(x) for x in predicted]


def generate_random_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    """Generate random numbers using attractor principles."""
    numbers = set()
    required_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Create attractor points
    attractors = [
        lottery_type_instance.min_number,
        (lottery_type_instance.min_number + lottery_type_instance.max_number) // 2,
        lottery_type_instance.max_number
    ]

    while len(numbers) < required_numbers:
        attractor = random.choice(attractors)
        noise = random.gauss(0, (lottery_type_instance.max_number - lottery_type_instance.min_number) / 10)
        num = int(attractor + noise)

        num = max(lottery_type_instance.min_number,
                  min(lottery_type_instance.max_number, num))

        if num not in numbers:
            numbers.add(num)

    # Ensure all numbers are native Python integers
    return [int(num) for num in sorted(numbers)]