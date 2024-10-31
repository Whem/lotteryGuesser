# adaptive_choreography_balance_prediction.py
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import statistics
import random
from itertools import combinations
from sklearn.cluster import KMeans


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Adaptive choreography algorithm that handles both simple and combined lottery types.
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
    """Generate a set of numbers using adaptive choreography algorithm."""
    # Get past draws
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:200])

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

    candidates = set()

    # 1. Movement Pattern Analysis
    movement_numbers = analyze_movement_patterns(
        past_numbers,
        min_num,
        max_num
    )
    candidates.update(movement_numbers[:required_numbers // 3])

    # 2. Balance Point Analysis
    balance_numbers = analyze_balance_points(
        past_numbers,
        min_num,
        max_num
    )
    candidates.update(balance_numbers[:required_numbers // 3])

    # 3. Choreographic Sequence Analysis
    sequence_numbers = analyze_choreographic_sequences(
        past_numbers,
        min_num,
        max_num
    )
    candidates.update(sequence_numbers[:required_numbers // 3])

    # Fill remaining slots using dynamic equilibrium
    while len(candidates) < required_numbers:
        weights = calculate_equilibrium_weights(
            past_numbers,
            min_num,
            max_num,
            candidates
        )

        available_numbers = set(range(min_num, max_num + 1)) - candidates
        if available_numbers:
            number_weights = [weights.get(num, 1.0) for num in available_numbers]
            selected = random.choices(list(available_numbers),
                                      weights=number_weights, k=1)[0]
            candidates.add(selected)

    return sorted(list(candidates))[:required_numbers]


def analyze_movement_patterns(past_draws: List[List[int]],
                              min_num: int,
                              max_num: int) -> List[int]:
    """Analyze movement patterns in number sequences."""
    movement_scores = defaultdict(float)

    if not past_draws or len(past_draws) < 2:
        return []

    # Create movement vectors
    vectors = []
    for i in range(len(past_draws) - 1):
        current = sorted(past_draws[i])
        next_draw = sorted(past_draws[i + 1])

        movements = []
        for j in range(min(len(current), len(next_draw))):
            movement = next_draw[j] - current[j]
            movements.append(movement)
        vectors.append(movements)

    try:
        # Standardize vector lengths
        max_len = max(len(v) for v in vectors)
        padded_vectors = [v + [0] * (max_len - len(v)) for v in vectors]

        # Cluster movements
        n_clusters = min(5, len(padded_vectors))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, n_init=10)
            clusters = kmeans.fit_predict(padded_vectors)

            # Score based on cluster centers
            for center in kmeans.cluster_centers_:
                last_draw = sorted(past_draws[-1])
                for movement in center:
                    for base in last_draw:
                        predicted = int(base + movement)
                        if min_num <= predicted <= max_num:
                            movement_scores[predicted] += 1

        # Add momentum analysis
        for i, vector in enumerate(vectors):
            if vector:
                momentum = sum(vector) / len(vector)
                weight = 1 / (i + 1)  # Recent movements weighted higher

                for base in sorted(past_draws[-1]):
                    predicted = int(base + momentum)
                    if min_num <= predicted <= max_num:
                        movement_scores[predicted] += weight

    except Exception as e:
        print(f"Movement analysis error: {e}")
        return []

    return sorted(movement_scores.keys(), key=movement_scores.get, reverse=True)


def generate_random_numbers(min_num: int,
                            max_num: int,
                            required_numbers: int) -> List[int]:
    """Generate random numbers using balanced distribution."""
    numbers = set()
    mean = (min_num + max_num) / 2
    std = (max_num - min_num) / 6

    while len(numbers) < required_numbers:
        num = int(random.gauss(mean, std))
        num = max(min_num, min(max_num, num))
        numbers.add(num)

    return sorted(list(numbers))


def analyze_balance_points(past_draws: List[List[int]],
                           min_num: int,
                           max_num: int) -> List[int]:
    """Find balanced number positions based on past draws."""
    if not past_draws:
        return []

    balance_scores = defaultdict(float)

    for draw in past_draws:
        if not draw:
            continue

        # Calculate center points
        mean = sum(draw) / len(draw)
        balance_scores[int(mean)] += 1

        # Find median points
        sorted_draw = sorted(draw)
        if len(sorted_draw) % 2 == 0:
            median = (sorted_draw[len(sorted_draw) // 2 - 1] +
                      sorted_draw[len(sorted_draw) // 2]) / 2
            balance_scores[int(median)] += 1
        else:
            median = sorted_draw[len(sorted_draw) // 2]
            balance_scores[median] += 1

        # Check pair midpoints
        for a, b in combinations(draw, 2):
            midpoint = (a + b) // 2
            if min_num <= midpoint <= max_num:
                balance_scores[midpoint] += 0.5

    return sorted(balance_scores.keys(), key=balance_scores.get, reverse=True)


def analyze_choreographic_sequences(past_draws: List[List[int]],
                                    min_num: int,
                                    max_num: int) -> List[int]:
    """Analyze number patterns and sequences."""
    if not past_draws or len(past_draws) < 2:
        return []

    sequence_scores = defaultdict(float)

    # Analyze position-based patterns
    for position in range(min(len(d) for d in past_draws)):
        numbers_in_position = [draw[position] for draw in past_draws if position < len(draw)]
        if numbers_in_position:
            avg_number = sum(numbers_in_position) / len(numbers_in_position)
            trend = (numbers_in_position[-1] - numbers_in_position[0]) / len(numbers_in_position)

            predicted = int(avg_number + trend)
            if min_num <= predicted <= max_num:
                sequence_scores[predicted] += 2

    # Analyze number gaps
    for draw in past_draws:
        sorted_draw = sorted(draw)
        for i in range(len(sorted_draw) - 1):
            gap = sorted_draw[i + 1] - sorted_draw[i]
            next_number = sorted_draw[-1] + gap
            if min_num <= next_number <= max_num:
                sequence_scores[next_number] += 1

    return sorted(sequence_scores.keys(), key=sequence_scores.get, reverse=True)


def calculate_equilibrium_weights(past_draws: List[List[int]],
                                  min_num: int,
                                  max_num: int,
                                  excluded_numbers: Set[int]) -> Dict[int, float]:
    """Calculate probability weights for remaining numbers."""
    weights = defaultdict(float)

    if not past_draws or not past_draws[0]:
        return weights

    # Calculate base statistics
    all_numbers = [num for draw in past_draws for num in draw]
    mean = statistics.mean(all_numbers)
    std_dev = statistics.stdev(all_numbers) if len(all_numbers) > 1 else 1

    for num in range(min_num, max_num + 1):
        if num in excluded_numbers:
            continue

        # Distance from mean
        z_score = abs((num - mean) / std_dev)
        weight = np.exp(-z_score / 2)

        # Frequency adjustment
        freq = all_numbers.count(num)
        freq_factor = 1 / (1 + freq)

        weights[num] = weight * freq_factor

    # Normalize weights
    total = sum(weights.values())
    if total > 0:
        for num in weights:
            weights[num] /= total

    return weights