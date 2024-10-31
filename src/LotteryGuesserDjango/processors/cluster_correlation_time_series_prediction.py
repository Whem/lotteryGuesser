# cluster_correlation_time_series_prediction.py
# Advanced clustering and correlation analysis with time series decomposition
# Combines cluster-based patterns with time series analysis and correlation detection

import numpy as np
from typing import List, Set, Dict, Tuple
from collections import Counter, defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import statistics
from sklearn.cluster import KMeans
from itertools import combinations
import random
from scipy.stats import pearsonr
import math


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Cluster correlation predictor for combined lottery types.
    Returns (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_number_set(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        True
    )

    # Generate additional numbers if needed
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
    """Generate numbers using cluster correlation analysis."""
    # Get historical data
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if not past_draws:
        return generate_random_numbers(lottery_type_instance)

    number_pool = set()

    # 1. Cluster Analysis
    cluster_numbers = perform_cluster_analysis(
        past_draws,
        min_num,
        max_num
    )
    number_pool.update(cluster_numbers[:required_numbers // 3])

    # 2. Time Series Analysis
    time_series_numbers = analyze_time_series(
        past_draws,
        min_num,
        max_num
    )
    number_pool.update(time_series_numbers[:required_numbers // 3])

    # 3. Correlation Analysis
    correlation_numbers = find_number_correlations(past_draws)
    number_pool.update(correlation_numbers[:required_numbers // 3])

    # Fill remaining slots
    while len(number_pool) < required_numbers:
        weights = calculate_adaptive_probabilities(
            past_draws,
            min_num,
            max_num,
            number_pool
        )

        available_numbers = set(range(min_num, max_num + 1)) - number_pool
        if available_numbers:
            number_weights = [weights.get(num, 1.0) for num in available_numbers]
            selected = random.choices(
                list(available_numbers),
                weights=number_weights,
                k=1
            )[0]
            number_pool.add(selected)

    return sorted(list(number_pool))[:required_numbers]


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get historical lottery data based on number type."""
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:150])

    if is_main:
        return [draw.lottery_type_number for draw in past_draws
                if isinstance(draw.lottery_type_number, list)]
    else:
        return [draw.additional_numbers for draw in past_draws
                if hasattr(draw, 'additional_numbers') and
                isinstance(draw.additional_numbers, list)]


def perform_cluster_analysis(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Advanced cluster analysis of historical draws."""
    cluster_scores = defaultdict(float)

    # Transform draws into feature vectors
    features = []
    for draw in past_draws:
        # Create position-based feature vector
        feature_vector = [0] * (max_num - min_num + 1)
        for num in draw:
            feature_vector[num - min_num] = 1
        features.append(feature_vector)

    if not features:
        return []

    # Perform clustering on recent draws
    n_clusters = min(8, len(features))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    recent_features = features[:30]  # Focus on recent patterns

    try:
        cluster_labels = kmeans.fit_predict(recent_features)

        # Analyze cluster centers
        for center in kmeans.cluster_centers_:
            # Find peaks in cluster centers
            peaks = sorted(enumerate(center), key=lambda x: x[1], reverse=True)
            for idx, value in peaks[:5]:  # Top 5 peaks
                actual_number = idx + min_num
                if min_num <= actual_number <= max_num:
                    cluster_scores[actual_number] += value

        # Analyze cluster transitions
        for i in range(len(cluster_labels) - 1):
            current_cluster = cluster_labels[i]
            next_cluster = cluster_labels[i + 1]
            current_draw = past_draws[i]
            next_draw = past_draws[i + 1]

            # Score numbers that appear in cluster transitions
            common_numbers = set(current_draw) & set(next_draw)
            for num in common_numbers:
                cluster_scores[num] += 1 / (abs(current_cluster - next_cluster) + 1)

    except Exception as e:
        print(f"Clustering error: {e}")
        return []

    return sorted(cluster_scores.keys(), key=cluster_scores.get, reverse=True)


def analyze_time_series(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Time series decomposition and analysis."""
    time_series_scores = defaultdict(float)

    # Create time series for each number position
    max_positions = max(len(draw) for draw in past_draws)
    position_series = [[] for _ in range(max_positions)]

    for draw in past_draws:
        for pos, num in enumerate(sorted(draw)):
            if pos < max_positions:
                position_series[pos].append(num)

    # Analyze trends in each position
    for pos, series in enumerate(position_series):
        if not series:
            continue

        # Calculate moving averages
        window_sizes = [3, 5, 7]
        for window in window_sizes:
            if len(series) < window:
                continue

            moving_avg = []
            for i in range(len(series) - window + 1):
                avg = sum(series[i:i + window]) / window
                moving_avg.append(avg)

            if moving_avg:
                # Predict next value using trend
                trend = moving_avg[-1] - moving_avg[0]
                predicted = int(moving_avg[-1] + trend / window)

                if min_num <= predicted <= max_num:
                    time_series_scores[predicted] += 1 / window

        # Analyze seasonality
        if len(series) >= 14:  # Need enough data for seasonal analysis
            for period in [7, 14]:  # Weekly and bi-weekly patterns
                seasonal_diffs = []
                for i in range(len(series) - period):
                    diff = series[i + period] - series[i]
                    seasonal_diffs.append(diff)

                if seasonal_diffs:
                    avg_diff = sum(seasonal_diffs) / len(seasonal_diffs)
                    predicted = int(series[-1] + avg_diff)

                    if min_num <= predicted <= max_num:
                        time_series_scores[predicted] += 2 / period

    return sorted(time_series_scores.keys(), key=time_series_scores.get, reverse=True)


def find_number_correlations(past_draws: List[List[int]]) -> List[int]:
    """Analyze correlations between numbers across draws."""
    correlation_scores = defaultdict(float)

    # Create number presence matrix
    all_numbers = sorted(list(set(num for draw in past_draws for num in draw)))
    number_matrix = []

    for draw in past_draws:
        row = [1 if num in draw else 0 for num in all_numbers]
        number_matrix.append(row)

    if not number_matrix:
        return []

    # Convert to numpy array for efficient computation
    number_matrix = np.array(number_matrix)

    # Calculate correlations between numbers
    try:
        for i in range(len(all_numbers)):
            for j in range(i + 1, len(all_numbers)):
                correlation, _ = pearsonr(number_matrix[:, i], number_matrix[:, j])

                # Score both numbers based on correlation strength
                weight = abs(correlation) * (1 + np.sign(correlation)) / 2  # Favor positive correlations
                correlation_scores[all_numbers[i]] += weight
                correlation_scores[all_numbers[j]] += weight

                # Look for lagged correlations
                for lag in range(1, 4):
                    if len(number_matrix) > lag:
                        lagged_corr, _ = pearsonr(number_matrix[:-lag, i], number_matrix[lag:, j])
                        lag_weight = abs(lagged_corr) * (1 + np.sign(lagged_corr)) / 2 * (1 / (lag + 1))
                        correlation_scores[all_numbers[j]] += lag_weight

    except Exception as e:
        print(f"Correlation error: {e}")
        return []

    return sorted(correlation_scores.keys(), key=correlation_scores.get, reverse=True)


def calculate_adaptive_probabilities(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        excluded_numbers: Set[int]
) -> Dict[int, float]:
    """Calculate adaptive probabilities for number selection."""
    probabilities = defaultdict(float)

    # Basic frequency analysis
    all_numbers = [num for draw in past_draws for num in draw]
    frequency = Counter(all_numbers)

    # Statistical measures
    mean = statistics.mean(all_numbers)
    std_dev = statistics.stdev(all_numbers)

    for num in range(min_num, max_num + 1):
        if num in excluded_numbers:
            continue

        # Base probability from frequency
        base_prob = frequency.get(num, 0) / len(past_draws)

        # Distance from mean adjustment
        z_score = abs((num - mean) / std_dev)
        distance_factor = math.exp(-z_score / 2)

        # Recent appearance factor
        last_seen = float('inf')
        for i, draw in enumerate(past_draws):
            if num in draw:
                last_seen = i
                break
        recency_factor = math.exp(-last_seen / 20) if last_seen != float('inf') else 0.1

        # Position analysis
        position_counts = Counter()
        for draw in past_draws[:20]:  # Focus on recent draws
            sorted_draw = sorted(draw)
            if num in sorted_draw:
                position_counts[sorted_draw.index(num)] += 1

        position_factor = 1.0
        if position_counts:
            most_common_pos = position_counts.most_common(1)[0][0]
            position_factor = 1.2 if most_common_pos in [0, len(past_draws[0]) - 1] else 1.0

        # Combine all factors
        probabilities[num] = base_prob * distance_factor * recency_factor * position_factor

    return probabilities


def generate_random_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    """Generate random numbers when no historical data is available."""
    numbers = set()
    while len(numbers) < lottery_type_instance.pieces_of_draw_numbers:
        numbers.add(random.randint(
            lottery_type_instance.min_number,
            lottery_type_instance.max_number
        ))
    return sorted(list(numbers))