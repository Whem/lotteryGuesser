# multidimensional_timeseries_harmonic_predictor.py
import numpy as np
from scipy.fftpack import fft
from statsmodels.tsa.seasonal import seasonal_decompose
from collections import Counter
from typing import List, Tuple, Optional, Dict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Generate lottery numbers using harmonic analysis."""
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
    """Generate numbers using multidimensional time series analysis."""
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if len(past_draws) < 20:
        return get_most_common_numbers(past_draws, required_numbers)

    return perform_prediction(past_draws, min_num, max_num, required_numbers)


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


def create_time_series(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int
) -> np.ndarray:
    """Create multidimensional time series."""
    time_series = np.zeros((len(past_draws), max_num - min_num + 1))
    for i, draw in enumerate(past_draws):
        for num in draw:
            time_series[i, num - min_num] = 1
    return time_series


def safe_decompose(series: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform safe time series decomposition."""
    try:
        period = min(10, len(series) - 1)
        result = seasonal_decompose(series, model='additive', period=period)
        return result.trend, result.seasonal, result.resid
    except Exception as e:
        print(f"Decomposition error: {str(e)}")
        return series, np.zeros_like(series), np.zeros_like(series)


def harmonic_analysis(series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform harmonic analysis using FFT."""
    fft_result = fft(series)
    frequencies = np.fft.fftfreq(len(series))
    magnitudes = np.abs(fft_result)
    return frequencies, magnitudes


def find_dominant_cycles(
        frequencies: np.ndarray,
        magnitudes: np.ndarray,
        top_n: int = 5
) -> List[float]:
    """Find dominant cycles in series."""
    sorted_indices = np.argsort(magnitudes)[::-1]
    top_frequencies = frequencies[sorted_indices[:top_n]]
    return [
        1 / freq if freq != 0 else len(frequencies)
        for freq in top_frequencies
    ]


def safe_extrapolate_trend(trend: np.ndarray, steps: int) -> float:
    """Safely extrapolate trend."""
    try:
        x = np.arange(len(trend))
        coeffs = np.polyfit(x, trend, deg=min(2, len(trend) - 1))
        poly = np.poly1d(coeffs)
        return poly(len(trend) + steps)
    except Exception as e:
        print(f"Extrapolation error: {str(e)}")
        return trend[-1]


def perform_prediction(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Perform prediction using time series analysis."""
    time_series = create_time_series(past_draws, min_num, max_num)
    predicted_probabilities = calculate_probabilities(
        time_series,
        min_num,
        max_num
    )

    return select_numbers(
        predicted_probabilities,
        min_num,
        required_numbers
    )


def calculate_probabilities(
        time_series: np.ndarray,
        min_num: int,
        max_num: int
) -> np.ndarray:
    """Calculate prediction probabilities."""
    probabilities = np.zeros(max_num - min_num + 1)

    for i in range(time_series.shape[1]):
        series = time_series[:, i]
        trend, seasonal, _ = safe_decompose(series)
        frequencies, magnitudes = harmonic_analysis(series)
        dominant_cycles = find_dominant_cycles(frequencies, magnitudes)

        # Calculate components
        future_trend = safe_extrapolate_trend(trend, 1)
        next_seasonal = seasonal[-1] if len(seasonal) > 0 else 0
        cycle_component = np.mean([
            series[int(min(-cycle, -1)) % len(series)]
            for cycle in dominant_cycles if cycle != 0
        ])

        predicted_value = future_trend + next_seasonal + cycle_component
        probabilities[i] = max(0, predicted_value)

    # Normalize probabilities
    sum_prob = np.sum(probabilities)
    if sum_prob > 0:
        probabilities /= sum_prob
    else:
        probabilities = np.ones_like(probabilities) / len(probabilities)

    return probabilities


def select_numbers(
        probabilities: np.ndarray,
        min_num: int,
        required_numbers: int
) -> List[int]:
    """Select numbers based on probabilities."""
    top_indices = np.argsort(probabilities)[::-1][:required_numbers]
    predicted_numbers = [int(index) + min_num for index in top_indices]
    return sorted(predicted_numbers)


def get_most_common_numbers(
        past_draws: List[List[int]],
        required_numbers: int
) -> List[int]:
    """Get most common numbers from past draws."""
    all_numbers = [num for draw in past_draws for num in draw]
    most_common = [num for num, _ in Counter(all_numbers).most_common(required_numbers)]
    return sorted(most_common)