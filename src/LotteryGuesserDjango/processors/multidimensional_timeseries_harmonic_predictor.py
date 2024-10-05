# multidimensional_timeseries_harmonic_predictor.py

import numpy as np
from scipy import signal
from scipy.fftpack import fft
from statsmodels.tsa.seasonal import seasonal_decompose
from django.apps import apps
from collections import Counter


def create_time_series(past_draws, min_num, max_num):
    """Create a multidimensional time series from past draws."""
    time_series = np.zeros((len(past_draws), max_num - min_num + 1))
    for i, draw in enumerate(past_draws):
        for num in draw:
            time_series[i, num - min_num] = 1
    return time_series


def safe_decompose(series):
    """Perform time series decomposition with error handling."""
    try:
        result = seasonal_decompose(series, model='additive', period=min(10, len(series) - 1))
        return result.trend, result.seasonal, result.resid
    except:
        # If decomposition fails, return the original series as trend
        return series, np.zeros_like(series), np.zeros_like(series)


def harmonic_analysis(series):
    """Perform harmonic analysis using FFT."""
    fft_result = fft(series)
    frequencies = np.fft.fftfreq(len(series))
    magnitudes = np.abs(fft_result)
    return frequencies, magnitudes


def find_dominant_cycles(frequencies, magnitudes):
    """Find dominant cycles in the series."""
    sorted_indices = np.argsort(magnitudes)[::-1]
    top_frequencies = frequencies[sorted_indices[:5]]  # Consider top 5 frequencies
    return [1 / freq if freq != 0 else len(frequencies) for freq in top_frequencies]


def safe_extrapolate_trend(trend, steps):
    """Safely extrapolate the trend using polynomial fitting."""
    x = np.arange(len(trend))
    try:
        coeffs = np.polyfit(x, trend, deg=min(2, len(trend) - 1))
        poly = np.poly1d(coeffs)
        return poly(len(trend) + steps)
    except:
        # If extrapolation fails, return the last known value
        return trend[-1]


def get_numbers(lottery_type_instance):
    lg_lottery_winner_number = apps.get_model('algorithms', 'lg_lottery_winner_number')

    min_num = int(lottery_type_instance.min_number)
    max_num = int(lottery_type_instance.max_number)
    total_numbers = int(lottery_type_instance.pieces_of_draw_numbers)

    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:100].values_list('lottery_type_number', flat=True))

    if len(past_draws) < 20:  # We need more data for this analysis
        # Return the most frequent numbers if not enough data
        all_numbers = [num for draw in past_draws for num in draw]
        return sorted(num for num, _ in Counter(all_numbers).most_common(total_numbers))

    time_series = create_time_series(past_draws, min_num, max_num)

    predicted_probabilities = np.zeros(max_num - min_num + 1)

    for i in range(time_series.shape[1]):
        series = time_series[:, i]

        # Time series decomposition
        trend, seasonal, _ = safe_decompose(series)

        # Harmonic analysis
        frequencies, magnitudes = harmonic_analysis(series)
        dominant_cycles = find_dominant_cycles(frequencies, magnitudes)

        # Extrapolate trend
        future_trend = safe_extrapolate_trend(trend, 1)

        # Predict next value considering trend, seasonality and dominant cycles
        next_seasonal = seasonal[-1] if len(seasonal) > 0 else 0
        cycle_component = np.mean([series[int(min(-cycle, -1)) % len(series)] for cycle in dominant_cycles])

        predicted_value = future_trend + next_seasonal + cycle_component
        predicted_probabilities[i] = max(0, predicted_value)  # Ensure non-negative probability

    # Normalize probabilities
    sum_prob = np.sum(predicted_probabilities)
    if sum_prob > 0:
        predicted_probabilities /= sum_prob
    else:
        # If all probabilities are zero, use uniform distribution
        predicted_probabilities = np.ones_like(predicted_probabilities) / len(predicted_probabilities)

    # Select top numbers based on predicted probabilities
    top_indices = np.argsort(predicted_probabilities)[::-1][:total_numbers]
    predicted_numbers = [int(index) + min_num for index in top_indices]

    return sorted(predicted_numbers)