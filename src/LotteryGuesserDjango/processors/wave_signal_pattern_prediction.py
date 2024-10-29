# wave_signal_pattern_prediction.py
# Advanced signal processing and wave analysis for lottery prediction
# Combines digital signal processing, wavelet analysis and frequency domain patterns

import numpy as np
from typing import List, Set, Dict, Tuple
from collections import Counter, defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import statistics
from scipy import signal
from scipy.fft import fft, ifft
import random
from scipy.signal import find_peaks
import math


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    """
    Wave and signal based pattern prediction algorithm.
    Utilizes signal processing techniques to identify patterns in number sequences.
    """
    # Get historical data for analysis
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True).order_by('-id')[:200])

    past_draws = [draw for draw in past_draws if isinstance(draw, list)]

    if not past_draws:
        return generate_random_numbers(lottery_type_instance)

    required_numbers = lottery_type_instance.pieces_of_draw_numbers
    candidates = set()

    # 1. Wave Pattern Analysis
    wave_numbers = analyze_wave_patterns(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    candidates.update(wave_numbers[:required_numbers // 3])

    # 2. Frequency Domain Analysis
    frequency_numbers = analyze_frequency_domain(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    candidates.update(frequency_numbers[:required_numbers // 3])

    # 3. Signal Peak Detection
    peak_numbers = detect_signal_peaks(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    candidates.update(peak_numbers[:required_numbers // 3])

    # Fill remaining slots using wave-based probability
    while len(candidates) < required_numbers:
        weights = calculate_wave_probabilities(
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


def analyze_wave_patterns(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze wave patterns in the number sequence."""
    wave_scores = defaultdict(float)

    # Convert draws to continuous signals
    signals = []
    for i in range(len(past_draws[0])):  # For each position
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
            # Apply window function
            windowed_signal = np.array(signal_data) * window

            # Analyze wave characteristics
            try:
                # Find local maxima and minima
                peaks, _ = find_peaks(windowed_signal)
                valleys, _ = find_peaks(-windowed_signal)

                if len(peaks) >= 2 and len(valleys) >= 2:
                    # Calculate wave periods
                    peak_period = np.mean(np.diff(peaks))
                    valley_period = np.mean(np.diff(valleys))

                    # Predict next peak and valley
                    next_peak = signal_data[peaks[-1]] + peak_period
                    next_valley = signal_data[valleys[-1]] + valley_period

                    if min_num <= next_peak <= max_num:
                        wave_scores[int(next_peak)] += 1.0
                    if min_num <= next_valley <= max_num:
                        wave_scores[int(next_valley)] += 0.8

                # Analyze wave envelope
                envelope = np.abs(signal.hilbert(windowed_signal))
                if len(envelope) > 0:
                    trend = np.polyfit(np.arange(len(envelope)), envelope, 2)
                    next_value = np.polyval(trend, len(envelope))
                    if min_num <= next_value <= max_num:
                        wave_scores[int(next_value)] += 1.2

            except Exception as e:
                print(f"Wave analysis error: {e}")
                continue

    return sorted(wave_scores.keys(), key=wave_scores.get, reverse=True)


def analyze_frequency_domain(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze frequency domain characteristics of the number sequence."""
    frequency_scores = defaultdict(float)

    # Convert draws to frequency domain
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

            # Find dominant frequencies
            dominant_freq_idx = np.argsort(frequencies)[-3:]  # Top 3 frequencies

            # Reconstruct signal and predict next values
            for idx in dominant_freq_idx:
                phase = np.angle(fft_result[idx])
                amplitude = frequencies[idx]

                # Project next value based on frequency components
                t = len(position_data)
                predicted = amplitude * np.cos(2 * np.pi * idx * t / len(frequencies) + phase)

                if min_num <= predicted <= max_num:
                    frequency_scores[int(predicted)] += amplitude / sum(frequencies)

            # Inverse FFT for time domain prediction
            filtered_fft = np.zeros_like(fft_result)
            filtered_fft[dominant_freq_idx] = fft_result[dominant_freq_idx]
            reconstructed = ifft(filtered_fft)

            # Use last reconstructed value as prediction
            if min_num <= abs(reconstructed[-1]) <= max_num:
                frequency_scores[int(abs(reconstructed[-1]))] += 1.0

        except Exception as e:
            print(f"Frequency analysis error: {e}")
            continue

    return sorted(frequency_scores.keys(), key=frequency_scores.get, reverse=True)


def detect_signal_peaks(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Detect significant peaks and patterns in the signal."""
    peak_scores = defaultdict(float)

    # Convert draws to multiple signal representations
    for window_size in [5, 10, 20]:
        for position in range(max(len(draw) for draw in past_draws)):
            signal_data = []
            for draw in past_draws:
                if position < len(draw):
                    signal_data.append(draw[position])

            if len(signal_data) < window_size:
                continue

            try:
                # Apply smoothing
                smoothed = signal.savgol_filter(signal_data, window_size, 3)

                # Find peaks with different properties
                peaks, properties = find_peaks(
                    smoothed,
                    height=(None, None),
                    threshold=(None, None),
                    distance=2,
                    prominence=1
                )

                if len(peaks) >= 2:
                    # Analyze peak characteristics
                    peak_heights = properties['peak_heights']
                    peak_distances = np.diff(peaks)

                    # Predict next peak based on patterns
                    avg_distance = np.mean(peak_distances)
                    next_position = peaks[-1] + avg_distance
                    if next_position < len(smoothed):
                        predicted_value = smoothed[int(next_position)]
                        if min_num <= predicted_value <= max_num:
                            peak_scores[int(predicted_value)] += 1.0

                    # Use peak height trend
                    height_trend = np.polyfit(peaks, peak_heights, 1)
                    next_height = np.polyval(height_trend, peaks[-1] + avg_distance)
                    if min_num <= next_height <= max_num:
                        peak_scores[int(next_height)] += 0.8

                # Analyze valleys (inverted peaks)
                valleys, valley_props = find_peaks(-smoothed)
                if len(valleys) >= 2:
                    valley_distances = np.diff(valleys)
                    next_valley_pos = valleys[-1] + np.mean(valley_distances)
                    if next_valley_pos < len(smoothed):
                        valley_prediction = smoothed[int(next_valley_pos)]
                        if min_num <= valley_prediction <= max_num:
                            peak_scores[int(valley_prediction)] += 0.7

            except Exception as e:
                print(f"Peak detection error: {e}")
                continue

    return sorted(peak_scores.keys(), key=peak_scores.get, reverse=True)


def calculate_wave_probabilities(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        excluded_numbers: Set[int]
) -> Dict[int, float]:
    """Calculate probabilities based on wave characteristics."""
    probabilities = defaultdict(float)

    # Create position-based signals
    position_signals = []
    for i in range(max(len(draw) for draw in past_draws)):
        signal_data = []
        for draw in past_draws:
            if i < len(draw):
                signal_data.append(draw[i])
        position_signals.append(signal_data)

    for num in range(min_num, max_num + 1):
        if num in excluded_numbers:
            continue

        score = 0.0
        count = 0

        for signal_data in position_signals:
            if not signal_data:
                continue

            try:
                # Calculate basic signal properties
                mean_value = np.mean(signal_data)
                std_value = np.std(signal_data)

                # Calculate distance from mean in terms of standard deviations
                z_score = abs((num - mean_value) / std_value if std_value > 0 else 0)

                # Add score based on normal distribution probability
                score += np.exp(-z_score ** 2 / 2)
                count += 1

                # Add frequency domain component
                fft_vals = np.abs(fft(signal_data))
                dominant_freq = np.argmax(fft_vals[1:]) + 1  # Skip DC component

                # Score based on resonance with dominant frequency
                resonance = np.cos(2 * np.pi * dominant_freq * num / (max_num - min_num + 1))
                score += (resonance + 1) / 2  # Normalize to [0,1]

            except Exception as e:
                print(f"Probability calculation error: {e}")
                continue

        if count > 0:
            probabilities[num] = score / (2 * count)  # Normalize by number of signals

    # Normalize probabilities
    total = sum(probabilities.values())
    if total > 0:
        for num in probabilities:
            probabilities[num] /= total

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