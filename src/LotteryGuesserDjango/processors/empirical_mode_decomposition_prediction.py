# empirical_mode_decomposition_prediction.py
import numpy as np
import random
from typing import List, Tuple, Set, Dict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
from collections import Counter

# Try to import PyEMD, use fallback if not available
try:
    from PyEMD import EMD
    HAS_EMD = True
except ImportError:
    HAS_EMD = False
    print("PyEMD not available, using mathematical fallback")


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Empirical Mode Decomposition predictor for combined lottery types.
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
    """Generate a set of numbers using EMD analysis."""
    # Get historical data
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if len(past_draws) < 20:
        return random_number_set(min_num, max_num, required_numbers)

    # Convert to numpy matrix
    draw_matrix = np.array(past_draws)

    # Perform EMD analysis
    predicted_numbers = perform_emd_analysis(
        draw_matrix,
        min_num,
        max_num,
        required_numbers
    )

    # Ensure unique numbers
    predicted_numbers = ensure_unique_numbers(
        predicted_numbers,
        past_draws,
        min_num,
        max_num,
        required_numbers
    )

    return sorted(predicted_numbers)


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get historical lottery data based on number type."""
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id'))

    if is_main:
        draws = [draw.lottery_type_number for draw in past_draws
                 if isinstance(draw.lottery_type_number, list)]
    else:
        draws = [draw.additional_numbers for draw in past_draws
                 if hasattr(draw, 'additional_numbers') and
                 isinstance(draw.additional_numbers, list)]

    # Ensure consistent length
    required_length = len(draws[0]) if draws else 0
    return [draw for draw in draws if len(draw) == required_length]


def perform_emd_analysis(
        draw_matrix: np.ndarray,
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Perform EMD analysis on the draw matrix or use mathematical fallback."""
    predicted_numbers = []

    if HAS_EMD:
        # Use real EMD analysis
        emd = EMD()
        
        # Analyze each position
        for i in range(draw_matrix.shape[1]):
            series = draw_matrix[:, i]

            try:
                # Apply EMD
                IMFs = emd.emd(series)

                # Reconstruct signal without first IMF (noise)
                if IMFs.shape[0] > 1:
                    reconstructed = np.sum(IMFs[1:], axis=0)
                else:
                    reconstructed = series

                # Predict next value
                if len(reconstructed) >= 2:
                    trend = reconstructed[-1] - reconstructed[-2]
                    next_value = reconstructed[-1] + trend
                else:
                    next_value = reconstructed[-1]

                # Round and adjust to range
                predicted_number = int(round(next_value))
                predicted_number = max(min_num, min(max_num, predicted_number))
                predicted_numbers.append(predicted_number)

            except Exception as e:
                print(f"EMD analysis error for position {i}: {str(e)}")
                # Fallback to trend analysis
                predicted_numbers.append(trend_analysis_fallback(series, min_num, max_num))
    else:
        # Use mathematical fallback - advanced trend analysis
        for i in range(draw_matrix.shape[1]):
            series = draw_matrix[:, i]
            predicted_number = advanced_trend_analysis(series, min_num, max_num)
            predicted_numbers.append(predicted_number)

    return predicted_numbers

def trend_analysis_fallback(series: np.ndarray, min_num: int, max_num: int) -> int:
    """Fallback trend analysis when EMD fails."""
    if len(series) < 2:
        return min_num + (max_num - min_num) // 2
    
    # Calculate trend using recent values
    recent_values = series[-5:] if len(series) >= 5 else series
    trend = np.mean(np.diff(recent_values)) if len(recent_values) > 1 else 0
    
    # Predict next value
    next_value = series[-1] + trend
    predicted_number = int(round(next_value))
    return max(min_num, min(max_num, predicted_number))

def advanced_trend_analysis(series: np.ndarray, min_num: int, max_num: int) -> int:
    """Advanced mathematical analysis without EMD."""
    if len(series) < 3:
        return random.randint(min_num, max_num)
    
    # Moving average with different windows
    short_ma = np.mean(series[-3:])
    long_ma = np.mean(series[-6:]) if len(series) >= 6 else np.mean(series)
    
    # Calculate momentum
    momentum = short_ma - long_ma
    
    # Fourier-like frequency analysis (simplified)
    diffs = np.diff(series)
    avg_change = np.mean(diffs)
    std_change = np.std(diffs)
    
    # Predict next value combining trend and oscillation
    base_prediction = series[-1] + avg_change
    oscillation = np.sin(len(series) * np.pi / 6) * std_change * 0.5
    
    next_value = base_prediction + momentum * 0.3 + oscillation
    predicted_number = int(round(next_value))
    
    return max(min_num, min(max_num, predicted_number))


def ensure_unique_numbers(
        predicted_numbers: List[int],
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Ensure we have enough unique numbers."""
    unique_numbers = set(predicted_numbers)

    if len(unique_numbers) < required_numbers:
        # Get frequency of past numbers
        frequency = Counter(num for draw in past_draws for num in draw)

        # Add most common numbers not already included
        common_numbers = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
        for num, _ in common_numbers:
            if num not in unique_numbers:
                unique_numbers.add(num)
            if len(unique_numbers) >= required_numbers:
                break

        # If still not enough, add random numbers
        while len(unique_numbers) < required_numbers:
            new_number = random.randint(min_num, max_num)
            unique_numbers.add(new_number)

    return list(unique_numbers)[:required_numbers]


def random_number_set(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate a random set of numbers."""
    return sorted(random.sample(range(min_num, max_num + 1), required_numbers))


def get_emd_statistics(past_draws: List[List[int]]) -> Dict:
    """
    Get comprehensive EMD statistics.

    Returns a dictionary containing:
    - imf_count: average number of IMFs per position
    - reconstruction_error: average reconstruction error
    - trend_strength: correlation with linear trend
    - noise_ratio: estimated noise ratio
    """
    if not past_draws or len(past_draws) < 2:
        return {}

    draw_matrix = np.array(past_draws)
    stats = {
        'imf_counts': [],
        'reconstruction_errors': [],
        'trend_correlations': [],
        'noise_ratios': []
    }

    emd = EMD()

    for i in range(draw_matrix.shape[1]):
        try:
            series = draw_matrix[:, i]
            IMFs = emd.emd(series)

            # Count IMFs
            stats['imf_counts'].append(IMFs.shape[0])

            # Calculate reconstruction error
            if IMFs.shape[0] > 1:
                reconstructed = np.sum(IMFs[1:], axis=0)
                error = np.mean(np.abs(series - reconstructed))
                stats['reconstruction_errors'].append(error)

                # Estimate noise ratio
                noise = IMFs[0]
                noise_ratio = np.std(noise) / np.std(series)
                stats['noise_ratios'].append(noise_ratio)

            # Calculate trend correlation
            trend = np.arange(len(series))
            correlation = np.corrcoef(series, trend)[0, 1]
            stats['trend_correlations'].append(correlation)

        except Exception as e:
            print(f"Error calculating EMD statistics for position {i}: {str(e)}")

    # Aggregate statistics
    return {
        'avg_imf_count': np.mean(stats['imf_counts']),
        'avg_reconstruction_error': np.mean(stats['reconstruction_errors']),
        'avg_trend_correlation': np.mean(stats['trend_correlations']),
        'avg_noise_ratio': np.mean(stats['noise_ratios'])
    }