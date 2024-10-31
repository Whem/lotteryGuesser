
import numpy as np
from typing import List, Tuple, Dict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Fourier transform predictor for combined lottery types.
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
    """Generate numbers using Fourier transform analysis."""
    try:
        # Get historical data
        past_draws = get_historical_data(lottery_type_instance, is_main)

        if not past_draws:
            return random_number_set(min_num, max_num, required_numbers)

        # Convert to number sequence and analyze
        number_sequence = np.array([
            number for draw in past_draws
            for number in draw
        ])

        # Get dominant frequencies
        dominant_frequencies = analyze_frequencies(
            number_sequence,
            required_numbers
        )

        # Generate predictions
        predicted_numbers = generate_predictions(
            dominant_frequencies,
            min_num,
            max_num,
            required_numbers
        )

        return sorted(predicted_numbers)

    except Exception as e:
        print(f"Error in Fourier analysis: {str(e)}")
        return random_number_set(min_num, max_num, required_numbers)


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get historical lottery data based on number type."""
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id'))

    if is_main:
        return [draw.lottery_type_number for draw in past_draws
                if isinstance(draw.lottery_type_number, list)]
    else:
        return [draw.additional_numbers for draw in past_draws
                if hasattr(draw, 'additional_numbers') and
                isinstance(draw.additional_numbers, list)]


def analyze_frequencies(
        number_sequence: np.ndarray,
        required_numbers: int
) -> np.ndarray:
    """Analyze frequencies using Fourier transform."""
    try:
        # Apply Fourier Transform
        fourier_result = np.fft.fft(number_sequence)
        frequencies = np.fft.fftfreq(number_sequence.size)

        # Get dominant frequencies
        dominant_indices = np.argsort(np.abs(fourier_result))[-required_numbers:]
        return frequencies[dominant_indices]

    except Exception as e:
        print(f"Error in frequency analysis: {str(e)}")
        return np.zeros(required_numbers)


def generate_predictions(
        frequencies: np.ndarray,
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate predictions from frequencies."""
    predicted_numbers = set()
    number_range = max_num - min_num + 1

    # Generate numbers from frequencies
    for freq in frequencies:
        try:
            # Normalize frequency to [0, 1]
            position = abs(freq) % 1
            number = int(position * number_range) + min_num

            # Ensure number is within range
            number = max(min_num, min(max_num, number))
            predicted_numbers.add(number)

        except Exception as e:
            print(f"Error generating prediction: {str(e)}")

    # Fill remaining numbers if needed
    fill_remaining_numbers(predicted_numbers, min_num, max_num, required_numbers)

    return list(predicted_numbers)


def fill_remaining_numbers(
        numbers: set,
        min_num: int,
        max_num: int,
        required_numbers: int
) -> None:
    """Fill missing numbers with random selection."""
    while len(numbers) < required_numbers:
        new_number = np.random.randint(min_num, max_num + 1)
        numbers.add(new_number)


def random_number_set(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate a random set of numbers."""
    return sorted(np.random.choice(
        range(min_num, max_num + 1),
        size=required_numbers,
        replace=False
    ))


def get_fourier_statistics(past_draws: List[List[int]], top_n: int = 5) -> Dict:
    """
    Get comprehensive Fourier analysis statistics.

    Returns:
    - dominant_frequencies: top frequencies and their magnitudes
    - periodicities: potential periodic patterns
    - spectral_density: power spectral density information
    """
    if not past_draws:
        return {}

    try:
        # Convert to number sequence
        number_sequence = np.array([
            number for draw in past_draws
            for number in draw
        ])

        # Perform Fourier analysis
        fourier_result = np.fft.fft(number_sequence)
        frequencies = np.fft.fftfreq(number_sequence.size)

        # Sort by magnitude
        sorted_indices = np.argsort(np.abs(fourier_result))[::-1]
        top_frequencies = frequencies[sorted_indices[:top_n]]
        top_magnitudes = np.abs(fourier_result[sorted_indices[:top_n]])

        # Calculate periodicities
        periodicities = 1 / np.abs(top_frequencies)
        periodicities = np.where(np.isinf(periodicities), 0, periodicities)

        # Calculate power spectral density
        psd = np.abs(fourier_result) ** 2

        stats = {
            'dominant_frequencies': dict(zip(
                top_frequencies.tolist(),
                top_magnitudes.tolist()
            )),
            'periodicities': periodicities.tolist(),
            'spectral_density': {
                'mean': float(np.mean(psd)),
                'max': float(np.max(psd)),
                'total_power': float(np.sum(psd))
            }
        }

        return stats

    except Exception as e:
        print(f"Error calculating Fourier statistics: {str(e)}")
        return {}