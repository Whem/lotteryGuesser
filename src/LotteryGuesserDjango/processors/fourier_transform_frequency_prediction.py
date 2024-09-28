import numpy as np
from typing import List
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list(
        'lottery_type_number', flat=True)

    # Flatten the list if it's a list of lists
    number_sequences = np.array([number for draw in past_draws for number in draw])

    # Apply Fourier Transform
    fourier_result = np.fft.fft(number_sequences)
    frequencies = np.fft.fftfreq(number_sequences.size)

    # Identify dominant frequencies
    dominant_indices = np.argsort(np.abs(fourier_result))[-lottery_type_instance.pieces_of_draw_numbers:]
    dominant_frequencies = frequencies[dominant_indices]

    # Use the dominant frequencies to generate predictions
    predicted_numbers = generate_numbers_from_frequencies(dominant_frequencies, lottery_type_instance)

    return sorted(predicted_numbers)

def generate_numbers_from_frequencies(frequencies: np.ndarray, lottery_type_instance: lg_lottery_type) -> List[int]:
    min_number, max_number = lottery_type_instance.min_number, lottery_type_instance.max_number
    number_range = max_number - min_number + 1

    predicted_numbers = set()
    for freq in frequencies:
        # Use the frequency to determine a position in the number range
        position = abs(freq) % 1  # Normalize to [0, 1]
        number = int(position * number_range) + min_number
        predicted_numbers.add(number)

    # If we don't have enough unique numbers, fill with random ones
    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        predicted_numbers.add(np.random.randint(min_number, max_number + 1))

    return list(predicted_numbers)

def analyze_fourier_patterns(past_draws: List[List[int]], top_n: int = 5) -> None:
    number_sequences = np.array([number for draw in past_draws for number in draw])
    fourier_result = np.fft.fft(number_sequences)
    frequencies = np.fft.fftfreq(number_sequences.size)

    # Sort frequencies by magnitude of Fourier coefficients
    sorted_indices = np.argsort(np.abs(fourier_result))[::-1]
    top_frequencies = frequencies[sorted_indices[:top_n]]
    top_magnitudes = np.abs(fourier_result[sorted_indices[:top_n]])

    print(f"Top {top_n} dominant frequencies:")
    for freq, mag in zip(top_frequencies, top_magnitudes):
        print(f"Frequency: {freq:.4f}, Magnitude: {mag:.2f}")

    # Analyze periodicity
    potential_periods = 1 / np.abs(top_frequencies)
    print("\nPotential periodicities in the number sequence:")
    for period in potential_periods:
        print(f"Every {period:.2f} numbers")

# Example usage of the analysis function:
# past_draws = list(lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True))
# analyze_fourier_patterns(past_draws)