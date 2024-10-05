# harmonic_oscillation_lottery_predictor.py

import numpy as np
from algorithms.models import lg_lottery_winner_number


def fourier_transform(data):
    """Compute the Fourier transform of the input data."""
    return np.fft.fft(data)


def generate_harmonic_sequence(frequencies, amplitudes, phases, length):
    """Generate a sequence based on harmonic oscillations."""
    t = np.arange(length)
    sequence = np.zeros(length)
    for freq, amp, phase in zip(frequencies, amplitudes, phases):
        if amp > 0:  # Only consider oscillations with non-zero amplitude
            sequence += amp * np.sin(2 * np.pi * freq * t + phase)
    return sequence


def get_numbers(lottery_type_instance):
    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    total_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Retrieve past winning numbers
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:100].values_list('lottery_type_number', flat=True))

    # Flatten the past draws into a single sequence
    flat_sequence = [num for draw in past_draws for num in draw]

    # Ensure we have enough data
    if len(flat_sequence) < total_numbers:
        return [int(num) for num in np.random.choice(range(min_num, max_num + 1), total_numbers, replace=False)]

    # Compute Fourier transform
    fft_result = fourier_transform(flat_sequence)

    # Find dominant frequencies
    frequencies = np.fft.fftfreq(len(fft_result))
    amplitudes = np.abs(fft_result)

    # Select top frequencies
    top_indices = np.argsort(amplitudes)[-5:]  # Select top 5 frequencies
    top_frequencies = frequencies[top_indices]
    top_amplitudes = amplitudes[top_indices]
    top_phases = np.angle(fft_result[top_indices])

    # Generate new sequence based on dominant frequencies
    harmonic_sequence = generate_harmonic_sequence(top_frequencies, top_amplitudes, top_phases, total_numbers * 2)

    # Handle case where harmonic_sequence is all zeros
    if np.all(harmonic_sequence == 0):
        return [int(num) for num in np.random.choice(range(min_num, max_num + 1), total_numbers, replace=False)]

    # Scale the sequence to the range of lottery numbers
    min_seq, max_seq = np.min(harmonic_sequence), np.max(harmonic_sequence)
    if min_seq == max_seq:
        scaled_sequence = np.full_like(harmonic_sequence, (min_num + max_num) / 2)
    else:
        scaled_sequence = (harmonic_sequence - min_seq) / (max_seq - min_seq)
        scaled_sequence = scaled_sequence * (max_num - min_num) + min_num

    # Round to nearest integers and ensure uniqueness
    predicted_numbers = []
    for num in scaled_sequence:
        rounded_num = int(round(num))
        rounded_num = max(min_num, min(rounded_num, max_num))  # Ensure number is within range
        if rounded_num not in predicted_numbers:
            predicted_numbers.append(rounded_num)
        if len(predicted_numbers) == total_numbers:
            break

    # If not enough unique numbers, fill with random selection
    if len(predicted_numbers) < total_numbers:
        remaining = set(range(min_num, max_num + 1)) - set(predicted_numbers)
        predicted_numbers += [int(num) for num in
                              np.random.choice(list(remaining), total_numbers - len(predicted_numbers), replace=False)]

    return sorted(predicted_numbers)