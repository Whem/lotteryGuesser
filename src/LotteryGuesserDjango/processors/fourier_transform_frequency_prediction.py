import numpy as np

from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    past_draws = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number', flat=True)
    number_sequences = np.array([draw for draw in past_draws])

    # Apply Fourier Transform
    fourier_result = np.fft.fft(number_sequences, axis=0)
    frequencies = np.fft.fftfreq(number_sequences.shape[0])

    # Identify dominant frequencies
    dominant_frequencies = np.argsort(np.abs(fourier_result), axis=0)[-lottery_type_instance.pieces_of_draw_numbers:]
    predicted_numbers = frequencies[dominant_frequencies]

    # Map frequencies back to number range
    predicted_numbers = (predicted_numbers - predicted_numbers.min()) / (predicted_numbers.max() - predicted_numbers.min())
    predicted_numbers *= (lottery_type_instance.max_number - lottery_type_instance.min_number)
    predicted_numbers += lottery_type_instance.min_number

    return np.round(predicted_numbers).astype(int)
