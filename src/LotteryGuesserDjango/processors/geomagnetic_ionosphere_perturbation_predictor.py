# geomagnetic_ionosphere_perturbation_predictor.py

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import skew, kurtosis
from django.apps import apps

def simulate_geomagnetic_field(past_draws, min_num, max_num):
    field = np.zeros(max_num - min_num + 1)
    for draw in past_draws:
        for num in draw:
            field[num - min_num] += 1
    return savgol_filter(field, 5, 3)  # Smooth the field

def ionospheric_perturbation(field, intensity=0.1):
    perturbation = np.random.normal(0, intensity, len(field))
    return field + perturbation

def magnetic_declination(field):
    declination = np.arctan2(np.diff(field), np.arange(1, len(field)))
    # Pad the declination array to match the size of the field
    return np.pad(declination, (0, 1), mode='edge')

def solar_wind_influence(field, factor=0.05):
    return field * (1 + factor * np.sin(np.linspace(0, 2*np.pi, len(field))))

def geomagnetic_storm_simulation(field, threshold=0.7):
    storm = np.random.random(len(field)) > threshold
    field[storm] *= 1.5
    return field

def analyze_field_characteristics(field):
    gradient = np.gradient(field)
    return {
        'mean': float(np.mean(field)),
        'std': float(np.std(field)),
        'skewness': float(skew(field)),
        'kurtosis': float(kurtosis(field)),
        'max_gradient': float(np.max(np.abs(gradient))),
        'energy': float(np.sum(field**2))
    }

def get_numbers(lottery_type_instance):
    lg_lottery_winner_number = apps.get_model('algorithms', 'lg_lottery_winner_number')

    min_num = int(lottery_type_instance.min_number)
    max_num = int(lottery_type_instance.max_number)
    total_numbers = int(lottery_type_instance.pieces_of_draw_numbers)

    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:100].values_list('lottery_type_number', flat=True))

    if len(past_draws) < 10:
        return sorted(set(np.random.choice(range(min_num, max_num + 1), total_numbers).tolist()))

    # Simulate geomagnetic field
    field = simulate_geomagnetic_field(past_draws, min_num, max_num)

    # Apply perturbations and influences
    field = ionospheric_perturbation(field)
    field = solar_wind_influence(field)
    field = geomagnetic_storm_simulation(field)

    # Analyze field characteristics
    characteristics = analyze_field_characteristics(field)

    # Calculate magnetic declination
    declination = magnetic_declination(field)

    # Ensure all arrays have the same length
    assert len(field) == len(declination), "Field and declination must have the same length"

    # Combine all factors for final prediction
    combined_field = field + 0.3 * declination + 0.2 * np.abs(characteristics['skewness']) + 0.1 * characteristics['energy']

    # Select numbers based on highest combined field values
    sorted_indices = np.argsort(combined_field)[::-1]
    predicted_numbers = [int(i + min_num) for i in sorted_indices[:total_numbers]]

    # Ensure uniqueness and correct range
    predicted_numbers = sorted(set(num for num in predicted_numbers if min_num <= num <= max_num))

    # If not enough unique numbers, fill with random selection
    while len(predicted_numbers) < total_numbers:
        new_num = int(np.random.randint(min_num, max_num + 1))
        if new_num not in predicted_numbers:
            predicted_numbers.append(new_num)

    return sorted(predicted_numbers)