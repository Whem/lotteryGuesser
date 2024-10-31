# geomagnetic_ionosphere_perturbation_predictor.py

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import skew, kurtosis
from django.apps import apps
from typing import List, Dict, Tuple, Set
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


def apply_field_modifications(field: np.ndarray) -> np.ndarray:
    """Applies all field modifications."""
    field = ionospheric_perturbation(field)
    field = solar_wind_influence(field)
    field = geomagnetic_storm_simulation(field)
    return field


def generate_predictions(
        field: np.ndarray,
        declination: np.ndarray,
        characteristics: Dict,
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generates predictions from field analysis."""
    # Combine factors
    combined_field = (
            field +
            0.3 * declination +
            0.2 * np.abs(characteristics['skewness']) +
            0.1 * characteristics['energy']
    )

    # Select numbers
    sorted_indices = np.argsort(combined_field)[::-1]
    predicted_numbers = [
        int(i + min_num)
        for i in sorted_indices[:required_numbers]
    ]

    # Ensure uniqueness and range
    predicted_numbers = sorted(set(
        num for num in predicted_numbers
        if min_num <= num <= max_num
    ))

    # Fill if needed
    while len(predicted_numbers) < required_numbers:
        new_num = int(np.random.randint(min_num, max_num + 1))
        if new_num not in predicted_numbers:
            predicted_numbers.append(new_num)

    return sorted(predicted_numbers)

def get_numbers(lottery_type_instance) -> Tuple[List[int], List[int]]:
    """Generates numbers using geomagnetic simulation for both main and additional sets."""
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
        lottery_type_instance,
        min_num: int,
        max_num: int,
        required_numbers: int,
        is_main: bool
) -> List[int]:
    """Generates numbers using geomagnetic field simulation."""
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if len(past_draws) < 10:
        return sorted(set(np.random.choice(
            range(min_num, max_num + 1),
            required_numbers
        ).tolist()))

    # Generate and analyze field
    field = simulate_geomagnetic_field(past_draws, min_num, max_num)
    field = apply_field_modifications(field)
    characteristics = analyze_field_characteristics(field)
    declination = magnetic_declination(field)

    # Generate predictions
    predicted_numbers = generate_predictions(
        field,
        declination,
        characteristics,
        min_num,
        max_num,
        required_numbers
    )

    return predicted_numbers


def get_historical_data(lottery_type_instance, is_main: bool) -> List[List[int]]:
    """Gets historical lottery data."""
    lg_lottery_winner_number = apps.get_model('algorithms', 'lg_lottery_winner_number')
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