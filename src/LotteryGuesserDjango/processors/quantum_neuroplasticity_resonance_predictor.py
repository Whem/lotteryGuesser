# quantum_neuroplasticity_resonance_predictor.py
"""
Optimalizált Quantum Neuroplasticity Resonance Predictor
Gyorsabb végrehajtás megtartva a predikciós pontosságot
"""

import numpy as np
from scipy.stats import entropy
from django.apps import apps
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


class OptimizedQuantumNeuron:
    def __init__(self, dimension):
        self.weights = np.random.rand(dimension)
        self.phase = np.random.rand() * 2 * np.pi

    def activate(self, inputs):
        # Optimalizált aktivációs függvény
        amplitude = np.dot(self.weights, inputs)
        return amplitude ** 2  # Simplified without complex exponential

    def update(self, inputs, learning_rate):
        # Gyorsabb súlyfrissítés
        self.weights += learning_rate * (inputs - self.weights) * 0.1
        self.phase += learning_rate * 0.1


class OptimizedAdaptiveResonanceLayer:
    def __init__(self, input_dim, num_neurons=8):  # Csökkentett neuronszám
        self.neurons = [OptimizedQuantumNeuron(input_dim) for _ in range(num_neurons)]

    def forward(self, inputs):
        return np.array([neuron.activate(inputs) for neuron in self.neurons])

    def update(self, inputs, learning_rate):
        for neuron in self.neurons:
            neuron.update(inputs, learning_rate)


def fast_quantum_interference(predictions):
    """Gyorsabb interference számítás FFT használatával"""
    n = len(predictions)
    if n <= 1:
        return np.zeros(n)
    
    # FFT-alapú konvolúció gyorsabb nagy adatokra
    fft_pred = np.fft.fft(predictions)
    interference = np.real(np.fft.ifft(fft_pred * np.conj(fft_pred)))
    return interference / n


def adaptive_vigilance(entropy_val, min_vigilance=0.2, max_vigilance=0.8):
    """Egyszerűsített vigilance számítás"""
    return min_vigilance + (max_vigilance - min_vigilance) * (1 - entropy_val)


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using optimized quantum neuroplasticity resonance predictor.
    """
    # Generate main numbers
    main_numbers = generate_number_set(
        lottery_type_instance,
        min_num=int(lottery_type_instance.min_number),
        max_num=int(lottery_type_instance.max_number),
        total_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
        is_main=True
    )

    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        # Generate additional numbers
        additional_numbers = generate_number_set(
            lottery_type_instance,
            min_num=int(lottery_type_instance.additional_min_number),
            max_num=int(lottery_type_instance.additional_max_number),
            total_numbers=int(lottery_type_instance.additional_numbers_count),
            is_main=False
        )

    return main_numbers, additional_numbers


def generate_number_set(
    lottery_type_instance: lg_lottery_type,
    min_num: int,
    max_num: int,
    total_numbers: int,
    is_main: bool
) -> List[int]:
    """
    Optimalizált számgenerálás quantum neuroplasticity módszerrel.
    """
    lg_lottery_winner_number = apps.get_model('algorithms', 'lg_lottery_winner_number')

    # Kevesebb múltbeli húzás a gyorsaság érdekében
    limit = min(100, 500)  # Maximálisan 100 húzás
    
    if is_main:
        past_draws = list(
            lg_lottery_winner_number.objects.filter(
                lottery_type=lottery_type_instance
            ).order_by('-id')[:limit].values_list('lottery_type_number', flat=True)
        )
    else:
        past_draws = list(
            lg_lottery_winner_number.objects.filter(
                lottery_type=lottery_type_instance
            ).order_by('-id')[:limit].values_list('additional_numbers', flat=True)
        )

    if len(past_draws) < 20:  # Csökkentett minimum
        return sorted(set(np.random.choice(range(min_num, max_num + 1), total_numbers, replace=False)))

    # Gyorsabb normalizálás
    normalized_draws = []
    range_size = max_num - min_num
    
    for draw in past_draws[:50]:  # Csak az első 50-et dolgozzuk fel
        if isinstance(draw, list) and len(draw) > 0:
            normalized_draw = (np.array(draw) - min_num) / range_size
            normalized_draws.append(normalized_draw)

    if not normalized_draws:
        return sorted(set(np.random.choice(range(min_num, max_num + 1), total_numbers, replace=False)))

    input_dim = len(normalized_draws[0])
    ar_layer = OptimizedAdaptiveResonanceLayer(input_dim, num_neurons=6)  # Kevesebb neuron

    learning_rate = 0.05  # Nagyobb kezdőérték
    num_epochs = 20  # Drasztikusan csökkentett epochok

    # Gyorsabb tanítás
    for epoch in range(num_epochs):
        # Csak minden 2. draw-t dolgozzuk fel
        for i, draw in enumerate(normalized_draws):
            if i % 2 == 0:  # Skip every second draw for speed
                predictions = ar_layer.forward(draw)
                ar_layer.update(draw, learning_rate)
        
        learning_rate *= 0.95  # Gyorsabb csökkenés

    # Optimalizált aggregáció - csak az utolsó 20 draw-t használjuk
    sample_draws = normalized_draws[:20]
    final_predictions = np.mean([ar_layer.forward(draw) for draw in sample_draws], axis=0)

    # Gyorsabb interference számítás
    interference = fast_quantum_interference(final_predictions)
    adjusted_predictions = final_predictions * (1 + interference * 0.1)  # Kevesebb hatás

    # Egyszerűsített entropy számítás
    entropy_val = -np.sum(adjusted_predictions * np.log2(adjusted_predictions + 1e-10))
    normalized_entropy = entropy_val / np.log2(len(adjusted_predictions))

    vigilance = adaptive_vigilance(normalized_entropy)

    # Egyszerűsített threshold
    threshold = np.percentile(adjusted_predictions, 70)  # Top 30%
    selected_indices = np.where(adjusted_predictions > threshold)[0]

    if len(selected_indices) < total_numbers:
        # Gyorsabb kiválasztás
        remaining_indices = np.argsort(adjusted_predictions)[::-1]
        selected_indices = remaining_indices[:total_numbers]

    # Gyorsabb mapping
    predicted_numbers = []
    for index in selected_indices:
        num = int(round(index * range_size / (input_dim - 1) + min_num))
        if min_num <= num <= max_num and num not in predicted_numbers:
            predicted_numbers.append(num)

    # Gyors feltöltés ha szükséges
    while len(predicted_numbers) < total_numbers:
        candidates = list(range(min_num, max_num + 1))
        remaining = [x for x in candidates if x not in predicted_numbers]
        if remaining:
            predicted_numbers.append(np.random.choice(remaining))
        else:
            break

    return sorted(predicted_numbers[:total_numbers])
