# quantum_neuroplasticity_resonance_predictor.py

import numpy as np
from scipy.stats import entropy
from django.apps import apps


class QuantumNeuron:
    def __init__(self, dimension):
        self.weights = np.random.rand(dimension)
        self.phase = np.random.rand() * 2 * np.pi

    def activate(self, inputs):
        amplitude = np.dot(self.weights, inputs)
        return np.abs(amplitude * np.exp(1j * self.phase)) ** 2

    def update(self, inputs, learning_rate):
        self.weights += learning_rate * (inputs - self.weights)
        self.phase += learning_rate * np.angle(np.sum(inputs))


class AdaptiveResonanceLayer:
    def __init__(self, input_dim, num_neurons):
        self.neurons = [QuantumNeuron(input_dim) for _ in range(num_neurons)]

    def forward(self, inputs):
        return [neuron.activate(inputs) for neuron in self.neurons]

    def update(self, inputs, learning_rate):
        for neuron in self.neurons:
            neuron.update(inputs, learning_rate)


def quantum_interference(predictions):
    interference = np.zeros(len(predictions))
    for i in range(len(predictions)):
        for j in range(len(predictions)):
            if i != j:
                interference[i] += np.cos(predictions[i] - predictions[j])
    return interference / len(predictions)


def adaptive_vigilance(entropy_val, min_vigilance=0.1, max_vigilance=0.9):
    return min_vigilance + (max_vigilance - min_vigilance) * (1 - entropy_val)


def get_numbers(lottery_type_instance):
    lg_lottery_winner_number = apps.get_model('algorithms', 'lg_lottery_winner_number')

    min_num = int(lottery_type_instance.min_number)
    max_num = int(lottery_type_instance.max_number)
    total_numbers = int(lottery_type_instance.pieces_of_draw_numbers)

    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:500].values_list('lottery_type_number', flat=True))

    if len(past_draws) < 50:
        return sorted(set(np.random.choice(range(min_num, max_num + 1), total_numbers)))

    # Normalize past draws
    normalized_draws = [(np.array(draw) - min_num) / (max_num - min_num) for draw in past_draws]

    input_dim = len(normalized_draws[0])
    ar_layer = AdaptiveResonanceLayer(input_dim, num_neurons=20)

    learning_rate = 0.01
    num_epochs = 100

    for epoch in range(num_epochs):
        for draw in normalized_draws:
            predictions = ar_layer.forward(draw)
            ar_layer.update(draw, learning_rate)

        learning_rate *= 0.99  # Decrease learning rate over time

    final_predictions = np.mean([ar_layer.forward(draw) for draw in normalized_draws], axis=0)

    interference = quantum_interference(final_predictions)
    adjusted_predictions = final_predictions * (1 + interference)

    # Calculate entropy of the adjusted predictions
    entropy_val = entropy(adjusted_predictions)
    normalized_entropy = entropy_val / np.log2(len(adjusted_predictions))

    vigilance = adaptive_vigilance(normalized_entropy)

    # Apply vigilance threshold
    threshold = np.mean(adjusted_predictions) + vigilance * np.std(adjusted_predictions)
    selected_indices = np.where(adjusted_predictions > threshold)[0]

    if len(selected_indices) < total_numbers:
        # If not enough numbers selected, add the highest remaining ones
        remaining_indices = np.argsort(adjusted_predictions)[::-1]
        selected_indices = np.unique(np.concatenate([selected_indices, remaining_indices]))[:total_numbers]

    # Map selected indices back to original number range
    predicted_numbers = [int(round(index * (max_num - min_num) + min_num)) for index in selected_indices]

    # Ensure we have the correct number of unique predictions within the valid range
    predicted_numbers = sorted(set(num for num in predicted_numbers if min_num <= num <= max_num))
    while len(predicted_numbers) < total_numbers:
        new_num = np.random.randint(min_num, max_num + 1)
        if new_num not in predicted_numbers:
            predicted_numbers.append(new_num)

    return sorted(predicted_numbers[:total_numbers])