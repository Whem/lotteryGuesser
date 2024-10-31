# neural_network_evolutionary_optimization.py

import numpy as np

from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import random  # Hiányzott a importálás

from typing import List, Set, Tuple
class SimpleNN:
    """Simple neural network with one hidden layer."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        self.z1 = np.dot(x, self.w1)
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.w2)
        return self.z2


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Neural evolution predictor for combined lottery types."""
    main_numbers = generate_number_set(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        True
    )

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
    """Generate numbers using neural evolution."""
    # Get historical data
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if len(past_draws) < 50:
        return random_selection(min_num, max_num, required_numbers)

    # Train neural network
    network = train_neural_network(
        past_draws,
        min_num,
        max_num,
        required_numbers
    )

    # Generate predictions
    predicted_numbers = generate_predictions(
        network,
        past_draws,
        min_num,
        max_num,
        required_numbers
    )

    return predicted_numbers


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get historical lottery data."""
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id'))

    if is_main:
        draws = [draw.lottery_type_number for draw in past_draws
                 if isinstance(draw.lottery_type_number, list)]
    else:
        draws = [draw.additional_numbers for draw in past_draws
                 if hasattr(draw, 'additional_numbers') and
                 isinstance(draw.additional_numbers, list)]

    return draws


def train_neural_network(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> SimpleNN:
    """Train neural network using evolutionary optimization."""
    window_size = 10
    input_size = window_size * required_numbers
    hidden_size = 20
    output_size = required_numbers

    population = [
        SimpleNN(input_size, hidden_size, output_size)
        for _ in range(50)
    ]

    for generation in range(100):
        fitness_scores = evaluate_population(
            population,
            past_draws,
            window_size,
            required_numbers,
            input_size,
            output_size
        )

        population = evolve_population(
            population,
            fitness_scores,
            input_size,
            hidden_size,
            output_size
        )

    return select_best_network(population, fitness_scores)


def evaluate_population(
        population: List[SimpleNN],
        past_draws: List[List[int]],
        window_size: int,
        required_numbers: int,
        input_size: int,
        output_size: int
) -> List[float]:
    """Evaluate fitness of neural network population."""
    fitness_scores = []

    for nn in population:
        predictions = []
        for i in range(len(past_draws) - window_size):
            input_window = past_draws[i:i + window_size]

            if not validate_window(input_window, required_numbers):
                continue

            input_data = np.array(input_window).flatten()
            if input_data.shape[0] != input_size:
                continue

            predictions.append(nn.forward(input_data))

        score = calculate_fitness(
            predictions,
            past_draws[window_size:],
            output_size
        )
        fitness_scores.append(score)

    return fitness_scores


def evolve_population(
        population: List[SimpleNN],
        fitness_scores: List[float],
        input_size: int,
        hidden_size: int,
        output_size: int
) -> List[SimpleNN]:
    """Evolve population using genetic algorithm."""
    if len(fitness_scores) < 10:
        top_performers_indices = np.argsort(fitness_scores)
    else:
        top_performers_indices = np.argsort(fitness_scores)[-10:]

    top_performers = [population[i] for i in top_performers_indices]
    new_population = top_performers.copy()

    while len(new_population) < len(population):
        parent1, parent2 = get_parents(top_performers)
        child = create_child(
            parent1,
            parent2,
            input_size,
            hidden_size,
            output_size
        )
        new_population.append(child)

    return new_population


def validate_window(
        window: List[List[int]],
        required_numbers: int
) -> bool:
    """Validate input window for neural network."""
    try:
        return all(
            isinstance(draw, list) and
            len(draw) == required_numbers
            for draw in window
        )
    except Exception as e:
        print(f"Window validation error: {str(e)}")
        return False


def calculate_fitness(
        predictions: List[np.ndarray],
        actual_draws: List[List[int]],
        output_size: int
) -> float:
    """Calculate fitness score for predictions."""
    if not predictions:
        return 0

    score = 0
    for j, pred in enumerate(predictions):
        if j >= len(actual_draws):
            break

        actual = np.array(actual_draws[j])
        if actual.shape[0] != output_size:
            continue

        pred_rounded = np.round(pred).astype(int)
        score += np.sum(np.abs(actual - pred_rounded))

    return 1 / (score + 1e-6)


def get_parents(
        top_performers: List[SimpleNN]
) -> Tuple[SimpleNN, SimpleNN]:
    """Select parents for breeding."""
    if len(top_performers) < 2:
        parent1 = parent2 = random.choice(top_performers)
    else:
        parent1, parent2 = random.sample(top_performers, 2)
    return parent1, parent2


def create_child(
        parent1: SimpleNN,
        parent2: SimpleNN,
        input_size: int,
        hidden_size: int,
        output_size: int
) -> SimpleNN:
    """Create child network through crossover."""
    child = SimpleNN(input_size, hidden_size, output_size)

    # Crossover weights with mutation
    child.w1 = crossover_weights(parent1.w1, parent2.w1)
    child.w2 = crossover_weights(parent1.w2, parent2.w2)

    return child


def crossover_weights(
        weights1: np.ndarray,
        weights2: np.ndarray,
        mutation_rate: float = 0.1
) -> np.ndarray:
    """Perform crossover of weights with mutation."""
    # Average weights
    child_weights = (weights1 + weights2) / 2

    # Add mutation
    mutation = np.random.randn(*child_weights.shape) * mutation_rate
    child_weights += mutation

    return child_weights


def select_best_network(
        population: List[SimpleNN],
        fitness_scores: List[float]
) -> SimpleNN:
    """Select best performing network."""
    if not fitness_scores:
        return random.choice(population)

    best_idx = np.argmax(fitness_scores)
    return population[best_idx]


def generate_predictions(
        network: SimpleNN,
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        required_numbers: int,
        window_size: int = 10
) -> List[int]:
    """Generate predictions using trained network."""
    try:
        # Prepare input window
        last_window = past_draws[-window_size:]
        if not validate_window(last_window, required_numbers):
            return random_selection(min_num, max_num, required_numbers)

        # Generate prediction
        input_data = np.array(last_window).flatten()
        prediction = network.forward(input_data)
        predicted_numbers = set()

        # Process predictions
        for num in np.round(prediction).astype(int):
            if min_num <= num <= max_num:
                predicted_numbers.add(int(num))

        # Fill missing numbers
        while len(predicted_numbers) < required_numbers:
            new_num = random.randint(min_num, max_num)
            predicted_numbers.add(new_num)

        return sorted(list(predicted_numbers)[:required_numbers])

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return random_selection(min_num, max_num, required_numbers)


def random_selection(min_num: int, max_num: int, count: int) -> List[int]:
    """Generate random number selection."""
    return sorted(random.sample(range(min_num, max_num + 1), count))