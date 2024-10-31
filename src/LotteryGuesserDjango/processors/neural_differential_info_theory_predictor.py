# neural_differential_info_theory_predictor.py

import random
import math
from collections import defaultdict
from django.apps import apps
from typing import List, Set, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
def safe_exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return float('inf')


def safe_sigmoid(x):
    if x < -709:  # log(sys.float_info.min)
        return 0
    elif x > 709:  # log(sys.float_info.max)
        return 1
    else:
        return 1 / (1 + safe_exp(-x))


def euler_integrate(ode, x0, t_span, num_steps):
    t_eval = [t_span[0] + i * (t_span[1] - t_span[0]) / num_steps for i in range(num_steps + 1)]
    trajectory = [x0]
    for t in t_eval[1:]:
        x_prev = trajectory[-1]
        x_next = ode.forward(x_prev, t)
        trajectory.append(x_next)
    return trajectory



def generate_numbers(ode, initial_state, t_span, num_steps, min_num, max_num, total_numbers):
    trajectory = euler_integrate(ode, initial_state, t_span, num_steps)
    final_state = trajectory[-1]

    # Map the final state to lottery numbers
    numbers = [int(min_num + (max_num - min_num) * safe_sigmoid(x)) for x in final_state]
    return numbers[:total_numbers]


class RobustNeuralODE:
    """Neural ODE implementation with robustness features."""

    def __init__(self, input_size: int, hidden_size: int):
        self.W1 = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.W2 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.b2 = [random.uniform(-1, 1) for _ in range(input_size)]

    def forward(self, x: List[float], t: float) -> List[float]:
        """Forward pass through the neural ODE."""
        h = [safe_sigmoid(sum(w[i] * x[i] for i in range(len(x))) + b)
             for w, b in zip(self.W1, self.b1)]
        dx = [safe_sigmoid(sum(w[i] * h[i] for i in range(len(h))) + b)
              for w, b in zip(self.W2, self.b2)]
        return [x[i] + t * dx[i] for i in range(len(x))]


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Generate numbers using neural ODE approach."""
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
    """Generate numbers using neural ODE and entropy analysis."""
    try:
        # Get historical data
        past_draws = get_historical_data(lottery_type_instance, is_main)

        if len(past_draws) < 10:
            return random_selection(min_num, max_num, required_numbers)

        # Process sequence
        normalized_sequence = normalize_sequence(
            past_draws,
            min_num,
            max_num
        )

        # Calculate entropy
        entropy = calculate_entropy(normalized_sequence, num_bins=10)

        # Initialize and run neural ODE
        predicted_numbers = run_neural_ode(
            entropy,
            min_num,
            max_num,
            required_numbers
        )

        # Ensure valid predictions
        predicted_numbers = validate_predictions(
            predicted_numbers,
            min_num,
            max_num,
            required_numbers,
            entropy
        )

        return sorted(predicted_numbers)

    except Exception as e:
        print(f"Error in neural ODE prediction: {str(e)}")
        return random_selection(min_num, max_num, required_numbers)


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get historical lottery data."""
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


def normalize_sequence(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int
) -> List[float]:
    """Normalize number sequence."""
    flat_sequence = [num for draw in past_draws for num in draw]
    return [(num - min_num) / (max_num - min_num) for num in flat_sequence]


def calculate_entropy(sequence: List[float], num_bins: int) -> float:
    """Calculate sequence entropy."""
    hist = defaultdict(int)
    for num in sequence:
        bin_index = min(int(num * num_bins), num_bins - 1)
        hist[bin_index] += 1

    total = len(sequence)
    entropy = 0
    for count in hist.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def run_neural_ode(
        entropy: float,
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Run neural ODE prediction."""
    input_size = required_numbers
    hidden_size = required_numbers * 2
    ode = RobustNeuralODE(input_size, hidden_size)

    initial_state = [random.uniform(0, min(entropy, 10))
                     for _ in range(input_size)]

    t_span = (0, min(entropy * 10, 100))
    num_steps = min(int(entropy * 100), 1000)

    trajectory = euler_integrate(ode, initial_state, t_span, num_steps)
    final_state = trajectory[-1]

    numbers = [
        int(min_num + (max_num - min_num) * safe_sigmoid(x))
        for x in final_state
    ]
    return numbers[:required_numbers * 2]


def validate_predictions(
        predicted_numbers: List[int],
        min_num: int,
        max_num: int,
        required_numbers: int,
        entropy: float
) -> List[int]:
    """Validate and ensure correct number of predictions."""
    # Ensure uniqueness and range
    predicted_numbers = list(set(
        num for num in predicted_numbers
        if min_num <= num <= max_num
    ))

    # Fill if needed
    if len(predicted_numbers) < required_numbers:
        remaining = set(range(min_num, max_num + 1)) - set(predicted_numbers)
        entropy_weights = [safe_exp(min(entropy, 10) * random.random())
                           for _ in range(len(remaining))]
        additional = random.choices(
            list(remaining),
            weights=entropy_weights,
            k=required_numbers - len(predicted_numbers)
        )
        predicted_numbers.extend(additional)

    return predicted_numbers[:required_numbers]