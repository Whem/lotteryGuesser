# robust_neural_differential_info_theory_predictor.py

import random
import math
from collections import defaultdict
from django.apps import apps


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


class RobustNeuralODE:
    def __init__(self, input_size, hidden_size):
        self.W1 = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.W2 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.b2 = [random.uniform(-1, 1) for _ in range(input_size)]

    def forward(self, x, t):
        h = [safe_sigmoid(sum(w[i] * x[i] for i in range(len(x))) + b) for w, b in zip(self.W1, self.b1)]
        dx = [safe_sigmoid(sum(w[i] * h[i] for i in range(len(h))) + b) for w, b in zip(self.W2, self.b2)]
        return [x[i] + t * dx[i] for i in range(len(x))]


def euler_integrate(ode, x0, t_span, num_steps):
    t_eval = [t_span[0] + i * (t_span[1] - t_span[0]) / num_steps for i in range(num_steps + 1)]
    trajectory = [x0]
    for t in t_eval[1:]:
        x_prev = trajectory[-1]
        x_next = ode.forward(x_prev, t)
        trajectory.append(x_next)
    return trajectory


def calculate_entropy(sequence, num_bins):
    hist = defaultdict(int)
    for num in sequence:
        bin_index = min(int(num * num_bins), num_bins - 1)
        hist[bin_index] += 1

    total = len(sequence)
    entropy = 0
    for count in hist.values():
        p = count / total
        entropy -= p * math.log2(p) if p > 0 else 0
    return entropy


def generate_numbers(ode, initial_state, t_span, num_steps, min_num, max_num, total_numbers):
    trajectory = euler_integrate(ode, initial_state, t_span, num_steps)
    final_state = trajectory[-1]

    # Map the final state to lottery numbers
    numbers = [int(min_num + (max_num - min_num) * safe_sigmoid(x)) for x in final_state]
    return numbers[:total_numbers]


def get_numbers(lottery_type_instance):
    try:
        # Dynamically import the model
        lg_lottery_winner_number = apps.get_model('algorithms', 'lg_lottery_winner_number')

        min_num = int(lottery_type_instance.min_number)
        max_num = int(lottery_type_instance.max_number)
        total_numbers = int(lottery_type_instance.pieces_of_draw_numbers)

        # Retrieve past winning numbers
        past_draws = list(lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('-id')[:100].values_list('lottery_type_number', flat=True))

        if len(past_draws) < 10:
            # If not enough past draws, return random numbers
            return random.sample(range(min_num, max_num + 1), total_numbers)

        # Flatten and normalize past draws
        flat_sequence = [num for draw in past_draws for num in draw]
        normalized_sequence = [(num - min_num) / (max_num - min_num) for num in flat_sequence]

        # Calculate entropy of the normalized sequence
        entropy = calculate_entropy(normalized_sequence, 10)

        # Initialize Neural ODE
        input_size = total_numbers
        hidden_size = total_numbers * 2
        ode = RobustNeuralODE(input_size, hidden_size)

        # Generate initial state based on entropy
        initial_state = [random.uniform(0, min(entropy, 10)) for _ in range(input_size)]

        # Set time span and number of steps based on entropy
        t_span = (0, min(entropy * 10, 100))
        num_steps = min(int(entropy * 100), 1000)

        # Generate numbers using Neural ODE
        predicted_numbers = generate_numbers(ode, initial_state, t_span, num_steps, min_num, max_num, total_numbers * 2)

        # Ensure uniqueness and correct range
        predicted_numbers = list(set(predicted_numbers))
        predicted_numbers = [num for num in predicted_numbers if min_num <= num <= max_num]

        # If not enough unique numbers, fill with entropy-guided random selection
        if len(predicted_numbers) < total_numbers:
            remaining = set(range(min_num, max_num + 1)) - set(predicted_numbers)
            entropy_weights = [safe_exp(min(entropy, 10) * random.random()) for _ in range(len(remaining))]
            additional_numbers = random.choices(list(remaining), weights=entropy_weights,
                                                k=total_numbers - len(predicted_numbers))
            predicted_numbers += additional_numbers

        return sorted(predicted_numbers[:total_numbers])

    except Exception as e:
        # Log the error (you might want to use a proper logging system)
        print(f"Error in robust_neural_differential_info_theory_predictor: {str(e)}")
        # Fall back to random number generation
        return random.sample(range(min_num, max_num + 1), total_numbers)