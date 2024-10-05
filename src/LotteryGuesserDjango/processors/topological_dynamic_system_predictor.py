# topological_dynamic_system_predictor.py

import random
import math
from collections import defaultdict
from algorithms.models import lg_lottery_winner_number


def create_point_cloud(past_draws, dimension):
    """Create a point cloud from past draws using time-delay embedding."""
    point_cloud = []
    for i in range(len(past_draws) - dimension + 1):
        point = [num for draw in past_draws[i:i + dimension] for num in draw]
        point_cloud.append(point)
    return point_cloud


def pairwise_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))


def calculate_persistent_homology(point_cloud, max_distance):
    """Simplified persistent homology calculation."""
    edges = []
    for i, p1 in enumerate(point_cloud):
        for j, p2 in enumerate(point_cloud[i + 1:], i + 1):
            distance = pairwise_distance(p1, p2)
            if distance <= max_distance:
                edges.append((i, j, distance))

    edges.sort(key=lambda x: x[2])

    components = list(range(len(point_cloud)))
    persistence = []

    for u, v, distance in edges:
        if components[u] != components[v]:
            old = components[u]
            new = components[v]
            persistence.append((old, new, distance))
            for i in range(len(components)):
                if components[i] == old:
                    components[i] = new

    return persistence


def lyapunov_exponent(sequence, embedding_dim, delay):
    """Calculate Lyapunov exponent to measure chaoticity."""
    N = len(sequence)
    M = N - (embedding_dim - 1) * delay

    y = sequence[:M]
    Y = [sequence[i:i + embedding_dim * delay:delay] for i in range(M)]

    epsilon = 1e-10
    lyap = 0
    for i in range(M):
        distances = [pairwise_distance(Y[i], Y[j]) for j in range(M) if i != j]
        nearest = min(distances)
        if nearest < epsilon:
            continue

        j = distances.index(nearest)
        d_n = abs(y[i] - y[j])
        if d_n < epsilon:
            continue

        lyap += math.log(d_n / nearest)

    return lyap / M if M > 0 else 0


def get_numbers(lottery_type_instance):
    try:
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

        # Create point cloud using time-delay embedding
        embedding_dim = 3
        point_cloud = create_point_cloud(past_draws, embedding_dim)

        # Calculate persistent homology
        max_distance = math.sqrt(total_numbers * (max_num - min_num) ** 2)
        persistence = calculate_persistent_homology(point_cloud, max_distance)

        # Extract topological features
        topological_features = [p[2] for p in persistence]

        # Calculate Lyapunov exponent
        flat_sequence = [num for draw in past_draws for num in draw]
        lyap_exp = lyapunov_exponent(flat_sequence, embedding_dim, 1)

        # Generate numbers based on topological features and chaoticity
        predicted_numbers = []
        for _ in range(total_numbers * 2):
            if topological_features:
                feature = random.choice(topological_features)
                num = int(min_num + (feature / max_distance) * (max_num - min_num))
            else:
                num = random.randint(min_num, max_num)

            # Apply chaotic perturbation
            num = int(num + lyap_exp * (random.random() - 0.5) * (max_num - min_num))
            num = max(min_num, min(num, max_num))

            if num not in predicted_numbers:
                predicted_numbers.append(num)

        # Ensure uniqueness and correct range
        predicted_numbers = list(set(predicted_numbers))
        predicted_numbers = [num for num in predicted_numbers if min_num <= num <= max_num]

        # If not enough unique numbers, fill with random selection
        if len(predicted_numbers) < total_numbers:
            remaining = set(range(min_num, max_num + 1)) - set(predicted_numbers)
            predicted_numbers += random.sample(list(remaining), total_numbers - len(predicted_numbers))

        return sorted(predicted_numbers[:total_numbers])

    except Exception as e:
        # Log the error (you might want to use a proper logging system)
        print(f"Error in topological_dynamic_system_predictor: {str(e)}")
        # Fall back to random number generation
        return random.sample(range(min_num, max_num + 1), total_numbers)