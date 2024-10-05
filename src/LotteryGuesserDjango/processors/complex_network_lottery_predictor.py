# complex_network_lottery_predictor.py

import random
from collections import defaultdict
from algorithms.models import lg_lottery_winner_number

def create_number_network(past_draws):
    """Create a network of numbers based on their co-occurrence in past draws."""
    network = defaultdict(lambda: defaultdict(int))
    for draw in past_draws:
        for i in range(len(draw)):
            for j in range(i + 1, len(draw)):
                network[draw[i]][draw[j]] += 1
                network[draw[j]][draw[i]] += 1
    return network

def calculate_node_strength(network):
    """Calculate the strength of each node in the network."""
    strength = defaultdict(int)
    for node, connections in network.items():
        strength[node] = sum(connections.values())
    return strength

def select_numbers_by_strength(strength, count, min_num, max_num):
    """Select numbers based on their strength in the network."""
    candidates = list(strength.items())
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = []
    for num, _ in candidates:
        if min_num <= num <= max_num and num not in selected:
            selected.append(num)
        if len(selected) == count:
            break
    return selected

def get_numbers(lottery_type_instance):
    try:
        min_num = int(lottery_type_instance.min_number)
        max_num = int(lottery_type_instance.max_number)
        total_numbers = int(lottery_type_instance.pieces_of_draw_numbers)

        # Retrieve past winning numbers
        past_draws = list(lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('-id')[:100].values_list('lottery_type_number', flat=True))

        if not past_draws:
            # If no past draws, return random numbers
            return random.sample(range(min_num, max_num + 1), total_numbers)

        # Create network from past draws
        network = create_number_network(past_draws)

        # Calculate node strengths
        strength = calculate_node_strength(network)

        # Select numbers based on network strength
        predicted_numbers = select_numbers_by_strength(strength, total_numbers, min_num, max_num)

        # If not enough numbers selected, fill with random selection
        if len(predicted_numbers) < total_numbers:
            remaining = set(range(min_num, max_num + 1)) - set(predicted_numbers)
            predicted_numbers += random.sample(list(remaining), total_numbers - len(predicted_numbers))

        return sorted(predicted_numbers)

    except Exception as e:
        # Log the error (you might want to use a proper logging system)
        print(f"Error in complex_network_lottery_predictor: {str(e)}")
        # Fall back to random number generation
        return random.sample(range(min_num, max_num + 1), total_numbers)