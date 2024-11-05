# complex_network_lottery_predictor.py

import random
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using complex network theory.
    Returns a single list containing both main and additional numbers.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A list containing predicted numbers (main numbers followed by additional numbers if applicable).
    """
    try:
        # Generate main numbers
        main_numbers = generate_numbers(
            lottery_type_instance=lottery_type_instance,
            number_field='lottery_type_number',
            min_num=lottery_type_instance.min_number,
            max_num=lottery_type_instance.max_number,
            total_numbers=lottery_type_instance.pieces_of_draw_numbers
        )

        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            # Generate additional numbers
            additional_numbers = generate_numbers(
                lottery_type_instance=lottery_type_instance,
                number_field='additional_numbers',
                min_num=lottery_type_instance.additional_min_number,
                max_num=lottery_type_instance.additional_max_number,
                total_numbers=lottery_type_instance.additional_numbers_count
            )

        # Return combined list
        return main_numbers , additional_numbers

    except Exception as e:
        print(f"Error in get_numbers: {str(e)}")
        return generate_random_numbers(
            lottery_type_instance.min_number,
            lottery_type_instance.max_number,
            lottery_type_instance.pieces_of_draw_numbers
        ), generate_random_numbers(
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count
        )


def generate_numbers(
        lottery_type_instance: lg_lottery_type,
        number_field: str,
        min_num: int,
        max_num: int,
        total_numbers: int
) -> List[int]:
    """
    Generates numbers using complex network analysis for either main or additional numbers.
    """
    try:
        # Retrieve past winning numbers
        past_draws_queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('-id')[:100]  # Limit to last 100 draws for efficiency

        past_draws = []
        for draw in past_draws_queryset:
            numbers = getattr(draw, number_field, None)
            if isinstance(numbers, list) and len(numbers) == total_numbers:
                try:
                    past_draws.append([int(num) for num in numbers])
                except (ValueError, TypeError):
                    continue

        if len(past_draws) < 5:  # Need minimum data for network analysis
            return generate_random_numbers(min_num, max_num, total_numbers)

        # Create co-occurrence network from past draws
        network = create_number_network(past_draws)

        # Calculate node strengths
        strength = calculate_node_strength(network)

        # Calculate edge weights for number pairs
        edge_weights = calculate_edge_weights(network)

        # Select numbers based on network analysis
        predicted_numbers = select_numbers_by_network_metrics(
            strength=strength,
            edge_weights=edge_weights,
            total_numbers=total_numbers,
            min_num=min_num,
            max_num=max_num
        )

        # If not enough numbers selected, fill with weighted random selection
        if len(predicted_numbers) < total_numbers:
            remaining = set(range(min_num, max_num + 1)) - set(predicted_numbers)
            weights = calculate_weights_from_strength(strength, remaining)

            while len(predicted_numbers) < total_numbers and remaining:
                selected = weighted_random_choice(weights, remaining)
                if selected is not None:
                    predicted_numbers.append(selected)
                    remaining.remove(selected)
                    weights.pop(selected, None)

        return sorted(predicted_numbers)[:total_numbers]

    except Exception as e:
        print(f"Error in generate_numbers: {str(e)}")
        return generate_random_numbers(min_num, max_num, total_numbers)


def create_number_network(past_draws: List[List[int]]) -> Dict[int, Dict[int, int]]:
    """Creates a network of numbers based on their co-occurrence in past draws."""
    network = defaultdict(lambda: defaultdict(int))
    try:
        for draw in past_draws:
            for i in range(len(draw)):
                for j in range(i + 1, len(draw)):
                    network[draw[i]][draw[j]] += 1
                    network[draw[j]][draw[i]] += 1
    except Exception as e:
        print(f"Error in create_number_network: {str(e)}")
    return network


def calculate_node_strength(network: Dict[int, Dict[int, int]]) -> Dict[int, float]:
    """Calculates the strength (weighted degree) of each node in the network."""
    strength = defaultdict(float)
    try:
        for node, connections in network.items():
            strength[node] = sum(connections.values())
    except Exception as e:
        print(f"Error in calculate_node_strength: {str(e)}")
    return strength


def calculate_edge_weights(network: Dict[int, Dict[int, int]]) -> Dict[tuple, float]:
    """Calculates normalized weights for edges between number pairs."""
    edge_weights = {}
    try:
        max_weight = max(max(weights.values()) for weights in network.values())
        if max_weight > 0:
            for node1, connections in network.items():
                for node2, weight in connections.items():
                    edge_weights[(node1, node2)] = weight / max_weight
    except Exception as e:
        print(f"Error in calculate_edge_weights: {str(e)}")
    return edge_weights


def select_numbers_by_network_metrics(
        strength: Dict[int, float],
        edge_weights: Dict[tuple, float],
        total_numbers: int,
        min_num: int,
        max_num: int
) -> List[int]:
    """Selects numbers based on both node strength and edge weights."""
    selected = []
    try:
        # Sort numbers by their network strength
        candidates = [(num, str_val) for num, str_val in strength.items()
                      if min_num <= num <= max_num]
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Select first number based on highest strength
        if candidates:
            selected.append(candidates[0][0])

        # Select remaining numbers based on combination of strength and connectivity
        while len(selected) < total_numbers and candidates:
            best_score = -1
            best_number = None

            for num, str_val in candidates:
                if num in selected:
                    continue

                # Calculate connectivity score with already selected numbers
                connectivity_score = sum(edge_weights.get((num, selected_num), 0) +
                                         edge_weights.get((selected_num, num), 0)
                                         for selected_num in selected)

                # Combine strength and connectivity scores
                total_score = (str_val + connectivity_score) / 2

                if total_score > best_score:
                    best_score = total_score
                    best_number = num

            if best_number is not None:
                selected.append(best_number)
            else:
                break

    except Exception as e:
        print(f"Error in select_numbers_by_network_metrics: {str(e)}")

    return selected


def calculate_weights_from_strength(
        strength: Dict[int, float],
        available_numbers: Set[int]
) -> Dict[int, float]:
    """Calculates selection weights based on network strength."""
    weights = {}
    try:
        max_strength = max(strength.values()) if strength else 1.0
        for num in available_numbers:
            weights[num] = (strength.get(num, 0) / max_strength) if max_strength > 0 else 1.0
    except Exception as e:
        print(f"Error in calculate_weights_from_strength: {str(e)}")
        weights = {num: 1.0 for num in available_numbers}
    return weights


def weighted_random_choice(weights: Dict[int, float], available_numbers: Set[int]) -> int:
    """Selects a random number based on weights."""
    try:
        if not available_numbers:
            return None

        numbers = list(available_numbers)
        if not numbers:
            return None

        weights_list = [weights.get(num, 0.0) for num in numbers]
        total_weight = sum(weights_list)

        if total_weight <= 0:
            return random.choice(numbers)

        weights_list = [w / total_weight for w in weights_list]
        return random.choices(numbers, weights=weights_list, k=1)[0]

    except Exception as e:
        print(f"Error in weighted_random_choice: {str(e)}")
        return random.choice(list(available_numbers)) if available_numbers else None


def generate_random_numbers(min_num: int, max_num: int, total_numbers: int) -> List[int]:
    """Generates random numbers safely."""
    try:
        if max_num < min_num or total_numbers <= 0:
            return []

        numbers = set()
        available_range = list(range(min_num, max_num + 1))

        if total_numbers > len(available_range):
            total_numbers = len(available_range)

        while len(numbers) < total_numbers:
            numbers.add(random.choice(available_range))

        return sorted(list(numbers))
    except Exception as e:
        print(f"Error in generate_random_numbers: {str(e)}")
        return list(range(min_num, min(min_num + total_numbers, max_num + 1)))