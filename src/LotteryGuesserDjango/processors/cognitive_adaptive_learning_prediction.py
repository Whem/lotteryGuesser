# cognitive_adaptive_learning_prediction.py
# Advanced cognitive pattern recognition with adaptive learning
# Combines cognitive science principles with reinforcement learning

import numpy as np
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import statistics
from scipy.optimize import minimize
import random
import math
from itertools import combinations, permutations
from sklearn.preprocessing import StandardScaler


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    """
    Cognitive adaptive learning algorithm that combines pattern recognition
    with dynamic reinforcement and memory-based learning.
    """
    # Get historical data with extended window for better learning
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True).order_by('-id')[:300])

    past_draws = [draw for draw in past_draws if isinstance(draw, list)]
    if not past_draws:
        return generate_random_numbers(lottery_type_instance)

    required_numbers = lottery_type_instance.pieces_of_draw_numbers
    candidates = set()

    # 1. Cognitive Pattern Recognition
    cognitive_numbers = analyze_cognitive_patterns(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    candidates.update(cognitive_numbers[:required_numbers // 3])

    # 2. Memory-Based Learning
    memory_numbers = perform_memory_learning(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    candidates.update(memory_numbers[:required_numbers // 3])

    # 3. Reinforcement Learning Analysis
    reinforcement_numbers = analyze_reinforcement_patterns(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number
    )
    candidates.update(reinforcement_numbers[:required_numbers // 3])

    # Fill remaining slots using adaptive cognitive weights
    while len(candidates) < required_numbers:
        weights = calculate_cognitive_weights(
            past_draws,
            lottery_type_instance.min_number,
            lottery_type_instance.max_number,
            candidates
        )

        available_numbers = set(range(
            lottery_type_instance.min_number,
            lottery_type_instance.max_number + 1
        )) - candidates

        if available_numbers:
            number_weights = [weights.get(num, 1.0) for num in available_numbers]
            selected = random.choices(list(available_numbers), weights=number_weights, k=1)[0]
            candidates.add(selected)

    return sorted(list(candidates))[:required_numbers]


def analyze_cognitive_patterns(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze cognitive patterns using psychological principles."""
    cognitive_scores = defaultdict(float)

    # Create cognitive memory representation
    short_term_memory = past_draws[:5]  # Recent draws
    medium_term_memory = past_draws[5:20]  # Medium-term patterns
    long_term_memory = past_draws[20:]  # Long-term patterns

    # Analyze gestalt patterns (proximity, similarity, continuity)
    for memory_set, weight in [(short_term_memory, 1.0),
                               (medium_term_memory, 0.6),
                               (long_term_memory, 0.3)]:
        for draw in memory_set:
            sorted_nums = sorted(draw)

            # Proximity analysis
            for i in range(len(sorted_nums) - 1):
                gap = sorted_nums[i + 1] - sorted_nums[i]
                next_in_sequence = sorted_nums[i + 1] + gap
                if min_num <= next_in_sequence <= max_num:
                    cognitive_scores[next_in_sequence] += weight * 0.5

            # Similarity analysis
            for num1, num2 in combinations(sorted_nums, 2):
                similarity = 1 / (1 + abs(num1 - num2))
                midpoint = (num1 + num2) // 2
                if min_num <= midpoint <= max_num:
                    cognitive_scores[midpoint] += weight * similarity

            # Continuity analysis
            if len(sorted_nums) >= 3:
                for i in range(len(sorted_nums) - 2):
                    trend = (sorted_nums[i + 1] - sorted_nums[i] +
                             sorted_nums[i + 2] - sorted_nums[i + 1]) / 2
                    next_value = sorted_nums[i + 2] + trend
                    if min_num <= next_value <= max_num:
                        cognitive_scores[int(next_value)] += weight * 0.7

    # Analyze cognitive biases
    anchoring_points = get_anchoring_points(past_draws)
    for anchor in anchoring_points:
        # Analyze numbers around cognitive anchors
        for offset in [-2, -1, 1, 2]:
            biased_num = anchor + offset
            if min_num <= biased_num <= max_num:
                cognitive_scores[biased_num] += 0.5 / abs(offset)

    return sorted(cognitive_scores.keys(), key=cognitive_scores.get, reverse=True)


def perform_memory_learning(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Implement memory-based learning analysis."""
    memory_scores = defaultdict(float)

    # Create memory layers
    memory_layers = {
        'episodic': past_draws[:10],  # Recent specific events
        'semantic': build_semantic_memory(past_draws),  # General patterns
        'procedural': extract_procedures(past_draws)  # Number generation rules
    }

    # Analyze episodic memory
    for i, draw in enumerate(memory_layers['episodic']):
        recency_weight = 1 / (i + 1)

        # Look for episodic patterns
        sorted_nums = sorted(draw)
        for j in range(len(sorted_nums) - 1):
            pattern = sorted_nums[j + 1] - sorted_nums[j]
            next_num = sorted_nums[j + 1] + pattern
            if min_num <= next_num <= max_num:
                memory_scores[next_num] += recency_weight

    # Analyze semantic memory
    semantic_patterns = memory_layers['semantic']
    for pattern, frequency in semantic_patterns.items():
        if isinstance(pattern, tuple) and len(pattern) >= 2:
            next_value = extrapolate_pattern(pattern, frequency)
            if min_num <= next_value <= max_num:
                memory_scores[next_value] += frequency

    # Apply procedural memory
    procedures = memory_layers['procedural']
    for procedure in procedures:
        try:
            predicted = apply_procedure(procedure, past_draws[-1], min_num, max_num)
            for num in predicted:
                if min_num <= num <= max_num:
                    memory_scores[num] += procedure['success_rate']
        except:
            continue

    return sorted(memory_scores.keys(), key=memory_scores.get, reverse=True)


def analyze_reinforcement_patterns(past_draws: List[List[int]], min_num: int, max_num: int) -> List[int]:
    """Analyze patterns using reinforcement learning principles."""
    reinforcement_scores = defaultdict(float)

    # Create state-action pairs
    state_actions = build_state_actions(past_draws)

    # Q-learning parameters
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor

    # Train Q-values
    Q_values = defaultdict(float)
    for state, actions in state_actions.items():
        for action in actions:
            # Calculate reward based on success in next draws
            reward = calculate_action_reward(action, past_draws)

            # Update Q-value
            current_q = Q_values[(state, action)]
            next_state = get_next_state(state, action)
            next_actions = state_actions.get(next_state, [])
            max_next_q = max([Q_values[(next_state, a)] for a in next_actions]) if next_actions else 0

            Q_values[(state, action)] = current_q + alpha * (reward + gamma * max_next_q - current_q)

    # Use Q-values to score numbers
    current_state = get_current_state(past_draws[-1] if past_draws else [])

    for action in range(min_num, max_num + 1):
        q_value = Q_values[(current_state, action)]
        reinforcement_scores[action] += q_value

        # Consider action sequences
        next_state = get_next_state(current_state, action)
        next_actions = state_actions.get(next_state, [])
        if next_actions:
            future_value = max([Q_values[(next_state, a)] for a in next_actions])
            reinforcement_scores[action] += gamma * future_value

    return sorted(reinforcement_scores.keys(), key=reinforcement_scores.get, reverse=True)


def build_semantic_memory(past_draws: List[List[int]]) -> Dict[Tuple[int, ...], float]:
    """Build semantic memory from past draws."""
    patterns = defaultdict(float)

    for i in range(len(past_draws) - 1):
        current = past_draws[i]
        next_draw = past_draws[i + 1]

        # Find recurring patterns
        for size in [2, 3, 4]:
            for combo in combinations(sorted(current), size):
                if any(n in next_draw for n in combo):
                    patterns[combo] += 1 / (i + 1)

    return patterns


def extract_procedures(past_draws: List[List[int]]) -> List[Dict]:
    """Extract number generation procedures from past data."""
    procedures = []

    # Look for successful procedures
    for i in range(len(past_draws) - 1):
        current = past_draws[i]
        next_draw = past_draws[i + 1]

        # Test different procedures
        potential_procedures = [
                                   {'type': 'increment', 'value': d} for d in range(1, 4)
                               ] + [
                                   {'type': 'multiply', 'value': m} for m in [1.5, 2, 3]
                               ] + [
                                   {'type': 'combine', 'operations': ['add', 'subtract']}
                               ]

        for proc in potential_procedures:
            success = test_procedure(proc, current, next_draw)
            if success > 0:
                proc['success_rate'] = success
                procedures.append(proc)

    return procedures


def calculate_cognitive_weights(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        excluded_numbers: Set[int]
) -> Dict[int, float]:
    """Calculate weights using cognitive principles."""
    weights = defaultdict(float)

    # Extract various cognitive features
    frequency = Counter(num for draw in past_draws for num in draw)
    total_numbers = sum(frequency.values())

    # Recent history analysis
    recent_draws = past_draws[:10]
    recent_frequency = Counter(num for draw in recent_draws for num in draw)

    for num in range(min_num, max_num + 1):
        if num in excluded_numbers:
            continue

        # Base probability from overall frequency
        base_weight = frequency.get(num, 0) / total_numbers

        # Recency bias
        recency_weight = recent_frequency.get(num, 0) / len(recent_draws) if recent_draws else 0

        # Cognitive anchoring
        anchor_points = get_anchoring_points(past_draws)
        anchor_weight = sum(1 / (1 + abs(num - anchor)) for anchor in anchor_points)

        # Pattern completion tendency
        pattern_weight = calculate_pattern_completion_weight(num, past_draws)

        # Combine weights with cognitive biases
        weights[num] = (
                0.3 * base_weight +
                0.3 * recency_weight +
                0.2 * anchor_weight +
                0.2 * pattern_weight
        )

    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        for num in weights:
            weights[num] /= total_weight

    return weights


def get_anchoring_points(past_draws: List[List[int]]) -> List[int]:
    """Identify cognitive anchoring points."""
    if not past_draws:
        return []

    all_numbers = [num for draw in past_draws for num in draw]
    if not all_numbers:
        return []

    anchors = []

    # Statistical anchors
    try:
        anchors.extend([
            int(statistics.mean(all_numbers)),
            int(statistics.median(all_numbers)),
            statistics.mode(all_numbers)
        ])
    except:
        pass

    # Psychological anchors (endpoints, midpoints)
    if past_draws:
        last_draw = past_draws[0]
        if last_draw:
            anchors.extend([min(last_draw), max(last_draw)])
            anchors.append(sum(last_draw) // len(last_draw))

    return list(set(anchors))


def test_procedure(procedure: Dict, current: List[int], next_draw: List[int]) -> float:
    """Test success rate of a procedure."""
    try:
        predicted = apply_procedure(procedure, current, min(next_draw), max(next_draw))
        return len(set(predicted) & set(next_draw)) / len(next_draw)
    except:
        return 0


def apply_procedure(procedure: Dict, numbers: List[int], min_num: int, max_num: int) -> List[int]:
    """Apply a procedure to generate new numbers."""
    result = []

    if procedure['type'] == 'increment':
        result = [n + procedure['value'] for n in numbers]
    elif procedure['type'] == 'multiply':
        result = [int(n * procedure['value']) for n in numbers]
    elif procedure['type'] == 'combine':
        for op in procedure['operations']:
            if op == 'add':
                result.extend([a + b for a, b in combinations(numbers, 2)])
            elif op == 'subtract':
                result.extend([abs(a - b) for a, b in permutations(numbers, 2)])

    return [n for n in result if min_num <= n <= max_num]


def calculate_pattern_completion_weight(num: int, past_draws: List[List[int]]) -> float:
    """Calculate weight based on pattern completion tendency."""
    if not past_draws:
        return 0

    weight = 0
    last_draw = past_draws[0]

    if last_draw:
        # Look for arithmetic sequences
        diffs = [num - n for n in last_draw]
        weight += sum(1 / (1 + abs(d)) for d in diffs) / len(last_draw)

        # Look for geometric relationships
        for n in last_draw:
            if n != 0:
                ratio = num / n
                weight += 1 / (1 + abs(ratio - round(ratio)))

    return weight


def generate_random_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    """Generate random numbers with cognitive biases when no historical data is available."""
    numbers = set()
    required_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Create cognitively biased ranges
    ranges = [
        (lottery_type_instance.min_number, lottery_type_instance.max_number // 3),  # Lower third
        (lottery_type_instance.max_number // 3, 2 * lottery_type_instance.max_number // 3),  # Middle third
        (2 * lottery_type_instance.max_number // 3, lottery_type_instance.max_number)  # Upper third
    ]

    # Distribution weights (people tend to pick middle numbers more often)
    range_weights = [0.3, 0.4, 0.3]

    while len(numbers) < required_numbers:
        # Select range based on cognitive bias
        selected_range = random.choices(ranges, weights=range_weights, k=1)[0]

        # Generate number within range
        number = random.randint(selected_range[0], selected_range[1])

        # Apply additional cognitive biases
        if random.random() < 0.3:  # 30% chance to pick "special" numbers
            if random.random() < 0.5:
                number = round(number, -1)  # Round to nearest 10
            else:
                number = round(number / 5) * 5  # Round to nearest 5

        if min_num <= number <= max_num and number not in numbers:
            numbers.add(number)

    return sorted(list(numbers))


def build_state_actions(past_draws: List[List[int]]) -> Dict[str, List[int]]:
    """Build state-action pairs for reinforcement learning."""
    state_actions = defaultdict(list)

    for i in range(len(past_draws) - 1):
        current_state = get_current_state(past_draws[i])
        next_draw = past_draws[i + 1]

        # Add successful actions to state
        state_actions[current_state].extend(next_draw)

        # Add derived actions
        for num1, num2 in combinations(sorted(next_draw), 2):
            derived = num2 - num1
            if derived > 0:
                state_actions[current_state].append(derived)

    # Remove duplicates and sort
    for state in state_actions:
        state_actions[state] = sorted(set(state_actions[state]))

    return state_actions


def get_current_state(numbers: List[int]) -> str:
    """Convert current numbers to state representation."""
    if not numbers:
        return "empty"

    # Create state features
    features = {
        'sum': sum(numbers),
        'mean': statistics.mean(numbers),
        'std': statistics.stdev(numbers) if len(numbers) > 1 else 0,
        'min': min(numbers),
        'max': max(numbers)
    }

    # Discretize features
    discretized = []
    for value in features.values():
        if isinstance(value, (int, float)):
            discretized.append(str(round(value, -1)))  # Round to nearest 10

    return "_".join(discretized)


def get_next_state(current_state: str, action: int) -> str:
    """Predict next state given current state and action."""
    if current_state == "empty":
        return str(action)

    try:
        # Parse current state
        state_values = [float(x) for x in current_state.split('_')]

        # Update state features with new action
        new_values = [
            state_values[0] + action,  # Update sum
            (state_values[1] * (len(state_values) - 1) + action) / len(state_values),  # Update mean
            state_values[2],  # Keep std (simplified)
            min(state_values[3], action),  # Update min
            max(state_values[4], action)  # Update max
        ]

        # Discretize new state
        return "_".join(str(round(v, -1)) for v in new_values)
    except:
        return current_state


def calculate_action_reward(action: int, past_draws: List[List[int]]) -> float:
    """Calculate reward for an action based on historical success."""
    if not past_draws:
        return 0

    reward = 0
    for draw in past_draws:
        if action in draw:
            reward += 1
            # Additional reward for correct position
            sorted_draw = sorted(draw)
            action_pos = sorted_draw.index(action)
            reward += 0.1 * (len(draw) - action_pos)  # Higher reward for early positions

    return reward / len(past_draws)


def extrapolate_pattern(pattern: Tuple[int, ...], frequency: float) -> int:
    """Extrapolate next number from a pattern."""
    if len(pattern) < 2:
        return pattern[0] if pattern else 0

    # Try different pattern types
    differences = [pattern[i + 1] - pattern[i] for i in range(len(pattern) - 1)]

    if len(set(differences)) == 1:
        # Arithmetic sequence
        return pattern[-1] + differences[0]

    ratios = []
    for i in range(len(pattern) - 1):
        if pattern[i] != 0:
            ratios.append(pattern[i + 1] / pattern[i])

    if ratios and len(set(ratios)) == 1:
        # Geometric sequence
        return int(pattern[-1] * ratios[0])

    # Fibonacci-like sequence
    if len(pattern) >= 3 and all(pattern[i + 2] == pattern[i + 1] + pattern[i] for i in range(len(pattern) - 2)):
        return pattern[-1] + pattern[-2]

    # Default to linear extrapolation
    return int(pattern[-1] + frequency * (pattern[-1] - pattern[0]) / len(pattern))


def calculate_pattern_strength(pattern: List[int], past_draws: List[List[int]]) -> float:
    """Calculate the strength of a pattern based on historical data."""
    if not pattern or not past_draws:
        return 0

    strength = 0
    pattern_set = set(pattern)

    for i, draw in enumerate(past_draws):
        draw_set = set(draw)
        intersection = pattern_set & draw_set

        if intersection:
            # Award points based on number of matching elements and recency
            strength += len(intersection) / len(pattern) * (1 / (i + 1))

            # Additional points for matching positions
            sorted_pattern = sorted(pattern)
            sorted_draw = sorted(draw)
            for num in intersection:
                pattern_pos = sorted_pattern.index(num)
                draw_pos = sorted_draw.index(num)
                if pattern_pos == draw_pos:
                    strength += 0.5 / (i + 1)

    return strength