# adaptive_multi_model_ensemble_predictor.py
import random
import math
from typing import List, Tuple, Set
from collections import defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


class BaseModel:
    def predict(self, past_draws: List[List[int]], min_num: int, max_num: int, total_numbers: int) -> List[int]:
        """Base prediction method."""
        pass


class FrequencyModel(BaseModel):
    def predict(self, past_draws: List[List[int]], min_num: int, max_num: int, total_numbers: int) -> List[int]:
        """Predict based on number frequency."""
        frequency = defaultdict(int)
        for draw in past_draws:
            for num in draw:
                frequency[num] += 1

        # Add weights for numbers that haven't appeared
        for num in range(min_num, max_num + 1):
            if num not in frequency:
                frequency[num] = 0

        sorted_nums = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:total_numbers]]


class GapModel(BaseModel):
    def predict(self, past_draws: List[List[int]], min_num: int, max_num: int, total_numbers: int) -> List[int]:
        """Predict based on number gaps."""
        last_seen = {num: float('inf') for num in range(min_num, max_num + 1)}
        for i, draw in enumerate(reversed(past_draws)):
            for num in draw:
                if last_seen[num] == float('inf'):
                    last_seen[num] = i

        sorted_nums = sorted(last_seen.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:total_numbers]]


class CycleModel(BaseModel):
    def predict(self, past_draws: List[List[int]], min_num: int, max_num: int, total_numbers: int) -> List[int]:
        """Predict based on number cycles."""
        cycles = defaultdict(list)
        for i in range(len(past_draws) - 1):
            for num in past_draws[i]:
                if num in past_draws[i + 1]:
                    cycles[num].append(i + 1)

        # Calculate average cycle length for each number
        avg_cycles = {}
        for num in range(min_num, max_num + 1):
            if num in cycles and cycles[num]:
                avg_cycles[num] = sum(cycles[num]) / len(cycles[num])
            else:
                avg_cycles[num] = float('inf')

        sorted_nums = sorted(avg_cycles.items(), key=lambda x: x[1])
        return [num for num, _ in sorted_nums[:total_numbers]]


class PatternModel(BaseModel):
    def predict(self, past_draws: List[List[int]], min_num: int, max_num: int, total_numbers: int) -> List[int]:
        """Predict based on number patterns."""
        if not past_draws or len(past_draws) < 2:
            return random.sample(range(min_num, max_num + 1), total_numbers)

        # Analyze patterns in consecutive draws
        patterns = defaultdict(int)
        for i in range(len(past_draws) - 1):
            for num1 in past_draws[i]:
                for num2 in past_draws[i + 1]:
                    diff = abs(num2 - num1)
                    patterns[diff] += 1

        # Find most common differences
        common_diffs = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:3]

        # Generate numbers based on patterns
        predictions = set()
        last_draw = past_draws[0]
        for base_num in last_draw:
            for diff, _ in common_diffs:
                candidates = [
                    base_num + diff if base_num + diff <= max_num else base_num - diff,
                    base_num - diff if base_num - diff >= min_num else base_num + diff
                ]
                for num in candidates:
                    if min_num <= num <= max_num:
                        predictions.add(num)

        return list(predictions)[:total_numbers]


def jaccard_similarity(set1: Set[int], set2: Set[int]) -> float:
    """Calculate Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate predictions for both main and additional numbers using ensemble approach.
    Returns (main_numbers, additional_numbers).
    """
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
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        total_numbers: int,
        is_main: bool
) -> List[int]:
    """Generate a set of numbers using ensemble prediction."""
    try:
        # Get past draws
        past_draws = list(lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('-id')[:100])

        # Extract appropriate number sets
        if is_main:
            past_numbers = [draw.lottery_type_number for draw in past_draws
                            if isinstance(draw.lottery_type_number, list)]
        else:
            past_numbers = [draw.additional_numbers for draw in past_draws
                            if hasattr(draw, 'additional_numbers') and
                            isinstance(draw.additional_numbers, list)]

        if len(past_numbers) < 10:
            return random.sample(range(min_num, max_num + 1), total_numbers)

        # Initialize models with weights
        models = [
            FrequencyModel(),
            GapModel(),
            CycleModel(),
            PatternModel()
        ]
        model_weights = [1 / len(models)] * len(models)

        # Generate predictions from each model
        predictions = []
        for model in models:
            try:
                pred = model.predict(past_numbers, min_num, max_num, total_numbers)
                if len(pred) >= total_numbers:
                    predictions.append(pred[:total_numbers])
                else:
                    # Fill missing numbers randomly
                    remaining = set(range(min_num, max_num + 1)) - set(pred)
                    pred.extend(random.sample(list(remaining), total_numbers - len(pred)))
                    predictions.append(pred)
            except Exception as e:
                print(f"Model prediction error: {str(e)}")
                predictions.append(random.sample(range(min_num, max_num + 1), total_numbers))

        # Evaluate model performance
        performance_window = min(10, len(past_numbers) - 1)
        for i in range(performance_window):
            actual_draw = set(past_numbers[i])
            for j, pred in enumerate(predictions):
                similarity = jaccard_similarity(set(pred), actual_draw)
                model_weights[j] *= (1 + similarity)

        # Normalize weights
        total_weight = sum(model_weights)
        if total_weight > 0:
            model_weights = [w / total_weight for w in model_weights]
        else:
            model_weights = [1 / len(models)] * len(models)

        # Combine predictions with weights
        number_scores = defaultdict(float)
        for pred, weight in zip(predictions, model_weights):
            for num in pred:
                number_scores[num] += weight

        # Select final numbers
        sorted_numbers = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
        final_numbers = [num for num, _ in sorted_numbers[:total_numbers]]

        # Ensure we have enough unique numbers
        if len(final_numbers) < total_numbers:
            remaining = set(range(min_num, max_num + 1)) - set(final_numbers)
            final_numbers.extend(random.sample(list(remaining), total_numbers - len(final_numbers)))

        return sorted(final_numbers)

    except Exception as e:
        print(f"Error in ensemble predictor: {str(e)}")
        return random.sample(range(min_num, max_num + 1), total_numbers)