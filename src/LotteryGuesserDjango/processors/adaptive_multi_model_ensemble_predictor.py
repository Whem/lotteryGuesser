# adaptive_multi_model_ensemble_predictor.py

import random
import math
from collections import defaultdict
from algorithms.models import lg_lottery_winner_number

class SimpleModel:
    def predict(self, past_draws, min_num, max_num, total_numbers):
        pass

class FrequencyModel(SimpleModel):
    def predict(self, past_draws, min_num, max_num, total_numbers):
        frequency = defaultdict(int)
        for draw in past_draws:
            for num in draw:
                frequency[num] += 1
        sorted_nums = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:total_numbers]]

class GapModel(SimpleModel):
    def predict(self, past_draws, min_num, max_num, total_numbers):
        last_seen = {num: 0 for num in range(min_num, max_num + 1)}
        for i, draw in enumerate(reversed(past_draws)):
            for num in draw:
                if last_seen[num] == 0:
                    last_seen[num] = i
        sorted_nums = sorted(last_seen.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:total_numbers]]

class CycleModel(SimpleModel):
    def predict(self, past_draws, min_num, max_num, total_numbers):
        cycles = defaultdict(list)
        for i in range(len(past_draws) - 1):
            for num in past_draws[i]:
                if num in past_draws[i+1]:
                    cycles[num].append(i+1)
        avg_cycles = {num: sum(c)/len(c) if c else float('inf') for num, c in cycles.items()}
        sorted_nums = sorted(avg_cycles.items(), key=lambda x: x[1])
        return [num for num, _ in sorted_nums[:total_numbers]]

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

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

        # Initialize models
        models = [FrequencyModel(), GapModel(), CycleModel()]
        model_weights = [1/len(models)] * len(models)

        # Generate predictions from each model
        predictions = [model.predict(past_draws, min_num, max_num, total_numbers) for model in models]

        # Calculate model performance based on recent draws
        performance_window = min(10, len(past_draws) - 1)
        for i in range(performance_window):
            actual_draw = set(past_draws[i])
            for j, pred in enumerate(predictions):
                similarity = jaccard_similarity(set(pred), actual_draw)
                model_weights[j] *= (1 + similarity)

        # Normalize weights
        total_weight = sum(model_weights)
        model_weights = [w / total_weight for w in model_weights]

        # Combine predictions
        number_scores = defaultdict(float)
        for pred, weight in zip(predictions, model_weights):
            for num in pred:
                number_scores[num] += weight

        # Select top numbers
        sorted_numbers = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
        predicted_numbers = [num for num, _ in sorted_numbers[:total_numbers]]

        # If not enough unique numbers, fill with random selection
        if len(predicted_numbers) < total_numbers:
            remaining = set(range(min_num, max_num + 1)) - set(predicted_numbers)
            predicted_numbers += random.sample(list(remaining), total_numbers - len(predicted_numbers))

        return sorted(predicted_numbers)

    except Exception as e:
        # Log the error (you might want to use a proper logging system)
        print(f"Error in adaptive_multi_model_ensemble_predictor: {str(e)}")
        # Fall back to random number generation
        return random.sample(range(min_num, max_num + 1), total_numbers)