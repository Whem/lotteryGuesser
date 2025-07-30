# ultimate_hybrid_prediction.py
"""
Végső Hibrid Predikció
Kombinálja a legjobb algoritmusokat intelligens súlyozással
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import Counter, defaultdict
import random
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

class UltimateHybridPredictor:
    def __init__(self):
        self.methods = {
            'frequency': 0.3,
            'temporal': 0.25,
            'gap_analysis': 0.2,
            'statistical': 0.15,
            'pattern': 0.1
        }
    
    def get_numbers(self, lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
        try:
            main_numbers = self._predict_numbers(
                lottery_type_instance, True,
                int(lottery_type_instance.min_number),
                int(lottery_type_instance.max_number),
                int(lottery_type_instance.pieces_of_draw_numbers)
            )
            
            additional_numbers = []
            if lottery_type_instance.has_additional_numbers:
                additional_numbers = self._predict_numbers(
                    lottery_type_instance, False,
                    int(lottery_type_instance.additional_min_number),
                    int(lottery_type_instance.additional_max_number),
                    int(lottery_type_instance.additional_numbers_count)
                )
            
            return sorted(main_numbers), sorted(additional_numbers)
        except Exception:
            return self._fallback(lottery_type_instance)
    
    def _predict_numbers(self, lottery_type_instance, is_main, min_num, max_num, count):
        historical_data = self._get_data(lottery_type_instance, is_main)
        if len(historical_data) < 20:
            return self._smart_random(min_num, max_num, count)
        
        predictions = {}
        predictions['frequency'] = self._frequency_method(historical_data, min_num, max_num, count)
        predictions['temporal'] = self._temporal_method(historical_data, min_num, max_num, count)
        predictions['gap_analysis'] = self._gap_method(historical_data, min_num, max_num, count)
        predictions['statistical'] = self._statistical_method(historical_data, min_num, max_num, count)
        predictions['pattern'] = self._pattern_method(historical_data, min_num, max_num, count)
        
        return self._ensemble_vote(predictions, min_num, max_num, count)
    
    def _get_data(self, lottery_type_instance, is_main):
        field = 'lottery_type_number' if is_main else 'additional_numbers'
        queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('-id').values_list(field, flat=True)[:100]
        
        return [draw for draw in queryset if isinstance(draw, list) and len(draw) > 0]
    
    def _frequency_method(self, data, min_num, max_num, count):
        counter = Counter()
        for i, draw in enumerate(data):
            weight = 0.95 ** i
            for num in draw:
                if min_num <= num <= max_num:
                    counter[num] += weight
        return [num for num, _ in counter.most_common(count)]
    
    def _temporal_method(self, data, min_num, max_num, count):
        if len(data) < 10:
            return self._smart_random(min_num, max_num, count)
        
        recent = data[:10]
        older = data[10:20] if len(data) >= 20 else []
        
        recent_freq = Counter(num for draw in recent for num in draw)
        older_freq = Counter(num for draw in older for num in draw) if older else Counter()
        
        trends = {}
        for num in range(min_num, max_num + 1):
            r_count = recent_freq.get(num, 0)
            o_count = older_freq.get(num, 0)
            trends[num] = r_count - o_count * 0.5
        
        return sorted(trends.keys(), key=lambda x: trends[x], reverse=True)[:count]
    
    def _gap_method(self, data, min_num, max_num, count):
        last_seen = {}
        for i, draw in enumerate(data):
            for num in draw:
                if min_num <= num <= max_num:
                    last_seen[num] = i
        
        gaps = {}
        for num in range(min_num, max_num + 1):
            gaps[num] = last_seen.get(num, len(data))
        
        return sorted(gaps.keys(), key=lambda x: gaps[x], reverse=True)[:count]
    
    def _statistical_method(self, data, min_num, max_num, count):
        all_nums = [num for draw in data for num in draw if min_num <= num <= max_num]
        if not all_nums:
            return self._smart_random(min_num, max_num, count)
        
        mean_val = np.mean(all_nums)
        std_val = np.std(all_nums)
        
        scores = {}
        for num in range(min_num, max_num + 1):
            z_score = abs(num - mean_val) / max(std_val, 1)
            scores[num] = 1 / (1 + z_score)  # Inverse z-score
        
        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:count]
    
    def _pattern_method(self, data, min_num, max_num, count):
        if len(data) < 5:
            return self._smart_random(min_num, max_num, count)
        
        patterns = defaultdict(list)
        for i in range(len(data) - 1):
            current = sorted(data[i])
            next_draw = data[i + 1]
            
            # Simple pattern: sum category
            sum_cat = 'low' if sum(current) < 150 else 'high'
            patterns[sum_cat].extend(next_draw)
        
        recent_sum = sum(data[0])
        recent_cat = 'low' if recent_sum < 150 else 'high'
        
        if recent_cat in patterns:
            candidates = [num for num in patterns[recent_cat] if min_num <= num <= max_num]
            counter = Counter(candidates)
            return [num for num, _ in counter.most_common(count)]
        
        return self._smart_random(min_num, max_num, count)
    
    def _ensemble_vote(self, predictions, min_num, max_num, count):
        votes = defaultdict(float)
        
        for method, numbers in predictions.items():
            weight = self.methods.get(method, 0.1)
            for i, num in enumerate(numbers[:count]):
                pos_weight = 1.0 / (i + 1)
                votes[num] += weight * pos_weight
        
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        selected = [num for num, _ in sorted_votes[:count]]
        
        if len(selected) < count:
            remaining = [n for n in range(min_num, max_num + 1) if n not in selected]
            random.shuffle(remaining)
            selected.extend(remaining[:count - len(selected)])
        
        return selected[:count]
    
    def _smart_random(self, min_num, max_num, count):
        center = (min_num + max_num) / 2
        std = (max_num - min_num) / 6
        
        numbers = set()
        for _ in range(count * 3):
            num = int(np.random.normal(center, std))
            if min_num <= num <= max_num:
                numbers.add(num)
            if len(numbers) >= count:
                break
        
        if len(numbers) < count:
            remaining = [n for n in range(min_num, max_num + 1) if n not in numbers]
            random.shuffle(remaining)
            numbers.update(remaining[:count - len(numbers)])
        
        return list(numbers)[:count]
    
    def _fallback(self, lottery_type_instance):
        main = self._smart_random(
            int(lottery_type_instance.min_number),
            int(lottery_type_instance.max_number),
            int(lottery_type_instance.pieces_of_draw_numbers)
        )
        
        additional = []
        if lottery_type_instance.has_additional_numbers:
            additional = self._smart_random(
                int(lottery_type_instance.additional_min_number),
                int(lottery_type_instance.additional_max_number),
                int(lottery_type_instance.additional_numbers_count)
            )
        
        return sorted(main), sorted(additional)

predictor = UltimateHybridPredictor()

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    return predictor.get_numbers(lottery_type_instance)
