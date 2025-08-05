# causal_inference_prediction.py
"""
Causal Inference Predikció
Oksági kapcsolatok feltárása és használata lottószám predikcióhoz
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict
import random
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


class CausalInferencePredictor:
    """
    Causal Inference alapú predikció
    """
    
    def __init__(self):
        self.causal_graph = {}
        self.intervention_effects = {}
        self.confounders = {}
        self.temporal_causality = {}
        
    def get_numbers(self, lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
        """Főbejárási pont"""
        try:
            main_numbers = self._generate_causal_numbers(
                lottery_type_instance,
                int(lottery_type_instance.min_number),
                int(lottery_type_instance.max_number),
                int(lottery_type_instance.pieces_of_draw_numbers),
                True
            )
            
            additional_numbers = []
            if lottery_type_instance.has_additional_numbers:
                additional_numbers = self._generate_causal_numbers(
                    lottery_type_instance,
                    int(lottery_type_instance.additional_min_number),
                    int(lottery_type_instance.additional_max_number),
                    int(lottery_type_instance.additional_numbers_count),
                    False
                )
            
            return sorted(main_numbers), sorted(additional_numbers)
            
        except Exception as e:
            print(f"Hiba a causal_inference_prediction-ben: {e}")
            return self._generate_fallback_numbers(lottery_type_instance)
    
    def _generate_causal_numbers(self, lottery_type_instance: lg_lottery_type,
                               min_num: int, max_num: int, required_numbers: int,
                               is_main: bool) -> List[int]:
        """Causal inference alapú számgenerálás"""
        historical_data = self._get_historical_data(lottery_type_instance, is_main)
        
        if len(historical_data) < 20:
            return self._generate_smart_random(min_num, max_num, required_numbers)
        
        # Causal graph építése
        self._build_causal_graph(historical_data, min_num, max_num)
        
        # Intervention effects számítása
        self._calculate_intervention_effects(historical_data, min_num, max_num)
        
        # Temporal causality elemzése
        self._analyze_temporal_causality(historical_data, min_num, max_num)
        
        # Causal predikció
        causal_predictions = self._generate_causal_predictions(
            historical_data, min_num, max_num, required_numbers
        )
        
        return causal_predictions
    
    def _build_causal_graph(self, historical_data: List[List[int]], 
                          min_num: int, max_num: int) -> None:
        """Causal graph építése"""
        # Conditional independence tesztek
        for num1 in range(min_num, max_num + 1):
            self.causal_graph[num1] = {}
            
            for num2 in range(min_num, max_num + 1):
                if num1 != num2:
                    # Causal strength számítása
                    causal_strength = self._calculate_causal_strength(
                        num1, num2, historical_data
                    )
                    
                    if causal_strength > 0.1:  # Threshold
                        self.causal_graph[num1][num2] = causal_strength
    
    def _calculate_causal_strength(self, cause: int, effect: int, 
                                 historical_data: List[List[int]]) -> float:
        """Causal strength számítása"""
        # P(effect|cause) - P(effect|not cause)
        cause_present = []
        cause_absent = []
        
        for draw in historical_data:
            if cause in draw:
                cause_present.append(effect in draw)
            else:
                cause_absent.append(effect in draw)
        
        if not cause_present or not cause_absent:
            return 0.0
        
        p_effect_given_cause = sum(cause_present) / len(cause_present)
        p_effect_given_not_cause = sum(cause_absent) / len(cause_absent)
        
        return abs(p_effect_given_cause - p_effect_given_not_cause)
    
    def _calculate_intervention_effects(self, historical_data: List[List[int]],
                                      min_num: int, max_num: int) -> None:
        """Intervention effects számítása"""
        for num in range(min_num, max_num + 1):
            # Do-calculus alapú intervention
            intervention_effect = self._do_calculus(num, historical_data)
            self.intervention_effects[num] = intervention_effect
    
    def _do_calculus(self, intervention_num: int, 
                    historical_data: List[List[int]]) -> float:
        """Do-calculus számítás"""
        # Egyszerűsített do-calculus
        total_effect = 0.0
        
        for draw in historical_data:
            if intervention_num in draw:
                # Intervention hatása a többi számra
                for other_num in draw:
                    if other_num != intervention_num:
                        causal_effect = self.causal_graph.get(intervention_num, {}).get(other_num, 0)
                        total_effect += causal_effect
        
        return total_effect / len(historical_data) if historical_data else 0.0
    
    def _analyze_temporal_causality(self, historical_data: List[List[int]],
                                  min_num: int, max_num: int) -> None:
        """Temporal causality elemzése"""
        for i in range(len(historical_data) - 1):
            current_draw = historical_data[i]
            next_draw = historical_data[i + 1]
            
            for cause_num in current_draw:
                for effect_num in next_draw:
                    if cause_num not in self.temporal_causality:
                        self.temporal_causality[cause_num] = {}
                    
                    if effect_num not in self.temporal_causality[cause_num]:
                        self.temporal_causality[cause_num][effect_num] = 0
                    
                    self.temporal_causality[cause_num][effect_num] += 1
    
    def _generate_causal_predictions(self, historical_data: List[List[int]],
                                   min_num: int, max_num: int, 
                                   required_numbers: int) -> List[int]:
        """Causal predictions generálása"""
        # Causal scores számítása
        causal_scores = {}
        
        for num in range(min_num, max_num + 1):
            score = 0.0
            
            # Direct causal effects
            score += self.intervention_effects.get(num, 0) * 0.4
            
            # Temporal causality
            if num in self.temporal_causality:
                temporal_score = sum(self.temporal_causality[num].values())
                score += temporal_score * 0.3
            
            # Graph centrality
            centrality = len(self.causal_graph.get(num, {}))
            score += centrality * 0.3
            
            causal_scores[num] = score
        
        # Top scoring numbers
        sorted_scores = sorted(causal_scores.items(), key=lambda x: x[1], reverse=True)
        predictions = [num for num, _ in sorted_scores[:required_numbers]]
        
        return predictions
    
    def _generate_smart_random(self, min_num: int, max_num: int, count: int) -> List[int]:
        """Intelligens véletlen generálás"""
        numbers = set()
        while len(numbers) < count:
            num = random.randint(min_num, max_num)
            numbers.add(num)
        return list(numbers)
    
    def _get_historical_data(self, lottery_type_instance: lg_lottery_type, 
                           is_main: bool) -> List[List[int]]:
        """Történeti adatok lekérése"""
        field_name = 'lottery_type_number' if is_main else 'additional_numbers'
        
        try:
            queryset = lg_lottery_winner_number.objects.filter(
                lottery_type=lottery_type_instance
            ).order_by('-id').values_list(field_name, flat=True)[:50]
            
            historical_data = []
            for draw in queryset:
                if isinstance(draw, list) and len(draw) > 0:
                    valid_numbers = [int(num) for num in draw if isinstance(num, (int, float))]
                    if valid_numbers:
                        historical_data.append(valid_numbers)
            
            return historical_data
        except Exception:
            return []
    
    def _generate_fallback_numbers(self, lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
        """Fallback számgenerálás"""
        main_numbers = self._generate_smart_random(
            int(lottery_type_instance.min_number),
            int(lottery_type_instance.max_number),
            int(lottery_type_instance.pieces_of_draw_numbers)
        )
        
        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            additional_numbers = self._generate_smart_random(
                int(lottery_type_instance.additional_min_number),
                int(lottery_type_instance.additional_max_number),
                int(lottery_type_instance.additional_numbers_count)
            )
        
        return sorted(main_numbers), sorted(additional_numbers)


# Globális instance
causal_predictor = CausalInferencePredictor()

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Főbejárási pont"""
    return causal_predictor.get_numbers(lottery_type_instance)
