# adaptive_neuro_fuzzy_predictor.py
"""
Adaptive Neuro-Fuzzy Predictor
Adaptív neuro-fuzzy rendszer ANFIS-szerű megközelítéssel fuzzy logika és neurális hálózatok kombinálásával
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from collections import defaultdict, deque
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FuzzySet:
    """Fuzzy halmaz osztály."""
    
    def __init__(self, name: str, params: Dict[str, float], membership_type: str = "gaussian"):
        self.name = name
        self.params = params
        self.membership_type = membership_type
    
    def membership_degree(self, x: float) -> float:
        """Tagsági fok számítása."""
        
        if self.membership_type == "gaussian":
            # Gauss-féle tagsági függvény
            center = self.params.get('center', 0.0)
            sigma = self.params.get('sigma', 1.0)
            
            return math.exp(-0.5 * ((x - center) / sigma) ** 2)
        
        elif self.membership_type == "triangular":
            # Háromszögletes tagsági függvény
            a = self.params.get('a', 0.0)
            b = self.params.get('b', 0.5)
            c = self.params.get('c', 1.0)
            
            if x <= a or x >= c:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a)
            else:
                return (c - x) / (c - b)
        
        elif self.membership_type == "trapezoidal":
            # Trapéz alakú tagsági függvény
            a = self.params.get('a', 0.0)
            b = self.params.get('b', 0.25)
            c = self.params.get('c', 0.75)
            d = self.params.get('d', 1.0)
            
            if x <= a or x >= d:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a)
            elif b < x <= c:
                return 1.0
            else:
                return (d - x) / (d - c)
        
        return 0.0


class FuzzyRule:
    """Fuzzy szabály osztály."""
    
    def __init__(self, rule_id: int, antecedents: List[Tuple[str, str]], 
                 consequent_params: Dict[str, float]):
        self.rule_id = rule_id
        self.antecedents = antecedents  # [(variable_name, fuzzy_set_name), ...]
        self.consequent_params = consequent_params  # TSK következmény paraméterek
        self.strength = 0.0
        
    def calculate_strength(self, fuzzy_variables: Dict[str, Dict[str, FuzzySet]], 
                          input_values: Dict[str, float]) -> float:
        """Szabály erősségének számítása."""
        
        strength = 1.0
        
        for var_name, fuzzy_set_name in self.antecedents:
            if var_name in fuzzy_variables and fuzzy_set_name in fuzzy_variables[var_name]:
                fuzzy_set = fuzzy_variables[var_name][fuzzy_set_name]
                input_val = input_values.get(var_name, 0.0)
                membership = fuzzy_set.membership_degree(input_val)
                strength = min(strength, membership)  # T-norm (minimum)
            else:
                strength = 0.0
                break
        
        self.strength = strength
        return strength
    
    def calculate_output(self, input_values: Dict[str, float]) -> float:
        """TSK-típusú kimenet számítása."""
        
        output = self.consequent_params.get('c0', 0.0)  # Konstans tag
        
        for var_name, value in input_values.items():
            param_name = f'c_{var_name}'
            if param_name in self.consequent_params:
                output += self.consequent_params[param_name] * value
        
        return output


class ANFIS:
    """Adaptive Neuro-Fuzzy Inference System."""
    
    def __init__(self, input_variables: List[str], min_num: int, max_num: int):
        self.input_variables = input_variables
        self.min_num = min_num
        self.max_num = max_num
        
        # Fuzzy változók és halmazok
        self.fuzzy_variables = {}
        self.fuzzy_rules = []
        
        # Tanulási paraméterek
        self.learning_rate = 0.01
        self.epochs = 50
        
        # Inicializálás
        self._initialize_fuzzy_sets()
        self._initialize_rules()
        
        # Teljesítmény követés
        self.training_errors = []
        
    def _initialize_fuzzy_sets(self):
        """Fuzzy halmazok inicializálása."""
        
        for var in self.input_variables:
            self.fuzzy_variables[var] = {}
            
            # Minden változóhoz 3 fuzzy halmaz: LOW, MEDIUM, HIGH
            self.fuzzy_variables[var]['LOW'] = FuzzySet(
                'LOW', 
                {'center': 0.2, 'sigma': 0.15}, 
                'gaussian'
            )
            
            self.fuzzy_variables[var]['MEDIUM'] = FuzzySet(
                'MEDIUM', 
                {'center': 0.5, 'sigma': 0.15}, 
                'gaussian'
            )
            
            self.fuzzy_variables[var]['HIGH'] = FuzzySet(
                'HIGH', 
                {'center': 0.8, 'sigma': 0.15}, 
                'gaussian'
            )
    
    def _initialize_rules(self):
        """Fuzzy szabályok inicializálása."""
        
        rule_id = 0
        
        # Összes lehetséges kombinációhoz szabályok
        fuzzy_set_names = ['LOW', 'MEDIUM', 'HIGH']
        
        if len(self.input_variables) == 1:
            # 1 változó esetén
            var = self.input_variables[0]
            for fs_name in fuzzy_set_names:
                consequent_params = {
                    'c0': random.uniform(-1, 1),
                    f'c_{var}': random.uniform(-1, 1)
                }
                
                rule = FuzzyRule(
                    rule_id,
                    [(var, fs_name)],
                    consequent_params
                )
                
                self.fuzzy_rules.append(rule)
                rule_id += 1
        
        elif len(self.input_variables) == 2:
            # 2 változó esetén
            var1, var2 = self.input_variables[0], self.input_variables[1]
            for fs1 in fuzzy_set_names:
                for fs2 in fuzzy_set_names:
                    consequent_params = {
                        'c0': random.uniform(-1, 1),
                        f'c_{var1}': random.uniform(-1, 1),
                        f'c_{var2}': random.uniform(-1, 1)
                    }
                    
                    rule = FuzzyRule(
                        rule_id,
                        [(var1, fs1), (var2, fs2)],
                        consequent_params
                    )
                    
                    self.fuzzy_rules.append(rule)
                    rule_id += 1
        
        else:
            # Több változó esetén egyszerűsített szabályok
            for fs_name in fuzzy_set_names:
                antecedents = [(var, fs_name) for var in self.input_variables]
                
                consequent_params = {'c0': random.uniform(-1, 1)}
                for var in self.input_variables:
                    consequent_params[f'c_{var}'] = random.uniform(-1, 1)
                
                rule = FuzzyRule(rule_id, antecedents, consequent_params)
                self.fuzzy_rules.append(rule)
                rule_id += 1
    
    def forward_pass(self, input_values: Dict[str, float]) -> float:
        """Forward pass az ANFIS-en keresztül."""
        
        # 1. réteg: Fuzzifikáció (már a membership_degree-ben)
        
        # 2. réteg: Szabály aktiváció
        rule_strengths = []
        rule_outputs = []
        
        for rule in self.fuzzy_rules:
            strength = rule.calculate_strength(self.fuzzy_variables, input_values)
            output = rule.calculate_output(input_values)
            
            rule_strengths.append(strength)
            rule_outputs.append(output)
        
        # 3. réteg: Normalizálás
        total_strength = sum(rule_strengths)
        if total_strength == 0:
            normalized_strengths = [1.0 / len(rule_strengths)] * len(rule_strengths)
        else:
            normalized_strengths = [s / total_strength for s in rule_strengths]
        
        # 4. réteg: Súlyozott összeg
        weighted_outputs = [ns * ro for ns, ro in zip(normalized_strengths, rule_outputs)]
        
        # 5. réteg: Végső kimenet
        final_output = sum(weighted_outputs)
        
        return final_output
    
    def train(self, training_data: List[Tuple[Dict[str, float], float]]):
        """ANFIS tanítás hibrid tanulási algoritmussal."""
        
        for epoch in range(self.epochs):
            epoch_error = 0.0
            
            for input_values, target_output in training_data:
                # Forward pass
                predicted_output = self.forward_pass(input_values)
                
                # Error számítás
                error = target_output - predicted_output
                epoch_error += error ** 2
                
                # Backward pass - consequent paraméterek frissítése
                self._update_consequent_parameters(input_values, error)
                
                # Premise paraméterek frissítése (egyszerűsített)
                self._update_premise_parameters(input_values, error)
            
            # Átlagos hiba
            avg_error = epoch_error / len(training_data)
            self.training_errors.append(avg_error)
            
            # Korai megállás
            if avg_error < 0.01:
                break
    
    def _update_consequent_parameters(self, input_values: Dict[str, float], error: float):
        """Consequent paraméterek frissítése."""
        
        # Szabály erősségek újraszámítása
        rule_strengths = []
        for rule in self.fuzzy_rules:
            strength = rule.calculate_strength(self.fuzzy_variables, input_values)
            rule_strengths.append(strength)
        
        # Normalizálás
        total_strength = sum(rule_strengths)
        if total_strength > 0:
            normalized_strengths = [s / total_strength for s in rule_strengths]
        else:
            normalized_strengths = [1.0 / len(rule_strengths)] * len(rule_strengths)
        
        # Consequent paraméterek frissítése
        for i, rule in enumerate(self.fuzzy_rules):
            strength = normalized_strengths[i]
            
            # Konstans tag frissítése
            rule.consequent_params['c0'] += self.learning_rate * error * strength
            
            # Lineáris tagok frissítése
            for var_name, value in input_values.items():
                param_name = f'c_{var_name}'
                if param_name in rule.consequent_params:
                    rule.consequent_params[param_name] += self.learning_rate * error * strength * value
    
    def _update_premise_parameters(self, input_values: Dict[str, float], error: float):
        """Premise paraméterek frissítése (egyszerűsített)."""
        
        # Fuzzy set paraméterek finomhangolása
        for var_name, value in input_values.items():
            if var_name in self.fuzzy_variables:
                for fs_name, fuzzy_set in self.fuzzy_variables[var_name].items():
                    if fuzzy_set.membership_type == 'gaussian':
                        # Gauss paraméterek frissítése
                        center = fuzzy_set.params['center']
                        sigma = fuzzy_set.params['sigma']
                        
                        membership = fuzzy_set.membership_degree(value)
                        
                        # Center frissítése
                        center_gradient = error * membership * (value - center) / (sigma ** 2)
                        fuzzy_set.params['center'] += self.learning_rate * center_gradient * 0.1
                        
                        # Sigma frissítése
                        sigma_gradient = error * membership * ((value - center) ** 2) / (sigma ** 3)
                        fuzzy_set.params['sigma'] += self.learning_rate * sigma_gradient * 0.1
                        
                        # Paraméterek korlátozása
                        fuzzy_set.params['center'] = max(0.0, min(1.0, fuzzy_set.params['center']))
                        fuzzy_set.params['sigma'] = max(0.05, min(0.5, fuzzy_set.params['sigma']))


class AdaptiveNeuroFuzzyPredictor:
    """Adaptív Neuro-Fuzzy prediktor főosztály."""
    
    def __init__(self, min_num: int, max_num: int, target_count: int):
        self.min_num = min_num
        self.max_num = max_num
        self.target_count = target_count
        
        # ANFIS modellek különböző featurékhez
        self.frequency_anfis = None
        self.trend_anfis = None
        self.pattern_anfis = None
        
        # Feature extractors
        self.feature_history = deque(maxlen=100)
        
        # Model súlyok
        self.model_weights = {'frequency': 0.4, 'trend': 0.3, 'pattern': 0.3}
        
    def extract_features(self, past_draws: List[List[int]]) -> Dict[str, List[Dict[str, float]]]:
        """Feature extraction a múltbeli húzásokból."""
        
        features = {
            'frequency': [],
            'trend': [],
            'pattern': []
        }
        
        if len(past_draws) < 5:
            return features
        
        # Normalizálási tartományok
        window_size = min(20, len(past_draws))
        
        for i in range(window_size, len(past_draws)):
            window = past_draws[i-window_size:i]
            target = past_draws[i]
            
            # Frequency features
            freq_features = self._extract_frequency_features(window)
            if freq_features:
                features['frequency'].append((freq_features, self._calculate_target_frequency(target)))
            
            # Trend features
            trend_features = self._extract_trend_features(window)
            if trend_features:
                features['trend'].append((trend_features, self._calculate_target_trend(window, target)))
            
            # Pattern features
            pattern_features = self._extract_pattern_features(window)
            if pattern_features:
                features['pattern'].append((pattern_features, self._calculate_target_pattern(target)))
        
        return features
    
    def _extract_frequency_features(self, window: List[List[int]]) -> Optional[Dict[str, float]]:
        """Frekvencia alapú feature extraction."""
        
        # Számok frekvenciája
        frequency = defaultdict(int)
        total_numbers = 0
        
        for draw in window:
            for num in draw:
                if self.min_num <= num <= self.max_num:
                    frequency[num] += 1
                    total_numbers += 1
        
        if total_numbers == 0:
            return None
        
        # Statisztikai jellemzők
        frequencies = list(frequency.values())
        
        features = {
            'avg_frequency': np.mean(frequencies) / total_numbers if frequencies else 0,
            'max_frequency': max(frequencies) / total_numbers if frequencies else 0,
            'frequency_std': np.std(frequencies) / total_numbers if len(frequencies) > 1 else 0,
            'unique_numbers': len(frequency) / (self.max_num - self.min_num + 1)
        }
        
        return features
    
    def _extract_trend_features(self, window: List[List[int]]) -> Optional[Dict[str, float]]:
        """Trend alapú feature extraction."""
        
        if len(window) < 3:
            return None
        
        # Idősor létrehozása (átlagok)
        averages = [np.mean(draw) for draw in window]
        maxes = [max(draw) for draw in window]
        mins = [min(draw) for draw in window]
        
        # Trend számítás
        x = np.arange(len(averages))
        
        # Lineáris trend az átlagokban
        avg_trend = np.polyfit(x, averages, 1)[0] if len(averages) > 1 else 0
        max_trend = np.polyfit(x, maxes, 1)[0] if len(maxes) > 1 else 0
        min_trend = np.polyfit(x, mins, 1)[0] if len(mins) > 1 else 0
        
        # Normalizálás
        range_size = self.max_num - self.min_num
        
        features = {
            'avg_trend': avg_trend / range_size,
            'max_trend': max_trend / range_size,
            'min_trend': min_trend / range_size,
            'trend_consistency': 1.0 - np.std([avg_trend, max_trend, min_trend]) / range_size
        }
        
        return features
    
    def _extract_pattern_features(self, window: List[List[int]]) -> Optional[Dict[str, float]]:
        """Mintázat alapú feature extraction."""
        
        # Számok közötti távolságok
        all_gaps = []
        consecutive_counts = []
        
        for draw in window:
            sorted_draw = sorted(draw)
            
            # Gaps között számok
            gaps = [sorted_draw[i+1] - sorted_draw[i] for i in range(len(sorted_draw)-1)]
            all_gaps.extend(gaps)
            
            # Egymást követő számok
            consecutive = sum(1 for gap in gaps if gap == 1)
            consecutive_counts.append(consecutive)
        
        if not all_gaps:
            return None
        
        # Páros/páratlan arány
        even_counts = []
        for draw in window:
            even_count = sum(1 for num in draw if num % 2 == 0)
            even_counts.append(even_count / len(draw))
        
        features = {
            'avg_gap': np.mean(all_gaps) / (self.max_num - self.min_num),
            'gap_std': np.std(all_gaps) / (self.max_num - self.min_num) if len(all_gaps) > 1 else 0,
            'consecutive_ratio': np.mean(consecutive_counts) / self.target_count,
            'even_ratio': np.mean(even_counts)
        }
        
        return features
    
    def _calculate_target_frequency(self, target_draw: List[int]) -> float:
        """Target frekvencia metrika."""
        # Egyszerű: a húzás átlagos száma normalizálva
        avg_number = np.mean(target_draw)
        return (avg_number - self.min_num) / (self.max_num - self.min_num)
    
    def _calculate_target_trend(self, window: List[List[int]], target_draw: List[int]) -> float:
        """Target trend metrika."""
        # Trend folytatás: mennyire követi a target az előző trendeket
        window_avg = np.mean([np.mean(draw) for draw in window])
        target_avg = np.mean(target_draw)
        
        # Normalizált különbség
        diff = (target_avg - window_avg) / (self.max_num - self.min_num)
        return 0.5 + diff  # [0, 1] tartományba normalizálás
    
    def _calculate_target_pattern(self, target_draw: List[int]) -> float:
        """Target mintázat metrika."""
        # Mintázat komplexitás
        sorted_draw = sorted(target_draw)
        gaps = [sorted_draw[i+1] - sorted_draw[i] for i in range(len(sorted_draw)-1)]
        
        # Gap változatosság
        gap_variety = len(set(gaps)) / len(gaps) if gaps else 0
        return gap_variety
    
    def train(self, past_draws: List[List[int]]):
        """ANFIS modellek tanítása."""
        
        if len(past_draws) < 10:
            logger.warning("Nincs elég adat az ANFIS tanításához")
            return
        
        # Feature extraction
        features = self.extract_features(past_draws)
        
        # Frequency ANFIS
        if features['frequency']:
            self.frequency_anfis = ANFIS(['avg_frequency', 'max_frequency', 'frequency_std', 'unique_numbers'], 
                                       self.min_num, self.max_num)
            self.frequency_anfis.train(features['frequency'])
        
        # Trend ANFIS
        if features['trend']:
            self.trend_anfis = ANFIS(['avg_trend', 'max_trend', 'min_trend', 'trend_consistency'], 
                                   self.min_num, self.max_num)
            self.trend_anfis.train(features['trend'])
        
        # Pattern ANFIS
        if features['pattern']:
            self.pattern_anfis = ANFIS(['avg_gap', 'gap_std', 'consecutive_ratio', 'even_ratio'], 
                                     self.min_num, self.max_num)
            self.pattern_anfis.train(features['pattern'])
        
        logger.info("ANFIS modellek sikeresen tanítva")
    
    def predict(self, past_draws: List[List[int]]) -> List[int]:
        """Adaptív neuro-fuzzy predikció."""
        
        if len(past_draws) < 5:
            return self._generate_fuzzy_fallback()
        
        # Jelenlegi feature extraction
        current_window = past_draws[-20:] if len(past_draws) >= 20 else past_draws
        
        freq_features = self._extract_frequency_features(current_window)
        trend_features = self._extract_trend_features(current_window)
        pattern_features = self._extract_pattern_features(current_window)
        
        # ANFIS predikciók
        predictions = {}
        
        if self.frequency_anfis and freq_features:
            freq_output = self.frequency_anfis.forward_pass(freq_features)
            predictions['frequency'] = freq_output
        
        if self.trend_anfis and trend_features:
            trend_output = self.trend_anfis.forward_pass(trend_features)
            predictions['trend'] = trend_output
        
        if self.pattern_anfis and pattern_features:
            pattern_output = self.pattern_anfis.forward_pass(pattern_features)
            predictions['pattern'] = pattern_output
        
        # Ensemble kombináció
        if predictions:
            combined_prediction = self._combine_anfis_predictions(predictions)
            final_numbers = self._convert_prediction_to_numbers(combined_prediction, past_draws)
        else:
            final_numbers = self._generate_fuzzy_fallback()
        
        return sorted(final_numbers[:self.target_count])
    
    def _combine_anfis_predictions(self, predictions: Dict[str, float]) -> float:
        """ANFIS predikciók kombinálása."""
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model_name, prediction in predictions.items():
            weight = self.model_weights.get(model_name, 0.33)
            weighted_sum += weight * prediction
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def _convert_prediction_to_numbers(self, prediction: float, past_draws: List[List[int]]) -> List[int]:
        """Fuzzy predikció konvertálása számokra."""
        
        numbers = set()
        
        # Központi szám a predikció alapján
        center = self.min_num + prediction * (self.max_num - self.min_num)
        center = int(round(center))
        
        # Fuzzy környezet a központ körül
        spread = max(3, (self.max_num - self.min_num) // 10)
        
        for offset in range(-spread, spread + 1):
            candidate = center + offset
            if self.min_num <= candidate <= self.max_num:
                # Fuzzy membership alapú valószínűség
                distance = abs(offset)
                membership = math.exp(-distance / spread)
                
                if random.random() < membership:
                    numbers.add(candidate)
                
                if len(numbers) >= self.target_count * 2:
                    break
        
        # Történeti mintázatok beépítése
        if past_draws:
            recent_numbers = set()
            for draw in past_draws[-5:]:
                recent_numbers.update(draw)
            
            # Egy részét a gyakori számokból
            frequent_numbers = list(recent_numbers)[:self.target_count // 2]
            numbers.update(frequent_numbers)
        
        # Kiegészítés ha szükséges
        while len(numbers) < self.target_count:
            # Fuzzy logika alapú kiegészítés
            candidate = self._fuzzy_number_generation(past_draws)
            numbers.add(candidate)
        
        return list(numbers)
    
    def _fuzzy_number_generation(self, past_draws: List[List[int]]) -> int:
        """Fuzzy logika alapú szám generálás."""
        
        if not past_draws:
            return random.randint(self.min_num, self.max_num)
        
        # Fuzzy szabályok alkalmazása
        # Szabály 1: Ha gyakori szám, akkor környéke valószínű
        frequent_numbers = []
        for draw in past_draws[-10:]:
            frequent_numbers.extend(draw)
        
        frequency = defaultdict(int)
        for num in frequent_numbers:
            frequency[num] += 1
        
        if frequency:
            # Leggyakoribb szám környéke
            most_frequent = max(frequency.keys(), key=lambda x: frequency[x])
            
            # Fuzzy környezet
            candidates = []
            for offset in [-2, -1, 0, 1, 2]:
                candidate = most_frequent + offset
                if self.min_num <= candidate <= self.max_num:
                    candidates.append(candidate)
            
            if candidates:
                return random.choice(candidates)
        
        # Fallback
        return random.randint(self.min_num, self.max_num)
    
    def _generate_fuzzy_fallback(self) -> List[int]:
        """Fuzzy logika alapú fallback."""
        
        numbers = set()
        
        # Fuzzy halmazok alapú generálás
        # LOW, MEDIUM, HIGH régiók
        low_range = (self.min_num, self.min_num + (self.max_num - self.min_num) // 3)
        med_range = (low_range[1], low_range[1] + (self.max_num - self.min_num) // 3)
        high_range = (med_range[1], self.max_num)
        
        ranges = [low_range, med_range, high_range]
        
        # Minden régióból legalább egy szám
        for low, high in ranges:
            if len(numbers) < self.target_count:
                num = random.randint(low, high)
                numbers.add(num)
        
        # Kiegészítés
        while len(numbers) < self.target_count:
            num = random.randint(self.min_num, self.max_num)
            numbers.add(num)
        
        return sorted(list(numbers)[:self.target_count])


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Adaptive Neuro-Fuzzy alapú lottószám predikció.
    Returns a tuple (main_numbers, additional_numbers).
    """
    try:
        # Fő számok generálása
        main_numbers = generate_neuro_fuzzy_numbers(
            lottery_type_instance,
            min_number=int(lottery_type_instance.min_number),
            max_number=int(lottery_type_instance.max_number),
            pieces_of_draw_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
            is_main=True
        )

        # Kiegészítő számok generálása
        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            additional_numbers = generate_neuro_fuzzy_numbers(
                lottery_type_instance,
                min_number=int(lottery_type_instance.additional_min_number),
                max_number=int(lottery_type_instance.additional_max_number),
                pieces_of_draw_numbers=int(lottery_type_instance.additional_numbers_count),
                is_main=False
            )

        return sorted(main_numbers), sorted(additional_numbers)
    
    except Exception as e:
        logger.error(f"Hiba a neuro-fuzzy predikció során: {e}")
        return generate_fallback_numbers(lottery_type_instance)


def generate_neuro_fuzzy_numbers(
    lottery_type_instance: lg_lottery_type,
    min_number: int,
    max_number: int,
    pieces_of_draw_numbers: int,
    is_main: bool
) -> List[int]:
    """Neuro-fuzzy számgenerálás."""
    
    # Történeti adatok lekérése
    past_draws = get_historical_data(lottery_type_instance, is_main)
    
    if len(past_draws) < 10:
        return generate_neuro_fuzzy_fallback(min_number, max_number, pieces_of_draw_numbers)
    
    # Adaptive neuro-fuzzy predictor
    predictor = AdaptiveNeuroFuzzyPredictor(
        min_num=min_number, 
        max_num=max_number, 
        target_count=pieces_of_draw_numbers
    )
    
    # Tanítás és predikció
    predictor.train(past_draws)
    predictions = predictor.predict(past_draws)
    
    return predictions


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Történeti adatok lekérése."""
    try:
        queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('-id')[:150]
        
        past_draws = []
        for draw in queryset:
            try:
                if is_main:
                    numbers = draw.lottery_type_number
                else:
                    numbers = getattr(draw, 'additional_numbers', None)
                
                if isinstance(numbers, list) and len(numbers) > 0:
                    valid_numbers = [int(num) for num in numbers if isinstance(num, (int, float))]
                    if valid_numbers:
                        past_draws.append(valid_numbers)
            except (ValueError, TypeError, AttributeError):
                continue
        
        return past_draws
    
    except Exception as e:
        logger.error(f"Hiba a történeti adatok lekérésében: {e}")
        return []


def generate_neuro_fuzzy_fallback(min_number: int, max_number: int, count: int) -> List[int]:
    """Neuro-fuzzy fallback számgenerálás."""
    
    # Egyszerű fuzzy logika
    numbers = set()
    
    # Fuzzy régiók definiálása
    region_size = (max_number - min_number + 1) // 3
    
    regions = [
        (min_number, min_number + region_size - 1),
        (min_number + region_size, min_number + 2 * region_size - 1),
        (min_number + 2 * region_size, max_number)
    ]
    
    # Minden régióból részben válogatás
    for i, (start, end) in enumerate(regions):
        # Fuzzy membership alapú számosság
        region_count = max(1, count // 3 + (1 if i < count % 3 else 0))
        
        for _ in range(region_count):
            if len(numbers) < count:
                num = random.randint(start, end)
                numbers.add(num)
    
    # Kiegészítés
    while len(numbers) < count:
        num = random.randint(min_number, max_number)
        numbers.add(num)
    
    return sorted(list(numbers)[:count])


def generate_fallback_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Fallback számgenerálás."""
    main_numbers = generate_neuro_fuzzy_fallback(
        int(lottery_type_instance.min_number),
        int(lottery_type_instance.max_number),
        int(lottery_type_instance.pieces_of_draw_numbers)
    )
    
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_neuro_fuzzy_fallback(
            int(lottery_type_instance.additional_min_number),
            int(lottery_type_instance.additional_max_number),
            int(lottery_type_instance.additional_numbers_count)
        )
    
    return sorted(main_numbers), sorted(additional_numbers) 