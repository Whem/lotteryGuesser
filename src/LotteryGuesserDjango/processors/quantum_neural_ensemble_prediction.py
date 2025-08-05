# quantum_neural_ensemble_prediction.py
"""
Kvantum-Neurális Ensemble Predikció
Kombinálja a kvantum-inspirált algoritmusokat neurális hálózatokkal
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict
import random
import math
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

from algorithms.models import lg_lottery_winner_number, lg_lottery_type


class QuantumNeuralEnsemble:
    """
    Kvantum-inspirált neurális ensemble predikció
    """
    
    def __init__(self):
        self.quantum_states = 8  # Kvantum állapotok száma
        self.neural_layers = [64, 32, 16]  # Neurális hálózat rétegek
        self.ensemble_size = 7  # Ensemble tagok száma
        self.coherence_threshold = 0.6  # Koherencia küszöb
        
        # Kvantum-inspirált paraméterek
        self.superposition_factor = 0.7
        self.entanglement_strength = 0.8
        self.decoherence_rate = 0.05
        
    def get_numbers(self, lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
        """
        Főbejárási pont a kvantum-neurális ensemble predikcióhoz
        """
        try:
            # Fő számok generálása
            main_numbers = self._generate_quantum_neural_numbers(
                lottery_type_instance,
                min_num=int(lottery_type_instance.min_number),
                max_num=int(lottery_type_instance.max_number),
                required_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
                is_main=True
            )
            
            # Kiegészítő számok generálása
            additional_numbers = []
            if lottery_type_instance.has_additional_numbers:
                additional_numbers = self._generate_quantum_neural_numbers(
                    lottery_type_instance,
                    min_num=int(lottery_type_instance.additional_min_number),
                    max_num=int(lottery_type_instance.additional_max_number),
                    required_numbers=int(lottery_type_instance.additional_numbers_count),
                    is_main=False
                )
            
            return sorted(main_numbers), sorted(additional_numbers)
            
        except Exception as e:
            print(f"Hiba a quantum_neural_ensemble_prediction-ben: {e}")
            return self._generate_fallback_numbers(lottery_type_instance)
    
    def _generate_quantum_neural_numbers(self, lottery_type_instance: lg_lottery_type,
                                       min_num: int, max_num: int, required_numbers: int,
                                       is_main: bool) -> List[int]:
        """
        Kvantum-neurális számgenerálás
        """
        # Történeti adatok lekérése
        historical_data = self._get_historical_data(lottery_type_instance, is_main)
        
        if len(historical_data) < 30:
            return self._quantum_random_generation(min_num, max_num, required_numbers)
        
        # Kvantum állapotok inicializálása
        quantum_states = self._initialize_quantum_states(historical_data, min_num, max_num)
        
        # Neurális ensemble predikciók
        neural_predictions = self._generate_neural_ensemble_predictions(
            historical_data, min_num, max_num, required_numbers
        )
        
        # Kvantum-neurális fúzió
        fused_predictions = self._quantum_neural_fusion(
            quantum_states, neural_predictions, min_num, max_num, required_numbers
        )
        
        # Koherencia ellenőrzés és finomhangolás
        final_numbers = self._coherence_optimization(
            fused_predictions, historical_data, min_num, max_num, required_numbers
        )
        
        return final_numbers
    
    def _initialize_quantum_states(self, historical_data: List[List[int]], 
                                 min_num: int, max_num: int) -> np.ndarray:
        """
        Kvantum állapotok inicializálása
        """
        num_range = max_num - min_num + 1
        quantum_states = np.zeros((self.quantum_states, num_range), dtype=complex)
        
        # Szuperpozíció létrehozása
        for state_idx in range(self.quantum_states):
            # Történeti adatok alapján amplitúdók számítása
            for i, draw in enumerate(historical_data[:20]):  # Legutóbbi 20 húzás
                weight = self.superposition_factor ** i
                for num in draw:
                    if min_num <= num <= max_num:
                        idx = num - min_num
                        # Komplex amplitúdó
                        phase = 2 * np.pi * (state_idx + 1) * num / num_range
                        quantum_states[state_idx, idx] += weight * np.exp(1j * phase)
        
        # Normalizálás
        for state_idx in range(self.quantum_states):
            norm = np.linalg.norm(quantum_states[state_idx])
            if norm > 0:
                quantum_states[state_idx] /= norm
        
        return quantum_states
    
    def _generate_neural_ensemble_predictions(self, historical_data: List[List[int]],
                                            min_num: int, max_num: int, 
                                            required_numbers: int) -> List[List[int]]:
        """
        Neurális ensemble predikciók generálása
        """
        predictions = []
        
        for ensemble_idx in range(self.ensemble_size):
            try:
                # Adatok előkészítése
                X, y = self._prepare_neural_data(historical_data, min_num, max_num)
                
                if len(X) < 10:
                    predictions.append(self._quantum_random_generation(min_num, max_num, required_numbers))
                    continue
                
                # Neurális hálózat tanítása
                model = MLPRegressor(
                    hidden_layer_sizes=self.neural_layers,
                    activation='tanh',
                    solver='adam',
                    alpha=0.01,
                    max_iter=500,
                    random_state=42 + ensemble_idx
                )
                
                model.fit(X, y)
                
                # Predikció
                last_features = X[-1:] if len(X) > 0 else np.zeros((1, X.shape[1]))
                predicted_numbers = model.predict(last_features)[0]
                
                # Számok konvertálása
                converted_numbers = []
                for pred in predicted_numbers:
                    num = max(min_num, min(max_num, int(round(pred))))
                    converted_numbers.append(num)
                
                # Duplikátumok eltávolítása és kiegészítés
                unique_numbers = list(set(converted_numbers))
                if len(unique_numbers) < required_numbers:
                    remaining = [n for n in range(min_num, max_num + 1) if n not in unique_numbers]
                    random.shuffle(remaining)
                    unique_numbers.extend(remaining[:required_numbers - len(unique_numbers)])
                
                predictions.append(unique_numbers[:required_numbers])
                
            except Exception:
                predictions.append(self._quantum_random_generation(min_num, max_num, required_numbers))
        
        return predictions
    
    def _prepare_neural_data(self, historical_data: List[List[int]], 
                           min_num: int, max_num: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Neurális hálózat adatok előkészítése
        """
        features = []
        targets = []
        
        for i in range(len(historical_data) - 1):
            current_draw = historical_data[i]
            next_draw = historical_data[i + 1]
            
            # Jellemzők kinyerése
            feature_vector = self._extract_features(current_draw, min_num, max_num)
            features.append(feature_vector)
            
            # Cél értékek (következő húzás)
            target_vector = [0] * (max_num - min_num + 1)
            for num in next_draw:
                if min_num <= num <= max_num:
                    target_vector[num - min_num] = 1
            targets.append(target_vector)
        
        return np.array(features), np.array(targets)
    
    def _extract_features(self, draw: List[int], min_num: int, max_num: int) -> List[float]:
        """
        Jellemzők kinyerése egy húzásból
        """
        features = []
        
        # Alapvető statisztikák
        features.extend([
            np.mean(draw),
            np.std(draw),
            np.min(draw),
            np.max(draw),
            len(draw)
        ])
        
        # Páros/páratlan arány
        even_count = sum(1 for num in draw if num % 2 == 0)
        features.append(even_count / len(draw))
        
        # Összeg kategóriák
        total_sum = sum(draw)
        features.append(total_sum)
        
        # Számok eloszlása tartományokban
        num_range = max_num - min_num + 1
        for i in range(5):  # 5 tartomány
            range_start = min_num + i * num_range // 5
            range_end = min_num + (i + 1) * num_range // 5
            count = sum(1 for num in draw if range_start <= num < range_end)
            features.append(count)
        
        # Gap statisztikák
        sorted_draw = sorted(draw)
        gaps = [sorted_draw[i+1] - sorted_draw[i] for i in range(len(sorted_draw)-1)]
        if gaps:
            features.extend([np.mean(gaps), np.std(gaps)])
        else:
            features.extend([0, 0])
        
        return features
    
    def _quantum_neural_fusion(self, quantum_states: np.ndarray, 
                             neural_predictions: List[List[int]],
                             min_num: int, max_num: int, 
                             required_numbers: int) -> List[int]:
        """
        Kvantum és neurális predikciók fúziója
        """
        # Kvantum mérés szimulálása
        quantum_probabilities = self._quantum_measurement(quantum_states)
        
        # Neurális predikciók súlyozása
        neural_votes = Counter()
        for prediction in neural_predictions:
            for num in prediction:
                neural_votes[num] += 1
        
        # Fúzió súlyozott kombinációval
        fused_scores = {}
        for num in range(min_num, max_num + 1):
            quantum_score = quantum_probabilities.get(num, 0)
            neural_score = neural_votes.get(num, 0) / len(neural_predictions)
            
            # Kvantum-neurális fúzió
            fused_scores[num] = (
                self.entanglement_strength * quantum_score +
                (1 - self.entanglement_strength) * neural_score
            )
        
        # Legmagasabb score-ú számok kiválasztása
        sorted_scores = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [num for num, _ in sorted_scores[:required_numbers]]
        
        return selected
    
    def _quantum_measurement(self, quantum_states: np.ndarray) -> Dict[int, float]:
        """
        Kvantum mérés szimulálása
        """
        probabilities = {}
        
        for state_idx in range(self.quantum_states):
            state = quantum_states[state_idx]
            # Valószínűség = |amplitúdó|²
            probs = np.abs(state) ** 2
            
            for num_idx, prob in enumerate(probs):
                num = num_idx + 1  # Feltételezve, hogy min_num = 1
                if num not in probabilities:
                    probabilities[num] = 0
                probabilities[num] += prob / self.quantum_states
        
        return probabilities
    
    def _coherence_optimization(self, predictions: List[int], 
                              historical_data: List[List[int]],
                              min_num: int, max_num: int, 
                              required_numbers: int) -> List[int]:
        """
        Koherencia optimalizálás
        """
        # Koherencia mérés
        coherence = self._measure_coherence(predictions, historical_data)
        
        if coherence < self.coherence_threshold:
            # Dekoherencia korrekció
            predictions = self._apply_decoherence_correction(
                predictions, historical_data, min_num, max_num, required_numbers
            )
        
        # Végső validáció
        final_predictions = self._validate_predictions(
            predictions, min_num, max_num, required_numbers
        )
        
        return final_predictions
    
    def _measure_coherence(self, predictions: List[int], 
                         historical_data: List[List[int]]) -> float:
        """
        Koherencia mérés
        """
        if not historical_data:
            return 0.0
        
        # Entrópia alapú koherencia mérés
        pred_counter = Counter(predictions)
        hist_counter = Counter(num for draw in historical_data[:10] for num in draw)
        
        # Normalizálás
        pred_probs = np.array(list(pred_counter.values())) / sum(pred_counter.values())
        hist_probs = np.array(list(hist_counter.values())) / sum(hist_counter.values())
        
        # Koherencia = 1 - relatív entrópia
        try:
            coherence = 1 - entropy(pred_probs, hist_probs)
            return max(0, min(1, coherence))
        except:
            return 0.5
    
    def _apply_decoherence_correction(self, predictions: List[int],
                                    historical_data: List[List[int]],
                                    min_num: int, max_num: int,
                                    required_numbers: int) -> List[int]:
        """
        Dekoherencia korrekció alkalmazása
        """
        # Történeti minták alapján korrekció
        recent_patterns = Counter(num for draw in historical_data[:5] for num in draw)
        
        corrected = []
        for num in predictions:
            # Dekoherencia alkalmazása
            if random.random() < self.decoherence_rate:
                # Cserélje ki történeti mintával
                if recent_patterns:
                    new_num = random.choices(
                        list(recent_patterns.keys()),
                        weights=list(recent_patterns.values())
                    )[0]
                    corrected.append(new_num)
                else:
                    corrected.append(num)
            else:
                corrected.append(num)
        
        return corrected
    
    def _validate_predictions(self, predictions: List[int], 
                            min_num: int, max_num: int, 
                            required_numbers: int) -> List[int]:
        """
        Predikciók validálása
        """
        # Duplikátumok eltávolítása
        unique_predictions = list(set(predictions))
        
        # Tartomány ellenőrzés
        valid_predictions = [num for num in unique_predictions if min_num <= num <= max_num]
        
        # Kiegészítés szükség esetén
        if len(valid_predictions) < required_numbers:
            remaining = [num for num in range(min_num, max_num + 1) if num not in valid_predictions]
            random.shuffle(remaining)
            valid_predictions.extend(remaining[:required_numbers - len(valid_predictions)])
        
        return valid_predictions[:required_numbers]
    
    def _quantum_random_generation(self, min_num: int, max_num: int, count: int) -> List[int]:
        """
        Kvantum-inspirált véletlen generálás
        """
        numbers = set()
        
        # Kvantum-inspirált véletlen generálás
        for _ in range(count * 3):
            # Szuperpozíció szimulálása
            phase = random.uniform(0, 2 * np.pi)
            amplitude = random.uniform(0, 1)
            
            # Kvantum szám generálás
            quantum_num = min_num + int(
                (max_num - min_num + 1) * (amplitude * np.cos(phase) + 1) / 2
            )
            
            if min_num <= quantum_num <= max_num:
                numbers.add(quantum_num)
            
            if len(numbers) >= count:
                break
        
        # Kiegészítés szükség esetén
        if len(numbers) < count:
            remaining = [num for num in range(min_num, max_num + 1) if num not in numbers]
            random.shuffle(remaining)
            numbers.update(remaining[:count - len(numbers)])
        
        return list(numbers)[:count]
    
    def _get_historical_data(self, lottery_type_instance: lg_lottery_type, 
                           is_main: bool) -> List[List[int]]:
        """
        Történeti adatok lekérése
        """
        field_name = 'lottery_type_number' if is_main else 'additional_numbers'
        
        try:
            queryset = lg_lottery_winner_number.objects.filter(
                lottery_type=lottery_type_instance
            ).order_by('-id').values_list(field_name, flat=True)[:150]
            
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
        """
        Fallback számgenerálás
        """
        main_numbers = self._quantum_random_generation(
            int(lottery_type_instance.min_number),
            int(lottery_type_instance.max_number),
            int(lottery_type_instance.pieces_of_draw_numbers)
        )
        
        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            additional_numbers = self._quantum_random_generation(
                int(lottery_type_instance.additional_min_number),
                int(lottery_type_instance.additional_max_number),
                int(lottery_type_instance.additional_numbers_count)
            )
        
        return sorted(main_numbers), sorted(additional_numbers)


# Globális instance
quantum_neural_ensemble = QuantumNeuralEnsemble()

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Főbejárási pont a kvantum-neurális ensemble predikcióhoz
    """
    return quantum_neural_ensemble.get_numbers(lottery_type_instance)
