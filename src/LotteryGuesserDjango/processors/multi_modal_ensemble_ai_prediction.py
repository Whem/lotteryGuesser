# multi_modal_ensemble_ai_prediction.py
"""
Multi-Modal Ensemble AI Predictor
Többmodális ensemble AI kombináló különböző gépi tanulási technikákat meta-tanulással
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict, Counter, deque
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import math
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Opcionális ML könyvtárak
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn nem elérhető")

class BasePredictor(ABC):
    """Alap prediktor interfész."""
    
    def __init__(self, name: str, min_num: int, max_num: int, target_count: int):
        self.name = name
        self.min_num = min_num
        self.max_num = max_num
        self.target_count = target_count
        self.confidence = 0.0
        self.performance_history = deque(maxlen=100)
    
    @abstractmethod
    def train(self, past_draws: List[List[int]]) -> None:
        """Modell tanítása."""
        pass
    
    @abstractmethod
    def predict(self, past_draws: List[List[int]]) -> List[int]:
        """Predikció generálása."""
        pass
    
    def evaluate_performance(self, predictions: List[int], actual: List[int]) -> float:
        """Predikció teljesítményének értékelése."""
        if not predictions or not actual:
            return 0.0
        
        # Jaccard index
        pred_set = set(predictions)
        actual_set = set(actual)
        
        intersection = len(pred_set & actual_set)
        union = len(pred_set | actual_set)
        
        performance = intersection / union if union > 0 else 0.0
        self.performance_history.append(performance)
        
        # Konfidencia frissítése
        self.confidence = np.mean(self.performance_history) if self.performance_history else 0.0
        
        return performance


class StatisticalPredictor(BasePredictor):
    """Statisztikai alapú prediktor."""
    
    def __init__(self, min_num: int, max_num: int, target_count: int):
        super().__init__("Statistical", min_num, max_num, target_count)
        self.frequency_weights = defaultdict(float)
        self.recency_weights = defaultdict(float)
        self.pattern_weights = defaultdict(float)
    
    def train(self, past_draws: List[List[int]]) -> None:
        """Statisztikai modellek tanítása."""
        
        # Frekvencia alapú súlyozás
        self._calculate_frequency_weights(past_draws)
        
        # Időbeli súlyozás
        self._calculate_recency_weights(past_draws)
        
        # Mintázat alapú súlyozás
        self._calculate_pattern_weights(past_draws)
    
    def _calculate_frequency_weights(self, past_draws: List[List[int]]):
        """Frekvencia alapú súlyok."""
        frequency = Counter()
        for draw in past_draws:
            for num in draw:
                if self.min_num <= num <= self.max_num:
                    frequency[num] += 1
        
        total = sum(frequency.values())
        for num, count in frequency.items():
            self.frequency_weights[num] = count / total if total > 0 else 0.0
    
    def _calculate_recency_weights(self, past_draws: List[List[int]]):
        """Időbeli súlyozás."""
        decay_factor = 0.95
        
        for i, draw in enumerate(past_draws):
            weight = decay_factor ** i
            for num in draw:
                if self.min_num <= num <= self.max_num:
                    self.recency_weights[num] += weight
        
        # Normalizálás
        total = sum(self.recency_weights.values())
        if total > 0:
            for num in self.recency_weights:
                self.recency_weights[num] /= total
    
    def _calculate_pattern_weights(self, past_draws: List[List[int]]):
        """Mintázat alapú súlyozás."""
        # Számok közötti távolságok elemzése
        for draw in past_draws:
            sorted_draw = sorted(draw)
            for i in range(len(sorted_draw) - 1):
                gap = sorted_draw[i + 1] - sorted_draw[i]
                self.pattern_weights[f"gap_{gap}"] += 1
        
        # Normalizálás
        total = sum(self.pattern_weights.values())
        if total > 0:
            for pattern in self.pattern_weights:
                self.pattern_weights[pattern] /= total
    
    def predict(self, past_draws: List[List[int]]) -> List[int]:
        """Statisztikai predikció."""
        
        # Kombinált súlyok számítása
        combined_weights = defaultdict(float)
        
        for num in range(self.min_num, self.max_num + 1):
            weight = (
                self.frequency_weights.get(num, 0.0) * 0.4 +
                self.recency_weights.get(num, 0.0) * 0.4 +
                self._calculate_pattern_score(num, past_draws) * 0.2
            )
            combined_weights[num] = weight
        
        # Legjobb számok kiválasztása
        sorted_numbers = sorted(combined_weights.items(), key=lambda x: x[1], reverse=True)
        predictions = [num for num, _ in sorted_numbers[:self.target_count * 2]]
        
        # Diverzitás biztosítása
        final_predictions = self._ensure_diversity(predictions)
        
        return final_predictions[:self.target_count]
    
    def _calculate_pattern_score(self, num: int, past_draws: List[List[int]]) -> float:
        """Mintázat pontszám számítása."""
        if not past_draws:
            return 0.0
        
        last_draw = past_draws[0] if past_draws else []
        score = 0.0
        
        for last_num in last_draw:
            gap = abs(num - last_num)
            pattern_key = f"gap_{gap}"
            score += self.pattern_weights.get(pattern_key, 0.0)
        
        return score / len(last_draw) if last_draw else 0.0
    
    def _ensure_diversity(self, numbers: List[int]) -> List[int]:
        """Diverzitás biztosítása."""
        diverse_numbers = []
        
        for num in numbers:
            if all(abs(num - existing) >= 2 for existing in diverse_numbers):
                diverse_numbers.append(num)
            
            if len(diverse_numbers) >= self.target_count:
                break
        
        # Kiegészítés ha szükséges
        while len(diverse_numbers) < self.target_count:
            remaining = [n for n in range(self.min_num, self.max_num + 1) 
                        if n not in diverse_numbers]
            if remaining:
                diverse_numbers.append(random.choice(remaining))
            else:
                break
        
        return diverse_numbers


class MLPredictor(BasePredictor):
    """Gépi tanulás alapú prediktor."""
    
    def __init__(self, min_num: int, max_num: int, target_count: int):
        super().__init__("MachineLearning", min_num, max_num, target_count)
        self.models = {}
        self.scaler = None
        self.feature_history = deque(maxlen=50)
    
    def train(self, past_draws: List[List[int]]) -> None:
        """ML modellek tanítása."""
        
        if not SKLEARN_AVAILABLE or len(past_draws) < 10:
            return
        
        # Feature engineering
        X, y = self._prepare_features(past_draws)
        
        if len(X) == 0:
            return
        
        # Scaler illesztése
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Különböző ML modellek tanítása
        self.models = {
            'rf': RandomForestClassifier(n_estimators=50, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=50, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'mlp': MLPClassifier(hidden_layer_sizes=(50, 25), random_state=42),
            'nb': GaussianNB()
        }
        
        for name, model in self.models.items():
            try:
                model.fit(X_scaled, y)
            except Exception as e:
                logger.error(f"Hiba a {name} modell tanításakor: {e}")
                del self.models[name]
    
    def _prepare_features(self, past_draws: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """Feature engineering."""
        if len(past_draws) < 6:
            return np.array([]), np.array([])
        
        X, y = [], []
        
        for i in range(len(past_draws) - 5):
            # Features: statisztikai jellemzők az utolsó 5 húzásból
            features = []
            
            recent_draws = past_draws[i:i+5]
            all_recent_numbers = [num for draw in recent_draws for num in draw]
            
            # Alapvető statisztikák
            features.extend([
                np.mean(all_recent_numbers),
                np.std(all_recent_numbers),
                np.median(all_recent_numbers),
                len(set(all_recent_numbers)),  # Egyedi számok száma
                max(all_recent_numbers) - min(all_recent_numbers)  # Range
            ])
            
            # Frekvencia jellemzők
            freq = Counter(all_recent_numbers)
            features.extend([
                max(freq.values()),  # Leggyakoribb szám frekvenciája
                len(freq),  # Különböző számok száma
                sum(1 for count in freq.values() if count > 1)  # Ismétlődő számok
            ])
            
            # Mintázat jellemzők
            for draw in recent_draws:
                sorted_draw = sorted(draw)
                if len(sorted_draw) > 1:
                    gaps = [sorted_draw[j+1] - sorted_draw[j] for j in range(len(sorted_draw)-1)]
                    features.extend([
                        np.mean(gaps),
                        np.std(gaps) if len(gaps) > 1 else 0
                    ])
                else:
                    features.extend([0, 0])
            
            X.append(features)
            
            # Target: következő húzás legnagyobb száma (egyszerűsítés)
            next_draw = past_draws[i+5]
            y.append(max(next_draw) if next_draw else self.min_num)
        
        return np.array(X), np.array(y)
    
    def predict(self, past_draws: List[List[int]]) -> List[int]:
        """ML alapú predikció."""
        
        if not self.models or not self.scaler or len(past_draws) < 5:
            return self._fallback_predict()
        
        # Features előkészítése az utolsó 5 húzásból
        recent_draws = past_draws[-5:]
        all_recent_numbers = [num for draw in recent_draws for num in draw]
        
        features = []
        
        # Ugyanaz a feature engineering mint a tanításnál
        features.extend([
            np.mean(all_recent_numbers),
            np.std(all_recent_numbers),
            np.median(all_recent_numbers),
            len(set(all_recent_numbers)),
            max(all_recent_numbers) - min(all_recent_numbers)
        ])
        
        freq = Counter(all_recent_numbers)
        features.extend([
            max(freq.values()),
            len(freq),
            sum(1 for count in freq.values() if count > 1)
        ])
        
        for draw in recent_draws:
            sorted_draw = sorted(draw)
            if len(sorted_draw) > 1:
                gaps = [sorted_draw[j+1] - sorted_draw[j] for j in range(len(sorted_draw)-1)]
                features.extend([
                    np.mean(gaps),
                    np.std(gaps) if len(gaps) > 1 else 0
                ])
            else:
                features.extend([0, 0])
        
        try:
            X = self.scaler.transform([features])
            
            # Ensemble predikció
            predictions = []
            for name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        # Valószínűség alapú predikció
                        probabilities = model.predict_proba(X)[0]
                        # A legnagyobb valószínűségű osztályok körül számokat generálunk
                        target_range = np.argmax(probabilities)
                        predictions.extend(self._generate_around_target(target_range))
                    else:
                        pred = model.predict(X)[0]
                        predictions.extend(self._generate_around_target(pred))
                        
                except Exception as e:
                    logger.error(f"Hiba a {name} modell predikciójában: {e}")
                    continue
            
            # Legjobb számok kiválasztása
            if predictions:
                prediction_counter = Counter(predictions)
                most_common = [num for num, _ in prediction_counter.most_common(self.target_count * 2)]
                valid_predictions = [num for num in most_common if self.min_num <= num <= self.max_num]
                
                return valid_predictions[:self.target_count] if valid_predictions else self._fallback_predict()
            
        except Exception as e:
            logger.error(f"ML predikció hiba: {e}")
        
        return self._fallback_predict()
    
    def _generate_around_target(self, target: float) -> List[int]:
        """Célérték körül számok generálása."""
        target_int = int(target)
        numbers = []
        
        # Célérték körül ±3 range-ben
        for offset in range(-3, 4):
            num = target_int + offset
            if self.min_num <= num <= self.max_num:
                numbers.append(num)
        
        return numbers
    
    def _fallback_predict(self) -> List[int]:
        """Fallback predikció."""
        return random.sample(range(self.min_num, self.max_num + 1), self.target_count)


class RuleBasedPredictor(BasePredictor):
    """Szabály alapú prediktor."""
    
    def __init__(self, min_num: int, max_num: int, target_count: int):
        super().__init__("RuleBased", min_num, max_num, target_count)
        self.rules = []
        self.rule_weights = {}
    
    def train(self, past_draws: List[List[int]]) -> None:
        """Szabályok tanulása."""
        
        # Alapvető szabályok definiálása
        self.rules = [
            self._fibonacci_rule,
            self._prime_rule,
            self._even_odd_balance_rule,
            self._sum_range_rule,
            self._consecutive_avoidance_rule,
            self._decade_distribution_rule
        ]
        
        # Szabályok súlyozása teljesítmény alapján
        for rule in self.rules:
            self.rule_weights[rule.__name__] = self._evaluate_rule(rule, past_draws)
    
    def _evaluate_rule(self, rule_func, past_draws: List[List[int]]) -> float:
        """Szabály teljesítményének értékelése."""
        if len(past_draws) < 10:
            return 0.5  # Alapértelmezett súly
        
        correct_predictions = 0
        total_predictions = 0
        
        # Utolsó 20 húzás tesztelése
        for i in range(min(20, len(past_draws) - 1)):
            test_history = past_draws[i+1:]
            actual = past_draws[i]
            
            try:
                predicted = rule_func(test_history)
                
                # Találatok számolása
                intersection = len(set(predicted) & set(actual))
                correct_predictions += intersection
                total_predictions += len(predicted)
                
            except Exception:
                continue
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.5
    
    def _fibonacci_rule(self, past_draws: List[List[int]]) -> List[int]:
        """Fibonacci számok szabálya."""
        fibonacci = [1, 1]
        while len(fibonacci) < 20:
            fibonacci.append(fibonacci[-1] + fibonacci[-2])
        
        valid_fib = [num for num in fibonacci if self.min_num <= num <= self.max_num]
        return valid_fib[:self.target_count]
    
    def _prime_rule(self, past_draws: List[List[int]]) -> List[int]:
        """Prímszámok szabálya."""
        primes = [num for num in range(self.min_num, self.max_num + 1) if self._is_prime(num)]
        return random.sample(primes, min(len(primes), self.target_count))
    
    def _even_odd_balance_rule(self, past_draws: List[List[int]]) -> List[int]:
        """Páros-páratlan egyensúly szabálya."""
        even_count = self.target_count // 2
        odd_count = self.target_count - even_count
        
        evens = [num for num in range(self.min_num, self.max_num + 1) if num % 2 == 0]
        odds = [num for num in range(self.min_num, self.max_num + 1) if num % 2 == 1]
        
        selected_evens = random.sample(evens, min(len(evens), even_count))
        selected_odds = random.sample(odds, min(len(odds), odd_count))
        
        return selected_evens + selected_odds
    
    def _sum_range_rule(self, past_draws: List[List[int]]) -> List[int]:
        """Összeg tartomány szabálya."""
        if not past_draws:
            return random.sample(range(self.min_num, self.max_num + 1), self.target_count)
        
        # Történeti összegek elemzése
        sums = [sum(draw) for draw in past_draws[-20:]]
        avg_sum = np.mean(sums)
        
        # Célösszeg körül számok generálása
        attempts = 0
        while attempts < 1000:
            candidates = random.sample(range(self.min_num, self.max_num + 1), self.target_count)
            if abs(sum(candidates) - avg_sum) < avg_sum * 0.2:  # 20% tolerancia
                return candidates
            attempts += 1
        
        return random.sample(range(self.min_num, self.max_num + 1), self.target_count)
    
    def _consecutive_avoidance_rule(self, past_draws: List[List[int]]) -> List[int]:
        """Egymást követő számok elkerülése."""
        candidates = list(range(self.min_num, self.max_num + 1))
        selected = []
        
        while len(selected) < self.target_count and candidates:
            num = random.choice(candidates)
            selected.append(num)
            
            # Szomszédos számok eltávolítása
            candidates = [c for c in candidates if abs(c - num) > 1]
        
        # Kiegészítés ha szükséges
        while len(selected) < self.target_count:
            remaining = [num for num in range(self.min_num, self.max_num + 1) 
                        if num not in selected]
            if remaining:
                selected.append(random.choice(remaining))
            else:
                break
        
        return selected
    
    def _decade_distribution_rule(self, past_draws: List[List[int]]) -> List[int]:
        """Évtized eloszlás szabálya."""
        decades = {}
        decade_size = (self.max_num - self.min_num + 1) // 10
        
        for i in range(10):
            start = self.min_num + i * decade_size
            end = min(start + decade_size - 1, self.max_num)
            decades[i] = list(range(start, end + 1))
        
        # Minden évtizedből legalább egy szám
        selected = []
        for decade_nums in decades.values():
            if decade_nums and len(selected) < self.target_count:
                selected.append(random.choice(decade_nums))
        
        # Kiegészítés
        while len(selected) < self.target_count:
            remaining = [num for num in range(self.min_num, self.max_num + 1) 
                        if num not in selected]
            if remaining:
                selected.append(random.choice(remaining))
            else:
                break
        
        return selected[:self.target_count]
    
    def _is_prime(self, n: int) -> bool:
        """Prímszám ellenőrzés."""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def predict(self, past_draws: List[List[int]]) -> List[int]:
        """Szabály alapú predikció."""
        
        all_predictions = []
        
        for rule in self.rules:
            try:
                rule_prediction = rule(past_draws)
                weight = self.rule_weights.get(rule.__name__, 0.5)
                
                # Súlyozás alapján ismétlések
                for _ in range(int(weight * 10) + 1):
                    all_predictions.extend(rule_prediction)
                    
            except Exception as e:
                logger.error(f"Hiba a {rule.__name__} szabályban: {e}")
                continue
        
        # Leggyakoribb számok kiválasztása
        if all_predictions:
            prediction_counter = Counter(all_predictions)
            most_common = [num for num, _ in prediction_counter.most_common(self.target_count * 2)]
            valid_predictions = [num for num in most_common if self.min_num <= num <= self.max_num]
            
            # Duplikátumok eltávolítása és kiegészítés
            unique_predictions = []
            for num in valid_predictions:
                if num not in unique_predictions:
                    unique_predictions.append(num)
                if len(unique_predictions) >= self.target_count:
                    break
            
            return unique_predictions[:self.target_count]
        
        return random.sample(range(self.min_num, self.max_num + 1), self.target_count)


class MetaLearner:
    """Meta-tanulás koordinátor."""
    
    def __init__(self, predictors: List[BasePredictor]):
        self.predictors = predictors
        self.meta_weights = {predictor.name: 1.0 for predictor in predictors}
        self.ensemble_history = deque(maxlen=50)
        
    def update_weights(self, predictions_dict: Dict[str, List[int]], actual: List[int]):
        """Meta-súlyok frissítése teljesítmény alapján."""
        
        for predictor_name, prediction in predictions_dict.items():
            # Teljesítmény értékelése
            performance = self._evaluate_prediction(prediction, actual)
            
            # Adaptív súly frissítés
            current_weight = self.meta_weights.get(predictor_name, 1.0)
            learning_rate = 0.1
            
            # Exponential moving average
            new_weight = current_weight * (1 - learning_rate) + performance * learning_rate
            self.meta_weights[predictor_name] = max(0.1, min(2.0, new_weight))  # Korlátok
    
    def _evaluate_prediction(self, prediction: List[int], actual: List[int]) -> float:
        """Predikció értékelése."""
        if not prediction or not actual:
            return 0.0
        
        pred_set = set(prediction)
        actual_set = set(actual)
        
        intersection = len(pred_set & actual_set)
        union = len(pred_set | actual_set)
        
        return intersection / union if union > 0 else 0.0
    
    def combine_predictions(self, predictions_dict: Dict[str, List[int]]) -> List[int]:
        """Predikciók kombinálása meta-súlyokkal."""
        
        weighted_counts = Counter()
        
        for predictor_name, prediction in predictions_dict.items():
            weight = self.meta_weights.get(predictor_name, 1.0)
            
            for num in prediction:
                weighted_counts[num] += weight
        
        # Legjobb számok kiválasztása
        most_common = [num for num, _ in weighted_counts.most_common()]
        
        return most_common


class MultiModalEnsembleAI:
    """Multi-modális ensemble AI fő osztály."""
    
    def __init__(self, min_num: int, max_num: int, target_count: int):
        self.min_num = min_num
        self.max_num = max_num
        self.target_count = target_count
        
        # Prediktorok inicializálása
        self.predictors = [
            StatisticalPredictor(min_num, max_num, target_count),
            RuleBasedPredictor(min_num, max_num, target_count)
        ]
        
        # ML prediktor hozzáadása ha elérhető
        if SKLEARN_AVAILABLE:
            self.predictors.append(MLPredictor(min_num, max_num, target_count))
        
        # Meta-tanulás
        self.meta_learner = MetaLearner(self.predictors)
        
        # Tanulási történet
        self.training_history = deque(maxlen=100)
    
    def train(self, past_draws: List[List[int]]):
        """Összes prediktor tanítása."""
        
        if len(past_draws) < 5:
            logger.warning("Nincs elég adat a tanításhoz")
            return
        
        # Minden prediktor tanítása
        for predictor in self.predictors:
            try:
                predictor.train(past_draws)
                logger.info(f"{predictor.name} prediktor sikeresen tanítva")
            except Exception as e:
                logger.error(f"Hiba a {predictor.name} prediktor tanításakor: {e}")
        
        # Meta-tanulás szimulációja a legutóbbi adatokon
        if len(past_draws) > 10:
            self._simulate_meta_learning(past_draws[-10:])
    
    def _simulate_meta_learning(self, test_draws: List[List[int]]):
        """Meta-tanulás szimulációja."""
        
        for i in range(len(test_draws) - 1):
            # Predikciók generálása
            history = test_draws[:i+1]
            actual = test_draws[i+1]
            
            predictions_dict = {}
            for predictor in self.predictors:
                try:
                    pred = predictor.predict(history)
                    predictions_dict[predictor.name] = pred
                except Exception as e:
                    logger.error(f"Hiba a {predictor.name} prediktor szimulációjában: {e}")
                    continue
            
            # Meta-súlyok frissítése
            if predictions_dict:
                self.meta_learner.update_weights(predictions_dict, actual)
    
    def predict(self, past_draws: List[List[int]]) -> List[int]:
        """Ensemble predikció."""
        
        if not past_draws:
            return random.sample(range(self.min_num, self.max_num + 1), self.target_count)
        
        # Minden prediktor predikciója
        predictions_dict = {}
        
        for predictor in self.predictors:
            try:
                pred = predictor.predict(past_draws)
                predictions_dict[predictor.name] = pred
            except Exception as e:
                logger.error(f"Hiba a {predictor.name} prediktor predikciójában: {e}")
                continue
        
        if not predictions_dict:
            return random.sample(range(self.min_num, self.max_num + 1), self.target_count)
        
        # Meta-kombinálás
        combined_predictions = self.meta_learner.combine_predictions(predictions_dict)
        
        # Finális szűrés és validálás
        final_predictions = self._finalize_predictions(combined_predictions)
        
        return final_predictions
    
    def _finalize_predictions(self, combined_predictions: List[int]) -> List[int]:
        """Finális predikciók validálása és optimalizálása."""
        
        # Érvényes számok szűrése
        valid_predictions = [num for num in combined_predictions 
                           if self.min_num <= num <= self.max_num]
        
        # Duplikátumok eltávolítása
        unique_predictions = []
        for num in valid_predictions:
            if num not in unique_predictions:
                unique_predictions.append(num)
            if len(unique_predictions) >= self.target_count:
                break
        
        # Kiegészítés ha szükséges
        while len(unique_predictions) < self.target_count:
            remaining = [num for num in range(self.min_num, self.max_num + 1) 
                        if num not in unique_predictions]
            if remaining:
                # Intelligens kiegészítés - középérték körül
                center = (self.min_num + self.max_num) / 2
                remaining.sort(key=lambda x: abs(x - center))
                unique_predictions.append(remaining[0])
            else:
                break
        
        return sorted(unique_predictions[:self.target_count])


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Multi-Modal Ensemble AI alapú lottószám predikció.
    Returns a tuple (main_numbers, additional_numbers).
    """
    try:
        # Fő számok generálása
        main_numbers = generate_multi_modal_numbers(
            lottery_type_instance,
            min_number=int(lottery_type_instance.min_number),
            max_number=int(lottery_type_instance.max_number),
            pieces_of_draw_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
            is_main=True
        )

        # Kiegészítő számok generálása
        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            additional_numbers = generate_multi_modal_numbers(
                lottery_type_instance,
                min_number=int(lottery_type_instance.additional_min_number),
                max_number=int(lottery_type_instance.additional_max_number),
                pieces_of_draw_numbers=int(lottery_type_instance.additional_numbers_count),
                is_main=False
            )

        return sorted(main_numbers), sorted(additional_numbers)
    
    except Exception as e:
        logger.error(f"Hiba a multi-modal ensemble predikció során: {e}")
        return generate_fallback_numbers(lottery_type_instance)


def generate_multi_modal_numbers(
    lottery_type_instance: lg_lottery_type,
    min_number: int,
    max_number: int,
    pieces_of_draw_numbers: int,
    is_main: bool
) -> List[int]:
    """Multi-modal ensemble számgenerálás."""
    
    # Történeti adatok lekérése
    past_draws = get_historical_data(lottery_type_instance, is_main)
    
    if len(past_draws) < 5:
        return generate_intelligent_fallback(min_number, max_number, pieces_of_draw_numbers)
    
    # Multi-modal ensemble AI
    ensemble_ai = MultiModalEnsembleAI(
        min_num=min_number, 
        max_num=max_number, 
        target_count=pieces_of_draw_numbers
    )
    
    # Tanítás és predikció
    ensemble_ai.train(past_draws)
    predictions = ensemble_ai.predict(past_draws)
    
    return predictions


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Történeti adatok lekérése."""
    try:
        queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('-id')[:100]
        
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


def generate_intelligent_fallback(min_number: int, max_number: int, count: int) -> List[int]:
    """Intelligens fallback számgenerálás."""
    
    # Kombinált heurisztikák
    predictions = set()
    
    # Fibonacci számok
    fibonacci = [1, 1]
    while len(fibonacci) < 20:
        fibonacci.append(fibonacci[-1] + fibonacci[-2])
    
    valid_fib = [num for num in fibonacci if min_number <= num <= max_number]
    predictions.update(valid_fib[:count//3])
    
    # Prímszámok
    primes = [num for num in range(min_number, max_number + 1) 
             if num > 1 and all(num % i != 0 for i in range(2, int(num**0.5) + 1))]
    predictions.update(primes[:count//3])
    
    # Normál eloszlás alapú
    center = (min_number + max_number) / 2
    std = (max_number - min_number) / 6
    
    while len(predictions) < count:
        num = int(np.random.normal(center, std))
        if min_number <= num <= max_number:
            predictions.add(num)
    
    return sorted(list(predictions)[:count])


def generate_fallback_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Fallback számgenerálás."""
    main_numbers = generate_intelligent_fallback(
        int(lottery_type_instance.min_number),
        int(lottery_type_instance.max_number),
        int(lottery_type_instance.pieces_of_draw_numbers)
    )
    
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_intelligent_fallback(
            int(lottery_type_instance.additional_min_number),
            int(lottery_type_instance.additional_max_number),
            int(lottery_type_instance.additional_numbers_count)
        )
    
    return sorted(main_numbers), sorted(additional_numbers) 