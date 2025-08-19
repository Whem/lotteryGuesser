# advanced_statistical_ensemble_predictor.py
"""
Advanced Statistical Ensemble Predictor
Fejlett statisztikai ensemble kombináló Bayesian, idősor, regressziós és eloszlás-alapú módszereket
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

# Scipy opcionális import
try:
    from scipy import stats
    from scipy.optimize import minimize
    from scipy.special import gamma, digamma
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy nem elérhető, alapvető implementációkra váltás")

class BaseStatisticalMethod(ABC):
    """Alap statisztikai módszer interfész."""
    
    def __init__(self, name: str, min_num: int, max_num: int, target_count: int):
        self.name = name
        self.min_num = min_num
        self.max_num = max_num
        self.target_count = target_count
        self.confidence = 0.5
        self.performance_history = deque(maxlen=50)
    
    @abstractmethod
    def fit(self, past_draws: List[List[int]]) -> None:
        """Modell illesztése a történeti adatokra."""
        pass
    
    @abstractmethod
    def predict_probabilities(self, past_draws: List[List[int]]) -> Dict[int, float]:
        """Valószínűségek predikciója minden számra."""
        pass
    
    def evaluate_performance(self, predictions: Dict[int, float], actual: List[int]) -> float:
        """Teljesítmény értékelése."""
        
        # Top predictions kiválasztása
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_predictions = [num for num, _ in sorted_preds[:self.target_count]]
        
        # Jaccard index
        pred_set = set(top_predictions)
        actual_set = set(actual)
        
        intersection = len(pred_set & actual_set)
        union = len(pred_set | actual_set)
        
        performance = intersection / union if union > 0 else 0.0
        self.performance_history.append(performance)
        
        # Konfidencia frissítése
        self.confidence = np.mean(self.performance_history) if self.performance_history else 0.5
        
        return performance


class BayesianMethod(BaseStatisticalMethod):
    """Bayesian statisztikai módszer."""
    
    def __init__(self, min_num: int, max_num: int, target_count: int):
        super().__init__("Bayesian", min_num, max_num, target_count)
        
        # Bayesian paraméterek
        self.prior_alpha = 1.0  # Dirichlet prior
        self.posterior_params = {}
        self.likelihood_cache = {}
        
    def fit(self, past_draws: List[List[int]]) -> None:
        """Bayesian modell illesztése."""
        
        # Prior inicializálása
        self.posterior_params = defaultdict(lambda: self.prior_alpha)
        
        # Likelihood számítása
        for draw in past_draws:
            for num in draw:
                if self.min_num <= num <= self.max_num:
                    self.posterior_params[num] += 1.0
        
        # Posterior normalizálás
        total_observations = sum(self.posterior_params.values())
        self.posterior_params = {
            num: count / total_observations 
            for num, count in self.posterior_params.items()
        }
        
        # Likelihood cache frissítése
        self._update_likelihood_cache(past_draws)
    
    def _update_likelihood_cache(self, past_draws: List[List[int]]):
        """Likelihood cache frissítése."""
        
        # Számok közötti feltételes valószínűségek
        conditional_probs = defaultdict(lambda: defaultdict(float))
        
        for draw in past_draws:
            for i, num1 in enumerate(draw):
                for j, num2 in enumerate(draw):
                    if i != j and self.min_num <= num1 <= self.max_num and self.min_num <= num2 <= self.max_num:
                        conditional_probs[num1][num2] += 1.0
        
        # Normalizálás
        for num1 in conditional_probs:
            total = sum(conditional_probs[num1].values())
            if total > 0:
                for num2 in conditional_probs[num1]:
                    conditional_probs[num1][num2] /= total
        
        self.likelihood_cache = conditional_probs
    
    def predict_probabilities(self, past_draws: List[List[int]]) -> Dict[int, float]:
        """Bayesian predikció."""
        
        probabilities = {}
        
        # Base posterior probabilities
        for num in range(self.min_num, self.max_num + 1):
            base_prob = self.posterior_params.get(num, self.prior_alpha / self.target_count)
            
            # Conditional likelihood updates
            likelihood_update = 1.0
            if past_draws:
                last_draw = past_draws[-1]
                for last_num in last_draw:
                    if last_num in self.likelihood_cache and num in self.likelihood_cache[last_num]:
                        likelihood_update *= (1.0 + self.likelihood_cache[last_num][num])
            
            # Posterior update
            probabilities[num] = base_prob * likelihood_update
        
        # Normalizálás
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {num: prob / total_prob for num, prob in probabilities.items()}
        
        return probabilities


class TimeSeriesMethod(BaseStatisticalMethod):
    """Idősor statisztikai módszer."""
    
    def __init__(self, min_num: int, max_num: int, target_count: int):
        super().__init__("TimeSeries", min_num, max_num, target_count)
        
        # Idősor paraméterek
        self.trend_coefficients = {}
        self.seasonal_patterns = {}
        self.autocorrelation = {}
        self.window_size = 20
        
    def fit(self, past_draws: List[List[int]]) -> None:
        """Idősor modell illesztése."""
        
        # Számok időbeli sorozatainak létrehozása
        number_series = defaultdict(list)
        
        for i, draw in enumerate(past_draws):
            for num in range(self.min_num, self.max_num + 1):
                # Binary time series: 1 if number appears, 0 otherwise
                appears = 1 if num in draw else 0
                number_series[num].append(appears)
        
        # Trend analysis minden számra
        for num, series in number_series.items():
            if len(series) >= 10:
                self.trend_coefficients[num] = self._calculate_trend(series)
                self.seasonal_patterns[num] = self._calculate_seasonality(series)
                self.autocorrelation[num] = self._calculate_autocorrelation(series)
    
    def _calculate_trend(self, series: List[int]) -> float:
        """Lineáris trend számítása."""
        n = len(series)
        if n < 2:
            return 0.0
        
        # Simple linear regression
        x = np.arange(n)
        y = np.array(series)
        
        # Slope calculation
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def _calculate_seasonality(self, series: List[int], period: int = 12) -> float:
        """Szezonalitás számítása."""
        if len(series) < period * 2:
            return 0.0
        
        # Autocorrelation at seasonal lag
        return self._autocorr_at_lag(series, period)
    
    def _calculate_autocorrelation(self, series: List[int]) -> Dict[int, float]:
        """Autokorrelációs függvény."""
        autocorr = {}
        max_lag = min(10, len(series) // 4)
        
        for lag in range(1, max_lag + 1):
            autocorr[lag] = self._autocorr_at_lag(series, lag)
        
        return autocorr
    
    def _autocorr_at_lag(self, series: List[int], lag: int) -> float:
        """Autokorrelációs együttható egy adott lag-nél."""
        if len(series) <= lag:
            return 0.0
        
        y1 = series[:-lag]
        y2 = series[lag:]
        
        if len(y1) == 0 or len(y2) == 0:
            return 0.0
        
        # Pearson correlation coefficient
        return np.corrcoef(y1, y2)[0, 1] if len(y1) > 1 else 0.0
    
    def predict_probabilities(self, past_draws: List[List[int]]) -> Dict[int, float]:
        """Idősor alapú predikció."""
        
        probabilities = {}
        
        for num in range(self.min_num, self.max_num + 1):
            prob = 0.5  # Base probability
            
            # Trend component
            if num in self.trend_coefficients:
                trend = self.trend_coefficients[num]
                prob += trend * 0.3
            
            # Seasonal component
            if num in self.seasonal_patterns:
                seasonal = self.seasonal_patterns[num]
                prob += seasonal * 0.2
            
            # Autocorrelation component
            if num in self.autocorrelation and past_draws:
                # Check recent appearances
                recent_window = past_draws[-5:]
                recent_appearances = sum(1 for draw in recent_window if num in draw)
                
                # Use autocorrelation to predict continuation
                if recent_appearances > 0:
                    autocorr_effect = 0.0
                    for lag, corr in self.autocorrelation[num].items():
                        if lag <= len(recent_window):
                            autocorr_effect += corr * (recent_appearances / len(recent_window))
                    
                    prob += autocorr_effect * 0.2
            
            # Normalize
            prob = max(0.01, min(0.99, prob))
            probabilities[num] = prob
        
        # Final normalization
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {num: prob / total_prob for num, prob in probabilities.items()}
        
        return probabilities


class RegressionMethod(BaseStatisticalMethod):
    """Regressziós statisztikai módszer."""
    
    def __init__(self, min_num: int, max_num: int, target_count: int):
        super().__init__("Regression", min_num, max_num, target_count)
        
        # Regressziós paraméterek
        self.feature_weights = {}
        self.intercepts = {}
        self.feature_means = {}
        self.feature_stds = {}
        
    def fit(self, past_draws: List[List[int]]) -> None:
        """Regressziós modell illesztése."""
        
        if len(past_draws) < 10:
            return
        
        # Feature engineering
        features, targets = self._prepare_regression_data(past_draws)
        
        if not features:
            return
        
        # Multiple regression for each target number
        for num in range(self.min_num, self.max_num + 1):
            if num in targets:
                y = targets[num]
                if len(y) > 5 and np.std(y) > 0:
                    # Simple linear regression
                    self.feature_weights[num] = self._fit_linear_regression(features, y)
    
    def _prepare_regression_data(self, past_draws: List[List[int]]) -> Tuple[List[List[float]], Dict[int, List[int]]]:
        """Regressziós adatok előkészítése."""
        
        features = []
        targets = defaultdict(list)
        
        for i in range(len(past_draws) - 1):
            # Features from current draw
            current_draw = past_draws[i]
            next_draw = past_draws[i + 1]
            
            # Statistical features
            feature_vector = [
                np.mean(current_draw),
                np.std(current_draw),
                np.median(current_draw),
                max(current_draw) - min(current_draw),  # Range
                len(set(current_draw)),  # Unique count
                sum(current_draw),  # Sum
                len([x for x in current_draw if x % 2 == 0]),  # Even count
                len([x for x in current_draw if x % 2 == 1])   # Odd count
            ]
            
            features.append(feature_vector)
            
            # Targets
            for num in range(self.min_num, self.max_num + 1):
                target_val = 1 if num in next_draw else 0
                targets[num].append(target_val)
        
        return features, targets
    
    def _fit_linear_regression(self, X: List[List[float]], y: List[int]) -> Dict[str, float]:
        """Egyszerű lineáris regresszió illesztése."""
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        # Feature normalization
        X_mean = np.mean(X_array, axis=0)
        X_std = np.std(X_array, axis=0)
        X_std[X_std == 0] = 1  # Avoid division by zero
        
        X_normalized = (X_array - X_mean) / X_std
        
        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(len(X_normalized)), X_normalized])
        
        # Least squares solution
        try:
            coefficients = np.linalg.lstsq(X_with_intercept, y_array, rcond=None)[0]
            
            return {
                'intercept': coefficients[0],
                'weights': coefficients[1:].tolist(),
                'feature_means': X_mean.tolist(),
                'feature_stds': X_std.tolist()
            }
        except np.linalg.LinAlgError:
            return {'intercept': 0.0, 'weights': [0.0] * len(X_mean), 
                   'feature_means': X_mean.tolist(), 'feature_stds': X_std.tolist()}
    
    def predict_probabilities(self, past_draws: List[List[int]]) -> Dict[int, float]:
        """Regressziós predikció."""
        
        if not past_draws:
            return {num: 1.0 / (self.max_num - self.min_num + 1) 
                   for num in range(self.min_num, self.max_num + 1)}
        
        probabilities = {}
        
        # Current features
        current_draw = past_draws[-1]
        current_features = [
            np.mean(current_draw),
            np.std(current_draw),
            np.median(current_draw),
            max(current_draw) - min(current_draw),
            len(set(current_draw)),
            sum(current_draw),
            len([x for x in current_draw if x % 2 == 0]),
            len([x for x in current_draw if x % 2 == 1])
        ]
        
        for num in range(self.min_num, self.max_num + 1):
            if num in self.feature_weights:
                weights_data = self.feature_weights[num]
                
                # Normalize features
                normalized_features = []
                for i, feature in enumerate(current_features):
                    mean_val = weights_data['feature_means'][i]
                    std_val = weights_data['feature_stds'][i]
                    normalized_val = (feature - mean_val) / std_val if std_val != 0 else 0
                    normalized_features.append(normalized_val)
                
                # Predict
                prediction = weights_data['intercept']
                for i, weight in enumerate(weights_data['weights']):
                    prediction += weight * normalized_features[i]
                
                # Sigmoid activation
                probability = 1 / (1 + np.exp(-prediction))
                probabilities[num] = probability
            else:
                probabilities[num] = 0.5
        
        # Normalization
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {num: prob / total_prob for num, prob in probabilities.items()}
        
        return probabilities


class DistributionMethod(BaseStatisticalMethod):
    """Eloszlás-alapú statisztikai módszer."""
    
    def __init__(self, min_num: int, max_num: int, target_count: int):
        super().__init__("Distribution", min_num, max_num, target_count)
        
        # Eloszlási paraméterek
        self.fitted_distributions = {}
        self.distribution_weights = {}
        
    def fit(self, past_draws: List[List[int]]) -> None:
        """Eloszlási modellek illesztése."""
        
        # Összes szám összegyűjtése
        all_numbers = []
        for draw in past_draws:
            all_numbers.extend(draw)
        
        if not all_numbers:
            return
        
        # Különböző eloszlások illesztése
        distributions_to_test = ['normal', 'uniform', 'exponential', 'beta']
        
        if SCIPY_AVAILABLE:
            distributions_to_test.extend(['gamma', 'lognormal', 'weibull'])
        
        best_distribution = None
        best_score = float('-inf')
        
        for dist_name in distributions_to_test:
            try:
                score = self._fit_distribution(all_numbers, dist_name)
                if score > best_score:
                    best_score = score
                    best_distribution = dist_name
            except Exception as e:
                logger.debug(f"Hiba a {dist_name} eloszlás illesztésében: {e}")
                continue
        
        if best_distribution:
            self.fitted_distributions['best'] = best_distribution
            self._calculate_distribution_probabilities(all_numbers, best_distribution)
    
    def _fit_distribution(self, data: List[int], distribution_name: str) -> float:
        """Eloszlás illesztése és goodness-of-fit számítása."""
        
        if distribution_name == 'normal':
            mean = np.mean(data)
            std = np.std(data)
            self.fitted_distributions['normal'] = {'mean': mean, 'std': std}
            
            # KS test simulation
            expected = np.random.normal(mean, std, len(data))
            return -np.mean((np.sort(data) - np.sort(expected))**2)
        
        elif distribution_name == 'uniform':
            min_val = min(data)
            max_val = max(data)
            self.fitted_distributions['uniform'] = {'min': min_val, 'max': max_val}
            
            # Uniform distribution score
            range_coverage = (max_val - min_val) / (self.max_num - self.min_num)
            return range_coverage
        
        elif distribution_name == 'exponential':
            rate = 1.0 / np.mean(data) if np.mean(data) > 0 else 1.0
            self.fitted_distributions['exponential'] = {'rate': rate}
            
            # Exponential fit score
            expected = np.random.exponential(1/rate, len(data))
            return -np.mean((np.sort(data) - np.sort(expected))**2)
        
        elif distribution_name == 'beta':
            # Simple beta distribution fitting
            normalized_data = [(x - self.min_num) / (self.max_num - self.min_num) for x in data]
            mean_norm = np.mean(normalized_data)
            var_norm = np.var(normalized_data)
            
            if var_norm > 0 and 0 < mean_norm < 1:
                alpha = mean_norm * ((mean_norm * (1 - mean_norm)) / var_norm - 1)
                beta = (1 - mean_norm) * ((mean_norm * (1 - mean_norm)) / var_norm - 1)
                
                self.fitted_distributions['beta'] = {'alpha': max(0.1, alpha), 'beta': max(0.1, beta)}
                return mean_norm * (1 - mean_norm)  # Simple score
        
        # SciPy distributions
        if SCIPY_AVAILABLE:
            if distribution_name == 'gamma':
                shape, loc, scale = stats.gamma.fit(data)
                self.fitted_distributions['gamma'] = {'shape': shape, 'loc': loc, 'scale': scale}
                return -stats.gamma.nnlf((shape, loc, scale), data)
            
            elif distribution_name == 'lognormal':
                shape, loc, scale = stats.lognorm.fit(data)
                self.fitted_distributions['lognormal'] = {'shape': shape, 'loc': loc, 'scale': scale}
                return -stats.lognorm.nnlf((shape, loc, scale), data)
        
        return float('-inf')
    
    def _calculate_distribution_probabilities(self, data: List[int], best_distribution: str):
        """Eloszlás alapú valószínűségek számítása."""
        
        self.distribution_weights = {}
        
        if best_distribution == 'normal' and 'normal' in self.fitted_distributions:
            params = self.fitted_distributions['normal']
            for num in range(self.min_num, self.max_num + 1):
                # Normal PDF
                prob = np.exp(-0.5 * ((num - params['mean']) / params['std'])**2)
                self.distribution_weights[num] = prob
        
        elif best_distribution == 'uniform':
            # Uniform weights
            for num in range(self.min_num, self.max_num + 1):
                self.distribution_weights[num] = 1.0
        
        elif best_distribution == 'exponential' and 'exponential' in self.fitted_distributions:
            params = self.fitted_distributions['exponential']
            for num in range(self.min_num, self.max_num + 1):
                # Exponential PDF
                prob = params['rate'] * np.exp(-params['rate'] * num)
                self.distribution_weights[num] = prob
        
        elif best_distribution == 'beta' and 'beta' in self.fitted_distributions:
            params = self.fitted_distributions['beta']
            for num in range(self.min_num, self.max_num + 1):
                # Beta PDF for normalized values
                x_norm = (num - self.min_num) / (self.max_num - self.min_num)
                if 0 < x_norm < 1:
                    prob = (x_norm**(params['alpha']-1)) * ((1-x_norm)**(params['beta']-1))
                    self.distribution_weights[num] = prob
                else:
                    self.distribution_weights[num] = 0.01
        
        # Normalization
        total = sum(self.distribution_weights.values())
        if total > 0:
            self.distribution_weights = {
                num: weight / total for num, weight in self.distribution_weights.items()
            }
    
    def predict_probabilities(self, past_draws: List[List[int]]) -> Dict[int, float]:
        """Eloszlás alapú predikció."""
        
        if not self.distribution_weights:
            # Uniform fallback
            uniform_prob = 1.0 / (self.max_num - self.min_num + 1)
            return {num: uniform_prob for num in range(self.min_num, self.max_num + 1)}
        
        # Adaptive weighting based on recent data
        if past_draws:
            recent_numbers = [num for draw in past_draws[-5:] for num in draw]
            
            # Adjust weights based on recent frequency
            adjusted_weights = {}
            for num in range(self.min_num, self.max_num + 1):
                base_weight = self.distribution_weights.get(num, 0.001)
                recent_freq = recent_numbers.count(num) / len(recent_numbers) if recent_numbers else 0
                
                # Combination of distribution weight and recent frequency
                adjusted_weights[num] = base_weight * 0.7 + recent_freq * 0.3
            
            # Normalization
            total = sum(adjusted_weights.values())
            if total > 0:
                adjusted_weights = {num: weight / total for num, weight in adjusted_weights.items()}
            
            return adjusted_weights
        
        return self.distribution_weights


class AdvancedStatisticalEnsemble:
    """Fejlett statisztikai ensemble koordinátor."""
    
    def __init__(self, min_num: int, max_num: int, target_count: int):
        self.min_num = min_num
        self.max_num = max_num
        self.target_count = target_count
        
        # Statisztikai módszerek
        self.methods = [
            BayesianMethod(min_num, max_num, target_count),
            TimeSeriesMethod(min_num, max_num, target_count),
            RegressionMethod(min_num, max_num, target_count),
            DistributionMethod(min_num, max_num, target_count)
        ]
        
        # Ensemble súlyok
        self.method_weights = {method.name: 1.0 for method in self.methods}
        
        # Meta-tanulás
        self.ensemble_history = deque(maxlen=50)
        self.cross_validation_scores = {}
        
    def train(self, past_draws: List[List[int]]):
        """Ensemble tanítás és validáció."""
        
        if len(past_draws) < 10:
            logger.warning("Nincs elég adat a statisztikai ensemble tanításához")
            return
        
        # Minden módszer tanítása
        for method in self.methods:
            try:
                method.fit(past_draws)
                logger.info(f"{method.name} módszer sikeresen tanítva")
            except Exception as e:
                logger.error(f"Hiba a {method.name} módszer tanításakor: {e}")
        
        # Cross-validation
        self._perform_cross_validation(past_draws)
        
        # Súlyok optimalizálása
        self._optimize_ensemble_weights()
    
    def _perform_cross_validation(self, past_draws: List[List[int]]):
        """Cross-validation végrehajtása."""
        
        cv_folds = min(5, len(past_draws) // 4)
        fold_size = len(past_draws) // cv_folds
        
        method_scores = defaultdict(list)
        
        for fold in range(cv_folds):
            # Train/test split
            test_start = fold * fold_size
            test_end = min((fold + 1) * fold_size, len(past_draws))
            
            train_data = past_draws[:test_start] + past_draws[test_end:]
            test_data = past_draws[test_start:test_end]
            
            if not train_data or not test_data:
                continue
            
            # Minden módszer tesztelése
            for method in self.methods:
                try:
                    method.fit(train_data)
                    
                    fold_score = 0.0
                    for test_draw in test_data:
                        probabilities = method.predict_probabilities(train_data)
                        performance = method.evaluate_performance(probabilities, test_draw)
                        fold_score += performance
                    
                    avg_fold_score = fold_score / len(test_data)
                    method_scores[method.name].append(avg_fold_score)
                    
                except Exception as e:
                    logger.error(f"CV hiba a {method.name} módszernél: {e}")
                    method_scores[method.name].append(0.0)
        
        # CV átlagok
        for method_name, scores in method_scores.items():
            self.cross_validation_scores[method_name] = np.mean(scores) if scores else 0.0
    
    def _optimize_ensemble_weights(self):
        """Ensemble súlyok optimalizálása CV eredmények alapján."""
        
        if not self.cross_validation_scores:
            return
        
        # Súlyok normalizálása CV pontszámok alapján
        total_score = sum(self.cross_validation_scores.values())
        
        if total_score > 0:
            for method_name, score in self.cross_validation_scores.items():
                # Exponenciális súlyozás a jobb teljesítményű módszereknek
                normalized_score = score / total_score
                self.method_weights[method_name] = np.exp(normalized_score * 2)
        
        # Súlyok renormalizálása
        total_weight = sum(self.method_weights.values())
        if total_weight > 0:
            self.method_weights = {
                name: weight / total_weight 
                for name, weight in self.method_weights.items()
            }
    
    def predict(self, past_draws: List[List[int]]) -> List[int]:
        """Ensemble predikció."""
        
        if not past_draws:
            return random.sample(range(self.min_num, self.max_num + 1), self.target_count)
        
        # Minden módszer predikciója
        method_probabilities = {}
        
        for method in self.methods:
            try:
                probabilities = method.predict_probabilities(past_draws)
                method_probabilities[method.name] = probabilities
            except Exception as e:
                logger.error(f"Hiba a {method.name} predikciójában: {e}")
                continue
        
        if not method_probabilities:
            return random.sample(range(self.min_num, self.max_num + 1), self.target_count)
        
        # Súlyozott kombináció
        combined_probabilities = self._combine_probabilities(method_probabilities)
        
        # Top számok kiválasztása
        sorted_numbers = sorted(combined_probabilities.items(), key=lambda x: x[1], reverse=True)
        predictions = [num for num, _ in sorted_numbers[:self.target_count * 2]]
        
        # Finális szűrés és validálás
        final_predictions = self._finalize_predictions(predictions)
        
        return final_predictions[:self.target_count]
    
    def _combine_probabilities(self, method_probabilities: Dict[str, Dict[int, float]]) -> Dict[int, float]:
        """Valószínűségek súlyozott kombinálása."""
        
        combined = defaultdict(float)
        
        for method_name, probabilities in method_probabilities.items():
            weight = self.method_weights.get(method_name, 0.0)
            
            for num, prob in probabilities.items():
                combined[num] += weight * prob
        
        # Normalizálás
        total_prob = sum(combined.values())
        if total_prob > 0:
            combined = {num: prob / total_prob for num, prob in combined.items()}
        
        return combined
    
    def _finalize_predictions(self, predictions: List[int]) -> List[int]:
        """Predikciók finalizálása diverzitással."""
        
        final_predictions = []
        
        for num in predictions:
            if self.min_num <= num <= self.max_num:
                # Diverzitás ellenőrzése
                if not final_predictions or all(abs(num - existing) >= 2 for existing in final_predictions):
                    final_predictions.append(num)
                
                if len(final_predictions) >= self.target_count:
                    break
        
        # Kiegészítés ha szükséges
        while len(final_predictions) < self.target_count:
            remaining = [num for num in range(self.min_num, self.max_num + 1) 
                        if num not in final_predictions]
            if remaining:
                # Statisztikai heurisztika alapú kiegészítés
                center = (self.min_num + self.max_num) / 2
                remaining.sort(key=lambda x: abs(x - center))
                final_predictions.append(remaining[0])
            else:
                break
        
        return sorted(final_predictions)


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Advanced Statistical Ensemble alapú lottószám predikció.
    Returns a tuple (main_numbers, additional_numbers).
    """
    try:
        # Fő számok generálása
        main_numbers = generate_statistical_ensemble_numbers(
            lottery_type_instance,
            min_number=int(lottery_type_instance.min_number),
            max_number=int(lottery_type_instance.max_number),
            pieces_of_draw_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
            is_main=True
        )

        # Kiegészítő számok generálása
        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            additional_numbers = generate_statistical_ensemble_numbers(
                lottery_type_instance,
                min_number=int(lottery_type_instance.additional_min_number),
                max_number=int(lottery_type_instance.additional_max_number),
                pieces_of_draw_numbers=int(lottery_type_instance.additional_numbers_count),
                is_main=False
            )

        return sorted(main_numbers), sorted(additional_numbers)
    
    except Exception as e:
        logger.error(f"Hiba a statistical ensemble predikció során: {e}")
        return generate_fallback_numbers(lottery_type_instance)


def generate_statistical_ensemble_numbers(
    lottery_type_instance: lg_lottery_type,
    min_number: int,
    max_number: int,
    pieces_of_draw_numbers: int,
    is_main: bool
) -> List[int]:
    """Statistical ensemble számgenerálás."""
    
    # Történeti adatok lekérése
    past_draws = get_historical_data(lottery_type_instance, is_main)
    
    if len(past_draws) < 10:
        return generate_statistical_fallback(min_number, max_number, pieces_of_draw_numbers)
    
    # Statistical ensemble
    ensemble = AdvancedStatisticalEnsemble(
        min_num=min_number, 
        max_num=max_number, 
        target_count=pieces_of_draw_numbers
    )
    
    # Tanítás és predikció
    ensemble.train(past_draws)
    predictions = ensemble.predict(past_draws)
    
    return predictions


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Történeti adatok lekérése."""
    try:
        queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('-id')[:200]
        
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


def generate_statistical_fallback(min_number: int, max_number: int, count: int) -> List[int]:
    """Statisztikai fallback számgenerálás."""
    
    # Kombinált statisztikai heurisztikák
    predictions = set()
    
    # Normal distribution based
    center = (min_number + max_number) / 2
    std = (max_number - min_number) / 6
    
    for _ in range(count // 2):
        num = int(np.random.normal(center, std))
        num = max(min_number, min(num, max_number))
        predictions.add(num)
    
    # Uniform distribution based
    remaining_count = count - len(predictions)
    uniform_samples = random.sample(range(min_number, max_number + 1), 
                                  min(remaining_count * 2, max_number - min_number + 1))
    
    for num in uniform_samples:
        if len(predictions) >= count:
            break
        predictions.add(num)
    
    # Kiegészítés ha szükséges
    while len(predictions) < count:
        num = random.randint(min_number, max_number)
        predictions.add(num)
    
    return sorted(list(predictions)[:count])


def generate_fallback_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Fallback számgenerálás."""
    main_numbers = generate_statistical_fallback(
        int(lottery_type_instance.min_number),
        int(lottery_type_instance.max_number),
        int(lottery_type_instance.pieces_of_draw_numbers)
    )
    
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_statistical_fallback(
            int(lottery_type_instance.additional_min_number),
            int(lottery_type_instance.additional_max_number),
            int(lottery_type_instance.additional_numbers_count)
        )
    
    return sorted(main_numbers), sorted(additional_numbers) 