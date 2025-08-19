# advanced_time_series_deep_learning_prediction.py
"""
Advanced Time Series Deep Learning Predictor
Fejlett idősor deep learning használva transformer, attention és LSTM/GRU kombinációját
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from collections import defaultdict, deque
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Dense, LSTM, GRU, Dropout, BatchNormalization, 
        Attention, MultiHeadAttention, LayerNormalization,
        Embedding, Add, GlobalAveragePooling1D
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow nem elérhető, fallback módba váltás")

class AdvancedTimeSeriesPredictor:
    """Fejlett idősor prediktor osztály."""
    
    def __init__(self, min_num: int, max_num: int, sequence_length: int = 20, target_count: int = 5):
        self.min_num = min_num
        self.max_num = max_num
        self.sequence_length = sequence_length
        self.target_count = target_count
        self.vocab_size = max_num - min_num + 1
        
        # Model komponensek
        self.transformer_model = None
        self.lstm_model = None
        self.attention_weights = None
        
        # Idősor jellemzők
        self.temporal_features = defaultdict(list)
        self.seasonal_patterns = {}
        self.trend_components = {}
        
    def build_transformer_model(self) -> Optional[Model]:
        """Transformer modell építése."""
        if not TF_AVAILABLE:
            return None
            
        try:
            # Input layer
            inputs = Input(shape=(self.sequence_length, self.vocab_size))
            
            # Positional encoding
            x = self._add_positional_encoding(inputs)
            
            # Multi-head attention layers
            attention_output = MultiHeadAttention(
                num_heads=8, 
                key_dim=64,
                dropout=0.1
            )(x, x)
            
            # Residual connection és normalizáció
            x = Add()([x, attention_output])
            x = LayerNormalization()(x)
            
            # Feed forward hálózat
            ffn_output = Dense(256, activation='relu')(x)
            ffn_output = Dropout(0.1)(ffn_output)
            ffn_output = Dense(self.vocab_size)(ffn_output)
            
            # Második residual connection
            x = Add()([x, ffn_output])
            x = LayerNormalization()(x)
            
            # Global pooling és output
            x = GlobalAveragePooling1D()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.2)(x)
            outputs = Dense(self.vocab_size, activation='sigmoid')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Transformer modell hiba: {e}")
            return None
    
    def build_lstm_model(self) -> Optional[Model]:
        """LSTM modell építése."""
        if not TF_AVAILABLE:
            return None
            
        try:
            inputs = Input(shape=(self.sequence_length, self.vocab_size))
            
            # Bidirectional LSTM layers
            x = tf.keras.layers.Bidirectional(
                LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
            )(inputs)
            x = BatchNormalization()(x)
            
            x = tf.keras.layers.Bidirectional(
                LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
            )(x)
            x = BatchNormalization()(x)
            
            # GRU layer kombinálása
            x = GRU(32, dropout=0.3, recurrent_dropout=0.3)(x)
            x = BatchNormalization()(x)
            
            # Dense layers
            x = Dense(64, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(32, activation='relu')(x)
            x = Dropout(0.2)(x)
            outputs = Dense(self.vocab_size, activation='sigmoid')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"LSTM modell hiba: {e}")
            return None
    
    def _add_positional_encoding(self, inputs):
        """Positional encoding hozzáadása."""
        if not TF_AVAILABLE:
            return inputs
            
        seq_len = self.sequence_length
        d_model = self.vocab_size
        
        # Positional encoding számítása
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        pos_encoding = tf.constant(pos_encoding, dtype=tf.float32)
        
        return inputs + pos_encoding
    
    def prepare_sequences(self, past_draws: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """Szekvenciák előkészítése deep learning modellekhez."""
        if len(past_draws) < self.sequence_length + 1:
            return np.array([]), np.array([])
        
        X, y = [], []
        
        # One-hot encoding készítése
        for i in range(len(past_draws) - self.sequence_length):
            # Input szekvencia
            sequence = []
            for j in range(i, i + self.sequence_length):
                one_hot = np.zeros(self.vocab_size)
                for num in past_draws[j]:
                    if self.min_num <= num <= self.max_num:
                        idx = num - self.min_num
                        one_hot[idx] = 1
                sequence.append(one_hot)
            X.append(sequence)
            
            # Target
            target = np.zeros(self.vocab_size)
            for num in past_draws[i + self.sequence_length]:
                if self.min_num <= num <= self.max_num:
                    idx = num - self.min_num
                    target[idx] = 1
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def extract_temporal_features(self, past_draws: List[List[int]]):
        """Időbeli jellemzők kinyerése."""
        
        # Szezonális minták
        for i, draw in enumerate(past_draws):
            week_of_year = (i % 52) + 1
            month = (i % 12) + 1
            quarter = (i % 4) + 1
            
            if week_of_year not in self.seasonal_patterns:
                self.seasonal_patterns[week_of_year] = defaultdict(int)
            if month not in self.seasonal_patterns:
                self.seasonal_patterns[month] = defaultdict(int)
            if quarter not in self.seasonal_patterns:
                self.seasonal_patterns[quarter] = defaultdict(int)
            
            for num in draw:
                if self.min_num <= num <= self.max_num:
                    self.seasonal_patterns[week_of_year][num] += 1
                    self.seasonal_patterns[month][num] += 1
                    self.seasonal_patterns[quarter][num] += 1
        
        # Trend komponensek
        window_sizes = [5, 10, 20]
        for window_size in window_sizes:
            self.trend_components[window_size] = self._calculate_moving_averages(past_draws, window_size)
    
    def _calculate_moving_averages(self, past_draws: List[List[int]], window_size: int) -> Dict[int, float]:
        """Mozgó átlagok számítása."""
        moving_averages = defaultdict(float)
        
        for i in range(len(past_draws) - window_size + 1):
            window_numbers = []
            for j in range(i, i + window_size):
                window_numbers.extend(past_draws[j])
            
            # Átlag számítása az ablakra
            for num in set(window_numbers):
                if self.min_num <= num <= self.max_num:
                    count = window_numbers.count(num)
                    moving_averages[num] += count / window_size
        
        # Normalizálás
        total_windows = len(past_draws) - window_size + 1
        for num in moving_averages:
            moving_averages[num] /= total_windows
        
        return moving_averages
    
    def train(self, past_draws: List[List[int]]):
        """Modell tanítása."""
        
        # Temporal features kinyerése
        self.extract_temporal_features(past_draws)
        
        # Szekvenciák előkészítése
        X, y = self.prepare_sequences(past_draws)
        
        if len(X) == 0:
            logger.warning("Nincs elég adat a modell tanításához")
            return
        
        # Modellek építése és tanítása
        if TF_AVAILABLE:
            try:
                # Transformer modell
                self.transformer_model = self.build_transformer_model()
                if self.transformer_model:
                    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
                    reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5)
                    
                    self.transformer_model.fit(
                        X, y,
                        epochs=50,
                        batch_size=min(32, len(X)),
                        validation_split=0.2,
                        callbacks=[early_stopping, reduce_lr],
                        verbose=0
                    )
                
                # LSTM modell
                self.lstm_model = self.build_lstm_model()
                if self.lstm_model:
                    self.lstm_model.fit(
                        X, y,
                        epochs=50,
                        batch_size=min(32, len(X)),
                        validation_split=0.2,
                        callbacks=[early_stopping, reduce_lr],
                        verbose=0
                    )
                
            except Exception as e:
                logger.error(f"Modell tanítási hiba: {e}")
    
    def predict(self, past_draws: List[List[int]]) -> List[int]:
        """Predikció generálása."""
        
        predictions = set()
        
        # Deep learning predikciók
        if TF_AVAILABLE and len(past_draws) >= self.sequence_length:
            dl_predictions = self._generate_deep_learning_predictions(past_draws)
            predictions.update(dl_predictions)
        
        # Temporal feature alapú predikciók
        temporal_predictions = self._generate_temporal_predictions()
        predictions.update(temporal_predictions)
        
        # Attention-based predikciók
        attention_predictions = self._generate_attention_predictions(past_draws)
        predictions.update(attention_predictions)
        
        # Kiegészítés ha szükséges
        while len(predictions) < self.target_count:
            # Trend alapú kiegészítés
            trend_number = self._generate_trend_based_number(predictions)
            predictions.add(trend_number)
        
        return sorted(list(predictions)[:self.target_count])
    
    def _generate_deep_learning_predictions(self, past_draws: List[List[int]]) -> List[int]:
        """Deep learning modellek predikciói."""
        predictions = []
        
        try:
            # Utolsó szekvencia előkészítése
            sequence = []
            for i in range(-self.sequence_length, 0):
                if abs(i) <= len(past_draws):
                    one_hot = np.zeros(self.vocab_size)
                    for num in past_draws[i]:
                        if self.min_num <= num <= self.max_num:
                            idx = num - self.min_num
                            one_hot[idx] = 1
                    sequence.append(one_hot)
            
            if len(sequence) == self.sequence_length:
                X = np.array([sequence])
                
                # Transformer predikció
                if self.transformer_model:
                    transformer_pred = self.transformer_model.predict(X, verbose=0)[0]
                    top_indices = np.argsort(transformer_pred)[-self.target_count:]
                    transformer_numbers = [idx + self.min_num for idx in top_indices]
                    predictions.extend(transformer_numbers)
                
                # LSTM predikció
                if self.lstm_model:
                    lstm_pred = self.lstm_model.predict(X, verbose=0)[0]
                    top_indices = np.argsort(lstm_pred)[-self.target_count:]
                    lstm_numbers = [idx + self.min_num for idx in top_indices]
                    predictions.extend(lstm_numbers)
        
        except Exception as e:
            logger.error(f"Deep learning predikció hiba: {e}")
        
        return predictions
    
    def _generate_temporal_predictions(self) -> List[int]:
        """Temporal features alapú predikciók."""
        predictions = []
        
        # Aktuális időszak szimulációja
        current_week = random.randint(1, 52)
        current_month = random.randint(1, 12)
        current_quarter = random.randint(1, 4)
        
        # Szezonális predikciók
        seasonal_candidates = []
        
        for time_period in [current_week, current_month, current_quarter]:
            if time_period in self.seasonal_patterns:
                pattern = self.seasonal_patterns[time_period]
                sorted_nums = sorted(pattern.items(), key=lambda x: x[1], reverse=True)
                seasonal_candidates.extend([num for num, _ in sorted_nums[:3]])
        
        predictions.extend(seasonal_candidates[:self.target_count])
        
        return predictions
    
    def _generate_attention_predictions(self, past_draws: List[List[int]]) -> List[int]:
        """Attention mechanizmus alapú predikciók."""
        if len(past_draws) < 5:
            return []
        
        # Számok közötti figyelmi súlyok
        attention_scores = defaultdict(float)
        
        # Utolsó néhány húzás figyelembevétele
        recent_draws = past_draws[-5:]
        
        for i, draw1 in enumerate(recent_draws):
            for j, draw2 in enumerate(recent_draws):
                if i != j:
                    # Figyelmi súly számítása a húzások között
                    attention_weight = math.exp(-(i - j)**2 / 2.0)
                    
                    for num1 in draw1:
                        for num2 in draw2:
                            if self.min_num <= num1 <= self.max_num and self.min_num <= num2 <= self.max_num:
                                # Kereszt-figyelem számítása
                                similarity = 1.0 / (1.0 + abs(num1 - num2))
                                attention_scores[num2] += attention_weight * similarity
        
        # Legjobb attention score-ok kiválasztása
        sorted_attention = sorted(attention_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [num for num, _ in sorted_attention[:self.target_count]]
    
    def _generate_trend_based_number(self, existing_predictions: set) -> int:
        """Trend alapú szám generálása."""
        
        available_numbers = [num for num in range(self.min_num, self.max_num + 1) 
                           if num not in existing_predictions]
        
        if not available_numbers:
            return random.randint(self.min_num, self.max_num)
        
        # Trend súlyok számítása
        trend_weights = []
        for num in available_numbers:
            weight = 0.0
            
            # Különböző ablakméretű trendek kombinálása
            for window_size, trends in self.trend_components.items():
                if num in trends:
                    weight += trends[num] * (1.0 / window_size)  # Kisebb ablak = nagyobb súly
            
            trend_weights.append(weight)
        
        # Súlyozott véletlen kiválasztás
        if sum(trend_weights) > 0:
            probabilities = np.array(trend_weights) / sum(trend_weights)
            selected_idx = np.random.choice(len(available_numbers), p=probabilities)
            return available_numbers[selected_idx]
        else:
            return random.choice(available_numbers)


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Advanced Time Series Deep Learning alapú lottószám predikció.
    Returns a tuple (main_numbers, additional_numbers).
    """
    try:
        # Fő számok generálása
        main_numbers = generate_advanced_ts_numbers(
            lottery_type_instance,
            min_number=int(lottery_type_instance.min_number),
            max_number=int(lottery_type_instance.max_number),
            pieces_of_draw_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
            is_main=True
        )

        # Kiegészítő számok generálása
        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            additional_numbers = generate_advanced_ts_numbers(
                lottery_type_instance,
                min_number=int(lottery_type_instance.additional_min_number),
                max_number=int(lottery_type_instance.additional_max_number),
                pieces_of_draw_numbers=int(lottery_type_instance.additional_numbers_count),
                is_main=False
            )

        return sorted(main_numbers), sorted(additional_numbers)
    
    except Exception as e:
        logger.error(f"Hiba az advanced time series predikció során: {e}")
        return generate_fallback_numbers(lottery_type_instance)


def generate_advanced_ts_numbers(
    lottery_type_instance: lg_lottery_type,
    min_number: int,
    max_number: int,
    pieces_of_draw_numbers: int,
    is_main: bool
) -> List[int]:
    """Advanced time series deep learning számgenerálás."""
    
    # Történeti adatok lekérése
    past_draws = get_historical_data(lottery_type_instance, is_main)
    
    if len(past_draws) < 10:
        return generate_smart_fallback(min_number, max_number, pieces_of_draw_numbers)
    
    # Advanced time series predictor
    predictor = AdvancedTimeSeriesPredictor(
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


def generate_smart_fallback(min_number: int, max_number: int, count: int) -> List[int]:
    """Intelligens fallback számgenerálás."""
    # Normál eloszlás alapú generálás
    center = (min_number + max_number) / 2
    std = (max_number - min_number) / 6
    
    numbers = set()
    attempts = 0
    
    while len(numbers) < count and attempts < 1000:
        num = int(np.random.normal(center, std))
        if min_number <= num <= max_number:
            numbers.add(num)
        attempts += 1
    
    # Kiegészítés ha szükséges
    if len(numbers) < count:
        remaining = [num for num in range(min_number, max_number + 1) if num not in numbers]
        random.shuffle(remaining)
        numbers.update(remaining[:count - len(numbers)])
    
    return sorted(list(numbers)[:count])


def generate_fallback_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Fallback számgenerálás."""
    main_numbers = generate_smart_fallback(
        int(lottery_type_instance.min_number),
        int(lottery_type_instance.max_number),
        int(lottery_type_instance.pieces_of_draw_numbers)
    )
    
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_smart_fallback(
            int(lottery_type_instance.additional_min_number),
            int(lottery_type_instance.additional_max_number),
            int(lottery_type_instance.additional_numbers_count)
        )
    
    return sorted(main_numbers), sorted(additional_numbers) 