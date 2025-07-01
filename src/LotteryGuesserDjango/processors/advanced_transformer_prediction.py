# advanced_transformer_prediction.py
"""
Modern Transformer-alapú lottószám predikció
Speciálisan az Eurojackpot-hoz optimalizálva, de univerzális
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerBlock(layers.Layer):
    """Multi-head self-attention Transformer block."""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class EurojackpotTransformerPredictor:
    """Advanced Transformer-based predictor for Eurojackpot lottery numbers."""
    
    def __init__(self, 
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 ff_dim: int = 256,
                 num_transformer_blocks: int = 4,
                 dropout_rate: float = 0.2,
                 sequence_length: int = 20):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout_rate = dropout_rate
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        
    def build_model(self, vocab_size: int, output_dim: int) -> keras.Model:
        """Build the Transformer model architecture."""
        inputs = layers.Input(shape=(self.sequence_length, 1))
        
        # Embedding layer
        embedding_layer = layers.Dense(self.embed_dim)(inputs)
        x = layers.Dropout(self.dropout_rate)(embedding_layer)
        
        # Positional encoding
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        positions = tf.cast(positions, tf.float32)
        position_embedding = layers.Dense(self.embed_dim)(
            tf.expand_dims(positions, axis=1)
        )
        x += position_embedding
        
        # Transformer blocks
        for _ in range(self.num_transformer_blocks):
            x = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, self.dropout_rate)(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Final dense layers
        x = layers.Dense(self.ff_dim, activation="relu")(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(output_dim, activation="linear")(x)
        
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )
        
        return model
    
    def prepare_sequences(self, data: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare input sequences for training."""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            sequence = data[i:i + self.sequence_length]
            target = data[i + self.sequence_length]
            
            # Flatten sequences
            flat_sequence = [num for draw in sequence for num in draw]
            X.append(flat_sequence)
            y.append(target)
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for Transformer input
        X = X.reshape(X.shape[0], self.sequence_length, -1)
        
        return X, y
    
    def extract_advanced_features(self, sequences: List[List[int]]) -> Dict[str, float]:
        """Extract advanced statistical features from number sequences."""
        all_numbers = [num for seq in sequences for num in seq]
        
        features = {
            'mean': np.mean(all_numbers),
            'std': np.std(all_numbers),
            'skewness': self._calculate_skewness(all_numbers),
            'kurtosis': self._calculate_kurtosis(all_numbers),
            'range_coverage': len(set(all_numbers)) / len(all_numbers) if all_numbers else 0,
            'odd_even_ratio': sum(1 for x in all_numbers if x % 2 == 1) / len(all_numbers) if all_numbers else 0,
            'consecutive_ratio': self._calculate_consecutive_ratio(sequences),
            'gap_variance': np.var([max(seq) - min(seq) for seq in sequences]) if sequences else 0,
        }
        
        return features
    
    def _calculate_skewness(self, data: List[int]) -> float:
        """Calculate skewness of the data."""
        if len(data) < 3:
            return 0.0
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return (n * sum(((x - mean) / std) ** 3 for x in data)) / ((n - 1) * (n - 2))
    
    def _calculate_kurtosis(self, data: List[int]) -> float:
        """Calculate kurtosis of the data."""
        if len(data) < 4:
            return 0.0
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return (n * (n + 1) * sum(((x - mean) / std) ** 4 for x in data)) / ((n - 1) * (n - 2) * (n - 3)) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    
    def _calculate_consecutive_ratio(self, sequences: List[List[int]]) -> float:
        """Calculate ratio of consecutive numbers in sequences."""
        total_consecutive = 0
        total_pairs = 0
        
        for seq in sequences:
            sorted_seq = sorted(seq)
            for i in range(len(sorted_seq) - 1):
                total_pairs += 1
                if sorted_seq[i + 1] - sorted_seq[i] == 1:
                    total_consecutive += 1
        
        return total_consecutive / total_pairs if total_pairs > 0 else 0.0


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate lottery numbers using advanced Transformer prediction.
    
    Args:
        lottery_type_instance: Lottery type configuration
        
    Returns:
        Tuple of (main_numbers, additional_numbers)
    """
    try:
        # Generate main numbers
        main_numbers = generate_number_set(
            lottery_type_instance,
            min_num=int(lottery_type_instance.min_number),
            max_num=int(lottery_type_instance.max_number),
            required_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
            is_main=True
        )
        
        # Generate additional numbers if needed
        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            additional_numbers = generate_number_set(
                lottery_type_instance,
                min_num=int(lottery_type_instance.additional_min_number),
                max_num=int(lottery_type_instance.additional_max_number),
                required_numbers=int(lottery_type_instance.additional_numbers_count),
                is_main=False
            )
        
        return main_numbers, additional_numbers
        
    except Exception as e:
        logger.error(f"Error in advanced_transformer_prediction: {e}")
        # Fallback to random selection
        return generate_fallback_numbers(lottery_type_instance)


def generate_number_set(
    lottery_type_instance: lg_lottery_type,
    min_num: int,
    max_num: int,
    required_numbers: int,
    is_main: bool
) -> List[int]:
    """Generate numbers using Transformer model."""
    
    # Get historical data
    past_draws = get_historical_data(lottery_type_instance, is_main)
    
    if len(past_draws) < 30:  # Not enough data for Transformer
        return generate_smart_fallback(min_num, max_num, required_numbers, past_draws)
    
    try:
        # Initialize predictor
        predictor = EurojackpotTransformerPredictor()
        
        # Prepare training data
        if len(past_draws) >= predictor.sequence_length + 1:
            X, y = predictor.prepare_sequences(past_draws)
            
            if X.shape[0] > 0:
                # Build and train model
                vocab_size = max_num - min_num + 1
                model = predictor.build_model(vocab_size, required_numbers)
                
                # Train with early stopping
                early_stop = keras.callbacks.EarlyStopping(
                    monitor='loss', patience=10, restore_best_weights=True
                )
                
                # Quick training for real-time prediction
                model.fit(
                    X, y,
                    epochs=50,
                    batch_size=min(32, X.shape[0]),
                    verbose=0,
                    callbacks=[early_stop],
                    validation_split=0.2 if X.shape[0] > 10 else 0
                )
                
                # Generate prediction
                last_sequence = X[-1:] if X.shape[0] > 0 else np.zeros((1, predictor.sequence_length, required_numbers))
                prediction = model.predict(last_sequence, verbose=0)[0]
                
                # Process prediction to valid numbers
                predicted_numbers = process_transformer_prediction(
                    prediction, min_num, max_num, required_numbers, past_draws
                )
                
                return sorted(predicted_numbers)
    
    except Exception as e:
        logger.warning(f"Transformer prediction failed: {e}, using fallback")
    
    # Fallback to intelligent random selection
    return generate_smart_fallback(min_num, max_num, required_numbers, past_draws)


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get historical lottery data."""
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:100])  # Last 100 draws
    
    if is_main:
        draws = [draw.lottery_type_number for draw in past_draws
                 if isinstance(draw.lottery_type_number, list)]
    else:
        draws = [draw.additional_numbers for draw in past_draws
                 if hasattr(draw, 'additional_numbers') and 
                 isinstance(draw.additional_numbers, list)]
    
    return [[int(num) for num in draw] for draw in draws if draw]


def process_transformer_prediction(
    prediction: np.ndarray,
    min_num: int,
    max_num: int,
    required_numbers: int,
    past_draws: List[List[int]]
) -> List[int]:
    """Process Transformer prediction into valid lottery numbers."""
    
    # Convert predictions to valid range
    predicted_numbers = []
    for pred in prediction:
        num = int(np.clip(np.round(pred), min_num, max_num))
        if num not in predicted_numbers:
            predicted_numbers.append(num)
    
    # Fill remaining with frequency-based selection
    if len(predicted_numbers) < required_numbers:
        # Get frequency distribution from historical data
        freq_dict = {}
        for draw in past_draws[-20:]:  # Recent draws
            for num in draw:
                freq_dict[num] = freq_dict.get(num, 0) + 1
        
        # Sort by frequency and recency
        candidates = [(num, freq) for num, freq in freq_dict.items() 
                     if min_num <= num <= max_num and num not in predicted_numbers]
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        for num, _ in candidates:
            predicted_numbers.append(num)
            if len(predicted_numbers) >= required_numbers:
                break
    
    # Final fallback with random selection
    while len(predicted_numbers) < required_numbers:
        num = random.randint(min_num, max_num)
        if num not in predicted_numbers:
            predicted_numbers.append(num)
    
    return predicted_numbers[:required_numbers]


def generate_smart_fallback(
    min_num: int, 
    max_num: int, 
    required_numbers: int, 
    past_draws: List[List[int]]
) -> List[int]:
    """Generate numbers using intelligent fallback strategy."""
    
    if not past_draws:
        return sorted(random.sample(range(min_num, max_num + 1), required_numbers))
    
    # Frequency-based selection with recency weighting
    freq_dict = {}
    for i, draw in enumerate(past_draws):
        weight = 1.0 + (i / len(past_draws))  # Recent draws have higher weight
        for num in draw:
            if min_num <= num <= max_num:
                freq_dict[num] = freq_dict.get(num, 0) + weight
    
    # Select based on weighted frequency
    sorted_nums = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    selected = [num for num, _ in sorted_nums[:required_numbers]]
    
    # Fill remaining with random selection
    while len(selected) < required_numbers:
        num = random.randint(min_num, max_num)
        if num not in selected:
            selected.append(num)
    
    return sorted(selected[:required_numbers])


def generate_fallback_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Generate fallback numbers when main algorithm fails."""
    main_numbers = sorted(random.sample(
        range(int(lottery_type_instance.min_number), 
              int(lottery_type_instance.max_number) + 1),
        int(lottery_type_instance.pieces_of_draw_numbers)
    ))
    
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = sorted(random.sample(
            range(int(lottery_type_instance.additional_min_number),
                  int(lottery_type_instance.additional_max_number) + 1),
            int(lottery_type_instance.additional_numbers_count)
        ))
    
    return main_numbers, additional_numbers 