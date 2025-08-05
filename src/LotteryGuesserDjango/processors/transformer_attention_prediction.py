# transformer_attention_prediction.py
"""
Transformer Attention Mechanizmus Predikció
Self-attention és multi-head attention használata lottószám predikcióhoz
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter
import random
import math
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


class TransformerAttentionPredictor:
    """
    Transformer alapú attention mechanizmus predikció
    """
    
    def __init__(self):
        self.d_model = 64  # Model dimenzió
        self.num_heads = 8  # Attention fejek száma
        self.d_k = self.d_model // self.num_heads  # Key dimenzió
        self.d_v = self.d_model // self.num_heads  # Value dimenzió
        self.max_seq_length = 50  # Maximális szekvencia hossz
        self.dropout_rate = 0.1
        
        # Pozíciós kódolás
        self.positional_encoding = self._create_positional_encoding()
        
        # Attention súlyok
        self.W_q = np.random.randn(self.d_model, self.d_model) * 0.1
        self.W_k = np.random.randn(self.d_model, self.d_model) * 0.1
        self.W_v = np.random.randn(self.d_model, self.d_model) * 0.1
        self.W_o = np.random.randn(self.d_model, self.d_model) * 0.1
        
        # Feed-forward hálózat
        self.W_ff1 = np.random.randn(self.d_model, self.d_model * 4) * 0.1
        self.W_ff2 = np.random.randn(self.d_model * 4, self.d_model) * 0.1
        
        # Layer normalization paraméterek
        self.gamma = np.ones(self.d_model)
        self.beta = np.zeros(self.d_model)
    
    def get_numbers(self, lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
        """
        Főbejárási pont a transformer attention predikcióhoz
        """
        try:
            # Fő számok generálása
            main_numbers = self._generate_transformer_numbers(
                lottery_type_instance,
                min_num=int(lottery_type_instance.min_number),
                max_num=int(lottery_type_instance.max_number),
                required_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
                is_main=True
            )
            
            # Kiegészítő számok generálása
            additional_numbers = []
            if lottery_type_instance.has_additional_numbers:
                additional_numbers = self._generate_transformer_numbers(
                    lottery_type_instance,
                    min_num=int(lottery_type_instance.additional_min_number),
                    max_num=int(lottery_type_instance.additional_max_number),
                    required_numbers=int(lottery_type_instance.additional_numbers_count),
                    is_main=False
                )
            
            return sorted(main_numbers), sorted(additional_numbers)
            
        except Exception as e:
            print(f"Hiba a transformer_attention_prediction-ben: {e}")
            return self._generate_fallback_numbers(lottery_type_instance)
    
    def _generate_transformer_numbers(self, lottery_type_instance: lg_lottery_type,
                                    min_num: int, max_num: int, required_numbers: int,
                                    is_main: bool) -> List[int]:
        """
        Transformer alapú számgenerálás
        """
        # Történeti adatok lekérése
        historical_data = self._get_historical_data(lottery_type_instance, is_main)
        
        if len(historical_data) < 10:
            return self._generate_smart_random(min_num, max_num, required_numbers)
        
        # Szekvencia előkészítése
        sequence = self._prepare_sequence(historical_data, min_num, max_num)
        
        # Embedding és pozíciós kódolás
        embedded_sequence = self._embed_sequence(sequence)
        
        # Multi-head self-attention
        attention_output = self._multi_head_attention(embedded_sequence)
        
        # Feed-forward hálózat
        ff_output = self._feed_forward(attention_output)
        
        # Predikció generálása
        predictions = self._generate_predictions(ff_output, min_num, max_num, required_numbers)
        
        # Attention-based refinement
        refined_predictions = self._attention_refinement(
            predictions, embedded_sequence, min_num, max_num, required_numbers
        )
        
        return refined_predictions
    
    def _create_positional_encoding(self) -> np.ndarray:
        """
        Pozíciós kódolás létrehozása
        """
        pos_encoding = np.zeros((self.max_seq_length, self.d_model))
        
        for pos in range(self.max_seq_length):
            for i in range(0, self.d_model, 2):
                pos_encoding[pos, i] = math.sin(pos / (10000 ** (i / self.d_model)))
                if i + 1 < self.d_model:
                    pos_encoding[pos, i + 1] = math.cos(pos / (10000 ** (i / self.d_model)))
        
        return pos_encoding
    
    def _prepare_sequence(self, historical_data: List[List[int]], 
                         min_num: int, max_num: int) -> List[List[float]]:
        """
        Szekvencia előkészítése transformer számára
        """
        sequence = []
        
        for draw in historical_data[:self.max_seq_length]:
            # Húzás reprezentáció
            draw_features = self._extract_draw_features(draw, min_num, max_num)
            sequence.append(draw_features)
        
        # Padding ha szükséges
        while len(sequence) < min(self.max_seq_length, 10):
            sequence.append([0.0] * len(sequence[0]) if sequence else [0.0] * self.d_model)
        
        return sequence
    
    def _extract_draw_features(self, draw: List[int], min_num: int, max_num: int) -> List[float]:
        """
        Húzás jellemzők kinyerése
        """
        features = []
        
        # Alapvető statisztikák
        features.extend([
            np.mean(draw) / max_num,  # Normalizált átlag
            np.std(draw) / max_num,   # Normalizált szórás
            min(draw) / max_num,      # Normalizált minimum
            max(draw) / max_num,      # Normalizált maximum
            len(draw) / 10,           # Normalizált darabszám
        ])
        
        # Páros/páratlan arány
        even_ratio = sum(1 for num in draw if num % 2 == 0) / len(draw)
        features.append(even_ratio)
        
        # Összeg normalizálva
        features.append(sum(draw) / (max_num * len(draw)))
        
        # Számok eloszlása kvartilisekben
        num_range = max_num - min_num + 1
        for i in range(4):
            range_start = min_num + i * num_range // 4
            range_end = min_num + (i + 1) * num_range // 4
            count = sum(1 for num in draw if range_start <= num < range_end)
            features.append(count / len(draw))
        
        # Gap statisztikák
        sorted_draw = sorted(draw)
        gaps = [sorted_draw[i+1] - sorted_draw[i] for i in range(len(sorted_draw)-1)]
        if gaps:
            features.extend([
                np.mean(gaps) / max_num,
                np.std(gaps) / max_num
            ])
        else:
            features.extend([0.0, 0.0])
        
        # Padding vagy truncation a d_model mérethez
        while len(features) < self.d_model:
            features.append(0.0)
        
        return features[:self.d_model]
    
    def _embed_sequence(self, sequence: List[List[float]]) -> np.ndarray:
        """
        Szekvencia embedding és pozíciós kódolás
        """
        embedded = np.array(sequence)
        
        # Pozíciós kódolás hozzáadása
        seq_len = len(embedded)
        if seq_len <= self.max_seq_length:
            embedded += self.positional_encoding[:seq_len, :embedded.shape[1]]
        
        return embedded
    
    def _multi_head_attention(self, x: np.ndarray) -> np.ndarray:
        """
        Multi-head self-attention mechanizmus
        """
        batch_size, seq_len, d_model = x.shape[0] if len(x.shape) == 3 else 1, x.shape[0], x.shape[1]
        
        # Reshape for batch processing
        if len(x.shape) == 2:
            x = x.reshape(1, seq_len, d_model)
        
        # Query, Key, Value mátrixok
        Q = np.dot(x, self.W_q).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = np.dot(x, self.W_k).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = np.dot(x, self.W_v).reshape(batch_size, seq_len, self.num_heads, self.d_v).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        attention_output = self._scaled_dot_product_attention(Q, K, V)
        
        # Concatenate heads
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        # Output projection
        output = np.dot(attention_output, self.W_o)
        
        # Residual connection és layer normalization
        output = self._layer_norm(output + x)
        
        return output.squeeze(0) if batch_size == 1 else output
    
    def _scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Scaled dot-product attention
        """
        # Attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
        
        # Softmax
        attention_weights = self._softmax(scores)
        
        # Apply dropout (simplified)
        if random.random() < self.dropout_rate:
            attention_weights *= (1 - self.dropout_rate)
        
        # Weighted values
        output = np.matmul(attention_weights, V)
        
        return output
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Softmax aktiváció
        """
        # Numerikus stabilitás érdekében
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        """
        Layer normalization
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + 1e-8) + self.beta
    
    def _feed_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Feed-forward hálózat
        """
        # Első réteg
        hidden = np.dot(x, self.W_ff1)
        hidden = self._relu(hidden)
        
        # Második réteg
        output = np.dot(hidden, self.W_ff2)
        
        # Residual connection és layer normalization
        output = self._layer_norm(output + x)
        
        return output
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """
        ReLU aktiváció
        """
        return np.maximum(0, x)
    
    def _generate_predictions(self, transformer_output: np.ndarray, 
                            min_num: int, max_num: int, required_numbers: int) -> List[int]:
        """
        Predikciók generálása transformer kimenetből
        """
        # Utolsó időlépés kimenete
        last_output = transformer_output[-1] if len(transformer_output.shape) == 2 else transformer_output
        
        # Számok valószínűségeinek számítása
        num_probs = {}
        
        for num in range(min_num, max_num + 1):
            # Szám reprezentáció
            num_repr = self._number_to_representation(num, min_num, max_num)
            
            # Attention score számítás
            attention_score = np.dot(last_output, num_repr) / (np.linalg.norm(last_output) * np.linalg.norm(num_repr) + 1e-8)
            
            # Valószínűség
            num_probs[num] = max(0, attention_score)
        
        # Softmax normalizálás
        total_prob = sum(num_probs.values())
        if total_prob > 0:
            for num in num_probs:
                num_probs[num] /= total_prob
        
        # Top számok kiválasztása
        sorted_probs = sorted(num_probs.items(), key=lambda x: x[1], reverse=True)
        selected = [num for num, _ in sorted_probs[:required_numbers]]
        
        return selected
    
    def _number_to_representation(self, num: int, min_num: int, max_num: int) -> np.ndarray:
        """
        Szám reprezentáció létrehozása
        """
        repr_vec = np.zeros(self.d_model)
        
        # Normalizált érték
        normalized_num = (num - min_num) / (max_num - min_num)
        
        # Sinusoidal encoding
        for i in range(0, self.d_model, 2):
            repr_vec[i] = math.sin(normalized_num * (i + 1))
            if i + 1 < self.d_model:
                repr_vec[i + 1] = math.cos(normalized_num * (i + 1))
        
        return repr_vec
    
    def _attention_refinement(self, predictions: List[int], embedded_sequence: np.ndarray,
                            min_num: int, max_num: int, required_numbers: int) -> List[int]:
        """
        Attention alapú finomhangolás
        """
        refined_scores = {}
        
        for num in predictions:
            # Szám reprezentáció
            num_repr = self._number_to_representation(num, min_num, max_num)
            
            # Attention súlyok számítása a teljes szekvenciával
            attention_scores = []
            for seq_step in embedded_sequence:
                score = np.dot(seq_step, num_repr) / (np.linalg.norm(seq_step) * np.linalg.norm(num_repr) + 1e-8)
                attention_scores.append(score)
            
            # Weighted average
            refined_scores[num] = np.mean(attention_scores)
        
        # További számok hozzáadása ha szükséges
        if len(predictions) < required_numbers:
            remaining_nums = [n for n in range(min_num, max_num + 1) if n not in predictions]
            
            for num in remaining_nums:
                num_repr = self._number_to_representation(num, min_num, max_num)
                attention_scores = []
                for seq_step in embedded_sequence:
                    score = np.dot(seq_step, num_repr) / (np.linalg.norm(seq_step) * np.linalg.norm(num_repr) + 1e-8)
                    attention_scores.append(score)
                refined_scores[num] = np.mean(attention_scores)
        
        # Top számok kiválasztása
        sorted_refined = sorted(refined_scores.items(), key=lambda x: x[1], reverse=True)
        final_numbers = [num for num, _ in sorted_refined[:required_numbers]]
        
        return final_numbers
    
    def _generate_smart_random(self, min_num: int, max_num: int, count: int) -> List[int]:
        """
        Intelligens véletlen generálás
        """
        center = (min_num + max_num) / 2
        std = (max_num - min_num) / 6
        
        numbers = set()
        attempts = 0
        
        while len(numbers) < count and attempts < 1000:
            num = int(np.random.normal(center, std))
            if min_num <= num <= max_num:
                numbers.add(num)
            attempts += 1
        
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
            ).order_by('-id').values_list(field_name, flat=True)[:100]
            
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
transformer_predictor = TransformerAttentionPredictor()

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Főbejárási pont a transformer attention predikcióhoz
    """
    return transformer_predictor.get_numbers(lottery_type_instance)
