# variational_autoencoder_prediction.py
"""
Variational Autoencoder Predikció
VAE használata lottószám minták tanulására és generálására
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter
import random
import math
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


class VariationalAutoencoderPredictor:
    """
    Variational Autoencoder alapú predikció
    """
    
    def __init__(self):
        self.input_dim = 90  # Max lottószám
        self.latent_dim = 16  # Latent space dimenzió
        self.hidden_dim = 64
        self.learning_rate = 0.001
        self.beta = 1.0  # KL divergence súly
        
        # Encoder súlyok
        self.W_enc1 = np.random.randn(self.input_dim, self.hidden_dim) * 0.1
        self.b_enc1 = np.zeros(self.hidden_dim)
        self.W_enc2 = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1
        self.b_enc2 = np.zeros(self.hidden_dim)
        
        # Latent space súlyok (mean és log_var)
        self.W_mu = np.random.randn(self.hidden_dim, self.latent_dim) * 0.1
        self.b_mu = np.zeros(self.latent_dim)
        self.W_logvar = np.random.randn(self.hidden_dim, self.latent_dim) * 0.1
        self.b_logvar = np.zeros(self.latent_dim)
        
        # Decoder súlyok
        self.W_dec1 = np.random.randn(self.latent_dim, self.hidden_dim) * 0.1
        self.b_dec1 = np.zeros(self.hidden_dim)
        self.W_dec2 = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1
        self.b_dec2 = np.zeros(self.hidden_dim)
        self.W_out = np.random.randn(self.hidden_dim, self.input_dim) * 0.1
        self.b_out = np.zeros(self.input_dim)
        
        # Tanítási történet
        self.training_history = []
    
    def get_numbers(self, lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
        """
        Főbejárási pont a VAE predikcióhoz
        """
        try:
            # Fő számok generálása
            main_numbers = self._generate_vae_numbers(
                lottery_type_instance,
                min_num=int(lottery_type_instance.min_number),
                max_num=int(lottery_type_instance.max_number),
                required_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
                is_main=True
            )
            
            # Kiegészítő számok generálása
            additional_numbers = []
            if lottery_type_instance.has_additional_numbers:
                additional_numbers = self._generate_vae_numbers(
                    lottery_type_instance,
                    min_num=int(lottery_type_instance.additional_min_number),
                    max_num=int(lottery_type_instance.additional_max_number),
                    required_numbers=int(lottery_type_instance.additional_numbers_count),
                    is_main=False
                )
            
            return sorted(main_numbers), sorted(additional_numbers)
            
        except Exception as e:
            print(f"Hiba a variational_autoencoder_prediction-ben: {e}")
            return self._generate_fallback_numbers(lottery_type_instance)
    
    def _generate_vae_numbers(self, lottery_type_instance: lg_lottery_type,
                            min_num: int, max_num: int, required_numbers: int,
                            is_main: bool) -> List[int]:
        """
        VAE alapú számgenerálás
        """
        # Történeti adatok lekérése
        historical_data = self._get_historical_data(lottery_type_instance, is_main)
        
        if len(historical_data) < 20:
            return self._generate_smart_random(min_num, max_num, required_numbers)
        
        # Adatok előkészítése
        training_data = self._prepare_training_data(historical_data, min_num, max_num)
        
        # VAE tanítása
        self._train_vae(training_data)
        
        # Új minták generálása
        generated_samples = self._generate_samples(num_samples=50)
        
        # Legjobb minták kiválasztása
        best_samples = self._select_best_samples(
            generated_samples, historical_data, min_num, max_num, required_numbers
        )
        
        # Ensemble más módszerekkel
        ensemble_numbers = self._ensemble_with_traditional_methods(
            best_samples, historical_data, min_num, max_num, required_numbers
        )
        
        return ensemble_numbers
    
    def _prepare_training_data(self, historical_data: List[List[int]], 
                             min_num: int, max_num: int) -> np.ndarray:
        """
        Tanítási adatok előkészítése
        """
        training_vectors = []
        
        for draw in historical_data:
            # One-hot encoding
            vector = np.zeros(self.input_dim)
            for num in draw:
                if 1 <= num <= self.input_dim:
                    vector[num - 1] = 1.0
            training_vectors.append(vector)
        
        return np.array(training_vectors)
    
    def _train_vae(self, training_data: np.ndarray, epochs: int = 100) -> None:
        """
        VAE tanítása
        """
        for epoch in range(epochs):
            total_loss = 0.0
            
            for i in range(len(training_data)):
                x = training_data[i]
                
                # Forward pass
                mu, log_var, z, x_reconstructed = self._forward_pass(x)
                
                # Loss számítás
                reconstruction_loss = self._reconstruction_loss(x, x_reconstructed)
                kl_loss = self._kl_divergence_loss(mu, log_var)
                total_loss_sample = reconstruction_loss + self.beta * kl_loss
                
                # Backward pass (egyszerűsített)
                self._backward_pass(x, x_reconstructed, mu, log_var, z)
                
                total_loss += total_loss_sample
            
            avg_loss = total_loss / len(training_data)
            self.training_history.append(avg_loss)
            
            # Early stopping
            if len(self.training_history) > 10:
                recent_losses = self.training_history[-10:]
                if max(recent_losses) - min(recent_losses) < 0.001:
                    break
    
    def _forward_pass(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        VAE forward pass
        """
        # Encoder
        h1 = self._relu(np.dot(x, self.W_enc1) + self.b_enc1)
        h2 = self._relu(np.dot(h1, self.W_enc2) + self.b_enc2)
        
        # Latent space paraméterek
        mu = np.dot(h2, self.W_mu) + self.b_mu
        log_var = np.dot(h2, self.W_logvar) + self.b_logvar
        
        # Reparameterization trick
        epsilon = np.random.normal(0, 1, self.latent_dim)
        z = mu + np.exp(0.5 * log_var) * epsilon
        
        # Decoder
        h3 = self._relu(np.dot(z, self.W_dec1) + self.b_dec1)
        h4 = self._relu(np.dot(h3, self.W_dec2) + self.b_dec2)
        x_reconstructed = self._sigmoid(np.dot(h4, self.W_out) + self.b_out)
        
        return mu, log_var, z, x_reconstructed
    
    def _reconstruction_loss(self, x_true: np.ndarray, x_pred: np.ndarray) -> float:
        """
        Rekonstrukciós loss (binary cross-entropy)
        """
        epsilon = 1e-8
        x_pred = np.clip(x_pred, epsilon, 1 - epsilon)
        loss = -np.sum(x_true * np.log(x_pred) + (1 - x_true) * np.log(1 - x_pred))
        return loss
    
    def _kl_divergence_loss(self, mu: np.ndarray, log_var: np.ndarray) -> float:
        """
        KL divergencia loss
        """
        kl_loss = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
        return kl_loss
    
    def _backward_pass(self, x: np.ndarray, x_reconstructed: np.ndarray, 
                      mu: np.ndarray, log_var: np.ndarray, z: np.ndarray) -> None:
        """
        Egyszerűsített backward pass (gradiens frissítés)
        """
        # Reconstruction loss gradiens
        reconstruction_grad = x_reconstructed - x
        
        # KL loss gradiens
        mu_grad = mu
        log_var_grad = 0.5 * (np.exp(log_var) - 1)
        
        # Súlyok frissítése (egyszerűsített)
        self.W_out -= self.learning_rate * np.outer(reconstruction_grad, z)
        self.b_out -= self.learning_rate * reconstruction_grad
        
        self.W_mu -= self.learning_rate * np.outer(mu_grad, z)
        self.b_mu -= self.learning_rate * mu_grad
        
        self.W_logvar -= self.learning_rate * np.outer(log_var_grad, z)
        self.b_logvar -= self.learning_rate * log_var_grad
    
    def _generate_samples(self, num_samples: int) -> List[np.ndarray]:
        """
        Új minták generálása a latent space-ből
        """
        samples = []
        
        for _ in range(num_samples):
            # Véletlen latent vektor
            z = np.random.normal(0, 1, self.latent_dim)
            
            # Decoder
            h3 = self._relu(np.dot(z, self.W_dec1) + self.b_dec1)
            h4 = self._relu(np.dot(h3, self.W_dec2) + self.b_dec2)
            x_generated = self._sigmoid(np.dot(h4, self.W_out) + self.b_out)
            
            samples.append(x_generated)
        
        return samples
    
    def _select_best_samples(self, samples: List[np.ndarray], historical_data: List[List[int]],
                           min_num: int, max_num: int, required_numbers: int) -> List[List[int]]:
        """
        Legjobb minták kiválasztása
        """
        scored_samples = []
        
        for sample in samples:
            # Valószínűségek alapján számok kiválasztása
            numbers = self._sample_to_numbers(sample, min_num, max_num, required_numbers)
            
            # Minta pontszámának számítása
            score = self._calculate_sample_score(numbers, historical_data)
            
            scored_samples.append((numbers, score))
        
        # Legjobb minták kiválasztása
        scored_samples.sort(key=lambda x: x[1], reverse=True)
        best_samples = [numbers for numbers, _ in scored_samples[:10]]
        
        return best_samples
    
    def _sample_to_numbers(self, sample: np.ndarray, min_num: int, max_num: int, 
                         required_numbers: int) -> List[int]:
        """
        VAE minta konvertálása számokra
        """
        # Valószínűségek normalizálása
        probabilities = sample / np.sum(sample) if np.sum(sample) > 0 else np.ones_like(sample) / len(sample)
        
        # Top valószínűségű számok
        top_indices = np.argsort(probabilities)[-required_numbers*2:][::-1]
        
        # Számok kiválasztása
        selected_numbers = []
        for idx in top_indices:
            num = idx + 1  # 1-based indexing
            if min_num <= num <= max_num and num not in selected_numbers:
                selected_numbers.append(num)
                if len(selected_numbers) >= required_numbers:
                    break
        
        # Kiegészítés szükség esetén
        if len(selected_numbers) < required_numbers:
            remaining = [num for num in range(min_num, max_num + 1) if num not in selected_numbers]
            random.shuffle(remaining)
            selected_numbers.extend(remaining[:required_numbers - len(selected_numbers)])
        
        return selected_numbers[:required_numbers]
    
    def _calculate_sample_score(self, numbers: List[int], historical_data: List[List[int]]) -> float:
        """
        Minta pontszámának számítása
        """
        score = 0.0
        
        # Frekvencia alapú pontszám
        frequency_counter = Counter(num for draw in historical_data for num in draw)
        for num in numbers:
            score += frequency_counter.get(num, 0)
        
        # Diverzitás pontszám
        if len(set(numbers)) == len(numbers):  # Nincs duplikátum
            score += 10.0
        
        # Statisztikai pontszám
        if numbers:
            mean_val = np.mean(numbers)
            historical_means = [np.mean(draw) for draw in historical_data]
            if historical_means:
                mean_similarity = 1.0 / (1.0 + abs(mean_val - np.mean(historical_means)))
                score += mean_similarity * 5.0
        
        # Gap pontszám
        sorted_numbers = sorted(numbers)
        gaps = [sorted_numbers[i+1] - sorted_numbers[i] for i in range(len(sorted_numbers)-1)]
        if gaps:
            avg_gap = np.mean(gaps)
            if 2 <= avg_gap <= 8:  # Optimális gap tartomány
                score += 3.0
        
        return score
    
    def _ensemble_with_traditional_methods(self, vae_samples: List[List[int]], 
                                         historical_data: List[List[int]],
                                         min_num: int, max_num: int, 
                                         required_numbers: int) -> List[int]:
        """
        Ensemble VAE-vel és hagyományos módszerekkel
        """
        # VAE eredmények aggregálása
        vae_counter = Counter()
        for sample in vae_samples:
            for num in sample:
                vae_counter[num] += 1
        
        # Hagyományos módszerek
        freq_numbers = self._frequency_method(historical_data, min_num, max_num, required_numbers)
        trend_numbers = self._trend_method(historical_data, min_num, max_num, required_numbers)
        gap_numbers = self._gap_method(historical_data, min_num, max_num, required_numbers)
        
        # Ensemble szavazás
        vote_counter = Counter()
        
        # VAE eredmények (40% súly)
        for num, count in vae_counter.items():
            vote_counter[num] += count * 0.4
        
        # Hagyományos módszerek (60% súly)
        methods = [
            (freq_numbers, 0.25),
            (trend_numbers, 0.2),
            (gap_numbers, 0.15)
        ]
        
        for numbers, weight in methods:
            for i, num in enumerate(numbers):
                position_weight = 1.0 / (i + 1)
                vote_counter[num] += weight * position_weight
        
        # Legmagasabb szavazatú számok
        top_numbers = [num for num, _ in vote_counter.most_common(required_numbers)]
        
        # Kiegészítés szükség esetén
        if len(top_numbers) < required_numbers:
            remaining = [num for num in range(min_num, max_num + 1) if num not in top_numbers]
            random.shuffle(remaining)
            top_numbers.extend(remaining[:required_numbers - len(top_numbers)])
        
        return top_numbers[:required_numbers]
    
    def _frequency_method(self, historical_data: List[List[int]], 
                         min_num: int, max_num: int, required_numbers: int) -> List[int]:
        """Frekvencia alapú módszer"""
        counter = Counter(num for draw in historical_data[:30] for num in draw)
        return [num for num, _ in counter.most_common(required_numbers)]
    
    def _trend_method(self, historical_data: List[List[int]], 
                     min_num: int, max_num: int, required_numbers: int) -> List[int]:
        """Trend alapú módszer"""
        if len(historical_data) < 20:
            return self._generate_smart_random(min_num, max_num, required_numbers)
        
        recent = Counter(num for draw in historical_data[:10] for num in draw)
        older = Counter(num for draw in historical_data[10:20] for num in draw)
        
        trends = {num: recent.get(num, 0) - older.get(num, 0) * 0.5 
                 for num in range(min_num, max_num + 1)}
        
        return sorted(trends.keys(), key=lambda x: trends[x], reverse=True)[:required_numbers]
    
    def _gap_method(self, historical_data: List[List[int]], 
                   min_num: int, max_num: int, required_numbers: int) -> List[int]:
        """Gap alapú módszer"""
        last_seen = {}
        for i, draw in enumerate(historical_data):
            for num in draw:
                last_seen[num] = i
        
        gaps = {num: last_seen.get(num, len(historical_data)) 
               for num in range(min_num, max_num + 1)}
        
        return sorted(gaps.keys(), key=lambda x: gaps[x], reverse=True)[:required_numbers]
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU aktiváció"""
        return np.maximum(0, x)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid aktiváció"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _generate_smart_random(self, min_num: int, max_num: int, count: int) -> List[int]:
        """Intelligens véletlen generálás"""
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
        """Történeti adatok lekérése"""
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
vae_predictor = VariationalAutoencoderPredictor()

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Főbejárási pont a VAE predikcióhoz"""
    return vae_predictor.get_numbers(lottery_type_instance)
