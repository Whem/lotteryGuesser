# quantum_machine_learning_prediction.py
"""
Quantum Machine Learning Lottery Predictor
Kvantum-inspirált gépi tanulás algoritmus speciális kvantum állapotok és interferencia szimulációval
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantumState:
    """Kvantum állapot reprezentáció."""
    def __init__(self, size: int):
        self.size = size
        # Komplex amplitúdók inicializálása
        self.amplitudes = np.random.complex128(np.random.randn(size) + 1j * np.random.randn(size))
        self.normalize()
    
    def normalize(self):
        """Kvantum állapot normalizálása."""
        norm = np.sqrt(np.sum(np.abs(self.amplitudes)**2))
        if norm > 0:
            self.amplitudes /= norm
    
    def measure(self) -> int:
        """Kvantum mérés - valószínűség alapú kollaps."""
        probabilities = np.abs(self.amplitudes)**2
        return np.random.choice(len(self.amplitudes), p=probabilities)
    
    def apply_rotation(self, angle: float, index: int):
        """Kvantum forgatás alkalmazása."""
        if 0 <= index < self.size:
            self.amplitudes[index] *= np.exp(1j * angle)
        self.normalize()


class QuantumCircuit:
    """Kvantum áramkör szimulátor."""
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.state_size = 2 ** num_qubits
        self.quantum_state = QuantumState(self.state_size)
    
    def hadamard_gate(self, qubit: int):
        """Hadamard kapu alkalmazása."""
        for i in range(self.state_size):
            if (i >> qubit) & 1:
                j = i ^ (1 << qubit)
                if j < i:
                    temp = (self.quantum_state.amplitudes[i] - self.quantum_state.amplitudes[j]) / np.sqrt(2)
                    self.quantum_state.amplitudes[j] = (self.quantum_state.amplitudes[i] + self.quantum_state.amplitudes[j]) / np.sqrt(2)
                    self.quantum_state.amplitudes[i] = temp
    
    def phase_gate(self, qubit: int, phase: float):
        """Fázis kapu alkalmazása."""
        for i in range(self.state_size):
            if (i >> qubit) & 1:
                self.quantum_state.amplitudes[i] *= np.exp(1j * phase)
    
    def measure_all(self) -> List[int]:
        """Összes qubit mérése."""
        measurement = self.quantum_state.measure()
        return [(measurement >> i) & 1 for i in range(self.num_qubits)]


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Kvantum gépi tanulás alapú lottószám predikció.
    Returns a tuple (main_numbers, additional_numbers).
    """
    try:
        # Fő számok generálása
        main_numbers = generate_quantum_ml_numbers(
            lottery_type_instance,
            min_number=int(lottery_type_instance.min_number),
            max_number=int(lottery_type_instance.max_number),
            pieces_of_draw_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
            is_main=True
        )

        # Kiegészítő számok generálása
        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            additional_numbers = generate_quantum_ml_numbers(
                lottery_type_instance,
                min_number=int(lottery_type_instance.additional_min_number),
                max_number=int(lottery_type_instance.additional_max_number),
                pieces_of_draw_numbers=int(lottery_type_instance.additional_numbers_count),
                is_main=False
            )

        return sorted(main_numbers), sorted(additional_numbers)
    
    except Exception as e:
        logger.error(f"Hiba a kvantum ML predikció során: {e}")
        return generate_fallback_numbers(lottery_type_instance)


def generate_quantum_ml_numbers(
    lottery_type_instance: lg_lottery_type,
    min_number: int,
    max_number: int,
    pieces_of_draw_numbers: int,
    is_main: bool
) -> List[int]:
    """Kvantum ML alapú számgenerálás."""
    
    # Történeti adatok lekérése
    past_draws = get_historical_data(lottery_type_instance, is_main)
    
    if len(past_draws) < 10:
        return generate_quantum_random(min_number, max_number, pieces_of_draw_numbers)
    
    # Kvantum tanulási algoritmus
    quantum_learner = QuantumMLLearner(min_number, max_number, pieces_of_draw_numbers)
    quantum_learner.train(past_draws)
    
    # Predikció generálása
    predictions = quantum_learner.predict()
    
    return sorted(predictions)


class QuantumMLLearner:
    """Kvantum gépi tanulás osztály."""
    
    def __init__(self, min_num: int, max_num: int, target_count: int):
        self.min_num = min_num
        self.max_num = max_num
        self.target_count = target_count
        self.num_range = max_num - min_num + 1
        
        # Kvantum paraméterek
        self.num_qubits = min(8, math.ceil(math.log2(self.num_range)))
        self.quantum_circuit = QuantumCircuit(self.num_qubits)
        
        # Tanulási paraméterek
        self.learning_rate = 0.1
        self.quantum_weights = np.random.random(self.num_range) * 2 * np.pi
        self.historical_patterns = defaultdict(float)
        
    def train(self, past_draws: List[List[int]]):
        """Kvantum tanulási folyamat."""
        
        # Történeti minták elemzése
        self._analyze_historical_patterns(past_draws)
        
        # Kvantum interferencia tanulás
        self._quantum_interference_learning(past_draws)
        
        # Adaptív súlyok optimalizálása
        self._optimize_quantum_weights(past_draws)
    
    def _analyze_historical_patterns(self, past_draws: List[List[int]]):
        """Történeti minták kvantum elemzése."""
        
        for draw in past_draws:
            # Számok közötti kvantum korreláció
            for i, num1 in enumerate(draw):
                for j, num2 in enumerate(draw):
                    if i != j and self.min_num <= num1 <= self.max_num and self.min_num <= num2 <= self.max_num:
                        correlation_key = f"{num1}-{num2}"
                        phase_diff = (num1 - num2) * np.pi / (self.max_num - self.min_num)
                        self.historical_patterns[correlation_key] += np.cos(phase_diff)
    
    def _quantum_interference_learning(self, past_draws: List[List[int]]):
        """Kvantum interferencia alapú tanulás."""
        
        for epoch in range(min(50, len(past_draws))):
            draw = past_draws[epoch % len(past_draws)]
            
            # Kvantum állapot előkészítése
            for i in range(self.num_qubits):
                self.quantum_circuit.hadamard_gate(i)
            
            # Történeti adatok beépítése kvantum fázisokba
            for num in draw:
                if self.min_num <= num <= self.max_num:
                    qubit_index = (num - self.min_num) % self.num_qubits
                    phase = self.quantum_weights[num - self.min_num]
                    self.quantum_circuit.phase_gate(qubit_index, phase)
            
            # Kvantum mérés és tanulás
            measurements = self.quantum_circuit.measure_all()
            
            # Súlyok frissítése kvantum gradiens alapján
            for i, measurement in enumerate(measurements):
                if measurement == 1:  # Pozitív mérési eredmény
                    for num in draw:
                        if self.min_num <= num <= self.max_num:
                            idx = num - self.min_num
                            gradient = np.sin(self.quantum_weights[idx]) * self.learning_rate
                            self.quantum_weights[idx] += gradient
    
    def _optimize_quantum_weights(self, past_draws: List[List[int]]):
        """Kvantum súlyok optimalizálása szimulált kvantum kinetikával."""
        
        # Kvantum energia függvény definiálása
        def quantum_energy(weights):
            energy = 0.0
            for draw in past_draws[-20:]:  # Legutóbbi 20 húzás
                for num in draw:
                    if self.min_num <= num <= self.max_num:
                        idx = num - self.min_num
                        # Kvantum harmonikus oszcillátor energia
                        energy += 0.5 * weights[idx]**2 + 0.25 * weights[idx]**4
            return energy
        
        # Kvantum gradiens descent
        for iteration in range(100):
            current_energy = quantum_energy(self.quantum_weights)
            
            # Kvantum zaj hozzáadása
            quantum_noise = np.random.normal(0, 0.01, len(self.quantum_weights))
            new_weights = self.quantum_weights + quantum_noise
            
            new_energy = quantum_energy(new_weights)
            
            # Kvantum alagút hatás szimulációja
            tunnel_probability = np.exp(-(new_energy - current_energy) / 0.1)
            
            if new_energy < current_energy or random.random() < tunnel_probability:
                self.quantum_weights = new_weights
    
    def predict(self) -> List[int]:
        """Kvantum predikció generálása."""
        
        predictions = set()
        
        # Több kvantum mérési ciklus
        for cycle in range(self.target_count * 3):
            
            # Kvantum állapot inicializálása
            quantum_state = QuantumState(self.num_range)
            
            # Kvantum súlyok alkalmazása
            for i in range(self.num_range):
                quantum_state.apply_rotation(self.quantum_weights[i], i)
            
            # Kvantum interferencia alkalmazása
            for pattern, strength in list(self.historical_patterns.items())[:10]:
                if '-' in pattern:
                    num1, num2 = map(int, pattern.split('-'))
                    if self.min_num <= num1 <= self.max_num and self.min_num <= num2 <= self.max_num:
                        idx1, idx2 = num1 - self.min_num, num2 - self.min_num
                        interference_angle = strength * np.pi / 4
                        quantum_state.apply_rotation(interference_angle, idx1)
                        quantum_state.apply_rotation(-interference_angle, idx2)
            
            # Kvantum mérés
            measured_index = quantum_state.measure()
            predicted_number = measured_index + self.min_num
            
            if self.min_num <= predicted_number <= self.max_num:
                predictions.add(predicted_number)
            
            if len(predictions) >= self.target_count:
                break
        
        # Kiegészítés ha szükséges
        while len(predictions) < self.target_count:
            # Kvantum bizonytalanság alapú kitöltés
            uncertainty_number = self._quantum_uncertainty_selection(predictions)
            predictions.add(uncertainty_number)
        
        return list(predictions)[:self.target_count]
    
    def _quantum_uncertainty_selection(self, existing_predictions: set) -> int:
        """Kvantum bizonytalanság elv alapú szám kiválasztás."""
        
        available_numbers = [num for num in range(self.min_num, self.max_num + 1) 
                           if num not in existing_predictions]
        
        if not available_numbers:
            return random.randint(self.min_num, self.max_num)
        
        # Heisenberg bizonytalanság szimulációja
        uncertainties = []
        for num in available_numbers:
            idx = num - self.min_num
            # Pozíció és impulzus bizonytalanság
            position_uncertainty = abs(self.quantum_weights[idx])
            momentum_uncertainty = 1.0 / (position_uncertainty + 0.001)
            # Heisenberg reláció: ΔxΔp ≥ ℏ/2
            total_uncertainty = position_uncertainty * momentum_uncertainty
            uncertainties.append(total_uncertainty)
        
        # Legnagyobb bizonytalanságú szám kiválasztása
        max_uncertainty_idx = np.argmax(uncertainties)
        return available_numbers[max_uncertainty_idx]


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


def generate_quantum_random(min_number: int, max_number: int, count: int) -> List[int]:
    """Kvantum véletlen számgenerálás kvantum zaj szimulációval."""
    
    # Kvantum zaj paraméterek
    num_range = max_number - min_number + 1
    quantum_circuit = QuantumCircuit(min(8, math.ceil(math.log2(num_range))))
    
    numbers = set()
    
    while len(numbers) < count:
        # Kvantum szuperpozíció létrehozása
        for i in range(quantum_circuit.num_qubits):
            quantum_circuit.hadamard_gate(i)
            # Kvantum zaj hozzáadása
            quantum_circuit.phase_gate(i, random.uniform(0, 2*np.pi))
        
        # Kvantum mérés
        measurement = quantum_circuit.quantum_state.measure()
        quantum_number = (measurement % num_range) + min_number
        
        if min_number <= quantum_number <= max_number:
            numbers.add(quantum_number)
    
    return sorted(list(numbers)[:count])


def generate_fallback_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Fallback számgenerálás kvantum hibák esetén."""
    main_numbers = generate_quantum_random(
        int(lottery_type_instance.min_number),
        int(lottery_type_instance.max_number),
        int(lottery_type_instance.pieces_of_draw_numbers)
    )
    
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_quantum_random(
            int(lottery_type_instance.additional_min_number),
            int(lottery_type_instance.additional_max_number),
            int(lottery_type_instance.additional_numbers_count)
        )
    
    return sorted(main_numbers), sorted(additional_numbers) 