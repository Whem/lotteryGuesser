# chaos_theory_fractal_predictor.py
"""
Chaos Theory & Fractal Predictor
Káosz elmélet és fraktál geometria alapú predikció Lorenz attraktor és kaotikus rendszerekkel
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

class LorenzAttractor:
    """Lorenz attraktor szimuláció."""
    
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0):
        self.sigma = sigma  # Prandtl szám
        self.rho = rho      # Rayleigh szám
        self.beta = beta    # Geometriai faktor
        
        # Kezdeti állapot
        self.x = 1.0
        self.y = 1.0
        self.z = 1.0
        
        # Trajektória
        self.trajectory = deque(maxlen=1000)
        self.dt = 0.01  # Időlépés
    
    def step(self) -> Tuple[float, float, float]:
        """Egy lépés a Lorenz rendszerben."""
        
        # Lorenz egyenletek
        dx_dt = self.sigma * (self.y - self.x)
        dy_dt = self.x * (self.rho - self.z) - self.y
        dz_dt = self.x * self.y - self.beta * self.z
        
        # Euler integráció
        self.x += dx_dt * self.dt
        self.y += dy_dt * self.dt
        self.z += dz_dt * self.dt
        
        # Trajektória tárolása
        point = (self.x, self.y, self.z)
        self.trajectory.append(point)
        
        return point
    
    def generate_sequence(self, length: int) -> List[Tuple[float, float, float]]:
        """Kaotikus szekvencia generálása."""
        sequence = []
        for _ in range(length):
            point = self.step()
            sequence.append(point)
        
        return sequence
    
    def normalize_to_range(self, points: List[Tuple[float, float, float]], 
                          min_val: int, max_val: int) -> List[int]:
        """Lorenz pontok normalizálása lottószám tartományba."""
        
        if not points:
            return []
        
        # Koordináták szétválasztása
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        z_coords = [p[2] for p in points]
        
        # Min-max normalizálás minden koordinátára
        coords = [x_coords, y_coords, z_coords]
        normalized_numbers = []
        
        for coord_list in coords:
            if len(coord_list) > 0:
                min_coord = min(coord_list)
                max_coord = max(coord_list)
                
                if max_coord != min_coord:
                    normalized = [(val - min_coord) / (max_coord - min_coord) for val in coord_list]
                    scaled = [int(min_val + norm * (max_val - min_val)) for norm in normalized]
                    normalized_numbers.extend(scaled)
        
        # Tartományon belüli számok szűrése
        valid_numbers = [num for num in normalized_numbers if min_val <= num <= max_val]
        
        return valid_numbers


class FractalGenerator:
    """Fraktál generátor osztály."""
    
    def __init__(self, min_num: int, max_num: int):
        self.min_num = min_num
        self.max_num = max_num
        
    def mandelbrot_sequence(self, width: int = 100, height: int = 100, 
                           max_iter: int = 50) -> List[int]:
        """Mandelbrot fraktál alapú számok."""
        
        numbers = []
        
        # Mandelbrot tartomány
        x_min, x_max = -2.0, 1.0
        y_min, y_max = -1.5, 1.5
        
        for i in range(height):
            for j in range(width):
                # Komplex szám
                c = complex(x_min + (x_max - x_min) * j / width,
                           y_min + (y_max - y_min) * i / height)
                
                # Mandelbrot iteráció
                z = 0
                iterations = 0
                
                while abs(z) <= 2 and iterations < max_iter:
                    z = z*z + c
                    iterations += 1
                
                # Iterációszám normalizálása
                if iterations < max_iter:
                    normalized = iterations / max_iter
                    number = int(self.min_num + normalized * (self.max_num - self.min_num))
                    if self.min_num <= number <= self.max_num:
                        numbers.append(number)
        
        return numbers
    
    def julia_sequence(self, c_real: float = -0.7, c_imag: float = 0.27015, 
                      width: int = 100, height: int = 100, max_iter: int = 50) -> List[int]:
        """Julia fraktál alapú számok."""
        
        numbers = []
        c = complex(c_real, c_imag)
        
        # Julia tartomány
        x_min, x_max = -2.0, 2.0
        y_min, y_max = -2.0, 2.0
        
        for i in range(height):
            for j in range(width):
                # Kezdő pont
                z = complex(x_min + (x_max - x_min) * j / width,
                           y_min + (y_max - y_min) * i / height)
                
                # Julia iteráció
                iterations = 0
                
                while abs(z) <= 2 and iterations < max_iter:
                    z = z*z + c
                    iterations += 1
                
                # Iterációszám normalizálása
                if iterations < max_iter:
                    normalized = iterations / max_iter
                    number = int(self.min_num + normalized * (self.max_num - self.min_num))
                    if self.min_num <= number <= self.max_num:
                        numbers.append(number)
        
        return numbers
    
    def sierpinski_triangle(self, iterations: int = 1000) -> List[int]:
        """Sierpinski háromszög fraktál."""
        
        # Háromszög csúcsok
        vertices = [(0, 0), (1, 0), (0.5, math.sqrt(3)/2)]
        
        # Kezdő pont
        current = (random.random(), random.random())
        numbers = []
        
        for _ in range(iterations):
            # Véletlen csúcs választása
            vertex = random.choice(vertices)
            
            # Középpont számítása
            current = ((current[0] + vertex[0]) / 2, (current[1] + vertex[1]) / 2)
            
            # Koordináták normalizálása
            x_norm = current[0]
            y_norm = current[1]
            
            # Számok generálása
            x_number = int(self.min_num + x_norm * (self.max_num - self.min_num))
            y_number = int(self.min_num + y_norm * (self.max_num - self.min_num))
            
            if self.min_num <= x_number <= self.max_num:
                numbers.append(x_number)
            if self.min_num <= y_number <= self.max_num:
                numbers.append(y_number)
        
        return numbers
    
    def dragon_curve(self, iterations: int = 10) -> List[int]:
        """Sárkány görbe fraktál."""
        
        # Kezdő irányok
        directions = [0]  # 0: jobb, 1: fel, 2: bal, 3: le
        
        # Dragon curve algoritmus
        for _ in range(iterations):
            new_directions = directions.copy()
            new_directions.append(1)  # Jobbra fordulás
            
            # Tükrözés és fordítás
            for i in range(len(directions) - 1, -1, -1):
                new_directions.append((directions[i] + 1) % 4)
            
            directions = new_directions
        
        # Útvonal követése
        x, y = 0, 0
        positions = [(x, y)]
        
        for direction in directions:
            if direction == 0:  # Jobb
                x += 1
            elif direction == 1:  # Fel
                y += 1
            elif direction == 2:  # Bal
                x -= 1
            elif direction == 3:  # Le
                y -= 1
            
            positions.append((x, y))
        
        # Normalizálás
        if positions:
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            numbers = []
            for x, y in positions:
                # X koordináta normalizálása
                if x_max != x_min:
                    x_norm = (x - x_min) / (x_max - x_min)
                    x_number = int(self.min_num + x_norm * (self.max_num - self.min_num))
                    if self.min_num <= x_number <= self.max_num:
                        numbers.append(x_number)
                
                # Y koordináta normalizálása
                if y_max != y_min:
                    y_norm = (y - y_min) / (y_max - y_min)
                    y_number = int(self.min_num + y_norm * (self.max_num - self.min_num))
                    if self.min_num <= y_number <= self.max_num:
                        numbers.append(y_number)
            
            return numbers
        
        return []


class ChaoticMap:
    """Kaotikus térkép osztály."""
    
    def __init__(self, map_type: str = "logistic"):
        self.map_type = map_type
        self.state = random.random()
        
    def logistic_map(self, r: float = 3.9) -> float:
        """Logisztikus térkép."""
        self.state = r * self.state * (1 - self.state)
        return self.state
    
    def henon_map(self, a: float = 1.4, b: float = 0.3, 
                  x: Optional[float] = None, y: Optional[float] = None) -> Tuple[float, float]:
        """Hénon térkép."""
        if x is None:
            x = self.state
        if y is None:
            y = random.random()
        
        x_new = 1 - a * x * x + y
        y_new = b * x
        
        self.state = x_new
        return x_new, y_new
    
    def tent_map(self, mu: float = 1.8) -> float:
        """Sátor térkép."""
        if self.state < 0.5:
            self.state = mu * self.state
        else:
            self.state = mu * (1 - self.state)
        
        return self.state
    
    def generate_sequence(self, length: int, **params) -> List[float]:
        """Kaotikus szekvencia generálása."""
        sequence = []
        
        for _ in range(length):
            if self.map_type == "logistic":
                value = self.logistic_map(params.get('r', 3.9))
                sequence.append(value)
            elif self.map_type == "henon":
                x, y = self.henon_map(params.get('a', 1.4), params.get('b', 0.3))
                sequence.extend([x, y])
            elif self.map_type == "tent":
                value = self.tent_map(params.get('mu', 1.8))
                sequence.append(value)
        
        return sequence


class ChaosTheoryFractalPredictor:
    """Káosz elmélet és fraktál prediktor főosztály."""
    
    def __init__(self, min_num: int, max_num: int, target_count: int):
        self.min_num = min_num
        self.max_num = max_num
        self.target_count = target_count
        
        # Kaotikus rendszerek
        self.lorenz = LorenzAttractor()
        self.fractal_gen = FractalGenerator(min_num, max_num)
        self.chaotic_maps = {
            'logistic': ChaoticMap('logistic'),
            'henon': ChaoticMap('henon'),
            'tent': ChaoticMap('tent')
        }
        
        # Múltbeli adatok analízise
        self.historical_chaos_metrics = {}
        self.attractor_parameters = {}
        
    def analyze_historical_chaos(self, past_draws: List[List[int]]):
        """Történeti adatok káosz elemzése."""
        
        if len(past_draws) < 10:
            return
        
        # Idősor létrehozása
        time_series = []
        for draw in past_draws:
            time_series.extend(draw)
        
        # Káosz metrikák számítása
        self.historical_chaos_metrics = {
            'lyapunov_exponent': self._estimate_lyapunov_exponent(time_series),
            'correlation_dimension': self._estimate_correlation_dimension(time_series),
            'entropy': self._calculate_entropy(time_series),
            'fractal_dimension': self._estimate_fractal_dimension(time_series)
        }
        
        # Attraktor paraméterek beállítása
        self._adapt_attractor_parameters()
    
    def _estimate_lyapunov_exponent(self, time_series: List[int]) -> float:
        """Lyapunov exponens becslése."""
        if len(time_series) < 20:
            return 0.0
        
        # Egyszerű közelítés: szomszédos pontok távolságának növekedése
        distances = []
        for i in range(len(time_series) - 1):
            distance = abs(time_series[i+1] - time_series[i])
            if distance > 0:
                distances.append(distance)
        
        if len(distances) < 2:
            return 0.0
        
        # Exponenciális növekedés mértéke
        log_distances = [math.log(d) for d in distances if d > 0]
        if len(log_distances) < 2:
            return 0.0
        
        # Lineáris trend a log távolságokban
        n = len(log_distances)
        x_mean = (n - 1) / 2
        y_mean = sum(log_distances) / n
        
        numerator = sum((i - x_mean) * (log_distances[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def _estimate_correlation_dimension(self, time_series: List[int]) -> float:
        """Korrelációs dimenzió becslése."""
        if len(time_series) < 10:
            return 1.0
        
        # Embedding dimenzió
        m = 3
        tau = 1  # Késleltetés
        
        # Embedded vectors létrehozása
        embedded = []
        for i in range(len(time_series) - (m-1) * tau):
            vector = [time_series[i + j * tau] for j in range(m)]
            embedded.append(vector)
        
        if len(embedded) < 2:
            return 1.0
        
        # Correlation integral közelítése
        threshold = np.std(time_series)
        correlations = 0
        total_pairs = 0
        
        for i in range(len(embedded)):
            for j in range(i + 1, len(embedded)):
                distance = math.sqrt(sum((embedded[i][k] - embedded[j][k])**2 for k in range(m)))
                if distance < threshold:
                    correlations += 1
                total_pairs += 1
        
        correlation_ratio = correlations / total_pairs if total_pairs > 0 else 0
        return math.log(correlation_ratio) / math.log(threshold) if correlation_ratio > 0 and threshold > 0 else 1.0
    
    def _calculate_entropy(self, time_series: List[int]) -> float:
        """Shannon entrópia számítása."""
        if not time_series:
            return 0.0
        
        # Gyakoriság számítása
        frequency = Counter(time_series)
        total = len(time_series)
        
        # Shannon entrópia
        entropy = 0.0
        for count in frequency.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _estimate_fractal_dimension(self, time_series: List[int]) -> float:
        """Fraktál dimenzió becslése box-counting módszerrel."""
        if len(time_series) < 4:
            return 1.0
        
        # 2D reprezentáció készítése
        points = [(i, time_series[i]) for i in range(len(time_series))]
        
        # Box sizes
        box_sizes = [1, 2, 4, 8, 16]
        counts = []
        
        for box_size in box_sizes:
            boxes = set()
            for x, y in points:
                box_x = int(x // box_size)
                box_y = int(y // box_size)
                boxes.add((box_x, box_y))
            counts.append(len(boxes))
        
        # Log-log fit a dimenzió becslésére
        if len(counts) >= 2 and all(c > 0 for c in counts):
            log_sizes = [math.log(1/size) for size in box_sizes]
            log_counts = [math.log(count) for count in counts]
            
            # Lineáris regresszió
            n = len(log_sizes)
            x_mean = sum(log_sizes) / n
            y_mean = sum(log_counts) / n
            
            numerator = sum((log_sizes[i] - x_mean) * (log_counts[i] - y_mean) for i in range(n))
            denominator = sum((log_sizes[i] - x_mean) ** 2 for i in range(n))
            
            return numerator / denominator if denominator != 0 else 1.0
        
        return 1.0
    
    def _adapt_attractor_parameters(self):
        """Attraktor paraméterek adaptálása történeti adatok alapján."""
        
        # Lorenz paraméterek adaptálása
        lyapunov = self.historical_chaos_metrics.get('lyapunov_exponent', 0.0)
        entropy = self.historical_chaos_metrics.get('entropy', 1.0)
        
        # Paraméterek skálázása
        self.lorenz.sigma = 10.0 + lyapunov * 5.0
        self.lorenz.rho = 28.0 + entropy * 10.0
        self.lorenz.beta = 8.0/3.0 + lyapunov * 2.0
        
        # Kaotikus térkép paraméterek
        self.attractor_parameters = {
            'logistic_r': min(4.0, 3.5 + entropy * 0.5),
            'henon_a': 1.4 + lyapunov * 0.3,
            'henon_b': 0.3 + entropy * 0.1,
            'tent_mu': min(2.0, 1.5 + lyapunov * 0.5)
        }
    
    def generate_chaotic_predictions(self) -> List[int]:
        """Kaotikus predikciók generálása."""
        
        all_predictions = []
        
        # Lorenz attraktor alapú számok
        lorenz_points = self.lorenz.generate_sequence(self.target_count * 10)
        lorenz_numbers = self.lorenz.normalize_to_range(lorenz_points, self.min_num, self.max_num)
        all_predictions.extend(lorenz_numbers[:self.target_count])
        
        # Kaotikus térképek
        for map_name, chaotic_map in self.chaotic_maps.items():
            params = {}
            if map_name == 'logistic':
                params['r'] = self.attractor_parameters.get('logistic_r', 3.9)
            elif map_name == 'henon':
                params['a'] = self.attractor_parameters.get('henon_a', 1.4)
                params['b'] = self.attractor_parameters.get('henon_b', 0.3)
            elif map_name == 'tent':
                params['mu'] = self.attractor_parameters.get('tent_mu', 1.8)
            
            sequence = chaotic_map.generate_sequence(self.target_count, **params)
            
            # Normalizálás
            normalized = []
            for value in sequence:
                if 0 <= value <= 1:
                    number = int(self.min_num + value * (self.max_num - self.min_num))
                    if self.min_num <= number <= self.max_num:
                        normalized.append(number)
            
            all_predictions.extend(normalized[:self.target_count // 2])
        
        return all_predictions
    
    def generate_fractal_predictions(self) -> List[int]:
        """Fraktál alapú predikciók."""
        
        all_predictions = []
        
        # Mandelbrot számok
        mandelbrot_nums = self.fractal_gen.mandelbrot_sequence(50, 50, 30)
        all_predictions.extend(mandelbrot_nums[:self.target_count])
        
        # Julia számok
        julia_nums = self.fractal_gen.julia_sequence(width=50, height=50, max_iter=30)
        all_predictions.extend(julia_nums[:self.target_count])
        
        # Sierpinski számok
        sierpinski_nums = self.fractal_gen.sierpinski_triangle(500)
        all_predictions.extend(sierpinski_nums[:self.target_count])
        
        # Dragon curve számok
        dragon_nums = self.fractal_gen.dragon_curve(8)
        all_predictions.extend(dragon_nums[:self.target_count])
        
        return all_predictions
    
    def predict(self, past_draws: List[List[int]]) -> List[int]:
        """Káosz és fraktál alapú predikció."""
        
        # Történeti káosz elemzés
        self.analyze_historical_chaos(past_draws)
        
        # Predikciók generálása
        chaotic_predictions = self.generate_chaotic_predictions()
        fractal_predictions = self.generate_fractal_predictions()
        
        # Kombinált predikció
        all_predictions = chaotic_predictions + fractal_predictions
        
        # Gyakoriság alapú szűrés
        prediction_counter = Counter(all_predictions)
        most_common = [num for num, _ in prediction_counter.most_common(self.target_count * 3)]
        
        # Érvényes számok szűrése
        valid_predictions = [num for num in most_common if self.min_num <= num <= self.max_num]
        
        # Diverzitás biztosítása
        diverse_predictions = self._ensure_chaos_diversity(valid_predictions)
        
        return diverse_predictions[:self.target_count]
    
    def _ensure_chaos_diversity(self, predictions: List[int]) -> List[int]:
        """Káosz-specifikus diverzitás biztosítása."""
        
        diverse_numbers = []
        
        for num in predictions:
            # Fraktál-alapú diverzitás: minden szám legalább fraktál dimenzió távolságra
            fractal_dim = self.historical_chaos_metrics.get('fractal_dimension', 1.5)
            min_distance = max(2, int(fractal_dim))
            
            if all(abs(num - existing) >= min_distance for existing in diverse_numbers):
                diverse_numbers.append(num)
            
            if len(diverse_numbers) >= self.target_count:
                break
        
        # Kiegészítés kaotikus mintázattal
        while len(diverse_numbers) < self.target_count:
            remaining = [num for num in range(self.min_num, self.max_num + 1) 
                        if num not in diverse_numbers]
            if remaining:
                # Káosz-alapú kiválasztás
                selected = self._chaotic_selection(remaining)
                diverse_numbers.append(selected)
            else:
                break
        
        return diverse_numbers
    
    def _chaotic_selection(self, candidates: List[int]) -> int:
        """Kaotikus kiválasztási stratégia."""
        
        if not candidates:
            return random.randint(self.min_num, self.max_num)
        
        # Logisztikus térkép alapú kiválasztás
        logistic_map = ChaoticMap('logistic')
        r = self.attractor_parameters.get('logistic_r', 3.9)
        
        chaotic_value = logistic_map.logistic_map(r)
        index = int(chaotic_value * len(candidates))
        index = min(index, len(candidates) - 1)
        
        return candidates[index]


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Chaos Theory & Fractal alapú lottószám predikció.
    Returns a tuple (main_numbers, additional_numbers).
    """
    try:
        # Fő számok generálása
        main_numbers = generate_chaos_fractal_numbers(
            lottery_type_instance,
            min_number=int(lottery_type_instance.min_number),
            max_number=int(lottery_type_instance.max_number),
            pieces_of_draw_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
            is_main=True
        )

        # Kiegészítő számok generálása
        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            additional_numbers = generate_chaos_fractal_numbers(
                lottery_type_instance,
                min_number=int(lottery_type_instance.additional_min_number),
                max_number=int(lottery_type_instance.additional_max_number),
                pieces_of_draw_numbers=int(lottery_type_instance.additional_numbers_count),
                is_main=False
            )

        return sorted(main_numbers), sorted(additional_numbers)
    
    except Exception as e:
        logger.error(f"Hiba a chaos fractal predikció során: {e}")
        return generate_fallback_numbers(lottery_type_instance)


def generate_chaos_fractal_numbers(
    lottery_type_instance: lg_lottery_type,
    min_number: int,
    max_number: int,
    pieces_of_draw_numbers: int,
    is_main: bool
) -> List[int]:
    """Chaos fractal számgenerálás."""
    
    # Történeti adatok lekérése
    past_draws = get_historical_data(lottery_type_instance, is_main)
    
    if len(past_draws) < 5:
        return generate_chaos_fallback(min_number, max_number, pieces_of_draw_numbers)
    
    # Chaos fractal predictor
    predictor = ChaosTheoryFractalPredictor(
        min_num=min_number, 
        max_num=max_number, 
        target_count=pieces_of_draw_numbers
    )
    
    # Predikció
    predictions = predictor.predict(past_draws)
    
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


def generate_chaos_fallback(min_number: int, max_number: int, count: int) -> List[int]:
    """Káosz-alapú fallback számgenerálás."""
    
    # Egyszerű logisztikus térkép
    chaotic_map = ChaoticMap('logistic')
    r = 3.9  # Kaotikus tartomány
    
    predictions = set()
    
    while len(predictions) < count:
        chaotic_value = chaotic_map.logistic_map(r)
        number = int(min_number + chaotic_value * (max_number - min_number))
        
        if min_number <= number <= max_number:
            predictions.add(number)
    
    return sorted(list(predictions))


def generate_fallback_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Fallback számgenerálás."""
    main_numbers = generate_chaos_fallback(
        int(lottery_type_instance.min_number),
        int(lottery_type_instance.max_number),
        int(lottery_type_instance.pieces_of_draw_numbers)
    )
    
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_chaos_fallback(
            int(lottery_type_instance.additional_min_number),
            int(lottery_type_instance.additional_max_number),
            int(lottery_type_instance.additional_numbers_count)
        )
    
    return sorted(main_numbers), sorted(additional_numbers) 