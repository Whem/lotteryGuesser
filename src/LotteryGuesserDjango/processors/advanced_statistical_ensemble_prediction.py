# advanced_statistical_ensemble_prediction.py
"""
Fejlett Statisztikai Ensemble Predikció
Kombinál több megközelítést robusztus validációval és adaptív tanulással
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Set
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import random
import math
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from algorithms.models import lg_lottery_winner_number, lg_lottery_type


class AdvancedStatisticalEnsemble:
    """
    Fejlett statisztikai ensemble predikció osztály
    Kombinálja a frekvencia elemzést, időbeli mintákat, és statisztikai validációt
    """
    
    def __init__(self):
        self.min_historical_draws = 50  # Minimum történeti húzások száma
        self.temporal_window_sizes = [10, 20, 50, 100]  # Különböző időablakok
        self.frequency_decay_factor = 0.95  # Időbeli súlyozás
        self.ensemble_weights = {
            'frequency_based': 0.25,
            'temporal_pattern': 0.25,
            'statistical_model': 0.20,
            'gap_analysis': 0.15,
            'clustering': 0.15
        }
    
    def get_numbers(self, lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
        """
        Főbejárási pont a fejlett statisztikai ensemble predikcióhoz
        """
        try:
            # Fő számok generálása
            main_numbers = self._generate_number_set(
                lottery_type_instance,
                min_num=int(lottery_type_instance.min_number),
                max_num=int(lottery_type_instance.max_number),
                required_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
                is_main=True
            )
            
            # Kiegészítő számok generálása
            additional_numbers = []
            if lottery_type_instance.has_additional_numbers:
                additional_numbers = self._generate_number_set(
                    lottery_type_instance,
                    min_num=int(lottery_type_instance.additional_min_number),
                    max_num=int(lottery_type_instance.additional_max_number),
                    required_numbers=int(lottery_type_instance.additional_numbers_count),
                    is_main=False
                )
            
            return sorted(main_numbers), sorted(additional_numbers)
            
        except Exception as e:
            print(f"Hiba az advanced_statistical_ensemble_prediction-ben: {e}")
            return self._generate_fallback_numbers(lottery_type_instance)
    
    def _generate_number_set(self, lottery_type_instance: lg_lottery_type, 
                           min_num: int, max_num: int, required_numbers: int, 
                           is_main: bool) -> List[int]:
        """
        Számhalmaz generálása ensemble módszerekkel
        """
        # Történeti adatok lekérése
        historical_data = self._get_historical_data(lottery_type_instance, is_main)
        
        if len(historical_data) < self.min_historical_draws:
            return self._generate_smart_random(min_num, max_num, required_numbers)
        
        # Ensemble predikciók generálása
        predictions = {}
        
        # 1. Frekvencia alapú predikció
        predictions['frequency_based'] = self._frequency_based_prediction(
            historical_data, min_num, max_num, required_numbers
        )
        
        # 2. Időbeli minta predikció
        predictions['temporal_pattern'] = self._temporal_pattern_prediction(
            historical_data, min_num, max_num, required_numbers
        )
        
        # 3. Statisztikai modell predikció
        predictions['statistical_model'] = self._statistical_model_prediction(
            historical_data, min_num, max_num, required_numbers
        )
        
        # 4. Gap elemzés predikció
        predictions['gap_analysis'] = self._gap_analysis_prediction(
            historical_data, min_num, max_num, required_numbers
        )
        
        # 5. Klaszterezés alapú predikció
        predictions['clustering'] = self._clustering_prediction(
            historical_data, min_num, max_num, required_numbers
        )
        
        # Ensemble szavazás
        final_numbers = self._ensemble_voting(
            predictions, min_num, max_num, required_numbers
        )
        
        # Validáció és finomhangolás
        final_numbers = self._validate_and_refine(
            final_numbers, historical_data, min_num, max_num, required_numbers
        )
        
        return final_numbers
    
    def _get_historical_data(self, lottery_type_instance: lg_lottery_type, 
                           is_main: bool) -> List[List[int]]:
        """
        Történeti adatok lekérése és előfeldolgozása
        """
        field_name = 'lottery_type_number' if is_main else 'additional_numbers'
        
        queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('-id').values_list(field_name, flat=True)
        
        historical_data = []
        for draw in queryset:
            if isinstance(draw, list) and len(draw) > 0:
                # Csak érvényes számokat tartunk meg
                valid_numbers = [int(num) for num in draw if isinstance(num, (int, float))]
                if valid_numbers:
                    historical_data.append(valid_numbers)
        
        return historical_data[:200]  # Legutóbbi 200 húzás
    
    def _frequency_based_prediction(self, historical_data: List[List[int]], 
                                  min_num: int, max_num: int, 
                                  required_numbers: int) -> List[int]:
        """
        Frekvencia alapú predikció időbeli súlyozással
        """
        weighted_counter = Counter()
        
        for i, draw in enumerate(historical_data):
            # Időbeli súlyozás (újabb húzások nagyobb súlyt kapnak)
            weight = self.frequency_decay_factor ** i
            for number in draw:
                if min_num <= number <= max_num:
                    weighted_counter[number] += weight
        
        # Leggyakoribb számok kiválasztása
        most_common = [num for num, _ in weighted_counter.most_common()]
        
        # Kiegészítés hiányzó számokkal
        if len(most_common) < required_numbers:
            missing_numbers = [num for num in range(min_num, max_num + 1) 
                             if num not in most_common]
            random.shuffle(missing_numbers)
            most_common.extend(missing_numbers)
        
        return most_common[:required_numbers]
    
    def _temporal_pattern_prediction(self, historical_data: List[List[int]], 
                                   min_num: int, max_num: int, 
                                   required_numbers: int) -> List[int]:
        """
        Időbeli minták elemzése és predikció
        """
        if len(historical_data) < 20:
            return self._generate_smart_random(min_num, max_num, required_numbers)
        
        # Különböző időablakok elemzése
        pattern_scores = defaultdict(float)
        
        for window_size in self.temporal_window_sizes:
            if len(historical_data) >= window_size:
                recent_window = historical_data[:window_size]
                older_window = historical_data[window_size:window_size*2] if len(historical_data) >= window_size*2 else []
                
                # Trend számítás
                recent_freq = Counter(num for draw in recent_window for num in draw)
                older_freq = Counter(num for draw in older_window for num in draw) if older_window else Counter()
                
                for num in range(min_num, max_num + 1):
                    recent_count = recent_freq.get(num, 0)
                    older_count = older_freq.get(num, 0)
                    
                    # Trend score (pozitív, ha növekvő trend)
                    if older_count > 0:
                        trend = (recent_count - older_count) / older_count
                    else:
                        trend = recent_count
                    
                    pattern_scores[num] += trend / len(self.temporal_window_sizes)
        
        # Legmagasabb trend score-ú számok kiválasztása
        sorted_numbers = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [num for num, _ in sorted_numbers[:required_numbers]]
        
        # Kiegészítés szükség esetén
        if len(selected) < required_numbers:
            remaining = [num for num in range(min_num, max_num + 1) if num not in selected]
            random.shuffle(remaining)
            selected.extend(remaining[:required_numbers - len(selected)])
        
        return selected[:required_numbers]
    
    def _statistical_model_prediction(self, historical_data: List[List[int]], 
                                    min_num: int, max_num: int, 
                                    required_numbers: int) -> List[int]:
        """
        Statisztikai modell alapú predikció
        """
        if len(historical_data) < 30:
            return self._generate_smart_random(min_num, max_num, required_numbers)
        
        try:
            # Adatok előkészítése
            df_data = []
            for i, draw in enumerate(historical_data):
                for num in draw:
                    if min_num <= num <= max_num:
                        df_data.append({
                            'number': num,
                            'draw_index': i,
                            'position': draw.index(num) if num in draw else 0,
                            'sum_of_draw': sum(draw),
                            'count_in_draw': len(draw)
                        })
            
            if not df_data:
                return self._generate_smart_random(min_num, max_num, required_numbers)
            
            df = pd.DataFrame(df_data)
            
            # Random Forest modell
            features = ['draw_index', 'position', 'sum_of_draw', 'count_in_draw']
            X = df[features]
            y = df['number']
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Predikció a következő húzásra
            next_draw_features = np.array([[0, 0, np.mean([sum(draw) for draw in historical_data[:10]]), 
                                          np.mean([len(draw) for draw in historical_data[:10]])]])
            
            # Több predikció generálása
            predictions = []
            for _ in range(required_numbers * 3):
                pred = model.predict(next_draw_features)[0]
                predictions.append(max(min_num, min(max_num, int(round(pred)))))
            
            # Leggyakoribb predikciók kiválasztása
            pred_counter = Counter(predictions)
            selected = [num for num, _ in pred_counter.most_common(required_numbers)]
            
            # Kiegészítés szükség esetén
            if len(selected) < required_numbers:
                remaining = [num for num in range(min_num, max_num + 1) if num not in selected]
                random.shuffle(remaining)
                selected.extend(remaining[:required_numbers - len(selected)])
            
            return selected[:required_numbers]
            
        except Exception:
            return self._generate_smart_random(min_num, max_num, required_numbers)
    
    def _gap_analysis_prediction(self, historical_data: List[List[int]], 
                               min_num: int, max_num: int, 
                               required_numbers: int) -> List[int]:
        """
        Gap elemzés alapú predikció
        """
        # Számok utolsó megjelenésének követése
        last_seen = {}
        gap_stats = defaultdict(list)
        
        for draw_index, draw in enumerate(historical_data):
            for num in range(min_num, max_num + 1):
                if num in draw:
                    if num in last_seen:
                        gap = draw_index - last_seen[num]
                        gap_stats[num].append(gap)
                    last_seen[num] = draw_index
        
        # Gap alapú score számítás
        gap_scores = {}
        for num in range(min_num, max_num + 1):
            if num in last_seen:
                current_gap = len(historical_data) - 1 - last_seen[num]
                if gap_stats[num]:
                    avg_gap = np.mean(gap_stats[num])
                    gap_scores[num] = current_gap / max(avg_gap, 1)
                else:
                    gap_scores[num] = current_gap
            else:
                gap_scores[num] = len(historical_data)  # Soha nem jelent meg
        
        # Legmagasabb gap score-ú számok
        sorted_by_gap = sorted(gap_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [num for num, _ in sorted_by_gap[:required_numbers]]
        
        return selected
    
    def _clustering_prediction(self, historical_data: List[List[int]], 
                             min_num: int, max_num: int, 
                             required_numbers: int) -> List[int]:
        """
        Klaszterezés alapú predikció
        """
        if len(historical_data) < 20:
            return self._generate_smart_random(min_num, max_num, required_numbers)
        
        try:
            # Húzások vektorizálása
            vectors = []
            for draw in historical_data[:50]:  # Legutóbbi 50 húzás
                vector = [0] * (max_num - min_num + 1)
                for num in draw:
                    if min_num <= num <= max_num:
                        vector[num - min_num] = 1
                vectors.append(vector)
            
            if len(vectors) < 10:
                return self._generate_smart_random(min_num, max_num, required_numbers)
            
            # K-means klaszterezés
            n_clusters = min(5, len(vectors) // 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(vectors)
            
            # Leggyakoribb klaszter keresése
            cluster_counts = Counter(clusters)
            most_common_cluster = cluster_counts.most_common(1)[0][0]
            
            # Klaszter centroid alapján számok kiválasztása
            centroid = kmeans.cluster_centers_[most_common_cluster]
            
            # Legmagasabb értékű pozíciók kiválasztása
            number_scores = [(i + min_num, score) for i, score in enumerate(centroid)]
            number_scores.sort(key=lambda x: x[1], reverse=True)
            
            selected = [num for num, _ in number_scores[:required_numbers]]
            
            return selected
            
        except Exception:
            return self._generate_smart_random(min_num, max_num, required_numbers)
    
    def _ensemble_voting(self, predictions: Dict[str, List[int]], 
                        min_num: int, max_num: int, 
                        required_numbers: int) -> List[int]:
        """
        Ensemble szavazás súlyozott módon
        """
        vote_scores = defaultdict(float)
        
        for method, numbers in predictions.items():
            weight = self.ensemble_weights.get(method, 0.1)
            for i, num in enumerate(numbers):
                # Pozíció alapú súlyozás (első helyen lévő számok nagyobb súlyt kapnak)
                position_weight = 1.0 / (i + 1)
                vote_scores[num] += weight * position_weight
        
        # Legmagasabb score-ú számok kiválasztása
        sorted_votes = sorted(vote_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [num for num, _ in sorted_votes[:required_numbers]]
        
        # Kiegészítés szükség esetén
        if len(selected) < required_numbers:
            remaining = [num for num in range(min_num, max_num + 1) if num not in selected]
            random.shuffle(remaining)
            selected.extend(remaining[:required_numbers - len(selected)])
        
        return selected[:required_numbers]
    
    def _validate_and_refine(self, numbers: List[int], historical_data: List[List[int]], 
                           min_num: int, max_num: int, required_numbers: int) -> List[int]:
        """
        Validáció és finomhangolás
        """
        # Duplikátumok eltávolítása
        unique_numbers = list(set(numbers))
        
        # Tartomány ellenőrzés
        valid_numbers = [num for num in unique_numbers if min_num <= num <= max_num]
        
        # Statisztikai validáció
        if len(historical_data) > 10:
            # Átlag és szórás számítás
            all_historical = [num for draw in historical_data[:20] for num in draw]
            hist_mean = np.mean(all_historical)
            hist_std = np.std(all_historical)
            
            # Outlier számok cseréje
            refined_numbers = []
            for num in valid_numbers:
                z_score = abs(num - hist_mean) / max(hist_std, 1)
                if z_score < 2.5:  # Nem outlier
                    refined_numbers.append(num)
            
            # Ha túl sok outlier-t távolítottunk el, visszatöltjük
            if len(refined_numbers) < required_numbers // 2:
                refined_numbers = valid_numbers
        else:
            refined_numbers = valid_numbers
        
        # Kiegészítés szükség esetén
        if len(refined_numbers) < required_numbers:
            remaining = [num for num in range(min_num, max_num + 1) 
                        if num not in refined_numbers]
            random.shuffle(remaining)
            refined_numbers.extend(remaining[:required_numbers - len(refined_numbers)])
        
        return refined_numbers[:required_numbers]
    
    def _generate_smart_random(self, min_num: int, max_num: int, 
                             required_numbers: int) -> List[int]:
        """
        Intelligens véletlen számgenerálás
        """
        # Normál eloszlás alapú generálás
        center = (min_num + max_num) / 2
        std = (max_num - min_num) / 6
        
        numbers = set()
        attempts = 0
        
        while len(numbers) < required_numbers and attempts < 1000:
            num = int(np.random.normal(center, std))
            if min_num <= num <= max_num:
                numbers.add(num)
            attempts += 1
        
        # Kiegészítés egyenletes eloszlással
        if len(numbers) < required_numbers:
            remaining = [num for num in range(min_num, max_num + 1) if num not in numbers]
            random.shuffle(remaining)
            numbers.update(remaining[:required_numbers - len(numbers)])
        
        return list(numbers)[:required_numbers]
    
    def _generate_fallback_numbers(self, lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
        """
        Fallback számgenerálás hiba esetén
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
ensemble_predictor = AdvancedStatisticalEnsemble()

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Főbejárási pont a fejlett statisztikai ensemble predikcióhoz
    """
    return ensemble_predictor.get_numbers(lottery_type_instance)
