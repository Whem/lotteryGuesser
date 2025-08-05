# graph_neural_network_prediction.py
"""
Graph Neural Network Predikció
GNN és Graph Attention Network használata lottószám kapcsolatok modellezésére
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from collections import Counter, defaultdict
import random
import math
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


class GraphNeuralNetworkPredictor:
    """
    Graph Neural Network alapú predikció
    """
    
    def __init__(self):
        self.hidden_dim = 64
        self.num_layers = 3
        self.num_heads = 4
        self.dropout_rate = 0.1
        self.learning_rate = 0.01
        
        # GNN réteg súlyok
        self.W_self = [np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1 for _ in range(self.num_layers)]
        self.W_neigh = [np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1 for _ in range(self.num_layers)]
        self.W_att = [np.random.randn(self.hidden_dim, self.num_heads) * 0.1 for _ in range(self.num_layers)]
        
        # Gráf struktúra
        self.adjacency_matrix = None
        self.node_features = None
        self.edge_weights = None
        
        # Aggregáció típusok
        self.aggregation_types = ['mean', 'max', 'sum', 'attention']
    
    def get_numbers(self, lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
        """
        Főbejárási pont a GNN predikcióhoz
        """
        try:
            # Fő számok generálása
            main_numbers = self._generate_gnn_numbers(
                lottery_type_instance,
                min_num=int(lottery_type_instance.min_number),
                max_num=int(lottery_type_instance.max_number),
                required_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
                is_main=True
            )
            
            # Kiegészítő számok generálása
            additional_numbers = []
            if lottery_type_instance.has_additional_numbers:
                additional_numbers = self._generate_gnn_numbers(
                    lottery_type_instance,
                    min_num=int(lottery_type_instance.additional_min_number),
                    max_num=int(lottery_type_instance.additional_max_number),
                    required_numbers=int(lottery_type_instance.additional_numbers_count),
                    is_main=False
                )
            
            return sorted(main_numbers), sorted(additional_numbers)
            
        except Exception as e:
            print(f"Hiba a graph_neural_network_prediction-ben: {e}")
            return self._generate_fallback_numbers(lottery_type_instance)
    
    def _generate_gnn_numbers(self, lottery_type_instance: lg_lottery_type,
                            min_num: int, max_num: int, required_numbers: int,
                            is_main: bool) -> List[int]:
        """
        GNN alapú számgenerálás
        """
        # Történeti adatok lekérése
        historical_data = self._get_historical_data(lottery_type_instance, is_main)
        
        if len(historical_data) < 15:
            return self._generate_smart_random(min_num, max_num, required_numbers)
        
        # Gráf építése
        self._build_number_graph(historical_data, min_num, max_num)
        
        # Node jellemzők inicializálása
        self._initialize_node_features(historical_data, min_num, max_num)
        
        # GNN forward pass
        node_embeddings = self._gnn_forward_pass()
        
        # Graph Attention Network
        attention_embeddings = self._graph_attention_network(node_embeddings)
        
        # Predikció generálása
        predictions = self._generate_graph_predictions(
            attention_embeddings, min_num, max_num, required_numbers
        )
        
        # Gráf alapú finomhangolás
        refined_predictions = self._graph_based_refinement(
            predictions, node_embeddings, min_num, max_num, required_numbers
        )
        
        return refined_predictions
    
    def _build_number_graph(self, historical_data: List[List[int]], 
                          min_num: int, max_num: int) -> None:
        """
        Számok közötti gráf építése
        """
        num_nodes = max_num - min_num + 1
        self.adjacency_matrix = np.zeros((num_nodes, num_nodes))
        self.edge_weights = np.zeros((num_nodes, num_nodes))
        
        # Co-occurrence alapú élek
        co_occurrence = defaultdict(int)
        
        for draw in historical_data:
            # Minden számpár között él
            for i, num1 in enumerate(draw):
                for j, num2 in enumerate(draw):
                    if i != j and min_num <= num1 <= max_num and min_num <= num2 <= max_num:
                        idx1, idx2 = num1 - min_num, num2 - min_num
                        co_occurrence[(idx1, idx2)] += 1
        
        # Adjacency mátrix feltöltése
        for (idx1, idx2), count in co_occurrence.items():
            self.adjacency_matrix[idx1][idx2] = 1
            self.edge_weights[idx1][idx2] = count
        
        # Normalizálás
        max_weight = np.max(self.edge_weights) if np.max(self.edge_weights) > 0 else 1
        self.edge_weights = self.edge_weights / max_weight
        
        # Szekvenciális kapcsolatok hozzáadása
        self._add_sequential_connections(historical_data, min_num, max_num)
        
        # Numerikus távolság alapú kapcsolatok
        self._add_distance_connections(min_num, max_num)
    
    def _add_sequential_connections(self, historical_data: List[List[int]], 
                                  min_num: int, max_num: int) -> None:
        """
        Szekvenciális kapcsolatok hozzáadása
        """
        for i in range(len(historical_data) - 1):
            current_draw = historical_data[i]
            next_draw = historical_data[i + 1]
            
            # Kapcsolatok a következő húzás számaival
            for num1 in current_draw:
                for num2 in next_draw:
                    if min_num <= num1 <= max_num and min_num <= num2 <= max_num:
                        idx1, idx2 = num1 - min_num, num2 - min_num
                        self.adjacency_matrix[idx1][idx2] = 1
                        self.edge_weights[idx1][idx2] += 0.5  # Szekvenciális bónusz
    
    def _add_distance_connections(self, min_num: int, max_num: int) -> None:
        """
        Numerikus távolság alapú kapcsolatok
        """
        for num1 in range(min_num, max_num + 1):
            for num2 in range(min_num, max_num + 1):
                if num1 != num2:
                    idx1, idx2 = num1 - min_num, num2 - min_num
                    distance = abs(num1 - num2)
                    
                    # Közeli számok között erősebb kapcsolat
                    if distance <= 5:
                        connection_strength = 1.0 / (distance + 1)
                        self.adjacency_matrix[idx1][idx2] = 1
                        self.edge_weights[idx1][idx2] += connection_strength * 0.3
    
    def _initialize_node_features(self, historical_data: List[List[int]], 
                                min_num: int, max_num: int) -> None:
        """
        Node jellemzők inicializálása
        """
        num_nodes = max_num - min_num + 1
        self.node_features = np.zeros((num_nodes, self.hidden_dim))
        
        # Frekvencia alapú jellemzők
        frequency_counter = Counter(num for draw in historical_data for num in draw)
        
        # Recency jellemzők
        recency = {}
        for i, draw in enumerate(historical_data):
            for num in draw:
                if num not in recency:
                    recency[num] = i
        
        # Gap jellemzők
        gaps = {}
        for num in range(min_num, max_num + 1):
            gaps[num] = recency.get(num, len(historical_data))
        
        # Node jellemzők feltöltése
        for num in range(min_num, max_num + 1):
            idx = num - min_num
            
            # Alapvető jellemzők
            features = [
                frequency_counter.get(num, 0) / len(historical_data),  # Normalizált frekvencia
                1.0 / (gaps[num] + 1),  # Recency score
                num / max_num,  # Normalizált érték
                (num % 2),  # Páros/páratlan
                (num % 3 == 0),  # Osztható 3-mal
                (num % 5 == 0),  # Osztható 5-tel
                (num % 7 == 0),  # Osztható 7-tel
            ]
            
            # Statisztikai jellemzők
            if frequency_counter:
                avg_freq = np.mean(list(frequency_counter.values()))
                std_freq = np.std(list(frequency_counter.values()))
                z_score = (frequency_counter.get(num, 0) - avg_freq) / max(std_freq, 1)
                features.append(z_score)
            else:
                features.append(0.0)
            
            # Pozíciós jellemzők
            position_scores = self._calculate_position_features(num, historical_data)
            features.extend(position_scores)
            
            # Padding vagy truncation
            while len(features) < self.hidden_dim:
                features.append(0.0)
            
            self.node_features[idx] = np.array(features[:self.hidden_dim])
    
    def _calculate_position_features(self, num: int, historical_data: List[List[int]]) -> List[float]:
        """
        Pozíció alapú jellemzők számítása
        """
        position_counts = [0] * 10  # Max 10 pozíció
        
        for draw in historical_data:
            if num in draw:
                pos = draw.index(num)
                if pos < len(position_counts):
                    position_counts[pos] += 1
        
        # Normalizálás
        total_appearances = sum(position_counts)
        if total_appearances > 0:
            position_features = [count / total_appearances for count in position_counts]
        else:
            position_features = [0.0] * 10
        
        return position_features[:5]  # Első 5 pozíció
    
    def _gnn_forward_pass(self) -> np.ndarray:
        """
        GNN forward pass
        """
        current_features = self.node_features.copy()
        
        for layer in range(self.num_layers):
            # Szomszédok aggregálása
            neighbor_features = self._aggregate_neighbors(current_features, layer)
            
            # Self-transformation
            self_transformed = np.dot(current_features, self.W_self[layer])
            
            # Neighbor transformation
            neighbor_transformed = np.dot(neighbor_features, self.W_neigh[layer])
            
            # Kombináció
            combined = self_transformed + neighbor_transformed
            
            # Aktiváció
            current_features = self._relu(combined)
            
            # Residual connection
            if layer > 0:
                current_features += self.node_features
            
            # Normalizálás
            current_features = self._layer_normalize(current_features)
        
        return current_features
    
    def _aggregate_neighbors(self, features: np.ndarray, layer: int) -> np.ndarray:
        """
        Szomszédok aggregálása
        """
        num_nodes = features.shape[0]
        aggregated = np.zeros_like(features)
        
        for node in range(num_nodes):
            neighbors = np.where(self.adjacency_matrix[node] > 0)[0]
            
            if len(neighbors) > 0:
                # Weighted aggregation
                neighbor_features = features[neighbors]
                weights = self.edge_weights[node][neighbors].reshape(-1, 1)
                
                # Különböző aggregáció típusok
                mean_agg = np.mean(neighbor_features * weights, axis=0)
                max_agg = np.max(neighbor_features * weights, axis=0)
                sum_agg = np.sum(neighbor_features * weights, axis=0)
                
                # Attention aggregation
                attention_weights = self._calculate_attention_weights(
                    features[node], neighbor_features, layer
                )
                att_agg = np.sum(neighbor_features * attention_weights.reshape(-1, 1), axis=0)
                
                # Kombinált aggregáció
                aggregated[node] = (mean_agg + max_agg + sum_agg + att_agg) / 4
            else:
                aggregated[node] = features[node]
        
        return aggregated
    
    def _calculate_attention_weights(self, query: np.ndarray, keys: np.ndarray, layer: int) -> np.ndarray:
        """
        Attention súlyok számítása
        """
        if len(keys) == 0:
            return np.array([])
        
        # Multi-head attention
        attention_scores = []
        
        for head in range(self.num_heads):
            # Query és key projekció
            q_proj = query
            k_proj = keys
            
            # Attention scores
            scores = np.dot(k_proj, q_proj) / math.sqrt(self.hidden_dim)
            attention_scores.append(scores)
        
        # Átlagolás a fejek között
        avg_scores = np.mean(attention_scores, axis=0)
        
        # Softmax
        exp_scores = np.exp(avg_scores - np.max(avg_scores))
        attention_weights = exp_scores / np.sum(exp_scores)
        
        return attention_weights
    
    def _graph_attention_network(self, node_embeddings: np.ndarray) -> np.ndarray:
        """
        Graph Attention Network réteg
        """
        num_nodes = node_embeddings.shape[0]
        attention_embeddings = np.zeros_like(node_embeddings)
        
        for node in range(num_nodes):
            # Szomszédok és saját maga
            neighbors = list(np.where(self.adjacency_matrix[node] > 0)[0])
            neighbors.append(node)  # Self-attention
            
            if len(neighbors) > 1:
                # Attention mechanizmus
                query = node_embeddings[node]
                keys = node_embeddings[neighbors]
                
                # Attention súlyok
                attention_weights = self._calculate_attention_weights(query, keys, 0)
                
                # Weighted combination
                attention_embeddings[node] = np.sum(
                    keys * attention_weights.reshape(-1, 1), axis=0
                )
            else:
                attention_embeddings[node] = node_embeddings[node]
        
        return attention_embeddings
    
    def _generate_graph_predictions(self, embeddings: np.ndarray, 
                                  min_num: int, max_num: int, required_numbers: int) -> List[int]:
        """
        Gráf alapú predikciók generálása
        """
        # Node scores számítása
        node_scores = {}
        
        for num in range(min_num, max_num + 1):
            idx = num - min_num
            embedding = embeddings[idx]
            
            # Score számítás (embedding magnitude)
            score = np.linalg.norm(embedding)
            
            # Szomszédok hatása
            neighbors = np.where(self.adjacency_matrix[idx] > 0)[0]
            neighbor_boost = 0
            
            for neighbor_idx in neighbors:
                neighbor_embedding = embeddings[neighbor_idx]
                similarity = np.dot(embedding, neighbor_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(neighbor_embedding) + 1e-8
                )
                neighbor_boost += similarity * self.edge_weights[idx][neighbor_idx]
            
            node_scores[num] = score + neighbor_boost * 0.3
        
        # Top számok kiválasztása
        sorted_scores = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        predictions = [num for num, _ in sorted_scores[:required_numbers]]
        
        return predictions
    
    def _graph_based_refinement(self, predictions: List[int], embeddings: np.ndarray,
                              min_num: int, max_num: int, required_numbers: int) -> List[int]:
        """
        Gráf alapú finomhangolás
        """
        refined_scores = {}
        
        # Predikciók pontszámainak finomhangolása
        for num in predictions:
            idx = num - min_num
            
            # Alappontszám
            base_score = np.linalg.norm(embeddings[idx])
            
            # Gráf struktúra alapú bónusz
            structure_bonus = self._calculate_structure_bonus(idx, predictions, min_num)
            
            # Diverzitás bónusz
            diversity_bonus = self._calculate_diversity_bonus(num, predictions)
            
            refined_scores[num] = base_score + structure_bonus + diversity_bonus
        
        # További számok hozzáadása ha szükséges
        if len(predictions) < required_numbers:
            remaining_nums = [n for n in range(min_num, max_num + 1) if n not in predictions]
            
            for num in remaining_nums:
                idx = num - min_num
                base_score = np.linalg.norm(embeddings[idx])
                structure_bonus = self._calculate_structure_bonus(idx, predictions, min_num)
                diversity_bonus = self._calculate_diversity_bonus(num, predictions)
                
                refined_scores[num] = base_score + structure_bonus + diversity_bonus
        
        # Végső kiválasztás
        sorted_refined = sorted(refined_scores.items(), key=lambda x: x[1], reverse=True)
        final_predictions = [num for num, _ in sorted_refined[:required_numbers]]
        
        return final_predictions
    
    def _calculate_structure_bonus(self, node_idx: int, predictions: List[int], min_num: int) -> float:
        """
        Gráf struktúra alapú bónusz számítás
        """
        bonus = 0.0
        
        # Kapcsolatok a már kiválasztott számokkal
        for pred_num in predictions:
            pred_idx = pred_num - min_num
            if self.adjacency_matrix[node_idx][pred_idx] > 0:
                bonus += self.edge_weights[node_idx][pred_idx] * 0.5
        
        return bonus
    
    def _calculate_diversity_bonus(self, num: int, predictions: List[int]) -> float:
        """
        Diverzitás bónusz számítás
        """
        if not predictions:
            return 0.0
        
        # Távolság alapú diverzitás
        min_distance = min(abs(num - pred) for pred in predictions)
        diversity_bonus = min_distance / 10.0  # Normalizálás
        
        return diversity_bonus
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU aktiváció"""
        return np.maximum(0, x)
    
    def _layer_normalize(self, x: np.ndarray) -> np.ndarray:
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + 1e-8)
    
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
gnn_predictor = GraphNeuralNetworkPredictor()

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Főbejárási pont a GNN predikcióhoz"""
    return gnn_predictor.get_numbers(lottery_type_instance)
