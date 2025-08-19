# deep_reinforcement_learning_predictor.py
"""
Deep Reinforcement Learning Lottery Predictor
Mélytanulás-alapú megerősítéses tanulás Q-learning és Actor-Critic módszerekkel
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict, deque
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LotteryEnvironment:
    """Lottery predikciós környezet RL ágensek számára."""
    
    def __init__(self, min_num: int, max_num: int, target_count: int):
        self.min_num = min_num
        self.max_num = max_num
        self.target_count = target_count
        self.vocab_size = max_num - min_num + 1
        
        # Állapot reprezentáció
        self.state_size = self.vocab_size * 3  # 3 korábbi húzás
        self.action_size = self.vocab_size  # Minden szám egy akció
        
        # Környezeti paraméterek
        self.current_state = np.zeros(self.state_size)
        self.episode_history = deque(maxlen=3)
        self.reward_history = deque(maxlen=100)
        
    def reset(self, historical_data: List[List[int]]) -> np.ndarray:
        """Környezet visszaállítása új epizódhoz."""
        
        # Utolsó 3 húzás betöltése állapotként
        self.episode_history.clear()
        
        if len(historical_data) >= 3:
            recent_draws = historical_data[-3:]
            for draw in recent_draws:
                self.episode_history.append(draw)
        else:
            # Véletlenszerű inicializálás
            for _ in range(3):
                random_draw = random.sample(range(self.min_num, self.max_num + 1), self.target_count)
                self.episode_history.append(random_draw)
        
        return self._encode_state()
    
    def _encode_state(self) -> np.ndarray:
        """Állapot kódolása neurális hálózat számára."""
        state = np.zeros(self.state_size)
        
        for i, draw in enumerate(self.episode_history):
            offset = i * self.vocab_size
            for num in draw:
                if self.min_num <= num <= self.max_num:
                    idx = num - self.min_num
                    state[offset + idx] = 1.0
        
        return state
    
    def step(self, action: int, actual_draw: List[int]) -> Tuple[np.ndarray, float, bool]:
        """Lépés végrehajtása a környezetben."""
        
        # Jutalom számítása
        reward = self._calculate_reward(action, actual_draw)
        
        # Új állapot frissítése
        self.episode_history.append(actual_draw)
        new_state = self._encode_state()
        
        # Epizód vége (mindig egy lépéses)
        done = True
        
        self.reward_history.append(reward)
        
        return new_state, reward, done
    
    def _calculate_reward(self, action: int, actual_draw: List[int]) -> float:
        """Jutalom számítása az akció és tényleges eredmény alapján."""
        
        # Alap jutalom: helyes találat esetén
        base_reward = 10.0 if action in actual_draw else -1.0
        
        # Bónusz jutalmak
        bonus_reward = 0.0
        
        # Frekvencia bónusz
        recent_numbers = [num for draw in self.episode_history for num in draw]
        frequency = recent_numbers.count(action)
        if frequency > 0:
            bonus_reward += frequency * 0.5
        
        # Diverzitás bónusz
        if self.episode_history:
            last_draw = self.episode_history[-1]
            min_distance = min(abs(action - num) for num in last_draw)
            if min_distance >= 3:  # Távol van az előző számoktól
                bonus_reward += 2.0
        
        # Összeg bónusz (cél összeg közelében)
        draw_sum = sum(actual_draw)
        expected_sum = (self.min_num + self.max_num) * self.target_count // 2
        sum_diff = abs(draw_sum - expected_sum) / expected_sum
        sum_bonus = max(0, 1.0 - sum_diff) * 3.0
        
        total_reward = base_reward + bonus_reward + sum_bonus
        
        return total_reward


class DQNAgent:
    """Deep Q-Network ágens."""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Hyperparaméterek
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory_size = 2000
        self.batch_size = 32
        self.gamma = 0.95  # Discount factor
        
        # Experience replay memory
        self.memory = deque(maxlen=self.memory_size)
        
        # Q-hálózat (egyszerű implementáció numpy-val)
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self._update_target_network()
        
    def _build_network(self) -> Dict[str, np.ndarray]:
        """Neurális hálózat építése (egyszerű FC hálózat)."""
        network = {
            'W1': np.random.randn(self.state_size, 256) * 0.1,
            'b1': np.zeros(256),
            'W2': np.random.randn(256, 128) * 0.1,
            'b2': np.zeros(128),
            'W3': np.random.randn(128, 64) * 0.1,
            'b3': np.zeros(64),
            'W4': np.random.randn(64, self.action_size) * 0.1,
            'b4': np.zeros(self.action_size)
        }
        return network
    
    def _forward_pass(self, state: np.ndarray, network: Dict[str, np.ndarray]) -> np.ndarray:
        """Forward pass a hálózaton."""
        # Layer 1
        z1 = np.dot(state, network['W1']) + network['b1']
        a1 = np.maximum(0, z1)  # ReLU
        
        # Layer 2
        z2 = np.dot(a1, network['W2']) + network['b2']
        a2 = np.maximum(0, z2)  # ReLU
        
        # Layer 3
        z3 = np.dot(a2, network['W3']) + network['b3']
        a3 = np.maximum(0, z3)  # ReLU
        
        # Output layer
        q_values = np.dot(a3, network['W4']) + network['b4']
        
        return q_values
    
    def get_action(self, state: np.ndarray, valid_actions: List[int] = None) -> int:
        """Akció kiválasztása epsilon-greedy stratégiával."""
        
        if valid_actions is None:
            valid_actions = list(range(self.action_size))
        
        if random.random() <= self.epsilon:
            # Exploration
            return random.choice(valid_actions)
        else:
            # Exploitation
            q_values = self._forward_pass(state, self.q_network)
            
            # Csak érvényes akciók közül választ
            valid_q_values = [(action, q_values[action]) for action in valid_actions]
            best_action = max(valid_q_values, key=lambda x: x[1])[0]
            
            return best_action
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Élmény tárolása replay memory-ban."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Experience replay tanulás."""
        if len(self.memory) < self.batch_size:
            return
        
        # Mini-batch mintavételezés
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            target = reward
            
            if not done:
                next_q_values = self._forward_pass(next_state, self.target_network)
                target += self.gamma * np.max(next_q_values)
            
            # Q-értékek frissítése (egyszerűsített backprop)
            current_q_values = self._forward_pass(state, self.q_network)
            target_q_values = current_q_values.copy()
            target_q_values[action] = target
            
            # Súlyok frissítése (egyszerű gradient update)
            self._update_weights(state, target_q_values)
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _update_weights(self, state: np.ndarray, target_q_values: np.ndarray):
        """Hálózat súlyainak frissítése (egyszerűsített)."""
        
        # Forward pass gradientekkel
        current_q_values = self._forward_pass(state, self.q_network)
        
        # Loss és gradient (MSE)
        error = target_q_values - current_q_values
        
        # Egyszerű gradient update (csak az utolsó réteg)
        self.q_network['W4'] += self.learning_rate * np.outer(state[:64], error)
        self.q_network['b4'] += self.learning_rate * error
    
    def _update_target_network(self):
        """Target hálózat frissítése."""
        for key in self.q_network:
            self.target_network[key] = self.q_network[key].copy()


class ActorCriticAgent:
    """Actor-Critic ágens."""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Actor network (policy)
        self.actor_network = self._build_actor_network()
        
        # Critic network (value function)
        self.critic_network = self._build_critic_network()
        
        # Paraméterek
        self.gamma = 0.95
        
    def _build_actor_network(self) -> Dict[str, np.ndarray]:
        """Actor hálózat (policy network)."""
        return {
            'W1': np.random.randn(self.state_size, 128) * 0.1,
            'b1': np.zeros(128),
            'W2': np.random.randn(128, 64) * 0.1,
            'b2': np.zeros(64),
            'W3': np.random.randn(64, self.action_size) * 0.1,
            'b3': np.zeros(self.action_size)
        }
    
    def _build_critic_network(self) -> Dict[str, np.ndarray]:
        """Critic hálózat (value network)."""
        return {
            'W1': np.random.randn(self.state_size, 128) * 0.1,
            'b1': np.zeros(128),
            'W2': np.random.randn(128, 64) * 0.1,
            'b2': np.zeros(64),
            'W3': np.random.randn(64, 1) * 0.1,
            'b3': np.zeros(1)
        }
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax aktiváció."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """Akció valószínűségek az Actor-ból."""
        # Actor forward pass
        z1 = np.dot(state, self.actor_network['W1']) + self.actor_network['b1']
        a1 = np.maximum(0, z1)  # ReLU
        
        z2 = np.dot(a1, self.actor_network['W2']) + self.actor_network['b2']
        a2 = np.maximum(0, z2)  # ReLU
        
        logits = np.dot(a2, self.actor_network['W3']) + self.actor_network['b3']
        
        return self._softmax(logits)
    
    def get_state_value(self, state: np.ndarray) -> float:
        """Állapot értéke a Critic-ból."""
        # Critic forward pass
        z1 = np.dot(state, self.critic_network['W1']) + self.critic_network['b1']
        a1 = np.maximum(0, z1)  # ReLU
        
        z2 = np.dot(a1, self.critic_network['W2']) + self.critic_network['b2']
        a2 = np.maximum(0, z2)  # ReLU
        
        value = np.dot(a2, self.critic_network['W3']) + self.critic_network['b3']
        
        return value[0]
    
    def get_action(self, state: np.ndarray, valid_actions: List[int] = None) -> int:
        """Akció kiválasztása policy alapján."""
        if valid_actions is None:
            valid_actions = list(range(self.action_size))
        
        action_probs = self.get_action_probabilities(state)
        
        # Csak érvényes akciók valószínűségeit vesszük figyelembe
        valid_probs = np.array([action_probs[action] for action in valid_actions])
        valid_probs = valid_probs / np.sum(valid_probs)  # Renormalizálás
        
        # Mintavételezés a valószínűség eloszlásból
        chosen_idx = np.random.choice(len(valid_actions), p=valid_probs)
        
        return valid_actions[chosen_idx]
    
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Actor-Critic frissítés."""
        
        # TD error számítása
        current_value = self.get_state_value(state)
        
        if done:
            td_target = reward
        else:
            next_value = self.get_state_value(next_state)
            td_target = reward + self.gamma * next_value
        
        td_error = td_target - current_value
        
        # Critic frissítése (egyszerűsített)
        self.critic_network['b3'] += self.learning_rate * td_error
        
        # Actor frissítése (policy gradient)
        action_probs = self.get_action_probabilities(state)
        
        # Log-likelihood gradient
        log_prob_gradient = np.zeros(self.action_size)
        log_prob_gradient[action] = 1.0 / max(action_probs[action], 1e-8)
        
        # Actor súlyok frissítése
        self.actor_network['b3'] += self.learning_rate * td_error * log_prob_gradient


class DeepRLPredictor:
    """Deep Reinforcement Learning Predictor fő osztály."""
    
    def __init__(self, min_num: int, max_num: int, target_count: int):
        self.min_num = min_num
        self.max_num = max_num
        self.target_count = target_count
        self.vocab_size = max_num - min_num + 1
        
        # Környezet
        self.env = LotteryEnvironment(min_num, max_num, target_count)
        
        # Ágensek
        self.dqn_agent = DQNAgent(self.env.state_size, self.env.action_size)
        self.ac_agent = ActorCriticAgent(self.env.state_size, self.env.action_size)
        
        # Ensemble súlyok
        self.agent_weights = {'dqn': 0.6, 'ac': 0.4}
        
        # Tanulási történet
        self.training_episodes = 0
        self.performance_history = deque(maxlen=100)
    
    def train(self, past_draws: List[List[int]]):
        """Reinforcement Learning tanítás."""
        
        if len(past_draws) < 10:
            logger.warning("Nincs elég adat az RL tanításhoz")
            return
        
        # Training episodes
        max_episodes = min(100, len(past_draws) - 5)
        
        for episode in range(max_episodes):
            # Környezet inicializálása
            state = self.env.reset(past_draws[:-(max_episodes-episode)])
            
            # Aktuális cél húzás
            if episode < len(past_draws) - 5:
                target_draw = past_draws[-(max_episodes-episode-1)]
            else:
                continue
            
            # Több lépés egy epizódban (több szám kiválasztása)
            episode_rewards = []
            
            for step in range(self.target_count):
                # DQN ágens lépése
                dqn_action = self.dqn_agent.get_action(state)
                next_state, reward, done = self.env.step(dqn_action, target_draw)
                
                # DQN tanulás
                self.dqn_agent.remember(state, dqn_action, reward, next_state, done)
                self.dqn_agent.replay()
                
                # Actor-Critic ágens lépése
                ac_action = self.ac_agent.get_action(state)
                ac_next_state, ac_reward, ac_done = self.env.step(ac_action, target_draw)
                
                # AC tanulás
                self.ac_agent.update(state, ac_action, ac_reward, ac_next_state, ac_done)
                
                episode_rewards.append((reward + ac_reward) / 2)
                state = next_state
            
            # Epizód teljesítmény
            avg_reward = np.mean(episode_rewards)
            self.performance_history.append(avg_reward)
            
            self.training_episodes += 1
            
            # Target network frissítése
            if episode % 10 == 0:
                self.dqn_agent._update_target_network()
        
        # Ágensek súlyainak dinamikus frissítése
        self._update_agent_weights()
        
        logger.info(f"RL tanítás befejezve: {self.training_episodes} epizód")
    
    def _update_agent_weights(self):
        """Ágensek súlyainak frissítése teljesítmény alapján."""
        
        if len(self.performance_history) < 10:
            return
        
        recent_performance = np.mean(list(self.performance_history)[-10:])
        
        # DQN teljesítmény alapú súlyozás
        if recent_performance > 0:
            self.agent_weights['dqn'] = min(0.8, self.agent_weights['dqn'] * 1.1)
            self.agent_weights['ac'] = 1.0 - self.agent_weights['dqn']
        else:
            self.agent_weights['ac'] = min(0.8, self.agent_weights['ac'] * 1.1)
            self.agent_weights['dqn'] = 1.0 - self.agent_weights['ac']
    
    def predict(self, past_draws: List[List[int]]) -> List[int]:
        """RL alapú predikció."""
        
        if not past_draws:
            return self._fallback_prediction()
        
        # Aktuális állapot
        state = self.env.reset(past_draws)
        
        # Ensemble predikció
        predictions = set()
        
        # DQN predikciók
        dqn_predictions = []
        for _ in range(self.target_count):
            valid_actions = [i for i in range(self.vocab_size) 
                           if (i + self.min_num) not in predictions]
            if valid_actions:
                action = self.dqn_agent.get_action(state, valid_actions)
                number = action + self.min_num
                dqn_predictions.append(number)
        
        # Actor-Critic predikciók
        ac_predictions = []
        for _ in range(self.target_count):
            valid_actions = [i for i in range(self.vocab_size) 
                           if (i + self.min_num) not in predictions]
            if valid_actions:
                action = self.ac_agent.get_action(state, valid_actions)
                number = action + self.min_num
                ac_predictions.append(number)
        
        # Súlyozott kombináció
        final_predictions = []
        
        # DQN súlyozva
        for pred in dqn_predictions:
            if len(final_predictions) < self.target_count:
                if random.random() < self.agent_weights['dqn']:
                    final_predictions.append(pred)
        
        # AC kiegészítés
        for pred in ac_predictions:
            if len(final_predictions) < self.target_count and pred not in final_predictions:
                if random.random() < self.agent_weights['ac']:
                    final_predictions.append(pred)
        
        # Hiányzó számok kiegészítése
        while len(final_predictions) < self.target_count:
            remaining = [num for num in range(self.min_num, self.max_num + 1) 
                        if num not in final_predictions]
            if remaining:
                # RL-alapú heurisztika
                selected = self._rl_heuristic_selection(remaining, past_draws)
                final_predictions.append(selected)
            else:
                break
        
        return sorted(final_predictions[:self.target_count])
    
    def _rl_heuristic_selection(self, candidates: List[int], past_draws: List[List[int]]) -> int:
        """RL-alapú heurisztikus kiválasztás."""
        
        if not candidates:
            return random.randint(self.min_num, self.max_num)
        
        # Reward-alapú pontozás minden kandidátra
        scores = []
        
        for candidate in candidates:
            score = 0.0
            
            # Frekvencia alapú reward szimuláció
            recent_numbers = [num for draw in past_draws[-10:] for num in draw]
            frequency = recent_numbers.count(candidate)
            score += frequency * 0.5
            
            # Diverzitás reward
            if past_draws:
                last_draw = past_draws[-1]
                min_distance = min(abs(candidate - num) for num in last_draw)
                if min_distance >= 3:
                    score += 2.0
            
            # Pozicionális reward
            position_score = 1.0 - abs(candidate - (self.min_num + self.max_num) / 2) / (self.max_num - self.min_num)
            score += position_score
            
            scores.append(score)
        
        # Softmax kiválasztás
        if max(scores) > min(scores):
            exp_scores = np.exp(np.array(scores) - max(scores))
            probabilities = exp_scores / np.sum(exp_scores)
            selected_idx = np.random.choice(len(candidates), p=probabilities)
            return candidates[selected_idx]
        else:
            return random.choice(candidates)
    
    def _fallback_prediction(self) -> List[int]:
        """Fallback predikció RL elvek alapján."""
        
        # Q-learning inspirált véletlen stratégia
        predictions = []
        
        for _ in range(self.target_count):
            # Exploration vs exploitation
            if random.random() < 0.3:  # Exploration
                num = random.randint(self.min_num, self.max_num)
            else:  # Exploitation (középérték körül)
                center = (self.min_num + self.max_num) / 2
                std = (self.max_num - self.min_num) / 8
                num = int(np.random.normal(center, std))
                num = max(self.min_num, min(num, self.max_num))
            
            if num not in predictions:
                predictions.append(num)
        
        # Kiegészítés ha szükséges
        while len(predictions) < self.target_count:
            remaining = [n for n in range(self.min_num, self.max_num + 1) if n not in predictions]
            if remaining:
                predictions.append(random.choice(remaining))
            else:
                break
        
        return sorted(predictions)


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Deep Reinforcement Learning alapú lottószám predikció.
    Returns a tuple (main_numbers, additional_numbers).
    """
    try:
        # Fő számok generálása
        main_numbers = generate_deep_rl_numbers(
            lottery_type_instance,
            min_number=int(lottery_type_instance.min_number),
            max_number=int(lottery_type_instance.max_number),
            pieces_of_draw_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
            is_main=True
        )

        # Kiegészítő számok generálása
        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            additional_numbers = generate_deep_rl_numbers(
                lottery_type_instance,
                min_number=int(lottery_type_instance.additional_min_number),
                max_number=int(lottery_type_instance.additional_max_number),
                pieces_of_draw_numbers=int(lottery_type_instance.additional_numbers_count),
                is_main=False
            )

        return sorted(main_numbers), sorted(additional_numbers)
    
    except Exception as e:
        logger.error(f"Hiba a deep RL predikció során: {e}")
        return generate_fallback_numbers(lottery_type_instance)


def generate_deep_rl_numbers(
    lottery_type_instance: lg_lottery_type,
    min_number: int,
    max_number: int,
    pieces_of_draw_numbers: int,
    is_main: bool
) -> List[int]:
    """Deep RL számgenerálás."""
    
    # Történeti adatok lekérése
    past_draws = get_historical_data(lottery_type_instance, is_main)
    
    if len(past_draws) < 10:
        return generate_rl_fallback(min_number, max_number, pieces_of_draw_numbers)
    
    # Deep RL predictor
    predictor = DeepRLPredictor(
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
        ).order_by('-id')[:150]
        
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


def generate_rl_fallback(min_number: int, max_number: int, count: int) -> List[int]:
    """RL-alapú fallback számgenerálás."""
    
    # Q-learning inspirált epsilon-greedy stratégia
    epsilon = 0.3
    predictions = []
    
    for _ in range(count):
        if random.random() < epsilon:
            # Exploration: véletlen szám
            num = random.randint(min_number, max_number)
        else:
            # Exploitation: optimális heurisztika
            center = (min_number + max_number) / 2
            # Reward-maximalizáló választás a központ körül
            std = (max_number - min_number) / 6
            num = int(np.random.normal(center, std))
            num = max(min_number, min(num, max_number))
        
        if num not in predictions:
            predictions.append(num)
    
    # Kiegészítés ha szükséges
    while len(predictions) < count:
        remaining = [n for n in range(min_number, max_number + 1) if n not in predictions]
        if remaining:
            predictions.append(random.choice(remaining))
        else:
            break
    
    return sorted(predictions)


def generate_fallback_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Fallback számgenerálás."""
    main_numbers = generate_rl_fallback(
        int(lottery_type_instance.min_number),
        int(lottery_type_instance.max_number),
        int(lottery_type_instance.pieces_of_draw_numbers)
    )
    
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_rl_fallback(
            int(lottery_type_instance.additional_min_number),
            int(lottery_type_instance.additional_max_number),
            int(lottery_type_instance.additional_numbers_count)
        )
    
    return sorted(main_numbers), sorted(additional_numbers) 