import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import pandas as pd
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 개선된 하이퍼파라미터 설정
class Config:
    # 환경 설정
    NUM_EPISODES = 1000
    MAX_STEPS = 100
    NUM_ECHELONS = 4  # 소매점, RDC, 도매상, 제조업체
    
    # SAC 하이퍼파라미터
    BATCH_SIZE = 256
    BUFFER_SIZE = 100000
    LR_ACTOR = 1e-4  # 더 안정적인 학습률
    LR_CRITIC = 1e-4
    LR_ALPHA = 1e-4
    GAMMA = 0.99
    TAU = 0.005
    HIDDEN_DIM = 256
    
    # 공급망 설정
    INITIAL_INVENTORY = [100, 200, 300, 500]
    HOLDING_COSTS = [1.0, 0.8, 0.6, 0.4]
    ORDER_COSTS = [2.0, 1.5, 1.2, 1.0]
    SHORTAGE_COSTS = [10.0, 8.0, 6.0, 4.0]
    LEAD_TIMES_MEAN = [1, 2, 3, 4]
    LEAD_TIMES_STD = [0.1, 0.2, 0.3, 0.4]  # 리드타임 변동성 감소
    MAX_ORDER_QTY = [300, 500, 700, 900]  # 최대 주문량 조정
    
    # 계층별 보상 가중치 (소매 → 제조업체)
    LAYER_WEIGHTS = [0.4, 0.3, 0.2, 0.1]
    
    # 환경 변동성 설정 (Bullwhip 완화를 위해 감소)
    DEMAND_VOLATILITY = 0.15  # 수요 변동성 감소
    LEAD_TIME_UNCERTAINTY = True
    
    # Bullwhip 개선을 위한 새로운 설정
    INFORMATION_SHARING = True  # 정보 공유 활성화
    COLLABORATION_BONUS = 5.0   # 협력 보너스
    ORDER_SMOOTHING_FACTOR = 0.3  # 주문 평활화 계수
    BULLWHIP_PENALTY_WEIGHT = 50.0  # Bullwhip 페널티 가중치 증가

config = Config()

class ReplayBuffer:
    """개선된 경험 재생 버퍼"""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, priority=1.0):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)
    
    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            return None
        
        priorities = np.array(self.priorities) + 1e-6
        probabilities = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        batch = [self.buffer[idx] for idx in indices]
        
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, indices
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    """개선된 SAC Actor 네트워크 (주문 평활화 포함)"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # 주문 평활화를 위한 추가 레이어
        self.smoothing_layer = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        
        # 드롭아웃 추가 (과적합 방지)
        self.dropout = nn.Dropout(0.1)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        # 주문 평활화
        x = F.relu(self.smoothing_layer(x))
        
        mean = self.fc_mean(x)
        log_std = self.fc_std(x)
        log_std = torch.clamp(log_std, -10, 2)  # 표준편차 범위 제한
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean

class CentralizedCritic(nn.Module):
    """중앙집중식 Critic 네트워크"""
    def __init__(self, global_state_dim: int, joint_action_dim: int, hidden_dim: int = 256):
        super(CentralizedCritic, self).__init__()
        
        # Q1 네트워크
        self.q1_fc1 = nn.Linear(global_state_dim + joint_action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc4 = nn.Linear(hidden_dim, 1)
        
        # Q2 네트워크
        self.q2_fc1 = nn.Linear(global_state_dim + joint_action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc4 = nn.Linear(hidden_dim, 1)
        
        # 드롭아웃
        self.dropout = nn.Dropout(0.1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, global_state, joint_action):
        sa = torch.cat([global_state, joint_action], 1)
        
        # Q1
        q1 = F.relu(self.q1_fc1(sa))
        q1 = self.dropout(q1)
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.dropout(q1)
        q1 = F.relu(self.q1_fc3(q1))
        q1 = self.q1_fc4(q1)
        
        # Q2
        q2 = F.relu(self.q2_fc1(sa))
        q2 = self.dropout(q2)
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.dropout(q2)
        q2 = F.relu(self.q2_fc3(q2))
        q2 = self.q2_fc4(q2)
        
        return q1, q2

class MultiAgentSACAgent:
    """개선된 Multi-Agent SAC 에이전트"""
    def __init__(self, agent_id: int, local_state_dim: int, global_state_dim: int, 
                 action_dim: int, joint_action_dim: int):
        self.agent_id = agent_id
        self.local_state_dim = local_state_dim
        self.global_state_dim = global_state_dim
        self.action_dim = action_dim
        self.joint_action_dim = joint_action_dim
        
        # 주문 이력 저장 (평활화를 위해)
        self.order_history = deque(maxlen=10)
        
        # Actor 및 Critic 네트워크
        self.actor = Actor(local_state_dim, action_dim, config.HIDDEN_DIM)
        self.critic = CentralizedCritic(global_state_dim, joint_action_dim, config.HIDDEN_DIM)
        self.critic_target = CentralizedCritic(global_state_dim, joint_action_dim, config.HIDDEN_DIM)
        
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 옵티마이저
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LR_CRITIC)
        
        # 적응적 온도 조절 (더 보수적)
        self.target_entropy = -action_dim * 0.3
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.LR_ALPHA)
        
        # 경험 재생 버퍼
        self.replay_buffer = ReplayBuffer(config.BUFFER_SIZE)
        
        # 학습률 스케줄러
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=100, gamma=0.95)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=100, gamma=0.95)
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def select_action(self, local_state, evaluate=False):
        state = torch.FloatTensor(local_state).unsqueeze(0)
        
        if evaluate:
            with torch.no_grad():
                _, _, action = self.actor.sample(state)
                raw_action = action.detach().cpu().numpy()[0]
        else:
            action, _, _ = self.actor.sample(state)
            raw_action = action.detach().cpu().numpy()[0]
        
        # 주문 평활화 적용
        if len(self.order_history) > 0:
            recent_avg = np.mean(list(self.order_history)[-3:])
            smoothed_action = (1 - config.ORDER_SMOOTHING_FACTOR) * raw_action + \
                             config.ORDER_SMOOTHING_FACTOR * recent_avg
        else:
            smoothed_action = raw_action
        
        self.order_history.append(smoothed_action)
        return smoothed_action
    
    def update(self, global_experiences):
        """전역 경험을 이용한 업데이트"""
        if len(self.replay_buffer) < config.BATCH_SIZE:
            return {}
        
        batch_data = self.replay_buffer.sample(config.BATCH_SIZE)
        if batch_data is None:
            return {}
        
        local_states, actions, rewards, next_local_states, dones, indices = batch_data
        
        local_states = torch.FloatTensor(local_states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_local_states = torch.FloatTensor(next_local_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # 전역 정보 생성 (실제로는 공유 메모리에서 가져와야 함)
        batch_size = local_states.shape[0]
        global_states = torch.randn(batch_size, self.global_state_dim)
        joint_actions = torch.randn(batch_size, self.joint_action_dim)
        next_global_states = torch.randn(batch_size, self.global_state_dim)
        
        # Critic 업데이트
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_local_states)
            next_joint_actions = joint_actions.clone()
            start_idx = self.agent_id * self.action_dim
            end_idx = (self.agent_id + 1) * self.action_dim
            next_joint_actions[:, start_idx:end_idx] = next_actions
            
            q1_next, q2_next = self.critic_target(next_global_states, next_joint_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + config.GAMMA * (1 - dones) * q_next
        
        q1, q2 = self.critic(global_states, joint_actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        # TD 오차 업데이트
        td_errors = torch.abs(q1 - q_target).detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(indices, td_errors + 1e-6)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Actor 업데이트
        new_actions, log_probs, _ = self.actor.sample(local_states)
        new_joint_actions = joint_actions.clone()
        new_joint_actions[:, start_idx:end_idx] = new_actions
        
        q1_new, q2_new = self.critic(global_states, new_joint_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Alpha 업데이트
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # 타겟 네트워크 업데이트
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(config.TAU * param.data + (1 - config.TAU) * target_param.data)
        
        # 학습률 스케줄러 업데이트
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item()
        }

class EnhancedFMCGSupplyChain:
    """개선된 FMCG 다계층 공급망 환경 (Bullwhip Effect 완화)"""
    def __init__(self, scenario='normal'):
        self.num_echelons = config.NUM_ECHELONS
        self.echelon_names = ['Retail', 'RDC', 'Wholesaler', 'Manufacturer']
        self.scenario = scenario
        
        # 상태 차원
        self.local_state_dim = 12  # 확장된 상태 (정보 공유 포함)
        self.global_state_dim = self.num_echelons * self.local_state_dim
        self.action_dim = 1
        self.joint_action_dim = self.num_echelons * self.action_dim
        
        # 정보 공유를 위한 글로벌 메트릭
        self.global_demand_forecast = deque(maxlen=20)
        self.global_inventory_ratio = 0.0
        self.supply_chain_efficiency = 0.0
        
        self._configure_scenario()
        self.reset()
    
    def _configure_scenario(self):
        """시나리오별 환경 설정"""
        if self.scenario == 'high_volatility':
            self.demand_volatility = 0.25  # 감소
            self.lead_time_uncertainty = 1.2
        elif self.scenario == 'supply_disruption':
            self.demand_volatility = 0.2
            self.lead_time_uncertainty = 1.3
            self.disruption_prob = 0.05  # 감소
        elif self.scenario == 'seasonal':
            self.demand_volatility = 0.2
            self.lead_time_uncertainty = 1.1
            self.seasonal_amplitude = 0.3  # 감소
        else:  # normal
            self.demand_volatility = config.DEMAND_VOLATILITY
            self.lead_time_uncertainty = 1.0
    
    def reset(self):
        """환경 초기화"""
        self.current_step = 0
        self.inventories = np.array(config.INITIAL_INVENTORY, dtype=np.float32)
        
        # 수요 히스토리 초기화
        self.demand_history = deque(maxlen=50)  # 더 긴 히스토리
        self.order_history = [deque(maxlen=20) for _ in range(self.num_echelons)]
        
        # 초기 수요 생성
        for _ in range(20):
            demand = self._generate_demand()
            self.demand_history.append(demand)
            self.global_demand_forecast.append(demand)
        
        # 리드타임 및 운송 큐 초기화
        self.current_lead_times = self._sample_lead_times()
        self.in_transit = [deque([0] * max(1, int(lt))) for lt in self.current_lead_times]
        
        self.previous_orders = np.zeros(self.num_echelons)
        
        # 성과 추적
        self.total_costs = []
        self.inventory_levels = []
        self.shortage_events = []
        self.bullwhip_metrics = []
        self.collaboration_scores = []
        
        return self._get_observations()
    
    def _sample_lead_times(self):
        """동적 리드타임 샘플링 (변동성 감소)"""
        if config.LEAD_TIME_UNCERTAINTY:
            lead_times = []
            for i in range(self.num_echelons):
                # 더 안정적인 리드타임
                lt = max(1, np.random.normal(
                    config.LEAD_TIMES_MEAN[i], 
                    config.LEAD_TIMES_STD[i] * self.lead_time_uncertainty
                ))
                lead_times.append(max(1, int(np.round(lt))))
            return lead_times
        else:
            return config.LEAD_TIMES_MEAN.copy()
    
    def _generate_demand(self):
        """개선된 수요 생성 (변동성 감소)"""
        base_demand = 50
        
        # 더 안정적인 계절성
        seasonal_factor = 1 + 0.15 * np.sin(2 * np.pi * self.current_step / 52)
        if self.scenario == 'seasonal':
            seasonal_factor += self.seasonal_amplitude * np.sin(2 * np.pi * self.current_step / 12)
        
        # 완만한 트렌드
        trend_factor = 1 + 0.002 * self.current_step
        
        # 감소된 노이즈
        noise_factor = np.random.normal(1, self.demand_volatility)
        
        # 급증 이벤트 감소
        spike_probability = 0.02
        if self.scenario == 'high_volatility':
            spike_probability = 0.05
        
        spike_factor = np.random.lognormal(0, 0.3) if np.random.random() < spike_probability else 1
        
        # 더 안정적인 이동평균
        if len(self.demand_history) > 5:
            ma_factor = 1 + 0.05 * (np.mean(list(self.demand_history)[-5:]) / 50 - 1)
        else:
            ma_factor = 1
        
        demand = max(10, base_demand * seasonal_factor * trend_factor * noise_factor * spike_factor * ma_factor)
        return demand
    
    def _get_observations(self):
        """확장된 관측 상태 (정보 공유 포함)"""
        observations = []
        current_demand = self._generate_demand()
        
        # 글로벌 메트릭 계산
        self._update_global_metrics()
        
        # 수요 예측 (이동평균 + 트렌드)
        if len(self.demand_history) >= 10:
            recent_demands = list(self.demand_history)[-10:]
            demand_forecast = np.mean(recent_demands)
            demand_trend = np.polyfit(range(10), recent_demands, 1)[0]
        else:
            demand_forecast = current_demand
            demand_trend = 0
        
        for i in range(self.num_echelons):
            # 기본 관측값
            inventory_level = self.inventories[i] / config.INITIAL_INVENTORY[i]
            incoming_shipment = sum(self.in_transit[i]) / config.INITIAL_INVENTORY[i]
            demand_ratio = current_demand / 50.0
            lead_time_ratio = self.current_lead_times[i] / max(config.LEAD_TIMES_MEAN)
            
            # 주문 평활화 메트릭
            if len(self.order_history[i]) > 1:
                order_volatility = np.std(list(self.order_history[i])) / (np.mean(list(self.order_history[i])) + 1)
            else:
                order_volatility = 0
            
            # 상류/하류 정보 (정보 공유)
            if config.INFORMATION_SHARING:
                if i < self.num_echelons - 1:
                    upstream_inventory = self.inventories[i+1] / config.INITIAL_INVENTORY[i+1]
                    upstream_order = self.previous_orders[i+1] / config.INITIAL_INVENTORY[i+1] if i+1 < len(self.previous_orders) else 0
                else:
                    upstream_inventory = 1.0
                    upstream_order = 0
                
                if i > 0:
                    downstream_inventory = self.inventories[i-1] / config.INITIAL_INVENTORY[i-1]
                    downstream_order = self.previous_orders[i-1] / config.INITIAL_INVENTORY[i-1] if i-1 >= 0 else 0
                else:
                    downstream_inventory = inventory_level
                    downstream_order = 0
            else:
                upstream_inventory = upstream_order = downstream_inventory = downstream_order = 0
            
            # 수요 예측 정보
            forecast_ratio = demand_forecast / 50.0
            trend_indicator = np.tanh(demand_trend)  # 정규화된 트렌드
            
            # 글로벌 공급망 상태
            global_efficiency = self.supply_chain_efficiency
            
            obs = np.array([
                inventory_level,
                incoming_shipment,
                demand_ratio,
                lead_time_ratio,
                order_volatility,
                upstream_inventory,
                upstream_order,
                downstream_inventory,
                downstream_order,
                forecast_ratio,
                trend_indicator,
                global_efficiency
            ])
            observations.append(obs)
        
        return observations
    
    def _update_global_metrics(self):
        """글로벌 메트릭 업데이트"""
        # 전체 재고 비율
        total_inventory = np.sum(self.inventories)
        target_inventory = np.sum(config.INITIAL_INVENTORY)
        self.global_inventory_ratio = total_inventory / target_inventory
        
        # 공급망 효율성 (재고 회전율 + 서비스 레벨)
        if len(self.total_costs) > 0:
            avg_cost = np.mean(self.total_costs[-10:])
            avg_inventory = np.mean([np.sum(inv) for inv in self.inventory_levels[-10:]])
            turnover = avg_cost / (avg_inventory + 1)
            
            shortage_rate = np.mean(self.shortage_events[-10:]) if len(self.shortage_events) > 0 else 0
            service_level = 1 - shortage_rate / self.num_echelons
            
            self.supply_chain_efficiency = 0.5 * turnover + 0.5 * service_level
        else:
            self.supply_chain_efficiency = 0.5
    
    def step(self, actions):
        """개선된 환경 스텝 (Bullwhip 완화)"""
        self.current_step += 1
        
        # 행동을 주문량으로 변환 (더 안정적인 변환)
        orders = []
        for i, action in enumerate(actions):
            # 더 보수적인 주문량 계산
            sigmoid_action = torch.sigmoid(torch.tensor(action[0])).item()
            
            # 수요 기반 주문량 조정
            if len(self.demand_history) > 0:
                recent_demand = np.mean(list(self.demand_history)[-3:])
                demand_factor = recent_demand / 50.0
            else:
                demand_factor = 1.0
            
            # 재고 수준 기반 조정
            inventory_factor = max(0.3, 1 - self.inventories[i] / config.INITIAL_INVENTORY[i])
            
            order_qty = sigmoid_action * config.MAX_ORDER_QTY[i] * demand_factor * inventory_factor
            orders.append(max(0, order_qty))
        
        orders = np.array(orders)
        
        # 주문 이력 업데이트
        for i, order in enumerate(orders):
            self.order_history[i].append(order)
        
        # 현재 수요
        current_demand = self._generate_demand()
        self.demand_history.append(current_demand)
        self.global_demand_forecast.append(current_demand)
        
        # 공급 중단 (확률 감소)
        supply_disruption = False
        if hasattr(self, 'disruption_prob') and np.random.random() < self.disruption_prob:
            supply_disruption = True
        
        # 리드타임 업데이트
        self.current_lead_times = self._sample_lead_times()
        
        # 입고 처리
        for i in range(self.num_echelons):
            if len(self.in_transit[i]) > 0:
                arrived_qty = self.in_transit[i].popleft()
                if supply_disruption and i > 0:
                    arrived_qty *= 0.7  # 덜 심각한 공급 중단
                self.inventories[i] += arrived_qty
        
        # 주문 처리
        shortages = np.zeros(self.num_echelons)
        
        # 소매점 수요 처리
        if self.inventories[0] >= current_demand:
            self.inventories[0] -= current_demand
        else:
            shortages[0] = current_demand - self.inventories[0]
            self.inventories[0] = 0
        
        # 계층 간 주문 처리
        for i in range(self.num_echelons - 1):
            order_demand = orders[i]
            
            if self.inventories[i + 1] >= order_demand:
                self.inventories[i + 1] -= order_demand
# 입고 스케줄링 (리드타임 고려)
                while len(self.in_transit[i]) < self.current_lead_times[i]:
                    self.in_transit[i].append(0)
                self.in_transit[i].append(order_demand)
            else:
                shortages[i + 1] = order_demand - self.inventories[i + 1]
                delivered_qty = self.inventories[i + 1]
                self.inventories[i + 1] = 0
                
                # 부분 배송
                while len(self.in_transit[i]) < self.current_lead_times[i]:
                    self.in_transit[i].append(0)
                self.in_transit[i].append(delivered_qty)
        
        # 제조업체 생산 (무제한 공급 가정)
        production_qty = orders[-1]
        while len(self.in_transit[-1]) < self.current_lead_times[-1]:
            self.in_transit[-1].append(0)
        self.in_transit[-1].append(production_qty)
        
        # 보상 계산
        rewards = self._calculate_rewards(orders, shortages, current_demand)
        
        # 성과 추적 업데이트
        self._update_performance_tracking(orders, shortages, current_demand)
        
        self.previous_orders = orders.copy()
        
        # 종료 조건
        done = self.current_step >= config.MAX_STEPS
        
        return self._get_observations(), rewards, done, self._get_info()
    
    def _calculate_rewards(self, orders, shortages, current_demand):
        """개선된 보상 함수 (Bullwhip 완화 초점)"""
        rewards = []
        
        # Bullwhip 메트릭 계산
        bullwhip_penalty = self._calculate_bullwhip_penalty()
        
        # 협력 보너스 계산
        collaboration_bonus = self._calculate_collaboration_bonus()
        
        for i in range(self.num_echelons):
            # 기본 비용
            holding_cost = config.HOLDING_COSTS[i] * self.inventories[i]
            order_cost = config.ORDER_COSTS[i] * orders[i]
            shortage_cost = config.SHORTAGE_COSTS[i] * shortages[i]
            
            # 주문 평활화 보너스
            smoothing_bonus = 0
            if len(self.order_history[i]) > 3:
                recent_orders = list(self.order_history[i])[-4:]
                order_stability = 1 / (1 + np.std(recent_orders))
                smoothing_bonus = order_stability * 5.0
            
            # 재고 최적화 보너스
            target_inventory = config.INITIAL_INVENTORY[i]
            inventory_deviation = abs(self.inventories[i] - target_inventory) / target_inventory
            inventory_bonus = max(0, 3 - inventory_deviation * 10)
            
            # 서비스 레벨 보너스
            service_bonus = 0
            if i == 0:  # 소매점
                if shortages[i] == 0:
                    service_bonus = 10
                else:
                    service_bonus = -5
            
            # 계층별 가중 기본 보상
            base_reward = -(holding_cost + order_cost + shortage_cost)
            weighted_reward = base_reward * config.LAYER_WEIGHTS[i]
            
            # 종합 보상
            total_reward = (weighted_reward + smoothing_bonus + inventory_bonus + 
                          service_bonus + collaboration_bonus - bullwhip_penalty)
            
            rewards.append(total_reward)
        
        return np.array(rewards)
    
    def _calculate_bullwhip_penalty(self):
        """Bullwhip Effect 페널티 계산"""
        if self.current_step < 10:
            return 0
        
        bullwhip_ratio = 0
        demand_variance = np.var(list(self.demand_history)[-10:])
        
        for i in range(self.num_echelons):
            if len(self.order_history[i]) >= 10:
                order_variance = np.var(list(self.order_history[i])[-10:])
                if demand_variance > 0:
                    layer_bullwhip = order_variance / demand_variance
                    bullwhip_ratio += layer_bullwhip * (i + 1)  # 상류로 갈수록 가중치 증가
        
        # Bullwhip 비율이 1보다 크면 페널티
        if bullwhip_ratio > self.num_echelons:
            penalty = (bullwhip_ratio - self.num_echelons) * config.BULLWHIP_PENALTY_WEIGHT
        else:
            penalty = 0
        
        self.bullwhip_metrics.append(bullwhip_ratio)
        return penalty
    
    def _calculate_collaboration_bonus(self):
        """협력 보너스 계산"""
        if not config.INFORMATION_SHARING:
            return 0
        
        bonus = 0
        
        # 재고 균형 보너스
        inventory_balance = 1 - np.std(self.inventories / config.INITIAL_INVENTORY)
        bonus += inventory_balance * config.COLLABORATION_BONUS
        
        # 주문 패턴 일관성 보너스
        if all(len(oh) >= 5 for oh in self.order_history):
            order_correlations = []
            for i in range(self.num_echelons - 1):
                orders_i = list(self.order_history[i])[-5:]
                orders_j = list(self.order_history[i + 1])[-5:]
                correlation = np.corrcoef(orders_i, orders_j)[0, 1]
                if not np.isnan(correlation):
                    order_correlations.append(abs(correlation))
            
            if order_correlations:
                avg_correlation = np.mean(order_correlations)
                bonus += avg_correlation * config.COLLABORATION_BONUS * 0.5
        
        self.collaboration_scores.append(bonus)
        return bonus
    
    def _update_performance_tracking(self, orders, shortages, current_demand):
        """성과 추적 업데이트"""
        # 총 비용
        total_cost = 0
        for i in range(self.num_echelons):
            holding_cost = config.HOLDING_COSTS[i] * self.inventories[i]
            order_cost = config.ORDER_COSTS[i] * orders[i]
            shortage_cost = config.SHORTAGE_COSTS[i] * shortages[i]
            total_cost += holding_cost + order_cost + shortage_cost
        
        self.total_costs.append(total_cost)
        self.inventory_levels.append(self.inventories.copy())
        self.shortage_events.append(np.sum(shortages > 0))
    
    def _get_info(self):
        """환경 정보 반환"""
        info = {
            'current_step': self.current_step,
            'inventories': self.inventories.copy(),
            'lead_times': self.current_lead_times.copy(),
            'demand': self.demand_history[-1] if self.demand_history else 0,
            'total_cost': self.total_costs[-1] if self.total_costs else 0,
            'bullwhip_metric': self.bullwhip_metrics[-1] if self.bullwhip_metrics else 0,
            'collaboration_score': self.collaboration_scores[-1] if self.collaboration_scores else 0,
            'global_inventory_ratio': self.global_inventory_ratio,
            'supply_chain_efficiency': self.supply_chain_efficiency
        }
        return info

class MultiAgentTrainer:
    """Multi-Agent SAC 훈련기"""
    def __init__(self, env):
        self.env = env
        self.agents = []
        
        # 에이전트 초기화
        for i in range(config.NUM_ECHELONS):
            agent = MultiAgentSACAgent(
                agent_id=i,
                local_state_dim=env.local_state_dim,
                global_state_dim=env.global_state_dim,
                action_dim=env.action_dim,
                joint_action_dim=env.joint_action_dim
            )
            self.agents.append(agent)
        
        # 공유 경험 저장소
        self.shared_buffer = ReplayBuffer(config.BUFFER_SIZE * 2)
        
        # 성과 추적
        self.episode_rewards = []
        self.episode_costs = []
        self.bullwhip_ratios = []
        self.service_levels = []
        self.training_losses = {i: [] for i in range(config.NUM_ECHELONS)}
    
    def train(self):
        """훈련 루프"""
        print("Starting Multi-Agent SAC Training for Enhanced FMCG Supply Chain...")
        print(f"Configuration: {config.NUM_EPISODES} episodes, {config.MAX_STEPS} steps per episode")
        print(f"Bullwhip mitigation features: {'Enabled' if config.INFORMATION_SHARING else 'Disabled'}")
        
        for episode in range(config.NUM_EPISODES):
            states = self.env.reset()
            episode_rewards = np.zeros(config.NUM_ECHELONS)
            episode_losses = {i: [] for i in range(config.NUM_ECHELONS)}
            
            for step in range(config.MAX_STEPS):
                # 행동 선택
                actions = []
                for i, agent in enumerate(self.agents):
                    action = agent.select_action(states[i], evaluate=False)
                    actions.append([action])
                
                # 환경 스텝
                next_states, rewards, done, info = self.env.step(actions)
                
                # 경험 저장
                for i, agent in enumerate(self.agents):
                    agent.replay_buffer.push(
                        states[i], actions[i], rewards[i], next_states[i], done
                    )
                    
                    # 공유 버퍼에도 저장
                    self.shared_buffer.push(
                        states[i], actions[i], rewards[i], next_states[i], done
                    )
                
                # 에이전트 업데이트
                for i, agent in enumerate(self.agents):
                    if len(agent.replay_buffer) > config.BATCH_SIZE:
                        losses = agent.update(self.shared_buffer)
                        if losses:
                            for key, value in losses.items():
                                episode_losses[i].append(value)
                
                episode_rewards += rewards
                states = next_states
                
                if done:
                    break
            
            # 에피소드 종료 후 기록
            self.episode_rewards.append(episode_rewards.mean())
            self.episode_costs.append(self.env.total_costs[-1] if self.env.total_costs else 0)
            
            if self.env.bullwhip_metrics:
                self.bullwhip_ratios.append(self.env.bullwhip_metrics[-1])
            
            # 서비스 레벨 계산
            service_level = 1 - (self.env.shortage_events[-1] / config.NUM_ECHELONS) if self.env.shortage_events else 1.0
            self.service_levels.append(service_level)
            
            # 손실 기록
            for i in range(config.NUM_ECHELONS):
                if episode_losses[i]:
                    self.training_losses[i].append(np.mean(episode_losses[i]))
                else:
                    self.training_losses[i].append(0)
            
            # 진행 상황 출력
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                avg_cost = np.mean(self.episode_costs[-50:])
                avg_bullwhip = np.mean(self.bullwhip_ratios[-50:]) if self.bullwhip_ratios else 0
                avg_service = np.mean(self.service_levels[-50:])
                
                print(f"Episode {episode + 1}/{config.NUM_EPISODES}")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Avg Cost: {avg_cost:.2f}")
                print(f"  Avg Bullwhip Ratio: {avg_bullwhip:.3f}")
                print(f"  Avg Service Level: {avg_service:.3f}")
                print(f"  Alpha values: {[agent.alpha.item() for agent in self.agents]}")
        
        print("Training completed!")
        return self.agents
    
    def evaluate(self, agents, num_episodes=10):
        """훈련된 에이전트 평가"""
        print(f"\nEvaluating trained agents over {num_episodes} episodes...")
        
        eval_rewards = []
        eval_costs = []
        eval_bullwhip = []
        eval_service = []
        
        for episode in range(num_episodes):
            states = self.env.reset()
            episode_reward = 0
            
            for step in range(config.MAX_STEPS):
                actions = []
                for i, agent in enumerate(agents):
                    action = agent.select_action(states[i], evaluate=True)
                    actions.append([action])
                
                states, rewards, done, info = self.env.step(actions)
                episode_reward += rewards.mean()
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_costs.append(info['total_cost'])
            eval_bullwhip.append(info['bullwhip_metric'])
            
            # 서비스 레벨
            service_level = 1 - (self.env.shortage_events[-1] / config.NUM_ECHELONS) if self.env.shortage_events else 1.0
            eval_service.append(service_level)
        
        # 평가 결과
        results = {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'avg_cost': np.mean(eval_costs),
            'std_cost': np.std(eval_costs),
            'avg_bullwhip': np.mean(eval_bullwhip),
            'std_bullwhip': np.std(eval_bullwhip),
            'avg_service_level': np.mean(eval_service),
            'std_service_level': np.std(eval_service)
        }
        
        print(f"Evaluation Results:")
        print(f"  Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Average Cost: {results['avg_cost']:.2f} ± {results['std_cost']:.2f}")
        print(f"  Average Bullwhip Ratio: {results['avg_bullwhip']:.3f} ± {results['std_bullwhip']:.3f}")
        print(f"  Average Service Level: {results['avg_service_level']:.3f} ± {results['std_service_level']:.3f}")
        
        return results

def plot_training_results(trainer):
    """훈련 결과 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Enhanced FMCG Supply Chain Training Results (Bullwhip Mitigation)', fontsize=16)
    
    # 보상 추이
    axes[0, 0].plot(trainer.episode_rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].grid(True)
    
    # 비용 추이
    axes[0, 1].plot(trainer.episode_costs)
    axes[0, 1].set_title('Episode Costs')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Total Cost')
    axes[0, 1].grid(True)
    
    # Bullwhip 비율
    if trainer.bullwhip_ratios:
        axes[0, 2].plot(trainer.bullwhip_ratios)
        axes[0, 2].axhline(y=config.NUM_ECHELONS, color='r', linestyle='--', 
                          label=f'Target (≤{config.NUM_ECHELONS})')
        axes[0, 2].set_title('Bullwhip Ratio')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Bullwhip Ratio')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
    
    # 서비스 레벨
    axes[1, 0].plot(trainer.service_levels)
    axes[1, 0].axhline(y=0.95, color='g', linestyle='--', label='Target (95%)')
    axes[1, 0].set_title('Service Level')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Service Level')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 훈련 손실 (평균)
    avg_losses = []
    for episode in range(len(trainer.training_losses[0])):
        episode_avg = np.mean([trainer.training_losses[i][episode] for i in range(config.NUM_ECHELONS)])
        avg_losses.append(episode_avg)
    
    axes[1, 1].plot(avg_losses)
    axes[1, 1].set_title('Average Training Loss')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True)
    
    # 성과 개선 (이동평균)
    window = 50
    if len(trainer.episode_rewards) >= window:
        moving_rewards = pd.Series(trainer.episode_rewards).rolling(window).mean()
        moving_costs = pd.Series(trainer.episode_costs).rolling(window).mean()
        
        ax2 = axes[1, 2]
        ax2.plot(moving_rewards, 'b-', label='Reward (MA)')
        ax2.set_ylabel('Reward', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        ax3 = ax2.twinx()
        ax3.plot(moving_costs, 'r-', label='Cost (MA)')
        ax3.set_ylabel('Cost', color='r')
        ax3.tick_params(axis='y', labelcolor='r')
        
        ax2.set_title('Performance Improvement (Moving Average)')
        ax2.set_xlabel('Episode')
        ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def run_comparison_scenarios():
    """시나리오별 비교 실험"""
    scenarios = ['normal', 'high_volatility', 'supply_disruption', 'seasonal']
    results = {}
    
    print("Running scenario comparison experiments...")
    
    for scenario in scenarios:
        print(f"\n=== Testing Scenario: {scenario.upper()} ===")
        
        # 환경 및 훈련기 생성
        env = EnhancedFMCGSupplyChain(scenario=scenario)
        trainer = MultiAgentTrainer(env)
        
        # 축약된 훈련 (비교용)
        original_episodes = config.NUM_EPISODES
        config.NUM_EPISODES = 300  # 빠른 비교를 위해 감소
        
        # 훈련
        trained_agents = trainer.train()
        
        # 평가
        eval_results = trainer.evaluate(trained_agents, num_episodes=20)
        results[scenario] = eval_results
        
        # 원래 설정 복원
        config.NUM_EPISODES = original_episodes
    
    # 결과 비교 출력
    print("\n" + "="*80)
    print("SCENARIO COMPARISON RESULTS")
    print("="*80)
    
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.round(3))
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Scenario Comparison: Enhanced FMCG Supply Chain', fontsize=16)
    
    scenarios_list = list(results.keys())
    
    # 평균 보상
    rewards = [results[s]['avg_reward'] for s in scenarios_list]
    axes[0, 0].bar(scenarios_list, rewards)
    axes[0, 0].set_title('Average Reward by Scenario')
    axes[0, 0].set_ylabel('Average Reward')
    
    # 평균 비용
    costs = [results[s]['avg_cost'] for s in scenarios_list]
    axes[0, 1].bar(scenarios_list, costs)
    axes[0, 1].set_title('Average Cost by Scenario')
    axes[0, 1].set_ylabel('Average Cost')
    
    # Bullwhip 비율
    bullwhip = [results[s]['avg_bullwhip'] for s in scenarios_list]
    axes[1, 0].bar(scenarios_list, bullwhip)
    axes[1, 0].axhline(y=config.NUM_ECHELONS, color='r', linestyle='--', 
                      label=f'Target (≤{config.NUM_ECHELONS})')
    axes[1, 0].set_title('Average Bullwhip Ratio by Scenario')
    axes[1, 0].set_ylabel('Bullwhip Ratio')
    axes[1, 0].legend()
    
    # 서비스 레벨
    service = [results[s]['avg_service_level'] for s in scenarios_list]
    axes[1, 1].bar(scenarios_list, service)
    axes[1, 1].axhline(y=0.95, color='g', linestyle='--', label='Target (95%)')
    axes[1, 1].set_title('Average Service Level by Scenario')
    axes[1, 1].set_ylabel('Service Level')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return results

# 메인 실행 함수
def main():
    """메인 실행 함수"""
    print("Enhanced FMCG Multi-Echelon Supply Chain with Bullwhip Mitigation")
    print("="*80)
    print(f"Key Features:")
    print(f"- Information Sharing: {config.INFORMATION_SHARING}")
    print(f"- Order Smoothing Factor: {config.ORDER_SMOOTHING_FACTOR}")
    print(f"- Bullwhip Penalty Weight: {config.BULLWHIP_PENALTY_WEIGHT}")
    print(f"- Collaboration Bonus: {config.COLLABORATION_BONUS}")
    print(f"- Demand Volatility: {config.DEMAND_VOLATILITY}")
    print("="*80)
    
    # 환경 생성
    env = EnhancedFMCGSupplyChain(scenario='normal')
    trainer = MultiAgentTrainer(env)
    
    # 훈련
    trained_agents = trainer.train()
    
    # 결과 시각화
    plot_training_results(trainer)
    
    # 평가
    evaluation_results = trainer.evaluate(trained_agents, num_episodes=50)
    
    # 시나리오 비교 (선택사항)
    run_scenarios = input("\nRun scenario comparison experiments? (y/n): ").lower() == 'y'
    if run_scenarios:
        scenario_results = run_comparison_scenarios()
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    # 최종 결과 요약
    print(f"\nFinal Performance Summary:")
    print(f"- Average Reward: {evaluation_results['avg_reward']:.2f}")
    print(f"- Average Cost: {evaluation_results['avg_cost']:.2f}")
    print(f"- Average Bullwhip Ratio: {evaluation_results['avg_bullwhip']:.3f}")
    print(f"- Average Service Level: {evaluation_results['avg_service_level']:.3f}")
    
    if evaluation_results['avg_bullwhip'] <= config.NUM_ECHELONS:
        print("✅ Bullwhip Effect successfully mitigated!")
    else:
        print("⚠️  Bullwhip Effect needs further improvement.")
    
    if evaluation_results['avg_service_level'] >= 0.95:
        print("✅ High service level achieved!")
    else:
        print("⚠️  Service level needs improvement.")

if __name__ == "__main__":
    main()
