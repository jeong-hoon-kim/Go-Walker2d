import gymnasium as gym
from gymnasium import Wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
import torch

# --- 1. 커스텀 보상 래퍼 클래스 ---

class PacedWalkingWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # --- 🏆 속도 제어 하이퍼파라미터 ---
        self.target_velocity = 1.75  # 목표 걷기 속도 (m/s)
        self.velocity_tolerance = 0.5 # 속도 허용 오차 (이 값이 작을수록 엄격해짐)
        self.velocity_reward_weight = 2 # 속도 보상의 최대 크기 (최대 보너스 점수)
        
        # --- 안정성 페널티 가중치 ---
        self.stability_weight = 0.3
        self.flight_penalty_weight = 1
        
        # MuJoCo 시뮬레이션에서 발과 바닥의 ID를 미리 찾아둡니다.
        self.left_foot_geom_id = self.env.unwrapped.model.geom('foot_left_geom').id
        self.right_foot_geom_id = self.env.unwrapped.model.geom('foot_geom').id
        self.floor_geom_id = self.env.unwrapped.model.geom('floor').id
        
    def _check_foot_contact(self):
        """두 발이 바닥에 닿아있는지 확인하는 함수"""
        left_contact = False
        right_contact = False
        
        for contact in self.env.unwrapped.data.contact:
            geom_pair = {contact.geom1, contact.geom2}
            
            if self.left_foot_geom_id in geom_pair and self.floor_geom_id in geom_pair:
                left_contact = True
            if self.right_foot_geom_id in geom_pair and self.floor_geom_id in geom_pair:
                right_contact = True
        
        return left_contact, right_contact

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        # 1. 기본 보상에서 '생존 보너스'와 '컨트롤 비용'만 가져옵니다.
        # 기존의 '전진 보상'은 더 이상 사용하지 않습니다.
        healthy_reward = info.get('reward_survive', 1.0)
        ctrl_cost = info.get('reward_ctrl', 0)

        # 2. 몸통 안정성 페널티 (유지)
        stability_penalty = self.stability_weight * (np.abs(obs[1]) + 0.1 * np.abs(obs[10]))
        
        # --- 🏆 3. '속도 상한선' 보너스 계산 ---
        
        # 현재 전진 속도를 obs 벡터에서 가져옵니다.
        current_velocity = obs[8]
        
        # 가우시안 함수를 이용해 보상 계산:
        # 현재 속도가 target_velocity에 가까울수록 보상이 velocity_reward_weight에 가까워지고,
        # 멀어질수록 0에 가까워집니다.
        velocity_bonus = self.velocity_reward_weight * \
                         np.exp(-np.square(current_velocity - self.target_velocity) / (2 * np.square(self.velocity_tolerance)))
                         
        # --- 🏆 '공중 체공' 페널티 계산 ---
        left_foot_on_ground, right_foot_on_ground = self._check_foot_contact()
        flight_penalty = 0
        if not left_foot_on_ground and not right_foot_on_ground:
            flight_penalty = self.flight_penalty_weight

        # 4. 모든 요소를 합산하여 최종 보상 계산
        new_reward = (
            velocity_bonus
            + healthy_reward 
            + ctrl_cost
            - stability_penalty
            - flight_penalty
        )
        
        return obs, new_reward, terminated, truncated, info

# --- 2. 커스텀 평가 콜백 클래스 정의 ---
class AdvancedEvalCallback(BaseCallback):
    def __init__(self, eval_env, save_path, eval_freq, n_eval_episodes, verbose):
        super(AdvancedEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        
        # 각 지표별 최고 기록을 저장할 변수
        self.best_mean_distance = -np.inf
        self.best_mean_stability = np.inf
        self.best_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            episode_distances, episode_stabilities = [], []
            for _ in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                done = False
                torso_angles = []
                final_distance = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated
                    
                    # Walker2d-v5의 obs[1]는 몸통 각도(torso angle)입니다. (v3, v4와 다름)
                    torso_angles.append(obs[1]) 
                    if done: 
                        final_distance = info.get('x_position', 0)
                
                episode_distances.append(final_distance)
                episode_stabilities.append(np.std(torso_angles))

            mean_distance = np.mean(episode_distances)
            mean_stability = np.mean(episode_stabilities)
            
            self.logger.record("eval/mean_distance", mean_distance)
            self.logger.record("eval/mean_stability", mean_stability)

            if self.verbose > 0:
                print(f"--- Timestep {self.num_timesteps}: Custom Eval ---")
                print(f"Avg Distance: {mean_distance:.2f} m, Avg Stability: {mean_stability:.4f}")
            # 최고 이동 거리 모델 저장
            if mean_distance > self.best_mean_distance:
                self.best_mean_distance = mean_distance
                self.model.save(os.path.join(self.save_path, "ppo_walker2d_best_distance.zip"))
                if self.verbose > 0: print(f"  >> New best distance model saved: {mean_distance:.2f} m")

            # 최고 안정성 모델 저장
            if mean_stability < self.best_mean_stability:
                self.best_mean_stability = mean_stability
                self.model.save(os.path.join(self.save_path, "ppo_walker2d_best_stability.zip"))
                if self.verbose > 0: print(f"  >> New best stability model saved: {mean_stability:.4f}")
            
            print("---------------------------------")
        
        return True

# --- 3. 메인 훈련 코드 ---
if __name__ == "__main__":
    # --- 설정 ---
    MODEL_NAME = "ppo_walker2d_custom_reward_v4"
    SAVE_PATH = f"results/{MODEL_NAME}/"
    LOG_PATH = "tensorboard_logs/"
    TOTAL_TIMESTEPS = 3000000
    SEED = 42

    set_random_seed(SEED)
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)

    # 훈련용 환경
    train_env = gym.make("Walker2d-v5")
    train_env = PacedWalkingWrapper(train_env)
    train_env = Monitor(train_env, SAVE_PATH)

    # 평가용 환경
    eval_env = gym.make("Walker2d-v5")

    # 사용 가능한 장치 확인 (GPU 우선)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 콜백 설정
    callback = AdvancedEvalCallback(
    eval_env=eval_env, 
    save_path=SAVE_PATH,
    eval_freq=20000,
    n_eval_episodes=5,
    verbose=1
)

    # 모델 생성하기
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1, 
        seed=SEED, 
        device=device,
        tensorboard_log=LOG_PATH 
    )

    # 모델 학습시키기
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        tb_log_name=MODEL_NAME
    )

    # 최종 모델 저장하기
    model.save(f"{SAVE_PATH}{MODEL_NAME}_final.zip")
    print("최종 모델 저장이 완료되었습니다.")

    train_env.close()
    eval_env.close()