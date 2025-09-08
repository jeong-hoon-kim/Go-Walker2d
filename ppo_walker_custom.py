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
import gymnasium as gym
from gymnasium import Wrapper
import numpy as np

class GaitRewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.time_step = 0
        
        # --- 걸음걸이 패턴을 위한 하이퍼파라미터 ---
        # 이 값들을 조정하며 최적의 걸음걸이를 찾아야 합니다.
        self.gait_frequency = 2.5  # 걸음걸이 속도 (Hz)
        self.hip_amplitude = 0.5   # 허벅지 관절의 움직임 폭 (radian)
        self.knee_amplitude = 0.5  # 무릎 관절의 움직임 폭 (radian)
        self.gait_reward_weight = 0.2 # 패턴 보상의 가중치

    def reset(self, **kwargs):
        self.time_step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.time_step += self.env.unwrapped.dt # 시뮬레이션 시간 업데이트

        # --- 🏆 걸음걸이 패턴 보상 (Gait Pattern Reward) ---
        
        # 1. 이상적인 목표 각도를 사인파로 계산
        # 현재 시간(t)을 기반으로 각 관절이 가져야 할 이상적인 각도를 계산합니다.
        phase = 2 * np.pi * self.gait_frequency * self.time_step
        
        # 오른쪽 다리의 목표 각도
        target_right_hip_angle = self.hip_amplitude * np.sin(phase)
        target_right_knee_angle = self.knee_amplitude * np.sin(phase + np.pi / 2) # 무릎은 위상이 약간 다름
        
        # 왼쪽 다리는 오른쪽과 180도(pi) 반대 위상
        target_left_hip_angle = self.hip_amplitude * np.sin(phase + np.pi)
        target_left_knee_angle = self.knee_amplitude * np.sin(phase + np.pi + np.pi / 2)

        # 2. 실제 관절 각도와 목표 각도의 차이 계산
        # obs 벡터에서 실제 관절 각도 값을 가져옵니다.
        actual_right_hip_angle = obs[2]
        actual_right_knee_angle = obs[3] # 종아리 관절이 무릎 역할
        actual_left_hip_angle = obs[5]
        actual_left_knee_angle = obs[6]

        # 3. 오차를 기반으로 페널티 계산 (오차가 작을수록 페널티가 적음)
        hip_error = (actual_right_hip_angle - target_right_hip_angle)**2 + \
                    (actual_left_hip_angle - target_left_hip_angle)**2
        knee_error = (actual_right_knee_angle - target_right_knee_angle)**2 + \
                     (actual_left_knee_angle - target_left_knee_angle)**2
                     
        # 오차가 클수록 큰 페널티를 부여 (보상 = -가중치 * 오차)
        gait_penalty = -self.gait_reward_weight * (hip_error + knee_error)

        # 4. 최종 보상에 합산
        new_reward = reward + gait_penalty
        
        return obs, new_reward, terminated, truncated, info

# --- 2. 커스텀 평가 콜백 클래스 정의 ---
class AdvancedEvalCallback(BaseCallback):
    def __init__(self, eval_env, save_path, eval_freq=20000, n_eval_episodes=5, verbose=1):
        super(AdvancedEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        
        self.best_mean_distance = -np.inf
        self.best_mean_stability = np.inf

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
                    torso_angles.append(obs[1])
                    if done: final_distance = info.get('x_position', 0)
                
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
                self.model.save(os.path.join(self.save_path, "best_distance_model.zip"))
                if self.verbose > 0: print(f"  >> New best distance model saved: {mean_distance:.2f} m")

            # 최고 안정성 모델 저장
            if mean_stability < self.best_mean_stability:
                self.best_mean_stability = mean_stability
                self.model.save(os.path.join(self.save_path, "best_stability_model.zip"))
                if self.verbose > 0: print(f"  >> New best stability model saved: {mean_stability:.4f}")
            print("---------------------------------")
        
        return True

# --- 3. 메인 훈련 코드 ---
if __name__ == "__main__":
    # --- 설정 ---
    MODEL_NAME = "ppo_walker2d_custom_reward_v1"
    SAVE_PATH = f"results/{MODEL_NAME}/"
    LOG_PATH = "tensorboard_logs/"
    TOTAL_TIMESTEPS = 3000000
    SEED = 42

    set_random_seed(SEED)
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)

    # 훈련용 환경
    train_env = gym.make("Walker2d-v5")
    train_env = GaitRewardWrapper(train_env)
    train_env = Monitor(train_env, SAVE_PATH)

    # 평가용 환경
    eval_env = gym.make("Walker2d-v5")

    # 사용 가능한 장치 확인 (GPU 우선)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 콜백 설정
    callback = AdvancedEvalCallback(eval_env, save_path=SAVE_PATH)

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