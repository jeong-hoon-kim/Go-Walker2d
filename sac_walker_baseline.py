import gymnasium as gym
from gymnasium import Wrapper
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
import torch

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
            episode_distances, episode_stabilities, episode_reward = [], [], []
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
                episode_reward.append(reward)

            mean_distance = np.mean(episode_distances)
            mean_stability = np.mean(episode_stabilities)
            mean_reward = np.mean(episode_reward)
            
            self.logger.record("eval/mean_distance", mean_distance)
            self.logger.record("eval/mean_stability", mean_stability)
            self.logger.record("eval/mean_reward", mean_reward)

            if self.verbose > 0:
                print(f"--- Timestep {self.num_timesteps}: Custom Eval ---")
                print(f"Avg Distance: {mean_distance:.2f} m, Avg Stability: {mean_stability:.4f}, Avg Reward: {mean_reward:.2f}")

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
            
            # 최고 보상 모델 저장
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.model.save(os.path.join(self.save_path, "ppo_walker2d_best_reward.zip"))
                if self.verbose > 0: print(f"  >> New best reward model saved: {mean_reward:.2f}")
            
            print("---------------------------------")
        
        return True

# --- 3. 메인 훈련 코드 ---
if __name__ == "__main__":
    MODEL_NAME = "sac_walker2d_baseline"
    SAVE_PATH = f"results/{MODEL_NAME}/"
    LOG_PATH = "tensorboard_logs/"
    TOTAL_TIMESTEPS = 3000000
    SEED = 42

    set_random_seed(SEED)
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)

    # 훈련용 환경
    train_env = gym.make("Walker2d-v5")
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
    model = SAC(
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