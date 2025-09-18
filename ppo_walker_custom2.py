import torch
import numpy as np
import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
import os
import datetime
import utils

class CustomRewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.time_step = 0

    def reset(self, **kwargs):
        self.time_step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.time_step += self.env.unwrapped.dt # 시뮬레이션 시간 업데이트

        new_reward = info["x_position"] * -1 # 뒤로걷기(테스트용)
        
        return obs, new_reward, terminated, truncated, info


# --- 커스텀 평가 콜백 클래스 정의 ---
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


def test_model(env, model_path, seed, video_folder):
    print(f"--- '{model_path}' 모델 테스트 시작 (시드: {seed}) ---")
    
    # 비디오 녹화 래퍼 적용
    os.makedirs(video_folder, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    video_prefix = f"{model_name}"
    env = RecordVideo(env, video_folder=video_folder, name_prefix=video_prefix, fps=30)

    # 훈련된 모델 불러오기
    set_random_seed(seed)
    model = PPO.load(model_path, env=env)
    
    # 평가 시작
    obs, info = env.reset(seed=seed)
    
    # 평가 지표 초기화
    torso_angles = []
    total_reward = 0
    final_distance = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 평가 지표 수집
        torso_angle = obs[1]
        torso_angles.append(torso_angle)
        total_reward += reward

        if done:
            final_distance = info.get('x_position', 0)
    # 최종 결과 계산 및 출력
    stability_score = np.std(torso_angles)

    print("\n--- 최종 평가 결과 ---")
    print(f"모델: {model_path}")
    print(f"최종 이동 거리: {final_distance:.2f} m")
    print(f"총 보상: {total_reward:.2f}")
    print(f"몸통 흔들림 (안정성): {stability_score:.4f} (낮을수록 안정적)")
    print(f"영상 저장 위치: {video_folder}{video_prefix}.mp4")
    print("-" * 30 + "\n")

    env.close()

# --- 메인 훈련 코드 ---
if __name__ == "__main__":
    FOLDER_NAME = "custom_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    SAVE_PATH = FOLDER_NAME + f"/results/"
    LOG_PATH = FOLDER_NAME + "/logs/"
    VIDEO_PATH = FOLDER_NAME + "/videos/"
    TOTAL_TIMESTEPS = 1000000
    
    # 시드 설정
    SEED = 42
    utils.set_seed(SEED)
    
    custom_xml_path = "C:/Users/Konyang/Go-Walker2d/walker2d_slope.xml" # 상대경로 왜 적용안되는지??
    # 훈련용 환경
    train_env = gym.make("Walker2d-v5", xml_file=custom_xml_path)
    train_env = Monitor(train_env, SAVE_PATH)
    train_env = CustomRewardWrapper(env=train_env)
    train_env.action_space.seed(SEED)
    train_env.reset(seed=SEED) # 환경 초기화 시 시드 설정
    # 평가용 환경
    eval_env = gym.make("Walker2d-v5", xml_file=custom_xml_path)
    eval_env = CustomRewardWrapper(env=eval_env)
    eval_env.action_space.seed(SEED)
    eval_env.reset(seed=SEED) # 환경 초기화 시 시드 설정
    # 환경 생성

    # 사용 가능한 장치 확인 (cpu 우선)
    device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 콜백 설정
    callback = AdvancedEvalCallback(
        eval_env, 
        save_path=SAVE_PATH, 
        eval_freq=20000, 
        n_eval_episodes=5, 
        verbose=1)

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
        tb_log_name="ppo_walker2d"
    )

    # 최종 모델 저장하기
    model.save(f"{SAVE_PATH}ppo_walker2d_final.zip")
    print("최종 모델 저장이 완료되었습니다.")

    train_env.close()
    eval_env.close()

    # 테스트
    test_model(
        env=eval_env,
        model_path=SAVE_PATH + "ppo_walker2d_best_distance",
        seed=SEED,
        video_folder=VIDEO_PATH
    )
    test_model(
        env=eval_env,
        model_path=SAVE_PATH + "ppo_walker2d_best_stability",
        seed=SEED,
        video_folder=VIDEO_PATH
    )
    test_model(
        env=eval_env,
        model_path=SAVE_PATH + "ppo_walker2d_best_reward",
        seed=SEED,
        video_folder=VIDEO_PATH
    )
    test_model(
        env=eval_env,
        model_path=SAVE_PATH + "ppo_walker2d_final",
        seed=SEED,
        video_folder=VIDEO_PATH
    )