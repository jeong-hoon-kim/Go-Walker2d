import gymnasium as gym
from gymnasium import Wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from gymnasium.wrappers import RecordVideo
import numpy as np
import os

class CustomRewardWrapper(Wrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

def test_model(model_path, seed, use_custom_wrapper, video_folder):
    print(f"--- '{model_path}' 모델 테스트 시작 (시드: {seed}) ---")

    # 1. 환경 생성
    env = gym.make("Walker2d-v5", render_mode="rgb_array")

    # 2. 커스텀 래퍼 적용 (필요 시)
    if use_custom_wrapper:
        print("커스텀 래퍼를 적용합니다.")
        env = CustomRewardWrapper(env)
    
    # 3. 비디오 녹화 래퍼 적용
    os.makedirs(video_folder, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    video_prefix = f"test_{model_name}_seed{seed}"
    env = RecordVideo(env, video_folder=video_folder, name_prefix=video_prefix, fps=30)

    # 4. 훈련된 모델 불러오기
    set_random_seed(seed)
    model = PPO.load(model_path, env=env)

    # 5. 평가 시작
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

    # 6. 최종 결과 계산 및 출력
    stability_score = np.std(torso_angles)

    print("\n--- 최종 평가 결과 ---")
    print(f"모델: {model_path}")
    print(f"최종 이동 거리: {final_distance:.2f} m")
    print(f"총 보상: {total_reward:.2f}")
    print(f"몸통 흔들림 (안정성): {stability_score:.4f} (낮을수록 안정적)")
    print(f"영상 저장 위치: {video_folder}{video_prefix}-video.mp4")
    print("-" * 30 + "\n")

    env.close()


if __name__ == "__main__":
    # --- 테스트 설정 ---
    MODEL_PATH = "results/ppo_walker2d_tensorboard/best_distance_model.zip" 
    SEED = 42
    VIDEO_FOLDER = "videos_test/"
    
    USE_CUSTOM_WRAPPER = False

    test_model(
        model_path=MODEL_PATH,
        seed=SEED,
        use_custom_wrapper=USE_CUSTOM_WRAPPER,
        video_folder=VIDEO_FOLDER
    )