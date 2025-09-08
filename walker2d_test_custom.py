import gymnasium as gym
from gymnasium import Wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from gymnasium.wrappers import RecordVideo
import numpy as np
import os

# --- 중요: 훈련 시 사용했던 커스텀 래퍼를 여기에 똑같이 정의해야 합니다. ---
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

def test_model(model_path, seed, use_custom_wrapper, video_folder):

    print(f"--- '{model_path}' 모델 테스트 시작 (시드: {seed}) ---")

    # 1. 환경 생성
    env = gym.make("Walker2d-v5", render_mode="rgb_array")

    # 2. 비디오 녹화 래퍼 적용
    os.makedirs(video_folder, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    video_prefix = f"test_{model_name}_seed{seed}"
    env = RecordVideo(env, video_folder=video_folder, name_prefix=video_prefix, fps=30)
    
    # --- 중요 ---
    if use_custom_wrapper:
        print("경고: 테스트 환경에 커스텀 보상 래퍼를 적용합니다. '총 보상' 점수가 훈련 기준과 같게 됩니다.")
        env = GaitRewardWrapper(env)

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
    print(f"총 보상 (테스트 환경 기준): {total_reward:.2f}")
    print(f"몸통 흔들림 (안정성): {stability_score:.4f} (낮을수록 안정적)")
    print(f"영상 저장 위치: {os.path.join(video_folder, video_prefix)}-video-0.mp4")
    print("-" * 30 + "\n")

    env.close()


if __name__ == "__main__":
    # --- 테스트 설정 ---
    MODEL_PATH = "results/ppo_walker2d_custom_reward_v1/best_distance_model.zip" 
    SEED = 42
    VIDEO_FOLDER = "videos_test_custom/"
    
    USE_CUSTOM_WRAPPER = False

    test_model(
        model_path=MODEL_PATH,
        seed=SEED,
        use_custom_wrapper=USE_CUSTOM_WRAPPER,
        video_folder=VIDEO_FOLDER
    )