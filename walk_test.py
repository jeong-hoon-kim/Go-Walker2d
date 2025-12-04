import gymnasium as gym
from gymnasium import Wrapper
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from gymnasium.envs.mujoco import walker2d_v5
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.utils import set_random_seed
from gymnasium.wrappers import RecordVideo
import numpy as np
import os
import utils

class CustomWalkerEnv(walker2d_v5.Walker2dEnv):
    """
    원본 Walker2dEnv를 상속받아 is_healthy 로직만 수정한 커스텀 환경입니다.
    """
    # @property 데코레이터를 사용하여 is_healthy를 메서드가 아닌 속성처럼 다룹니다.
    @property
    def is_healthy(self):
        """
        여기에서 새로운 'healthy' 조건을 정의합니다.
        원본 로직을 참고하여 수정하거나 완전히 새로 작성할 수 있습니다.
        """
        
        # 원본 Walker2d-v5의 is_healthy 로직 (참고용)
        # z, angle = self.data.qpos[1:3]

        # min_z, max_z = self._healthy_z_range
        # min_angle, max_angle = self._healthy_angle_range

        # healthy_z = min_z < z < max_z
        # healthy_angle = min_angle < angle < max_angle
        # is_healthy = healthy_z and healthy_angle

        # return is_healthy

        z, angle = self.data.qpos[1:3]

        min_z, max_z = (0.8, 200.0) # 수정된 z 범위
        min_angle, max_angle = self._healthy_angle_range

        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle
        is_healthy = healthy_z and healthy_angle

        return is_healthy


## --- 모델 테스트와 영상 녹화 ---
def test_model(xml, model_path, seed, video_folder):
    print(f"--- '{model_path}' 모델 테스트 시작 (시드: {seed}) ---")
    
    # 환경 생성
    custom_xml_path = xml
    env = CustomWalkerEnv(render_mode="rgb_array", xml_file=custom_xml_path)
    env = TimeLimit(env, max_episode_steps=1000)
    env = gym.make("Walker2d-v5", render_mode="rgb_array", xml_file=custom_xml_path)
    
    # 비디오 녹화 래퍼 적용
    os.makedirs(video_folder, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    video_prefix = f"{model_name}"
    env = RecordVideo(env, video_folder=video_folder, name_prefix=video_prefix, fps=30)
    
    # 훈련된 모델 불러오기
    set_random_seed(seed)
    model = SAC.load(model_path, env=env)
    
    # 평가 시작
    obs, info = env.reset(seed=seed)
    
    # 평가 지표 초기화
    torso_angles = []
    total_reward = 0.0
    final_distance = 0.0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 평가 지표 수집
        torso_angle = obs[1]
        torso_angles.append(torso_angle)
        total_reward += float(reward)

        if done:
            final_distance = info.get('x_position', 0)
    # 최종 결과 계산 및 출력
    stability_score = np.std(torso_angles)

    utils.print_log("\n--- 최종 평가 결과 ---", "videos_test_custom/"+model_name)
    utils.print_log(f"모델: {model_path}", "videos_test_custom/"+model_name)
    utils.print_log(f"최종 이동 거리: {final_distance:.2f} m", "videos_test_custom/"+model_name)
    utils.print_log(f"총 보상: {total_reward:.2f}", "videos_test_custom/"+model_name)
    utils.print_log(f"몸통 흔들림 (안정성): {stability_score:.4f} (낮을수록 안정적)", "videos_test_custom/"+model_name)
    utils.print_log(f"영상 저장 위치: {video_folder}{video_prefix}.mp4", "videos_test_custom/"+model_name)
    utils.print_log("-" * 30 + "\n", "videos_test_custom/"+model_name)

    env.close()


if __name__ == "__main__":
    # --- 테스트 설정 ---
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    XML_PATH = os.path.join(current_dir, 'xml/walker2d_slope.xml')
    MODEL_PATH = "custom_SAC_2025-12-03_05-22-57/results/sac_walker2d_final.zip" 
    SEED = 42
    VIDEO_FOLDER = "videos_test_custom/"
    
    USE_CUSTOM_WRAPPER = False

    test_model(
        xml=XML_PATH,
        model_path=MODEL_PATH,
        seed=SEED,
        video_folder=VIDEO_FOLDER
    )