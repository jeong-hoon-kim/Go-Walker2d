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

## --- 커스텀 리워드 래퍼 정의 ---
class CustomRewardWrapper(Wrapper):
    def __init__(self, env, flight_penalty_weight=1):
        super().__init__(env)
        self.flight_penalty_weight = flight_penalty_weight
        
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
        obs, reward, terminated, truncated, info = self.env.step(action)

        # --- 🏆 '공중 체공' 페널티 계산 ---
        left_foot_on_ground, right_foot_on_ground = self._check_foot_contact()
        flight_penalty = 0
        if not left_foot_on_ground and not right_foot_on_ground:
            flight_penalty = -self.flight_penalty_weight
        

        # 3. 기존 보상에 모든 페널티를 추가합니다.
        new_reward = reward + flight_penalty
        
        return obs, new_reward, terminated, truncated, info


## --- 커스텀 평가 콜백 클래스 정의 ---
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
                episode_total_reward = 0.0 # 누적 보상 초기화

                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated
                    
                    episode_total_reward += reward # 누적 보상 업데이트

                    # Walker2d-v5의 obs[1]는 몸통 각도(torso angle)입니다. (v3, v4와 다름)
                    torso_angles.append(obs[1]) 
                    if done: 
                        final_distance = info.get('x_position', 0)
                
                episode_distances.append(final_distance)
                episode_stabilities.append(np.std(torso_angles))
                episode_reward.append(episode_total_reward)

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

## --- 모델 테스트와 영상 녹화 ---
def test_model(xml, model_path, seed, video_folder):
    print(f"--- '{model_path}' 모델 테스트 시작 (시드: {seed}) ---")
    
    # 환경 생성
    custom_xml_path = xml
    env = gym.make("Walker2d-v5", render_mode="rgb_array", xml_file=custom_xml_path)
    env = CustomRewardWrapper(env=env)
    
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

    utils.print_log("\n--- 최종 평가 결과 ---", model_path)
    utils.print_log(f"모델: {model_path}", model_path)
    utils.print_log(f"최종 이동 거리: {final_distance:.2f} m", model_path)
    utils.print_log(f"총 보상: {total_reward:.2f}", model_path)
    utils.print_log(f"몸통 흔들림 (안정성): {stability_score:.4f} (낮을수록 안정적)", model_path)
    utils.print_log(f"영상 저장 위치: {video_folder}{video_prefix}.mp4", model_path)
    utils.print_log("-" * 30 + "\n", model_path)

    env.close()

# --- 메인 훈련 코드 ---
if __name__ == "__main__":
    FOLDER_NAME = "custom_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    SAVE_PATH = FOLDER_NAME + f"/results/"
    TENSORBOARD_PATH = FOLDER_NAME + "/tensorboard/"
    VIDEO_PATH = FOLDER_NAME + "/videos/"
    TOTAL_TIMESTEPS = 3000000
    
    # xml 파일 경로 설정
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    custom_xml_path = os.path.join(current_dir, 'xml/walker2d_base.xml')
    
    # 시드 설정
    SEED = 42
    utils.set_seed(SEED)

    # 훈련용 환경
    train_env = gym.make("Walker2d-v5", xml_file=custom_xml_path)
    train_env = Monitor(train_env, SAVE_PATH)
    train_env = CustomRewardWrapper(env=train_env)
    train_env.reset(seed=SEED) # 환경 초기화 시 시드 설정
    train_env.action_space.seed(SEED)

    # 평가용 환경
    eval_env = gym.make("Walker2d-v5", xml_file=custom_xml_path)
    eval_env = CustomRewardWrapper(env=eval_env)
    eval_env.reset(seed=SEED) # 환경 초기화 시 시드 설정
    eval_env.action_space.seed(SEED)

    # cpu 사용
    device = "cpu"
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
        tensorboard_log=TENSORBOARD_PATH 
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
        xml=custom_xml_path,
        model_path=SAVE_PATH + "ppo_walker2d_best_distance",
        seed=SEED,
        video_folder=VIDEO_PATH
    )
    test_model(
        xml=custom_xml_path,
        model_path=SAVE_PATH + "ppo_walker2d_best_stability",
        seed=SEED,
        video_folder=VIDEO_PATH
    )
    test_model(
        xml=custom_xml_path,
        model_path=SAVE_PATH + "ppo_walker2d_best_reward",
        seed=SEED,
        video_folder=VIDEO_PATH
    )
    test_model(
        xml=custom_xml_path,
        model_path=SAVE_PATH + "ppo_walker2d_final",
        seed=SEED,
        video_folder=VIDEO_PATH
    )