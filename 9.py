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
    def __init__(self, env):
        super().__init__(env)
        
        self.knee_bend_weight = 0.3
        
        self.knee_angle_min_rad = 30.0 * (np.pi / 180.0)
        self.knee_angle_max_rad = 90.0 * (np.pi / 180.0)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        # 1. 기본 환경의 step을 먼저 호출합니다.
        # 💡 base_terminated는 '높이'와 '각도'가 모두 포함된 값입니다.
        obs, base_reward, base_terminated, base_truncated, info = self.env.step(action)

        # 2. 💡 '건강함(healthy)'을 '높이' 없이 '각도'로만 재정의합니다.
        #    obs[1]은 몸통의 각도(qpos[2])입니다.
        is_angle_healthy = (obs[1] > -1.0) & (obs[1] < 1.0)
        
        # 3. 💡 우리가 원하는 새로운 '종료(terminated)' 조건을 계산합니다.
        #    오직 '각도'가 범위를 벗어났을 때만 종료시킵니다.
        new_terminated = (not is_angle_healthy)

        # 4. 💡 보상을 '완전히' 재조립합니다.
        #    base_reward를 사용하지 않습니다. (높이 때문에 healthy_reward가 0이 됐을 수 있으므로)
        
        # 4a. 기본 보상 컴포넌트를 info에서 가져옵니다.
        forward_reward = info.get('reward_run', 0.0)
        ctrl_cost = info.get('reward_ctrl', 0.0)
        
        # 4b. '생존 보상'을 우리의 '각도' 기준으로 새로 계산합니다.
        base_healthy_reward_value = self.env.unwrapped.healthy_reward # (보통 1.0)
        healthy_reward = 0.0
        if is_angle_healthy: # '각도'가 건강할 때만 생존 보너스를 줍니다.
            healthy_reward = base_healthy_reward_value

        # 4c. 사용자 정의 보상/페널티를 가져옵니다. (기존 코드와 동일)
        tilt_penalty = -2 * np.abs(obs[1])
        shake_penalty = -0.5 * np.abs(obs[10])
        
        right_thigh_vel = obs[11] # 오른쪽 허벅지 속도
        left_thigh_vel = obs[14]  # 왼쪽 허벅지 속도
        right_knee_angle = np.abs(obs[3])
        left_knee_angle = np.abs(obs[6])
        
        knee_bend_bonus = 0
        if right_thigh_vel > 0 and \
           (self.knee_angle_min_rad < right_knee_angle < self.knee_angle_max_rad):
            knee_bend_bonus += self.knee_bend_weight
        if left_thigh_vel > 0 and \
           (self.knee_angle_min_rad < left_knee_angle < self.knee_angle_max_rad):
            knee_bend_bonus += self.knee_bend_weight
            
        # 4d. 모든 보상을 합산합니다.
        new_reward = (
            forward_reward      # 기본 전진 보상
            + healthy_reward    # '각도' 기준 생존 보상
            + ctrl_cost         # 기본 컨트롤 비용
            + tilt_penalty      # 커스텀 페널티
            + shake_penalty     # 커스텀 페널티
            + knee_bend_bonus   # 커스텀 보너스
        )
        
        # 5. 💡 'Monitor' 래퍼 오류 방지용 리셋 신호 처리 (중요)
        #    이전 대화에서 다룬 'RuntimeError'를 방지하는 코드입니다.
        
        # 5a. 우리의 '각도' 기준 종료는 'terminated'로 설정합니다.
        final_terminated = new_terminated
        
        # 5b. 'TimeLimit'에 의한 'truncated'는 그대로 존중합니다.
        #     *또한*, 기본 환경이 '높이' 때문에 종료(base_terminated=True)됐지만,
        #     우리는 각도 때문에 종료가 아니라고(new_terminated=False) 판단한
        #     '데드락' 상태일 때, 'truncated=True'로 위장하여 VecEnv의 리셋을 강제합니다.
        final_truncated = base_truncated or (base_terminated and not new_terminated)

        # 6. 최종 신호들을 반환합니다.
        return obs, new_reward, final_terminated, final_truncated, info


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
    custom_xml_path = os.path.join(current_dir, 'xml/walker2d_slope.xml')
    
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