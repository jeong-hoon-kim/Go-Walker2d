import torch
import numpy as np
import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.wrappers import RecordVideo
from gymnasium.envs.mujoco import walker2d_v5
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
import os
import datetime
import utils


## --- 커스텀 리워드 래퍼 정의 ---
class PhasedGaitWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # --- 🏆 하이퍼파라미터 ---
        # 기존 파라미터들
        self.target_velocity = 1.75
        self.velocity_tolerance = 0.5
        self.velocity_reward_weight = 2.0
        self.stability_weight = 0.3
        self.flight_penalty_weight = 1.0
        
        # --- 추가된 부분: 분면 보상 관련 ---
        self.phase_reward_weight = 0.05 # 분면 보상 가중치 (원래 0.005에서 0.05로 상향 조정 권장)
        self.current_phase = 0 # 현재 보행 단계 (0은 초기/정의되지 않은 상태)
        self.phase_definition_threshold = 0.1 # 각속도가 0에 가깝다고 판단할 임계값

        # 발과 바닥의 ID
        # Mujoco 환경의 모델 구조에 따라 ID 가져오는 부분은 그대로 유지
        self.left_foot_geom_id = self.env.unwrapped.model.geom('foot_left_geom').id
        self.right_foot_geom_id = self.env.unwrapped.model.geom('foot_geom').id
        self.floor_geom_id = self.env.unwrapped.model.geom('floor').id
        
    def _check_foot_contact(self):
        # ... (기존과 동일) ...
        left_contact, right_contact = False, False
        for contact in self.env.unwrapped.data.contact:
            geom_pair = {contact.geom1, contact.geom2}
            if self.left_foot_geom_id in geom_pair and self.floor_geom_id in geom_pair: left_contact = True
            if self.right_foot_geom_id in geom_pair and self.floor_geom_id in geom_pair: right_contact = True
        return left_contact, right_contact

    # --- 🏆 개선된 부분: 6단계 보행 단계 판단 함수 ---
    def _get_current_phase(self, obs, left_contact, right_contact):
        # 허벅지 관절 각속도 (Thigh Angular Velocity)
        # obs[11]: 오른쪽 허벅지 각속도, obs[14]: 왼쪽 허벅지 각속도
        # 걷는 방향으로의 회전을 양수, 뒤로 미는 동작을 음수로 가정합니다.
        right_thigh_vel = obs[11]
        left_thigh_vel = obs[14]

        threshold = self.phase_definition_threshold
        
        # --- 6단계 보행 주기 정의 (오른발 주기: 1->2->3, 왼발 주기: 4->5->6) ---
        
        # 1. Initial Double Support (IDS, 오른발 착지 직후)
        # L/R 모두 접촉 & 오른발 스윙 끝/정지 & 왼발은 아직 뒤로 밀고 있음
        if left_contact and right_contact and \
           (right_thigh_vel < threshold and right_thigh_vel > -threshold) and left_thigh_vel < -threshold:
            return 1
            
        # 2. Right Single Support (RSS, 오른발 단독 지지 - 왼발 스윙 시작)
        # 왼발 공중 & 오른발 지지 & 왼발이 앞으로 스윙
        elif not left_contact and right_contact and \
             right_thigh_vel <= threshold and left_thigh_vel > threshold:
            return 2

        # 3. Terminal Double Support (TDS, 왼발 착지 직전)
        # L/R 모두 접촉 & 오른발이 뒤로 밀어내고 & 왼발 스윙이 끝나 착지 준비 중
        elif left_contact and right_contact and \
             right_thigh_vel < -threshold and (left_thigh_vel < threshold and left_thigh_vel > -threshold):
            return 3 
            
        # --- 대칭 단계 (왼발 주기) ---
        
        # 4. Initial Double Support (IDS, 왼발 착지 직후)
        # L/R 모두 접촉 & 왼발 스윙 끝/정지 & 오른발은 아직 뒤로 밀고 있음
        elif left_contact and right_contact and \
             (left_thigh_vel < threshold and left_thigh_vel > -threshold) and right_thigh_vel < -threshold:
            return 4
            
        # 5. Left Single Support (LSS, 왼발 단독 지지 - 오른발 스윙 시작)
        # 오른발 공중 & 왼발 지지 & 오른발이 앞으로 스윙
        elif left_contact and not right_contact and \
             left_thigh_vel <= threshold and right_thigh_vel > threshold:
            return 5

        # 6. Terminal Double Support (TDS, 오른발 착지 직전)
        # L/R 모두 접촉 & 왼발이 뒤로 밀어내고 & 오른발 스윙이 끝나 착지 준비 중
        elif left_contact and right_contact and \
             left_thigh_vel < -threshold and (right_thigh_vel < threshold and right_thigh_vel > -threshold):
            return 6
            
        else: # 두 발 공중 또는 정의된 6단계에 해당하지 않는 애매한 상태
            return 0

    def reset(self, **kwargs):
        self.current_phase = 0 # 단계 초기화
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        healthy_reward = info.get('reward_survive', 1.0)
        ctrl_cost = info.get('reward_ctrl', 0)
        # Stability: 롤링(obs[1])과 피칭 각속도(obs[10]) 페널티
        stability_penalty = -self.stability_weight * (np.abs(obs[1]) + 0.1 * np.abs(obs[10]))
        
        current_velocity = obs[8] # 전방 속도
        velocity_bonus = self.velocity_reward_weight * \
                          np.exp(-np.square(current_velocity - self.target_velocity) / (2 * np.square(self.velocity_tolerance)))
        
        left_foot_on_ground, right_foot_on_ground = self._check_foot_contact()
        flight_penalty = 0
        if not left_foot_on_ground and not right_foot_on_ground:
            flight_penalty = -self.flight_penalty_weight

        # --- 🏆 개선된 부분: 6단계 분면 보상 (Phase Reward) 계산 ---
        previous_phase = self.current_phase
        current_phase = self._get_current_phase(obs, left_foot_on_ground, right_foot_on_ground)
        
        phase_reward = 0
        # 6단계 순환: 1->2, 2->3, 3->4, 4->5, 5->6, 6->1
        expected_next_phase = (previous_phase % 6) + 1 # 1부터 6까지 순환 (6 다음은 1)

        if previous_phase != 0 and current_phase == expected_next_phase: 
            # 올바른 순서로 전이: +1 보상
            phase_reward = 1
        elif current_phase == previous_phase or current_phase == 0 or previous_phase == 0: 
            # 단계 유지, 유효하지 않은 단계(0), 또는 초기 상태: 보상/페널티 없음
            phase_reward = 0
        else: 
            # 잘못된 순서로 전이 (예: 1->3 또는 1->6): -1 페널티
            phase_reward = -1 
            
        # 현재 단계를 다음 스텝을 위해 저장
        if current_phase != 0: # 유효한 단계일 때만 업데이트
            self.current_phase = current_phase

        # --- 최종 보상 계산 ---
        new_reward = (
            velocity_bonus
            + healthy_reward
            + ctrl_cost
            + stability_penalty
            + flight_penalty
            + (self.phase_reward_weight * phase_reward) # <-- 분면 보상 추가
        )
        
        # info에 현재 단계 추가 (디버깅용)
        info['current_phase'] = self.current_phase
        info['phase_reward'] = self.phase_reward_weight * phase_reward
        
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
                self.model.save(os.path.join(self.save_path, "sac_walker2d_best_distance.zip"))
                if self.verbose > 0: print(f"  >> New best distance model saved: {mean_distance:.2f} m")

            # 최고 안정성 모델 저장
            if mean_stability < self.best_mean_stability:
                self.best_mean_stability = mean_stability
                self.model.save(os.path.join(self.save_path, "sac_walker2d_best_stability.zip"))
                if self.verbose > 0: print(f"  >> New best stability model saved: {mean_stability:.4f}")
            
            # 최고 보상 모델 저장
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.model.save(os.path.join(self.save_path, "sac_walker2d_best_reward.zip"))
                if self.verbose > 0: print(f"  >> New best reward model saved: {mean_reward:.2f}")
            
            print("---------------------------------")
        
        return True


## --- 모델 테스트와 영상 녹화 ---
def test_model(xml, model_path, seed, video_folder):
    print(f"--- '{model_path}' 모델 테스트 시작 (시드: {seed}) ---")
    
    # 환경 생성
    custom_xml_path = xml
    env = gym.make("Walker2d-v5", xml_file=custom_xml_path)
    env = PhasedGaitWrapper(env=env)
    
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
    FOLDER_NAME = "custom_SAC_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    SAVE_PATH = FOLDER_NAME + f"/results/"
    TENSORBOARD_PATH = FOLDER_NAME + "/tensorboard/"
    VIDEO_PATH = FOLDER_NAME + "/videos/"
    TOTAL_TIMESTEPS = 3000000
    
    # xml 파일 경로 설정
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    custom_xml_path = os.path.join(current_dir, 'xml/walker2d_hurdle.xml')

    # 시드 설정
    SEED = 42
    utils.set_seed(SEED)

    # 훈련용 환경
    train_env = gym.make("Walker2d-v5", xml_file=custom_xml_path)
    train_env = TimeLimit(train_env, max_episode_steps=1000)
    train_env = Monitor(train_env, SAVE_PATH)
    train_env = PhasedGaitWrapper(env=train_env)
    train_env.reset(seed=SEED) # 환경 초기화 시 시드 설정
    train_env.action_space.seed(SEED)

    # 평가용 환경
    eval_env = gym.make("Walker2d-v5", xml_file=custom_xml_path)
    eval_env = TimeLimit(eval_env, max_episode_steps=1000)
    eval_env = PhasedGaitWrapper(env=eval_env)
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
    model = SAC(
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
        tb_log_name="sac_walker2d"
    )

    # 최종 모델 저장하기
    model.save(f"{SAVE_PATH}sac_walker2d_final.zip")
    print("최종 모델 저장이 완료되었습니다.")

    train_env.close()
    eval_env.close()

    # 테스트
    test_model(
        xml=custom_xml_path,
        model_path=SAVE_PATH + "sac_walker2d_best_distance",
        seed=SEED,
        video_folder=VIDEO_PATH
    )
    test_model(
        xml=custom_xml_path,
        model_path=SAVE_PATH + "sac_walker2d_best_stability",
        seed=SEED,
        video_folder=VIDEO_PATH
    )
    test_model(
        xml=custom_xml_path,
        model_path=SAVE_PATH + "sac_walker2d_best_reward",
        seed=SEED,
        video_folder=VIDEO_PATH
    )
    test_model(
        xml=custom_xml_path,
        model_path=SAVE_PATH + "sac_walker2d_final",
        seed=SEED,
        video_folder=VIDEO_PATH

    )
