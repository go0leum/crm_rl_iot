from stable_baselines3 import DQN
from DQN_Agent import ConstructionEnv
def action_sequence(step):
    sequence = [1, 2, 4, 6, 1, 5, 7, 3, 4, 6, 2, 5, 7, 8]
    return sequence[step]

def evaluate_agent(model, env, num_episodes=5):
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\n에피소드 {episode + 1} 시작")
        
        while not done:
            # 30 스텝 이후 에피소드 종료
            if step >= 30:
                print("\n최대 스텝 수(30)에 도달하여 에피소드를 종료합니다.")
                break
                
            action, _ = model.predict(obs, deterministic=True)
            # numpy array를 정수로 변환
            action = int(action)
            # if step < 14:
            #     action = action_sequence(step)

            # 현재 상태와 행동 출력
            print(f"\n스텝 {step + 1}:")
            print(f"에이전트 위치: ({obs[0]}, {obs[1]})")
            print(f"보유 리소스: {obs[2:4]}")
            print(f"보유 장비: {obs[4:6]}")
            print(f"프로젝트 상태: {obs[6:9]}")
            print(f"현재 요일: {obs[9]}")
            
            # 행동 설명
            action_desc = {
                0: "위로 이동",
                1: "오른쪽으로 이동",
                2: "아래로 이동",
                3: "왼쪽으로 이동",
                4: "리소스 픽업",
                5: "리소스 드롭",
                6: "장비 픽업",
                7: "장비 드롭",
                8: "태스크 실행"
            }
            print(f"선택한 행동: {action_desc[action]}")
            
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
            print(f"보상: {reward}")
            step += 1
            
        print(f"\n에피소드 {episode + 1} 종료")
        print(f"총 보상: {episode_reward}")
        print(f"총 스텝 수: {step}")

if __name__ == "__main__":
    # 환경 생성
    env = ConstructionEnv()
    
    # 학습된 모델 로드
    loaded_model = DQN.load("construction_dqn")
    
    # 에이전트 평가
    evaluate_agent(loaded_model, env) 