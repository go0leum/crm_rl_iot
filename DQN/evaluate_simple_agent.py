from stable_baselines3 import DQN
from Simple_DQN_Agent import SimpleConstructionEnv

def evaluate_agent(model, env, num_episodes=1):
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\n에피소드 {episode + 1} 시작")
        
        while not done:
            if step >= 35:
                print("\n최대 스텝 수(35)에 도달하여 에피소드를 종료합니다.")
                break
                
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            # 현재 상태와 행동 출력
            print(f"\n스텝 {step + 1}:")
            print(f"에이전트 위치: ({obs[0]}, {obs[1]})")
            print(f"리소스 상태:")
            print(f"- 리소스1: {'보유 중' if obs[2] == 1 else '미보유'}")
            print(f"- 리소스2: {'보유 중' if obs[3] == 1 else '미보유'}")
            
            # 프로젝트 상태 설명 수정
            status = {
                0: "준비 안됨",
                1: "준비 완료",
                2: "완료"
            }
            print("\n프로젝트 상태:")
            print(f"프로젝트 1:")
            print(f"- 태스크 1 (리소스1 필요):")
            print(f"  - 리소스1: {status[obs[4]]}")
            print(f"- 태스크 2 (리소스1, 리소스2 필요):")
            print(f"  - 리소스1: {status[obs[6]]}")
            print(f"  - 리소스2: {status[obs[7]]}")
            
            print(f"프로젝트 2:")
            print(f"- 태스크 1 (리소스1, 리소스2 필요):")
            print(f"  - 리소스1: {status[obs[8]]}")
            print(f"  - 리소스2: {status[obs[9]]}")
            print(f"- 태스크 2 (리소스1 필요):")
            print(f"  - 리소스1: {status[obs[10]]}")
            
            # 행동 설명
            action_desc = {
                0: "위로 이동",
                1: "오른쪽으로 이동",
                2: "아래로 이동",
                3: "왼쪽으로 이동",
                4: "리소스1 픽업/드롭",
                5: "리소스2 픽업/드롭",
                6: "태스크 실행"
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
    env = SimpleConstructionEnv()
    
    # 학습된 모델 로드
    try:
        loaded_model = DQN.load("simple_construction_dqn")
        print("학습된 모델을 성공적으로 불러왔습니다.")
        
        # 에이전트 평가
        evaluate_agent(loaded_model, env)
        
    except Exception as e:
        print(f"모델을 불러오는 데 실패했습니다: {e}") 