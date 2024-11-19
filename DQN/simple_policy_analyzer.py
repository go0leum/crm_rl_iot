from stable_baselines3 import DQN
from Simple_DQN_Agent import SimpleConstructionEnv

def analyze_policy(model, env):
    # 모든 가능한 상태 조합 생성
    for x in range(3):  # 3x3 그리드
        for y in range(3):
            for resource in [0, 1]:  # 리소스 보유 상태
                for proj1 in [0, 1, 2]:  # 프로젝트1 상태 (준비안됨/준비됨/완료)
                    for proj2 in [0, 1, 2]:  # 프로젝트2 상태
                        # 상태 벡터 생성
                        state = [x, y, resource, proj1, proj2]
                        
                        # 모델의 예측 행동 얻기
                        action, _ = model.predict(state, deterministic=True)
                        action = int(action)
                        
                        # 상태와 행동 출력
                        print("\n현재 상태:")
                        print(f"위치: ({x}, {y})")
                        print(f"리소스 보유: {resource}")
                        print(f"프로젝트1 상태: {proj1}")
                        print(f"프로젝트2 상태: {proj2}")
                        
                        action_desc = {
                            0: "위로 이동",
                            1: "오른쪽으로 이동",
                            2: "아래로 이동",
                            3: "왼쪽으로 이동",
                            4: "리소스 픽업",
                            5: "리소스 드롭",
                            6: "태스크 실행"
                        }
                        print(f"선택된 행동: {action_desc[action]}")

def analyze_specific_state(model, state):
    """특정 상태에서의 정책을 분석합니다."""
    action, _ = model.predict(state, deterministic=True)
    action = int(action)
    
    print("\n분석할 상태:")
    print(f"위치: ({state[0]}, {state[1]})")
    print(f"리소스 보유: {state[2]}")
    print(f"프로젝트1 상태: {state[3]}")
    print(f"프로젝트2 상태: {state[4]}")
    
    action_desc = {
        0: "위로 이동",
        1: "오른쪽으로 이동",
        2: "아래로 이동",
        3: "왼쪽으로 이동",
        4: "리소스 픽업",
        5: "리소스 드롭",
        6: "태스크 실행"
    }
    print(f"선택된 행동: {action_desc[action]}")

if __name__ == "__main__":
    # 환경 생성
    env = SimpleConstructionEnv()
    
    # 학습된 모델 로드
    loaded_model = DQN.load("simple_construction_dqn")
    
    # 전체 정책 분석
    print("전체 정책 분석 시작...")
    analyze_policy(loaded_model, env)
    
    # 특정 상태 분석 예시
    # print("\n특정 상태 분석 예시:")
    # example_state = [0, 0, 0, 0, 0]  # 초기 상태
    # analyze_specific_state(loaded_model, example_state) 