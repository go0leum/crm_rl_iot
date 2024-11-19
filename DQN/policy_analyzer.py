from stable_baselines3 import DQN
from DQN_Agent import ConstructionEnv

def analyze_policy(model, env):
    # 모든 가능한 상태 조합 생성
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            for res1 in [0, 1]:  # 리소스1 보유 여부
                for res2 in [0, 1]:  # 리소스2 보유 여부
                    for eq1 in [0, 1]:  # 장비1 보유 여부
                        for eq2 in [0, 1]:  # 장비2 보유 여부
                            for proj1 in [0, 1]:  # 프로젝트1 상태
                                for proj2 in [0, 1]:  # 프로젝트2 상태
                                    for proj3 in [0, 1]:  # 프로젝트3 상태
                                        for day in range(7):  # 요일
                                            # 상태 벡터 생성
                                            state = [x, y, res1, res2, eq1, eq2, proj1, proj2, proj3, day]
                                            
                                            # 모델의 예측 행동 얻기
                                            action, _ = model.predict(state, deterministic=True)
                                            action = int(action)
                                            
                                            # 상태와 행동 출력
                                            print("\n현재 상태:")
                                            print(f"위치: ({x}, {y})")
                                            print(f"리소스: [{res1}, {res2}]")
                                            print(f"장비: [{eq1}, {eq2}]")
                                            print(f"프로젝트: [{proj1}, {proj2}, {proj3}]")
                                            print(f"요일: {day}")
                                            
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
                                            print(f"선택된 행동: {action_desc[action]}")

def analyze_specific_state(model, state):
    """특정 상태에서의 정책을 분석합니다."""
    action, _ = model.predict(state, deterministic=True)
    action = int(action)
    
    print("\n분석할 상태:")
    print(f"위치: ({state[0]}, {state[1]})")
    print(f"리소스: [{state[2]}, {state[3]}]")
    print(f"장비: [{state[4]}, {state[5]}]")
    print(f"프로젝트: [{state[6]}, {state[7]}, {state[8]}]")
    print(f"요일: {state[9]}")
    
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
    print(f"선택된 행동: {action_desc[action]}")

if __name__ == "__main__":
    # 환경 생성
    env = ConstructionEnv()
    
    # 학습된 모델 로드
    loaded_model = DQN.load("construction_dqn")
    
    # 전체 정책 분석
    # print("전체 정책 분석 시작...")
    # analyze_policy(loaded_model, env)
    
    # 특정 상태 분석 예시
    print("\n특정 상태 분석 예시:")
    example_state = [2, 0, 0, 0, 0, 0, 2, 2, 1, 4]  # 초기 상태
    analyze_specific_state(loaded_model, example_state) 