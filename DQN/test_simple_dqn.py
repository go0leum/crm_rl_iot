import numpy as np
from Simple_DQN_Agent import SimpleConstructionEnv

def test_state_transition():
    env = SimpleConstructionEnv()
    
    # 테스트 케이스 1: 초기 상태에서 오른쪽으로 이동
    initial_state = env.reset()[0]
    action = 1  # 오른쪽 이동
    next_state, reward, done, _, _ = env.step(action)
    print("테스트 1 - 오른쪽 이동")
    print(f"초기 상태: {initial_state}")
    print(f"다음 상태: {next_state}")
    print(f"리워드: {reward}\n")
    
    # 테스트 케이스 2: 리소스 위치에서 픽업
    env.reset()
    # 리소스 위치(1,1)로 이동
    env.step(2)  # 아래로
    env.step(1)  # 오른쪽으로
    state_before_pickup = env.state.copy()
    action = 4  # 리소스 픽업
    next_state, reward, done, _, _ = env.step(action)
    print("테스트 2 - 리소스 픽업")
    print(f"픽업 전 상태: {state_before_pickup}")
    print(f"픽업 후 상태: {next_state}")
    print(f"리워드: {reward}\n")
    
    # 테스트 케이스 3: 프로젝트 위치에서 리소스 드롭
    env.reset()
    env.state[env.IDX_RESOURCE] = 1  # 리소스 보유 상태로 설정
    env.agent_pos = [2, 2]  # 첫 번째 프로젝트 위치로 이동
    env.state[env.IDX_POS_X:env.IDX_POS_Y+1] = env.agent_pos
    state_before_drop = env.state.copy()
    action = 5  # 리소스 드롭
    next_state, reward, done, _, _ = env.step(action)
    print("테스트 3 - 리소스 드롭")
    print(f"드롭 전 상태: {state_before_drop}")
    print(f"드롭 후 상태: {next_state}")
    print(f"리워드: {reward}\n")
    
    # 테스트 케이스 4: 프로젝트 실행
    env.reset()
    env.agent_pos = [2, 2]  # 첫 번째 프로젝트 위치
    env.state[env.IDX_POS_X:env.IDX_POS_Y+1] = env.agent_pos
    env.state[env.IDX_PROJECT_START+1] = 1  # 프로젝트 준비 상태로 설정
    state_before_execute = env.state.copy()
    action = 6  # 태스크 실행
    next_state, reward, done, _, _ = env.step(action)
    print("테스트 4 - 프로젝트 실행")
    print(f"실행 전 상태: {state_before_execute}")
    print(f"실행 후 상태: {next_state}")
    print(f"리워드: {reward}")
    
    # 테스트 케이스 5: 두 프로젝트 완료 시나리오
    env.reset()
    # 첫 번째 프로젝트 완료 상태로 설정
    env.state[env.IDX_PROJECT_START] = 2  # 첫 번째 프로젝트 완료
    env.agent_pos = [2, 2]  # 두 번째 프로젝트 위치
    env.state[env.IDX_POS_X:env.IDX_POS_Y+1] = env.agent_pos
    env.state[env.IDX_PROJECT_START+1] = 1  # 두 번째 프로젝트 준비 상태
    
    state_before_complete = env.state.copy()
    action = 6  # 두 번째 프로젝트 실행
    next_state, reward, done, _, _ = env.step(action)
    print("\n테스트 5 - 두 프로젝트 완료")
    print(f"완료 전 상태: {state_before_complete}")
    print(f"완료 후 상태: {next_state}")
    print(f"리워드: {reward}")
    print(f"에피소드 종료: {done}")

if __name__ == "__main__":
    test_state_transition() 