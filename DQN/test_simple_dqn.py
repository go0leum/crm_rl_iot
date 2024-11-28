import numpy as np
from Simple_DQN_Agent import SimpleConstructionEnv

def test_project_completion():
    env = SimpleConstructionEnv()
    env.reset()
    
    print("=== 프로젝트 완료 시나리오 테스트 ===")
    
    # 프로젝트 1의 태스크1 완료 상태로 설정
    env.state[env.IDX_PROJECT_START] = 2  # 태스크1 완료
    
    # 프로젝트 1의 태스크2 완료 상태로 설정
    env.state[env.IDX_PROJECT_START + 2] = 2  # 태스크2 리소스1 완료
    env.state[env.IDX_PROJECT_START + 3] = 2  # 태스크2 리소스2 완료
    
    # 프로젝트 2의 리소스 준비 상태로 설정
    base_idx = env.IDX_PROJECT_START + 4
    env.state[base_idx] = 1     # 리소스1 준비됨
    env.state[base_idx + 1] = 1 # 리소스2 준비됨
    
    # 에이전트를 프로젝트 2의 위치로 이동
    env.agent_pos = env.project_positions[1]  # 프로젝트 2 위치
    env.state[env.IDX_POS_X:env.IDX_POS_Y+1] = env.agent_pos
    
    print("초기 상태:")
    print(f"에이전트 위치: {env.agent_pos}")
    print(f"프로젝트1 상태: 태스크1={env.state[env.IDX_PROJECT_START]}, 태스크2={env.state[env.IDX_PROJECT_START+2:env.IDX_PROJECT_START+4]}")
    print(f"프로젝트2 상태: {env.state[base_idx:base_idx+2]}")
    
    # 프로젝트 2 실행
    action = 6  # 태스크 실행 액션
    next_state, reward, done, _, _ = env.step(action)
    
    print("\n태스크 실행 후:")
    print(f"리워드: {reward}")
    print(f"완료 여부: {done}")
    print(f"프로젝트2 상태: {next_state[base_idx:base_idx+2]}")
    
    # 모든 프로젝트가 완료되었는지 확인
    all_complete = env._all_projects_complete()
    print(f"\n모든 프로젝트 완료 여부: {all_complete}")

def test_resource_drop():
    env = SimpleConstructionEnv()
    env.reset()
    
    print("=== 리소스 드롭 시나리오 테스트 ===")
    
    # 에이전트가 리소스1을 보유한 상태로 설정
    env.state[env.IDX_RESOURCE_1] = 1  # 리소스1 보유 상태
    
    # 프로젝트 2의 초기 상태 설정
    base_idx = env.IDX_PROJECT_START + 4
    env.state[base_idx] = 0     # 리소스1 준비 안됨
    env.state[base_idx + 1] = 0 # 리소스2 준비 안됨
    
    # 에이전트를 프로젝트 2의 위치로 이동
    env.agent_pos = env.project_positions[1]  # 프로젝트 2 위치
    env.state[env.IDX_POS_X:env.IDX_POS_Y+1] = env.agent_pos
    
    print("초기 상태:")
    print(f"에이전트 위치: {env.agent_pos}")
    print(f"에이전트 리소스 보유: {env.state[env.IDX_RESOURCE_1]}")
    print(f"프로젝트2 상태: {env.state[base_idx:base_idx+2]}")
    
    # 리소스1 드롭 액션 실행
    action = 5  # 리소스1 드롭 액션
    next_state, reward, done, _, _ = env.step(action)
    
    print("\n리소스1 드롭 후:")
    print(f"리워드: {reward}")
    print(f"완료 여부: {done}")
    print(f"에이전트 리소스 보유: {next_state[env.IDX_RESOURCE_1]}")
    print(f"프로젝트2 상태: {next_state[base_idx:base_idx+2]}")

if __name__ == "__main__":
    test_project_completion()
    test_resource_drop()