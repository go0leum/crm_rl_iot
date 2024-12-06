from stable_baselines3 import DQN
from Simple_DQN_Agent import SimpleConstructionEnv
import os

def new_training(env, total_steps=120000):
    """새로운 모델 학습을 수행하는 함수"""
    print("=== 새로운 모델 학습 시작 ===")
    
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        buffer_size=50000,
        learning_starts=50000,
        batch_size=64,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        target_update_interval=1000,
        verbose=1,
        tensorboard_log="./crm_rl_iot/DQN/log/dqn_tensorboard/"
    )
    
    model.learn(total_timesteps=total_steps)
    model.save("../weights/simple_construction_dqn")
    print("학습 완료. 모델 저장됨: simple_construction_dqn")
    
    return model

def continue_training(env, model_path="simple_construction_dqn", additional_steps=50000):
    """기존 모델을 불러와 추가 학습을 수행하는 함수"""
    print("=== 기존 모델 추가 학습 시작 ===")
    
    if not os.path.exists(f"{model_path}.zip"):
        print(f"Error: {model_path}.zip 파일을 찾을 수 없습니다.")
        return None
        
    try:
        model = DQN.load(
            model_path,
            env=env,
            tensorboard_log="./crm_rl_iot/DQN/log/dqn_tensorboard/",
            # 필요한 경우 하이퍼파라미터 수정
            learning_rate=1e-4,
            exploration_fraction=0.2,
            exploration_initial_eps=0.5,
            exploration_final_eps=0.05
        )
        
        print("기존 모델 불러오기 완료")
        model.learn(
            total_timesteps=additional_steps,
            reset_num_timesteps=False
        )
        
        # 기존 모델 파일에 덮어쓰기
        model.save(model_path)
        print(f"추가 학습 완료. 모델 저장됨: {model_path}")
        
        return model
        
    except Exception as e:
        print(f"모델 로딩 중 오류 발생: {e}")
        return None

def main():
    env = SimpleConstructionEnv()
    
    while True:
        print("\n=== DQN 학습 메뉴 ===")
        print("1. 새로운 모델 학습")
        print("2. 기존 모델 추가 학습")
        print("3. 종료")
        
        choice = input("선택하세요 (1-3): ")
        
        if choice == "1":
            steps = input("학습 스텝 수를 입력하세요 (기본값: 120000): ")
            steps = int(steps) if steps.isdigit() else 120000
            new_training(env, steps)
            
        elif choice == "2":
            model_path = input("불러올 모델 경로를 입력하세요 (기본값: simple_construction_dqn): ")
            model_path = model_path if model_path else "simple_construction_dqn"
            
            steps = input("추가 학습 스텝 수를 입력하세요 (기본값: 50000): ")
            steps = int(steps) if steps.isdigit() else 50000
            
            continue_training(env, model_path, steps)
            
        elif choice == "3":
            print("프로그램을 종료합니다.")
            break
            
        else:
            print("잘못된 선택입니다. 다시 선택해주세요.")

if __name__ == "__main__":
    main() 