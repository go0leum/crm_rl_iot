# 학습된 모델 사용하기

## 모델 load하기

- `loaded_model = DQN.load("simple_construction_dqn")`로 모델을 load하여 사용할 수 있습니다.

## Agent의 action 가져오기

- `action, _ = model.predict(obs, deterministic=True)`로 obs라는 상태가 주어졌을 때, 최적의 action을 가져올 수 있습니다.
- `action = int(action)`을 통해서 형 변환을 해주어야 합니다.

## Env의 observation space 가져오기

- 앞선 Agent의 `model.predict`의 인자인 obs를 가져오기 위해서 `obs, _ = env.reset()`를 통해 초기 observation space를 환경으로 부터 얻어와야 합니다.

## Env step을 통해 다음 상태 가져오기

- `obs, reward, done, _, _ = env.step(action)` 코드를 통해서 action에 대한 다음 상태를 가져올 수 있습니다.
