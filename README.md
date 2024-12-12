# Construction Resource Management using Reinforcement Learning

## Project Overview
Efficient resource allocation in construction projects plays a critical role in influencing cost, duration, and quality. It is essential to allocate various construction resources based on the needs, timing, location, and availability at each site.

In this project, we aim to define tasks that may arise in construction site projects and find optimal equipment operation process policies through reinforcement learning. This approach is based on the "Simulation of autonomous resource allocation through deep reinforcement learning-based portfolio-project integration."

### Key Objectives
**Resource Allocation**: Assigning construction resources based on the requirements, availability, and timing at each site.

**Task Definition**: Defining tasks in construction site projects.

**Reinforcement Learning**: Utilizing reinforcement learning to find the optimal policy for resource management.

**Optimization**: Conducting reinforcement learning to minimize the time required to complete given projects with limited resources.

## MDP Definition

### Environment

#### Time
- The total working time and daily working time are given.
- At the end of each working day, the agent's location and available resources change.

#### Project
- One or more projects are assigned.
- There are specific locations on the map where projects can be performed.
- Each project has one or more tasks.
- Completing all tasks within a project results in the completion of the project.

#### Task
- Each task requires one or more resources.
- The agent must transport the required resources to the task location to perform the task.
- Once all required resources are transported, executing the task action completes the task.

#### Resource
- One or more resources are provided.
- Specific locations on the map provide resources.
- The agent can load resources only at these specific locations.

#### Map
- The map includes start positions, project locations, resource locations, and obstacles.
- The agent moves to the start position at the beginning of each workday.
- Projects and resources have specific locations on the map.

### Action
Actions include moving, resource pickup, resource drop, and executing tasks. Each task has an associated time cost.

| number | action           | time cost | number | action          | time cost |
|--------|------------------|-----------|--------|-----------------|-----------|
| 0      | move left        | 1         | 4      | resource pickup | 5         |
| 1      | move right       | 1         | 5      | resource drop   | 5         |
| 2      | move up          | 1         | 6      | execute task    | 10        |
| 3      | move down        | 1         |        |                 |           |

### Reward
The agent receives positive rewards for actions that progress the tasks and negative rewards for actions that consume time without progress.

#### Positive Rewards

| Scenario                                | Reward |
|-----------------------------------------|--------|
| Completing all projects within total time | 1000   |
| Successfully performing resource pickup/drop | 10    |
| Successfully completing a task           | 100    |

#### Negative Rewards

| Scenario                         | Reward |
|----------------------------------|--------|
| Exceeding total working time     | -300   |
| Moving outside the map boundary  | -1     |
| Performing a move action         | -5     |
| Unsuccessful resource pickup/drop | -5     |

## Training & Test

### Q-Learning

#### observation space

| Name           | Size   | Definition                                   |
| -------------- | ------ | -------------------------------------------- |
| Agent Position | 26 x 12 | Indicates the coordinates where the agent is currently located. |

### DQN 

#### Simple DQN

| Name                      | Size        | Definition                                                                                                           |
|---------------------------|-------------|----------------------------------------------------------------------------------------------------------------------|
| Agent Position            | 5 x 5       | Indicates the coordinates where the agent is currently located.                                                      |
| Agent Resource            | 2 x 2       | Indicates the resources the agent possesses. Since there are 2 types of resources, each resource's possession status must be indicated. For example, [0, 1] means the agent has only the second resource. |
| Date type                 | 2           | Indicates the information of resources that can be picked up in the current time zone. If [0], only resource 1 can be picked up; if [1], only resource 2 can be picked up. |
| Project-Task-Resource status | (3 x 3) ^ 4 | Indicates the status of each task resource in the project. The resources can have statuses of 0: Not Ready, 1: Ready, 2: Completed. Since there are 2 types of resources, a (3 x 3) state space is needed. This must be recorded for each task in each project, with 2 projects and 2 tasks each, requiring a total of 4 pairs. For example, if the statuses are as follows: \n[0, 1]: (Resource 2 of Project 1, Task 1 is ready) \n[1, 1]: (Both resources of Project 1, Task 2 are ready) \n[2, 2]: (Project 2, Task 1 is completed) \n[0, 0]: (Resources of Project 2, Task 2 are not ready) |

#### Complex DQN
