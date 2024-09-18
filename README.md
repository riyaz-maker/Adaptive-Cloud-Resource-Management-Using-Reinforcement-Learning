# Adaptive Cloud Resource Management Using Reinforcement Learning

This project uses Reinforcement Learning (PPO) to dynamically optimize cloud resource scaling based on workloads.

## Project Structure
- **environment.py**: Custom cloud resource management environment.
- **train.py**: Training the PPO agent.
- **evaluate.py**: Evaluating the trained PPO agent.
- **requirements.txt**: List of dependencies.

## Setup Instructions

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the agent:
   ```bash
   python train.py
   ```

3. Evaluate the trained agent:
   ```bash
   python evaluate.py
   ```

## Project Components
1. **Cloud Resource Environment**: A custom Gymnasium environment simulating dynamic cloud workloads.
2. **PPO Agent**: Proximal Policy Optimization agent to learn resource scaling strategies.
