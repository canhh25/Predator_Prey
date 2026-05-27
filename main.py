import time
def test_random_agents():
    print("Testing random agents in the gridworld environment...")
    time.sleep(1) 
    from env.gridworld import PreyAlgoEnv
    env = PreyAlgoEnv()
    obs, infos = env.reset()
    print("Initial Observations:")
    for agent, ob in obs.items():
        print(f"{agent}: {ob}")
    env.render()
    step_count = 0
    total_reward = {agent: 0.0 for agent in env.agents}
    while env.agents:
            step_count += 1
            print(f"\n--- Bước {step_count} ---")
            
            # Bốc ngẫu nhiên 1 hành động (0, 1, 2, 3, 4) cho từng con cá mập
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            
            # In ra để xem chúng nó định đi hướng nào
            print("Hành động ngẫu nhiên:", actions)
            
            # Truyền hành động vào môi trường
            observations, rewards, terminations, truncations, infos = env.step(actions)
            for agent, reward in rewards.items():
                total_reward[agent] += reward
            # Vẽ bàn cờ ra màn hình
            env.render()
            
            time.sleep(2)

    print(f"\n=== GAME OVER SAU {step_count} BƯỚC! ===")
    print("Phần thưởng cuối cùng (Rewards):", total_reward)

if __name__ == "__main__":
    test_random_agents()