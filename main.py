import time
from agents.prey_algo import Prey 
from agents.predator_algo import Predator
from env.gridworld import GridWorldEnv
def main():
    print("Starting Predator-Prey Simulation...")
    env = GridWorldEnv()
    obs, infos = env.reset()
    env.render()
    prey_agent = Prey()
    predator_agents = {agent: Predator() for agent in env.agents}
    step_count = 0
    while env.agents:
        step_count += 1
        print(f"\n--- Bước {step_count} ---")
        actions = {}
        for agent in env.agents:
            actions[agent] = predator_agents[agent].get_action(obs[agent])
        actions["prey"] = prey_agent.get_action(obs[env.agents[0]])
        env.render()
        obs, rewards, terminations, truncations, infos = env.step(actions)
        time.sleep(2) 
    print(f"Game over! Total steps: {step_count}")
if __name__ == "__main__":
    main()