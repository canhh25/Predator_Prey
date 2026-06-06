import time
from agents.prey_algo import Prey 
from agents.predator_algo import Predator
from mpe2 import simple_tag_v3
import pygame

def main():
    print("Starting Predator-Prey Simulation...")
    env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=3, num_obstacles=2, continuous_actions=True, render_mode="human",num_agent_neighbors = 2, max_cycles=30)
    obs, infos = env.reset()
    env.render()
    prey_agent = Prey()
    predator_agents = {agent: Predator(agent) for agent in env.agents if "adversary" in agent}    
    step_count = 0
    ack_rewards = {agent: 0.0 for agent in env.agents}
    clock = pygame.time.Clock()
    while env.agents:
        step_count += 1
        actions = {}
        
        for agent in env.agents:
            if "adversary" in agent:
                actions[agent] = predator_agents[agent].get_action(obs[agent])
            else:
                actions[agent] = prey_agent.get_action(obs[agent])
            
        obs, rewards, terminations, truncations, infos = env.step(actions)
        for agent, r in rewards.items():
            ack_rewards[agent] += r
            
        env.render()
        clock.tick(30)
    print(f"Game over! Total steps: {step_count}")
    for agent, r in ack_rewards.items():
            print(f"  - {agent}: {r}")
if __name__ == "__main__":
    main()