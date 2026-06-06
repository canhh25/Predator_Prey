import time
from agents.prey_algo import Prey 
from agents.predator_algo import Predator
from env.worldmap import WorldMap
import pygame
def main():
    print("Starting Predator-Prey Simulation...")
    env = WorldMap()
    obs, infos = env.reset()
    env.render()
    prey_agent = Prey()
    predator_agents = {agent: Predator(agent) for agent in env.agents}
    step_count = 0
    ack_rewards = {agent: 0.0 for agent in env.agents}
    clock = pygame.time.Clock()
    while env.agents:
        step_count += 1
        actions = {}
        for agent in env.agents:
            actions[agent] = predator_agents[agent].get_action(obs[agent])
        actions["prey"] = prey_agent.get_action(obs[env.agents[0]])
        obs, rewards, terminations, truncations, infos = env.step(actions)
        for agent, r in rewards.items():
            ack_rewards[agent] += r
        env.render()
        clock.tick(30) 
    print(f"Game over! Total steps: {step_count}")
    for agent, r in rewards.items():
            print(f"  - {agent}: {r}")
if __name__ == "__main__":
    main()