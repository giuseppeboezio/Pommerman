from agent import PlannerAgent
from pommerman import agents, make
import time


def main():

    my_agent = PlannerAgent()
    config = 'PommeFFACompetition-v0'
    agent_list = [my_agent, agents.SimpleAgent(), agents.SimpleAgent(), agents.SimpleAgent()]
    game_state_file = None

    # Make the "Free-For-All" environment using the agent list
    env = make(config, agent_list, game_state_file)

    state = env.reset()
    done = False
    while not done:
        env.render()
        time.sleep(1)

        actions = env.act(state)
        state, reward, done, info = env.step(actions)


main()