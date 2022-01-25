from agent import PlannerAgent
from pommerman import agents, make
import time
from pommerman.agents import BaseAgent
from pommerman import constants


# agent which returns always stop as move
class StopAgent(BaseAgent):

    def __init__(self):
        super(StopAgent, self).__init__()

    def act(self, obs, action_space):
        return constants.Action.Stop


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
        time.sleep(1.5)

        actions = env.act(state)
        state, reward, done, info = env.step(actions)


main()