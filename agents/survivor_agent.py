import torch
from pommerman import make
from pommerman import agents, constants
from heuristic_modules import module_1 as m1
from heuristic_modules.module_2.module_2 import preprocess_state_avoidance
from heuristic_modules import network
import numpy as np
import time


class SurvivorAgent(agents.BaseAgent):

    def __init__(self, model, threshold):
        super(SurvivorAgent, self).__init__()

        # network to predict the probability to avoid a bomb and threshold to do a safe action
        self.model = model
        self.threshold = threshold

        # path to follow to put a bomb
        self.path = []

        # flag to keep the intention of the agent (reach target position, put a bomb)
        self.target_defined = False

    def act(self, obs, action_space):

        if not self.target_defined:

            # use heuristics to find the target and Dijkstra's algorithm to find the path

            # generation of the target position
            b_matrix = m1.get_destroyed_boxes(obs)
            d_matrix, nodes = m1.get_distances(obs)
            final_matrix = m1.combine_masks(b_matrix, d_matrix)
            target_position = np.unravel_index(np.argmax(final_matrix, axis=None), final_matrix.shape)

            # generation of the path to follow

            for node in nodes:

                position = node.get_position()
                if position[0] == target_position[0] and position[1] == target_position[1]:

                    parent = node.get_parent()
                    # obtaining the path using results of Dijkstra's algorithm
                    while parent is not None:

                        self.path.append(node)
                        node = parent

                    break

            # reverse the element to get the path from the root node to the next one
            self.path.reverse()

            # remove first element because it is the current node
            self.path.pop()
            self.target_defined = True

        # checking whether the position is the target
        if len(self.path) == 0:

            self.target_defined = False
            action = constants.Action.Bomb.value

        else:

            # getting position of next element and checking whether it is safe to move there
            next_position = self.path[0].get_position()

            input_raw = preprocess_state_avoidance(obs['board'], next_position)
            input_net = torch.from_numpy(input_raw)

            with torch.no_grad():
                self.model.eval()

                prob = self.model(input_net)

            if prob > self.threshold:

                self.path.pop(0)
                ag_pos = obs['position']

                if ag_pos[0] == next_position[0] and ag_pos[1] < next_position[1]:
                    action = constants.Action.Right.value
                elif ag_pos[0] == next_position[0] and ag_pos[1] < next_position[1]:
                    action = constants.Action.Left.value
                elif ag_pos[1] == next_position[1] and ag_pos[0] < next_position[0]:
                    action = constants.Action.Down.value
                else:
                    action = constants.Action.Up.value

            else:

                action = constants.Action.Stop.value

            return action


def main():

    path = "heuristic_modules/module_2/model_weights.pth"
    model = network.DiscriminatorNet(64)
    model.load_state_dict(torch.load(path))
    threshold = 0.6

    my_agent = SurvivorAgent(model, threshold)

    agent_list = [my_agent, agents.RandomAgent(), agents.SimpleAgent(), agents.RandomAgent()]

    # Make the "Free-For-All" environment using the agent list
    env = make('PommeFFACompetition-v0', agent_list)

    state = env.reset()
    done = False
    while not done:

        env.render()
        time.sleep(0.7)

        actions = env.act(state)
        state, reward, done, info = env.step(actions)


if __name__ == '__main__':
    main()