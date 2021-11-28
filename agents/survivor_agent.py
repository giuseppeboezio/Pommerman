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

        # path to follow to put a bomb (each element is a tuple)
        self.path = []

        # flag to keep the intention of the agent (reach target position, put a bomb)
        self.target_defined = False

    def act(self, obs, action_space):

        ag_pos = obs['position']

        if not self.target_defined:

            # use heuristics to find the target and Dijkstra's algorithm to find the path

            # generation of the target position
            b_matrix = m1.get_destroyed_boxes(obs)
            d_matrix, nodes = m1.get_distances(obs)
            final_matrix = m1.combine_masks(b_matrix, d_matrix)

            # avoid to stay in the same position
            final_matrix[ag_pos[0], ag_pos[1]] = -np.inf
            target_position = np.unravel_index(np.argmax(final_matrix, axis=None), final_matrix.shape)

            # generation of the path to follow

            for node in nodes:

                position = node.get_position()
                if position[0] == target_position[0] and position[1] == target_position[1]:

                    parent = node.get_parent()
                    node_pth = node
                    # obtaining the path using results of Dijkstra's algorithm
                    while parent is not None:

                        self.path.append(node_pth.get_position())
                        node_pth = parent
                        parent = node_pth.get_parent()

                    break

            # reverse the element to get the path from the root node to the next one
            self.path.reverse()

            self.target_defined = True

        print("Path:")
        for elem in self.path:
            print(f"{elem}")

        # checking whether the position is the target
        if len(self.path) == 0:

            self.target_defined = False
            action = constants.Action.Bomb

        else:

            # getting position of next element and checking whether it is safe to move there
            next_position = self.path[0]

            input_raw = preprocess_state_avoidance(obs['board'], next_position)
            input_net = torch.from_numpy(input_raw)
            # casting of the type and reshaping of the tensor
            input_net = input_net.float()
            input_net = torch.reshape(input_net, (1, 1, constants.BOARD_SIZE, constants.BOARD_SIZE))
            print(input_net.shape)
            # input_net /= torch.max(input_net)

            with torch.no_grad():

                prob = self.model(input_net)

            if prob > 0:

                self.path.pop(0)

                if ag_pos[0] == next_position[0] and ag_pos[1] < next_position[1]:
                    action = constants.Action.Right
                elif ag_pos[0] == next_position[0] and ag_pos[1] < next_position[1]:
                    action = constants.Action.Left
                elif ag_pos[1] == next_position[1] and ag_pos[0] < next_position[0]:
                    action = constants.Action.Down
                else:
                    action = constants.Action.Up

            else:

                action = constants.Action.Stop

            print(f"Action: {action}")

            return action


def main():

    path = "C:/Users/boezi/PycharmProjects/Pommerman/heuristic_modules/module_2/model_weights.pth"
    model = network.DiscriminatorNet(16)
    model.load_state_dict(torch.load(path))
    threshold = 0.6

    my_agent = SurvivorAgent(model, threshold)

    config = 'PommeFFACompetition-v0'
    agent_list = [my_agent, agents.RandomAgent(), agents.SimpleAgent(), agents.RandomAgent()]
    game_state_file = None

    # Make the "Free-For-All" environment using the agent list
    env = make(config, agent_list, game_state_file)

    state = env.reset()
    done = False
    while not done:

        print("Board for agent0")
        print(state[0]['board'])
        env.render()
        time.sleep(0.5)

        actions = env.act(state)
        state, reward, done, info = env.step(actions)


if __name__ == '__main__':
    main()