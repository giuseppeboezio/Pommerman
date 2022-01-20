from pommerman import constants
from pommerman.constants import Item
import colorama


class ExplosionField:

    def __init__(self, position, life):
        super(ExplosionField, self).__init__()
        self.position = position
        self.life = life
        self.field = []

    def get_position(self):
        return self.position

    def get_life(self):
        return self.life

    def set_danger_pos(self, positions):
        self.field = list(positions)

    def get_dang_pos(self):
        return self.field


def get_dangerous_positions(board, blast_strength, life, position):
    # positions affected by the blast strength
    left = [(position[0], position[1] - j) for j in range(1, blast_strength + 1)]
    right = [(position[0], position[1] + j) for j in range(1, blast_strength + 1)]
    up = [(position[0] - j, position[1]) for j in range(1, blast_strength + 1)]
    down = [(position[0] + j, position[1]) for j in range(1, blast_strength + 1)]
    positions = left + right + up + down
    # removing positions outside the board
    acceptable_pos = list(positions)
    for pos in positions:
        if pos[0] < 0 or pos[1] < 0 or pos[0] > constants.BOARD_SIZE - 1 or \
                pos[1] > constants.BOARD_SIZE - 1:
            acceptable_pos.remove(pos)
    # removing positions not affected by the flames because there is a wooden or destructible walls
    # left direction
    left = [pos for pos in acceptable_pos if pos[1] < position[1]]
    for elem in left:
        # checking whether the element on the position is a rigid wall or a wooden wall
        if board[elem[0], elem[1]] == Item.Rigid.value or board[elem[0], elem[1]] == Item.Wood.value:
            remove_list = [pos for pos in left if pos[1] < elem[1]]
            # removing positions beyond the rigid wall because they cannot be reached by the flames
            for item in remove_list:
                if item in acceptable_pos:
                    acceptable_pos.remove(item)
    # right direction
    right = [pos for pos in acceptable_pos if pos[1] > position[1]]
    for elem in right:
        # checking whether the element on the position is a rigid wall
        if board[elem[0], elem[1]] == Item.Rigid.value or board[elem[0], elem[1]] == Item.Wood.value:
            remove_list = [pos for pos in right if pos[1] > elem[1]]
            # removing positions beyond the rigid wall because they cannot be reached by the flames
            for item in remove_list:
                if item in acceptable_pos:
                    acceptable_pos.remove(item)
    # up direction
    up = [pos for pos in acceptable_pos if pos[0] < position[0]]
    for elem in up:
        # checking whether the element on the position is a rigid wall
        if board[elem[0], elem[1]] == Item.Rigid.value or board[elem[0], elem[1]] == Item.Wood.value:
            remove_list = [pos for pos in up if pos[0] < elem[0]]
            # removing positions beyond the rigid wall because they cannot be reached by the flames
            for item in remove_list:
                if item in acceptable_pos:
                    acceptable_pos.remove(item)
    # down direction
    down = [pos for pos in acceptable_pos if pos[0] > position[0]]
    for elem in down:
        # checking whether the element on the position is a rigid wall
        if board[elem[0], elem[1]] == Item.Rigid.value or board[elem[0], elem[1]] == Item.Wood.value:
            remove_list = [pos for pos in down if pos[0] > elem[0]]
            # removing positions beyond the rigid wall because they cannot be reached by the flames
            for item in remove_list:
                if item in acceptable_pos:
                    acceptable_pos.remove(item)
    # adding positions to the main list
    dang_pos = acceptable_pos

    explosion_field = ExplosionField(position, life)
    explosion_field.set_danger_pos(dang_pos)

    return explosion_field


def color_sign(x):
    if x == 0:
        c = colorama.Fore.LIGHTBLACK_EX
    elif x == 1:
        c = colorama.Fore.BLACK
    elif x == 2:
        c = colorama.Fore.BLUE
    elif x == 3:
        c = colorama.Fore.RED
    elif x == 4:
        c = colorama.Fore.RED
    elif x == 10:
        c = colorama.Fore.YELLOW
    elif x == 11:
        c = colorama.Fore.CYAN
    elif x == 12:
        c = colorama.Fore.GREEN
    elif x == 13:
        c = colorama.Fore.MAGENTA
    else:
        c = colorama.Fore.WHITE
    x = '{0: <2}'.format(x)
    return f'{c}{x}{colorama.Fore.RESET}'


def show_board(board):

    for i in range(board.shape[0]):
        print("[", end='\t')
        for j in range(board.shape[1]):
            print(color_sign(int(board[i,j])), end='\t')
        print(']')
