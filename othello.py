import random

EMPTY = 2
PLAYER1 = 0
PLAYER2 = 1
PLAYER_NAMES = ["O", "0", "."]
OTHER_PLAYER = {PLAYER1: PLAYER2, PLAYER2: PLAYER1}
depth = 3

# game logic stuff
# game core, main logic, function.. idk, i just bad at naming stuff -haris
def copy_board(board):
    return [row[:] for row in board] # why do we needed it? cuz minimax "would try and simulating" many possibilities and we need to copy the board so it doesnt broke the current game state CMIIW -haris


def create_state(board = None, boardSize = 8, nextPlayerToMove = PLAYER1): 
    if board is None:
        board = [[EMPTY] * boardSize for i in range(boardSize)]
        middle = boardSize // 2
        board[middle - 1][middle - 1] = PLAYER1
        board[middle][middle] = PLAYER1
        board[middle - 1][middle] = PLAYER2
        board[middle][middle - 1] = PLAYER2

    return {
        "board": board,
        "boardSize": boardSize,
        "nextPlayerToMove": nextPlayerToMove
    }
    # im not commenting this cause, wdym? it just creating the state did it so hard to understand? -haris


def clone_state(state):
    return create_state(
        copy_board(state["board"]),
        state["boardSize"],
        state["nextPlayerToMove"]
    )


def state_to_string(state):
    string_board = []
    string_board.append("  #-----------------#")
    row_num = 8
    for row in state["board"]:
        cells = " ".join(PLAYER_NAMES[c] for c in row)
        string_board.append(f"{row_num} | {cells} |")
        row_num -= 1

    string_board.append("  #-----------------#")
    string_board.append("    A B C D E F G H")

    return "\n".join(string_board)

def coordinate_translator(x, y): # cuz why not?
    x_translate = ["A", "B", "C", "D", "E", "F", "G", "H"]
    y_translate = [8, 7, 6, 5, 4, 3, 2, 1]

    return f"{x_translate[x]}{y_translate[y]}"

def score(state):
    s = 0
    for row in state["board"]:
        for cell in row:
            if cell == PLAYER1:
                s += 1
            elif cell == PLAYER2:
                s -= 1
    return s

def print_score(state): # maya are responsible for writing this
    o_score = 0
    x_score = 0

    for row in state["board"]:
        for cell in row:
            if cell == PLAYER1:
                o_score += 1
            elif cell == PLAYER2:
                x_score += 1

    print(f"Score --> O: {o_score} | 0: {x_score}")

def generate_moves(state, player=None): # learned
    if player is None:
        player = state["nextPlayerToMove"]

    board = state["board"]
    size = state["boardSize"]
    moves = []

    dirs = [(-1, -1), (0, -1), (1, -1),
            (-1, 0),           (1, 0),
            (-1, 1),  (0, 1),  (1, 1)]

    for y in range(size):
        for x in range(size):
            if board[y][x] != EMPTY:
                continue

            for direction_x, direction_y in dirs:
                current_x, current_y = x + direction_x, y + direction_y # current location of checking bruh wdym?
                found_opponent = False

                while 0 <= current_x < size and 0 <= current_y < size:
                    if board[current_y][current_x] == OTHER_PLAYER[player]:
                        found_opponent = True
                    elif board[current_y][current_x] == player and found_opponent:
                        moves.append((player, x, y))
                        break
                    else:
                        break
                    current_x += direction_x
                    current_y += direction_y

                if moves and moves[-1][1] == x and moves[-1][2] == y: # makes sure there's no duplicated move with same coordinates ay? -haris
                    break

    return moves


def apply_move(state, move):
    if move is None:
        state["nextPlayerToMove"] = OTHER_PLAYER[state["nextPlayerToMove"]]
        return

    player, x, y = move
    board = state["board"]
    size = state["boardSize"]

    board[y][x] = player
    state["nextPlayerToMove"] = OTHER_PLAYER[player]

    dirs = [(-1, -1), (0, -1), (1, -1),
            (-1, 0),           (1, 0),
            (-1, 1),  (0, 1),  (1, 1)]

    for dx, dy in dirs:
        nx, ny = x + dx, y + dy
        to_flip = []

        while 0 <= nx < size and 0 <= ny < size:
            if board[ny][nx] == OTHER_PLAYER[player]:
                to_flip.append((nx, ny))
            elif board[ny][nx] == player:
                for fx, fy in to_flip:
                    board[fy][fx] = player
                break
            else:
                break
            nx += dx
            ny += dy


def apply_move_cloning(state, move):
    new_state = clone_state(state)
    apply_move(new_state, move)
    return new_state

#Fuzzy Logic maybe? -ipan
def mobility(state, player):
    return len(generate_moves(state, player))


def corner_control(state, player):
    board = state["board"]
    size = state["boardSize"]
    corners = [(0,0), (0,size-1), (size-1,0), (size-1,size-1)]
    return sum(1 for x, y in corners if board[y][x] == player)


def fuzzify(value, min_val, max_val):
    if value <= min_val:
        return 0.0
    if value >= max_val:
        return 1.0
    return (value - min_val) / (max_val - min_val)


def fuzzy_evaluation(state):
    material = score(state)
    mob = mobility(state, PLAYER1) - mobility(state, PLAYER2)
    corner = corner_control(state, PLAYER1) - corner_control(state, PLAYER2)

    material_f = fuzzify(material, -20, 20)
    mobility_f = fuzzify(mob, -10, 10)
    corner_f = fuzzify(corner, -4, 4)

    return (material_f * 0.4 + mobility_f * 0.3 + corner_f * 0.3) * 100


def negamax(state, depth): # idk i just trying things i found in the internet, but this kinda fire..
    moves = generate_moves(state)

    if depth == 0 or not moves:
        return fuzzy_evaluation(state), None # changed to fuzzy evaluation hehe -ipan
    
    best_score = -9999
    best_move = None

    for move in moves:
        new_state = apply_move_cloning(state, move)
        score_rating, _ = negamax(new_state, depth - 1)
        score_rating = -score_rating

        if score_rating > best_score:
            best_score = score_rating
            best_move = move

    return best_score, best_move

def minimax (state, depth):
    moves = generate_moves(state)

    if depth == 0 or not moves: # end of depth
        return score(state), None

    current_player = state["nextPlayerToMove"]

    if current_player == PLAYER1:
        best_score = -9999
        best_move = None

        for move in moves:
            new_state = apply_move_cloning(state, move)
            eval_score, _ = minimax(new_state, depth - 1)

            if eval_score > best_score:
                best_score = eval_score
                best_move = move

        return best_score, best_move

    else:
        best_score = 9999
        best_move = None

        for move in moves:
            new_state = apply_move_cloning(state, move)
            eval_score, _ = minimax(new_state, depth - 1)

            if eval_score < best_score:
                best_score = eval_score
                best_move = move

        return best_score, best_move

def game_over(state):
    return not generate_moves(state, PLAYER1) and not generate_moves(state, PLAYER2)


def winner(state):
    s = score(state)
    if s > 0:
        return PLAYER_NAMES[PLAYER1]
    elif s < 0:
        return PLAYER_NAMES[PLAYER2]
    return "DRAW"


# player turn handler
def human_player(state):
    moves = generate_moves(state)
    if not moves:
        return None

    for i, m in enumerate(moves):
        # print(f"{i}: Player {PLAYER_NAMES[m[0]]} -> ({m[1]},{m[2]})") # hard to read, translate to chess coordinate later -haris
        print(f"{i}: Player {PLAYER_NAMES[m[0]]} -> {coordinate_translator(m[1], m[2])}")

    while True:
        try:
            idx = int(input("Choose move (0-9): "))
            return moves[idx]
        except:
            print("Invalid input")


def bot_random(state):
    moves = generate_moves(state)
    return random.choice(moves) if moves else None

def bot_negamax(state):
    _, move = negamax(state, depth)
    return move

def bot_minimax(state):
    _, move = minimax(state, depth)
    return move

def create_player(name):
    if name == "human":
        return human_player
    if name == "random":
        return bot_random
    if name == "minimax":
        return bot_minimax
    if name == "negamax":
        return bot_negamax
    return bot_random


# game loop
def play_game(initial_state, player1, player2):
    state = clone_state(initial_state)
    players = [player1, player2]
    player_index = 0

    while not game_over(state):
        print("\nCurrent state,", PLAYER_NAMES[state["nextPlayerToMove"]], "to move:")
        print(state_to_string(state),"\n")
        print_score(state)

        move = players[player_index](state)
        if move:
            print(f"Player {PLAYER_NAMES[move[0]]} -> ({move[1]},{move[2]})")

        state = apply_move_cloning(state, move)
        player_index = (player_index + 1) % 2

    print("\n*** Final winner:", winner(state), "***")
    print(state_to_string(state))
    print("\nFinal Score:")
    print_score(state)


# ...
player_1 = "minimax"
player_2 = input("silahkan pilih lawan (human/random/minimax/negamax): ").lower() # when im writing this i think i need to update the menu before game started -haris

if player_2 == "minimax" or player_2 == "negamax":
    while True:
        try :
            depth = int(input("masukkan depth (2 - 5): "))
            if 2 <= depth <= 5:
                break
            elif depth > 5:
                print("please, are you trying to crash my device \n") # why? cause i have tried to run 100 depth and suddenly it got crashed.
            else:
                print("Depth harus di kisaran 2 sampai 5.\n")
        except:
            print("Invalid input\n")

state = create_state()
play_game(
    state,
    create_player(player_1),
    create_player(player_2)
)