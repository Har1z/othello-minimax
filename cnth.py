import random
import numpy as np # for faster array manipulation -robby
import skfuzzy as fuzz # for fuzzy logic -robby
from skfuzzy import control as ctrl # for fuzzy logic control system -robby
import sys # for exiting the program -robby
import time # for delay effect -robby

EMPTY = 0
BLACK = 1 # Player 1
WHITE = 2 # Player 2
SIZE = 8

# Mapping untuk tampilan di terminal -robby
PLAYER_NAMES = {EMPTY: ".", BLACK: "●", WHITE: "○"} 
OTHER_PLAYER = {BLACK: WHITE, WHITE: BLACK}

# papan othello dengan nilai bobot untuk evaluasi -robby
BASE_WEIGHTS = np.array([
    [100, -20, 10,  5,  5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [ 10,  -2, -1, -1, -1, -1,  -2,  10],
    [  5,  -2, -1, -1, -1, -1,  -2,   5],
    [  5,  -2, -1, -1, -1, -1,  -2,   5],
    [ 10,  -2, -1, -1, -1, -1,  -2,  10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [100, -20, 10,  5,  5, 10, -20, 100],
])
# Global Variable untuk menyimpan output Fuzzy terakhir -robby
current_corner_weight = 100 
# Definisi Variable Input/Output -robby
game_progress = ctrl.Antecedent(np.arange(0, 101, 1), 'game_progress')
corner_importance = ctrl.Consequent(np.arange(0, 201, 1), 'corner_importance')
# Aturan game bahwa semakin ke akhir game, corner control jadi semakin penting -robby
game_progress['early'] = fuzz.trimf(game_progress.universe, [0, 0, 50])
game_progress['mid']   = fuzz.trimf(game_progress.universe, [20, 50, 80])
game_progress['late']  = fuzz.trimf(game_progress.universe, [50, 100, 100])

corner_importance['normal'] = fuzz.trimf(corner_importance.universe, [0, 50, 100])
corner_importance['high']   = fuzz.trimf(corner_importance.universe, [50, 100, 150])
corner_importance['critical'] = fuzz.trimf(corner_importance.universe, [100, 200, 200])

rule1 = ctrl.Rule(game_progress['early'], corner_importance['high'])
rule2 = ctrl.Rule(game_progress['mid'], corner_importance['critical'])
rule3 = ctrl.Rule(game_progress['late'], corner_importance['critical'])
# Mesin aturan fuzzy logic - robby
strategy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
strategy_sim = ctrl.ControlSystemSimulation(strategy_ctrl)

def update_fuzzy_strategy(board): # Function to trigger fuzzy calculation -robby
    global current_corner_weight
    filled = np.count_nonzero(board)
    percent = (filled / 64) * 100
    try:
        strategy_sim.input['game_progress'] = percent
        strategy_sim.compute()
        current_corner_weight = strategy_sim.output['corner_importance']
    except:
        current_corner_weight = 100

# gw matiin biar g bentrok -robby
# def copy_board(board):
#     return [row[:] for row in board] # why do we needed it? cuz minimax "would try and simulating" many possibilities and we need to copy the board so it doesnt broke the current game state CMIIW -haris

def create_state():   # gw ganti pake numpy - robby
    board = np.zeros((8,8), dtype=int) # board 8x8 pake numpy -robby
    # set posisi awal hitam dan putih
    board[3,3], board[4,4] = WHITE, WHITE 
    board[3,4], board[4,3] = BLACK, BLACK 
    # gw ganti ini menggunakan lib numpy biar lebih cepat - robby
    return {
        "board": board,
        "nextPlayerToMove": BLACK,
        "depth": 3 # default depth
    }

def clone_state(state):
    # why do we needed it? cuz minimax "would try and simulating" many possibilities -haris
    return {
        "board": np.copy(state["board"]), # fixed for numpy array -robby
        "nextPlayerToMove": state["nextPlayerToMove"],
        "depth": state.get("depth", 3)
    }

def is_on_board(r, c): # Helper function -robby
    return 0 <= r < SIZE and 0 <= c < SIZE

# Gw matiin nyoba ganti logic nya -robby
# def state_to_string(state):
#     string_board = []
#     string_board.append("  #-----------------#")
#     row_num = 8
#     for row in state["board"]:
#         cells = " ".join(PLAYER_NAMES[c] for c in row)
#         string_board.append(f"{row_num} | {cells} |")
#         row_num -= 1
#     string_board.append("  #-----------------#")
#     string_board.append("    A B C D E F G H")
#     return "\n".join(string_board)


def state_to_string(state): # aku ubah ya bwang ui nya -robby
    string_board = []
    string_board.append("   A B C D E F G H") 
    string_board.append("  +---------------+")
    row_num = 0
    for row in state["board"]:  # aku ganti biar 0-7 bukan 8-1 -robby
        cells = " ".join(PLAYER_NAMES[c] for c in row) # Fixed spacing -robby
        string_board.append(f"{row_num} |{cells}|") # Fixed spacing -robby
        row_num += 1
    string_board.append("  +---------------+")
    return "\n".join(string_board)

def coordinate_translator(x, y): # cuz why not? -haris
    # Fixed logic to match standard Othello board (A-H, 0-7) -robby
    x_translate = ["A", "B", "C", "D", "E", "F", "G", "H"]
    return f"{x_translate[x]}{y}"

# nyoba pake perhitungan score yang baru dengan numpy -robby
# def score(state):
#     s = 0
#     for row in state["board"]:
#         for cell in row:
#             if cell == PLAYER1:
#                 s += 1
#             elif cell == PLAYER2:
#                 s -= 1
#     return s

def print_score(state): # maya are responsible for writing this
    board = state["board"]
    o_score = np.count_nonzero(board == BLACK) # Fixed using numpy count -robby
    x_score = np.count_nonzero(board == WHITE)
    print(f"Score --> Hitam ({PLAYER_NAMES[BLACK]}): {o_score} | Putih ({PLAYER_NAMES[WHITE]}): {x_score}")

def generate_moves(state, player=None): # learned -haris
    # Optimized using Numpy logic for performance -robby
    if player is None:
        player = state["nextPlayerToMove"]

    board = state["board"]
    moves = []
    opponent = OTHER_PLAYER[player]
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                  (0, 1), (1, -1), (1, 0), (1, 1)]
    
    # Check valid moves logic (adapted for numpy) -robby
    rows, cols = np.where(board == EMPTY)
    for r, c in zip(rows, cols):
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            found_opp = False
            while is_on_board(nr, nc):
                if board[nr, nc] == opponent:
                    found_opp = True
                elif board[nr, nc] == player:
                    if found_opp:
                        moves.append((player, c, r)) # Format: (Player, X/Col, Y/Row)
                    break
                else:
                    break
                nr += dr
                nc += dc
            # makes sure there's no duplicated move with same coordinates ay? -haris
            if len(moves) > 0 and moves[-1][1] == c and moves[-1][2] == r:
                break 
    return moves

def apply_move(state, move):
    # Logic updated to handle Numpy Board -robby
    if move is None:
        state["nextPlayerToMove"] = OTHER_PLAYER[state["nextPlayerToMove"]]
        return

    player, x, y = move # x=col, y=row
    board = state["board"]
    
    board[y, x] = player # Numpy indexing [row, col] -robby
    state["nextPlayerToMove"] = OTHER_PLAYER[player]

    dirs = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

    for dx, dy in dirs:
        nx, ny = x + dx, y + dy
        to_flip = []
        while is_on_board(ny, nx):
            if board[ny, nx] == OTHER_PLAYER[player]:
                to_flip.append((nx, ny))
            elif board[ny, nx] == player:
                for fx, fy in to_flip:
                    board[fy, fx] = player
                break
            else:
                break
            nx += dx
            ny += dy

def apply_move_cloning(state, move):
    new_state = clone_state(state)
    apply_move(new_state, move)
    return new_state

# code nya gw matiin biar g bentrok sama fuzzy logic baru -robby
#Fuzzy Logic maybe? -ipan
# def mobility(state, player):
#     return len(generate_moves(state, player))

# def corner_control(state, player):
#     board = state["board"]
#     size = state["boardSize"]
#     corners = [(0,0), (0,size-1), (size-1,0), (size-1,size-1)]
#     return sum(1 for x, y in corners if board[y][x] == player)

# def fuzzify(value, min_val, max_val):
#     if value <= min_val:
#         return 0.0
#     if value >= max_val:
#         return 1.0
#     return (value - min_val) / (max_val - min_val)

# def fuzzy_evaluation(state):
#     material = score(state)
#     mob = mobility(state, PLAYER1) - mobility(state, PLAYER2)
#     corner = corner_control(state, PLAYER1) - corner_control(state, PLAYER2)
#     material_f = fuzzify(material, -20, 20)
#     mobility_f = fuzzify(mob, -10, 10)
#     corner_f = fuzzify(corner, -4, 4)
#     return (material_f * 0.4 + mobility_f * 0.3 + corner_f * 0.3) * 100

# Pengganti fuzzy evaluation dengan Logic Baru -robby
def evaluate_board(state, player):
    board = state["board"]
    opponent = OTHER_PLAYER[player]
    
    # Update bobot pojok menggunakan hasil Fuzzy Logic -robby
    current_weights = np.copy(BASE_WEIGHTS)
    corners = [(0,0), (0,7), (7,0), (7,7)]
    for r, c in corners:
        current_weights[r, c] = current_corner_weight 
        
    # Hitung skor posisi menggunakan Numpy Masking (Cepat) -robby
    player_score = np.sum(current_weights[board == player])
    opp_score = np.sum(current_weights[board == opponent])
    
    return player_score - opp_score

# minimax baru dengan alpha beta pruning - robby
def minimax_ab(state, depth, alpha, beta, maximizing_player):
    moves = generate_moves(state, state["nextPlayerToMove"])
    
    # jika depth habis atau tidak ada langkah valid - robby
    if depth == 0 or not moves:
        player_check = state["nextPlayerToMove"] if maximizing_player else OTHER_PLAYER[state["nextPlayerToMove"]]
        return evaluate_board(state, player_check), None

    if maximizing_player: # bot mencoba memaksimalkan score nya
        max_eval = -np.inf
        best_move = None
        for move in moves:
            new_state = apply_move_cloning(state, move)
            eval_val, _ = minimax_ab(new_state, depth-1, alpha, beta, False)
            
            if eval_val > max_eval:
                max_eval = eval_val
                best_move = move
            alpha = max(alpha, eval_val)
            if beta <= alpha: break # Pruning logic -robby
        return max_eval, best_move
        
    else: # lawan mencoba meminimalkan score AI
        min_eval = np.inf
        best_move = None
        for move in moves:
            new_state = apply_move_cloning(state, move)
            eval_val, _ = minimax_ab(new_state, depth-1, alpha, beta, True)
            
            if eval_val < min_eval:
                min_eval = eval_val
                best_move = move
            beta = min(beta, eval_val)
            if beta <= alpha: break # Pruning logic -robby
        return min_eval, best_move


# gw coba negamax nya diganti sama alpha beta minimax -robby
# def negamax(state, depth): # idk i just trying things i found in the internet, but this kinda fire..
#     moves = generate_moves(state)
#     if depth == 0 or not moves:
#         return fuzzy_evaluation(state), None # changed to fuzzy evaluation hehe -ipan
    
#     best_score = -9999
#     best_move = None
#     for move in moves:
#         new_state = apply_move_cloning(state, move)
#         score_rating, _ = negamax(new_state, depth - 1)
#         score_rating = -score_rating
#         if score_rating > best_score:
#             best_score = score_rating
#             best_move = move
#     return best_score, best_move

# def minimax (state, depth):
#     moves = generate_moves(state)
#     if depth == 0 or not moves: # end of depth
#         return score(state), None
#     current_player = state["nextPlayerToMove"]
#     if current_player == PLAYER1:
#         best_score = -9999
#         best_move = None
#         for move in moves:
#             new_state = apply_move_cloning(state, move)
#             eval_score, _ = minimax(new_state, depth - 1)
#             if eval_score > best_score:
#                 best_score = eval_score
#                 best_move = move
#         return best_score, best_move
#     else:
#         best_score = 9999
#         best_move = None
#         for move in moves:
#             new_state = apply_move_cloning(state, move)
#             eval_score, _ = minimax(new_state, depth - 1)
#             if eval_score < best_score:
#                 best_score = eval_score
#                 best_move = move
#         return best_score, best_move

# menentukan akhir game
def game_over(state):
    # cek apakah kedua player tidak punya langkah valid -robby
    p1_moves = generate_moves(state, BLACK)
    p2_moves = generate_moves(state, WHITE)
    return not p1_moves and not p2_moves
# menentukan pemenang
def winner(state):
    board = state["board"]
    b = np.count_nonzero(board == BLACK)
    w = np.count_nonzero(board == WHITE)
    if b > w: return "HITAM (Black)"
    elif w > b: return "PUTIH (White)"
    return "DRAW"
# untuk player atau user
def human_player(state):
    moves = generate_moves(state)
    if not moves: return None

    # hard to read, translate to chess coordinate later -haris
    # Translated using coordinate_translator -robby
    print("Langkah Valid:")
    for i, m in enumerate(moves):
        print(f"[{i}] -> {coordinate_translator(m[1], m[2])}")

    while True:
        try:
            idx = int(input(f"Pilih langkah (0-{len(moves)-1}): "))
            if 0 <= idx < len(moves):
                return moves[idx]
            else: print("Angka di luar jangkauan.")
        except:
            print("Invalid input")

def bot_random(state):
    moves = generate_moves(state)
    return random.choice(moves) if moves else None

# bot minimax dan negamax gw matikan -robby
# def bot_negamax(state):
#     _, move = negamax(state, depth)
#     return move

# def bot_minimax(state):
#     _, move = minimax(state, depth)
#     return move


def bot_dewa(state): # bot baru -robby
    # Update Fuzzy Logic variables before thinking -robby
    update_fuzzy_strategy(state["board"]) 
    
    print(f"AI Thinking... (Depth: {state['depth']}, CornerWeight: {current_corner_weight:.0f})")
    
    # Call Minimax Alpha Beta -robby
    _, move = minimax_ab(state, state['depth'], -np.inf, np.inf, True)
    
    time.sleep(0.8) # aesthetic delay -robby
    return move
#  untuk memilih player - robby
def create_player(name):
    if name == "human": return human_player
    if name == "random": return bot_random
    if name == "dewa": return bot_dewa # Combined logic -robby
    return bot_random
# main game loop - robby
def play_game(initial_state, player1_func, player2_func):
    state = clone_state(initial_state)
    players = {BLACK: player1_func, WHITE: player2_func}
    
    while not game_over(state):
        current_p = state["nextPlayerToMove"]
        print("\n" + "="*30)
        print(f"Giliran: {PLAYER_NAMES[current_p]}") # Display current player
        print(state_to_string(state))
        print_score(state)

        move = players[current_p](state)
        
        if move:
            print(f"Jalan ke: {coordinate_translator(move[1], move[2])}") # Translate coordinates for display -robby
            state = apply_move_cloning(state, move)
        else: 
            print("Tidak ada jalan. Pass.")
            state["nextPlayerToMove"] = OTHER_PLAYER[current_p]
            time.sleep(1)

    print("\n*** GAME OVER ***")
    print(state_to_string(state))
    print(f"Pemenang: {winner(state)}")
# menu utama - roby
if __name__ == "__main__": # main menu
    # when im writing this i think i need to update the menu before game started -haris
    print("=== OTHELLO FUZZY+MINIMAX ===")
    
    player_2_input = input("Pilih Lawan (human/random/dewa): ").lower() #bot dewa pake minimax + fuzzy logic -robby
    
    depth_val = 3
    if player_2_input == "dewa":
        while True:
            try:
                # why? cause i have tried to run 100 depth and suddenly it got crashed. -haris
                # Adjusted limit for safety -robby
                d = int(input("Masukkan depth (2-5): "))
                if 2 <= d <= 5:
                    depth_val = d
                    break
                else: print("Depth harus 2-5.")
            except:
                print("Invalid number.")
    
    state = create_state()
    state["depth"] = depth_val # Store depth in state -robby
    
    # Player 1 selalu Human (Hitam), Player 2 sesuai pilihan (Putih)
    play_game(state, create_player("human"), create_player(player_2_input))