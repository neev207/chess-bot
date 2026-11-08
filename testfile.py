import chess
import random

board = chess.Board()
while True:
    if board.is_game_over():
        print(board.result())
        break
    w_move = input("Enter your move: ")
    if w_move=='-1':
        print("Well played!")
        break
    print("You played:", w_move)
    
    board.push_san(w_move)

    if board.is_game_over():
        print(board.result())
        break
        
    legal_moves = list(board.legal_moves)
    b_move = random.choice(legal_moves)
    
    print("Computer played:", b_move)
    
    board.push(b_move) 
    print(board)
print(board)    
print("Game over!")