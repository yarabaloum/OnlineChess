from flask import Flask, request, jsonify
import os
import uuid
import cv2
import numpy as np
import base64
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model
model = load_model("model_32.h5")
image_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)
ch = ['h', 'g', 'f', 'e', 'd', 'c', 'b', 'a']
clas = ['B_Bishop', 'B_King', 'B_Knight', 'B_Pawn', 'B_Queen', 'B_Rook',
        'Empty', 'W_Bishop', 'W_King', 'W_Knight', 'W_Pawn', 'W_Queen', 'W_Rook']

def predict_image(img_array):
    img_array = cv2.resize(img_array, (32, 32))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = image_gen.preprocessing_function(img_array)
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return clas[predicted_class]

def get_chessboard_state(image, rows, cols, square_width, square_height):
    chessboard_state = {}
    for i in range(rows):
        for j in range(cols):
            x_start, y_start = j * square_width, i * square_height
            x_end, y_end = x_start + square_width, y_start + square_height
            square = image[y_start:y_end, x_start:x_end]
            piece = predict_image(square)
            square_name = f"{ch[j]}{i+1}"
            chessboard_state[square_name] = piece
    return chessboard_state

def detect_moves(state_1, state_2):
    from_squares = []
    to_squares = []
    capture = False

    for square, piece_1 in state_1.items():
        piece_2 = state_2.get(square)
        if piece_1 != "Empty" and piece_2 == "Empty":
            from_squares.append((square, piece_1))

    for square, piece_2 in state_2.items():
        piece_1 = state_1.get(square)
        if piece_1 != piece_2 and piece_2 != "Empty" and not any(p == piece_2 for _, p in from_squares):
            continue
        if piece_1 != piece_2 and piece_2 != "Empty":
            to_squares.append((square, piece_2))
            if piece_1 != "Empty":
                capture = True

    moves = []
    if len(from_squares) == 1 and len(to_squares) == 1:
        from_square, piece = from_squares[0]
        to_square, _ = to_squares[0]
        move_text = f"{piece} {from_square} → {to_square}"
        if capture:
            move_text += " (capture)"
        moves.append((from_square, to_square, move_text))

    elif len(from_squares) == 2 and len(to_squares) == 2:
        for square, piece in from_squares:
            if "King" in piece:
                king_from = square
                for to_sq, to_piece in to_squares:
                    if "King" in to_piece:
                        king_to = to_sq
                        move_text = f"{piece} {king_from} → {king_to} (castling)"
                        moves.append((king_from, king_to, move_text))
                        break
    return moves

def process_video_with_movement_and_labels(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    rows, cols = 8, 8
    square_height = height // rows
    square_width = width // cols

    prev_state = None
    move_list = []
    current_move_display = None
    move_display_frames = int(fps * 3)
    move_display_counter = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 10 == 0:
            current_state = get_chessboard_state(frame, rows, cols, square_width, square_height)
            if prev_state is not None:
                moves = detect_moves(prev_state, current_state)
                if moves:
                    for _, _, move_text in moves:
                        move_list.append(move_text)
                        current_move_display = move_text
                        move_display_counter = move_display_frames
                        print("Detected move:", move_text)
            prev_state = current_state

        # Draw labels on every square (grid + class name)
        for i in range(rows):
            for j in range(cols):
                x_start, y_start = j * square_width, i * square_height
                x_end, y_end = x_start + square_width, y_start + square_height
                square = frame[y_start:y_end, x_start:x_end]
                square_rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
                label = predict_image(square_rgb)

                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 1)
                cv2.putText(frame, label, (x_start + 3, y_start + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        if move_display_counter > 0 and current_move_display:
            cv2.putText(frame, current_move_display, (10, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            move_display_counter -= 1

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    return move_list

@app.route('/upload', methods=['POST'])
def upload_video():
    video_file = request.files['video']
    filename = str(uuid.uuid4()) + ".mp4"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, "output_" + filename)

    video_file.save(input_path)
    print("Processing:", input_path)
    move_list = process_video_with_movement_and_labels(input_path, output_path)

    with open(output_path, 'rb') as f:
        encoded_video = base64.b64encode(f.read()).decode('utf-8')

    return jsonify({
        "video": encoded_video,
        "moves": move_list,
        "filename": os.path.basename(output_path)
    })

if __name__ == '__main__':
    app.run(debug=True)























    