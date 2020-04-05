from flask import Flask, request, session
from flask_socketio import SocketIO
from flask_socketio import join_room, emit
import numpy as np
import MLpart
# ========= Project Constants =========== #

ROOM_ID = 'roomId'
JOIN = 'join'
IDENTIFY = 'identify'
PLAYER_NAME = 'playerName'
DRAW_OBJECT = 'drawObject'
GAME_OVER = 'gameOver'
WINNER = 'winner'
START = 'start'
DEFINITION = 'definition'
CHALLENGE = 'challenge'
# ======================================= #

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')
room_map = {}
room_targets = {}

ml_module = MLpart.get_default_model()  # todo import ML module


def find_room(room_name):
    rooms = [key for key in room_map.keys() if room_name in key]
    if len(rooms) == 0:
        return None
    return room_name[0]


def create_matrix(draw_object):
    size = (draw_object['width'], draw_object['width'])
    matrix = np.zeros(size)
    for line in draw_object['lines']:
        for point in line['points']:
            matrix[int(round(point['x'])), int(round(point['y']))] = 1
    return matrix


def get_random_def():
    definition = ml_module.get_random_definition()
    return definition


@socketio.on('connect')
def connect():
    print("client connected to socket")


@socketio.on(START)
def start(data):
    """Send a starting definition to a specific room"""
    try:
        room_name = data[ROOM_ID]  # Get client room id
    except KeyError:
        print("bad request format")
        return '-1'

    target = get_random_def()
    room_map[room_name]['definition'] = target
    emit(CHALLENGE, {DEFINITION: target}, room=room_name)


@socketio.on(JOIN)
def client_join(data):
    """Add new client to its room"""
    session_id = request.sid
    try:
        room_name = data[ROOM_ID]  # Get client room id
        player_name = data[PLAYER_NAME]
    except KeyError:
        print("bad request format")
        return '-1'
    join_room(room=room_name, sid=session_id)  # Add client to its room

    if room_name not in room_map:
        room_map[room_name] = {
            'players': [],
            'definition': None
        }
    room_map[room_name]['players'].append({'session_id': session_id, 'player_name': player_name})  # Save in room map
    print(player_name + ' has entered the room ' + str(room_name))


@socketio.on(IDENTIFY)
def identify_image(data):
    """identify user image"""
    try:  # Parse request
        draw_object = data[DRAW_OBJECT]
        user_name = data[PLAYER_NAME]
        room_name = data[ROOM_ID]
    except KeyError:
        print("bad request format")
        return '-1'

    print(data)
    over = False
    room = room_map[room_name]
    if room is None:
        print("bad room name" + room_name)
        return '-1'

    generated_matrix = create_matrix(draw_object)
    success = ml_module.classify_specific_definition(generated_matrix, room_map[room_name]['definition'])
    # success = True
    if success:  # Announce game over
        data = {WINNER: user_name}
        emit(GAME_OVER, data, room=room_name)


if __name__ == '__main__':
    room_map = {}  # Initialize
    socketio.run(app)
