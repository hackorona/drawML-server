from flask import Flask, request, session
from flask_socketio import SocketIO
from flask_socketio import join_room, emit
# ========= Project Constants =========== #

ROOM_ID = 'roomId'
JOIN = 'join'
IDENTIFY = 'identify'
USERNAME = 'username'
IMAGE_MATRIX = 'matrix'
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

ml_module = ''  # todo import ML module


def find_room(user_name):
    keys = [key for key, value in room_map.items() if user_name in value]
    if len(keys) == 0:
        return None
    return keys[0]


def get_random_def():
    return "Dummy def"
    # return ml_module.get_random_def()  # todo import


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
    room_targets[room_name] = target
    emit(CHALLENGE, {DEFINITION: target}, room=room_name)


@socketio.on(JOIN)
def client_join(data):
    """Add new client to its room"""
    session_id = request.sid
    try:
        room_name = data[ROOM_ID]  # Get client room id
    except KeyError:
        print("bad request format")
        return '-1'
    join_room(room=room_name, sid=session_id)  # Add client to its room

    if session_id not in room_map:
        room_map[session_id] = []
    room_map[session_id].append(session_id)  # Save in room map
    print(session_id + ' has entered the room ' + str(room_name))


def check_answer(answer, room):
    target = room_targets[room]
    # Todo check if answer is close enough to target
    pass


@socketio.on(IDENTIFY)
def identify_image(data):
    """identify user image"""
    try:  # Parse request
        matrix = data[IMAGE_MATRIX]
        user_name = data[USERNAME]
    except KeyError:
        print("bad request format")
        return '-1'

    over = False
    room_name = find_room(user_name=user_name)
    if room_name is None:
        print("bad username")
        return '-1'

    # answer = ml_module.predict(matrix)  # todo import
    # over = check_answer(answer, room_name)  # Todo implement

    if over:  # Announce game over
        data = {WINNER: user_name}
        emit(GAME_OVER, data, room=room_name)


if __name__ == '__main__':
    room_map = {}  # Initialize
    socketio.run(app)