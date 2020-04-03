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
# ======================================= #

app = Flask(__name__)
socketio = SocketIO(app)
room_map = {}

ml_module = '' # todo import ML module


def find_room(user_name):
    keys = [key for key, value in room_map.items() if user_name in value]
    if len(keys) == 0:
        return None
    return keys[0]


def get_random_def():
    return ml_module.get_random_def()  # todo import

@socketio.on(JOIN)
def client_join(data):
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


@socketio.on(IDENTIFY)
def identify_image(data):
    """identify user image"""
    try:  # Parse request
        matrix = data[IMAGE_MATRIX]
        user_name = data[USERNAME]
    except KeyError:
        print("bad request format")
        return '-1'

    answer = ml_module.predict(matrix)
    if answer:  # Announce game over
        room_name = find_room(user_name=user_name)
        if room_name is None:
            print("bad username")
            return '-1'
        data = {WINNER: user_name}
        emit(GAME_OVER, data, room=room_name)




@socketio.on('connect')
def connected():
    print("client connected to socket")



if __name__ == '__main__':
    room_map = {}  # Initialize
    socketio.run(app)