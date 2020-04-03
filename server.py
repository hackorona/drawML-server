from flask import Flask, request, session
from flask_socketio import SocketIO
from flask_socketio import join_room, leave_room
# ========= Project Constants =========== #

ROOM_ID = 'roomId'
JOIN = 'join'
IDENTIFY = 'identify'
USERNAME = 'username'
IMAGE_MATRIX = 'matrix'
# ======================================= #

app = Flask(__name__)
socketio = SocketIO(app)
room_map = {}

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

socketio.on(IDENTIFY)
def identify_image(data):
    try:
        matrix = data[IMAGE_MATRIX]
        user_name = data[USERNAME]
    except KeyError:
        print("bad request format")
        return '-1'

    



@socketio.on('connect')
def connected():
    print("client connected to socket")



if __name__ == '__main__':
    room_map = {}  # Initialize
    socketio.run(app)