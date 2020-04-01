from flask import Flask, request, session
from flask_socketio import SocketIO
from flask_socketio import join_room, leave_room

app = Flask(__name__)
socketio = SocketIO(app)



# @socketio.on('join')
# def on_join(data):
#     username = data['username']
#     room = data['room']
#     join_room(room)
#     send(username + ' has entered the room.', room=room)
#
# @socketio.on('leave')
# def on_leave(data):
#     username = data['username']
#     room = data['room']
#     leave_room(room)
#     send(username + ' has left the room.', room=room)

@app.route('/hi')  # API route
def main():
    return 'hi'

@socketio.on('connect')
def connected():
    print("client connected to socket")

@socketio.on('new_client')
def handel_event():
    session_id = request.sid
    # Save the session id to emit to this client in the future

    room = session.get('room')
    join_room(room)
    socketio.emit('status',  {'msg': session.get('name') + ' has entered the room.'}, room=session_id)
    print("new client")

def update_clients():
    msg = 'new data'
    socketio.emit(msg)

if __name__ == '__main__':
    socketio.run(app)