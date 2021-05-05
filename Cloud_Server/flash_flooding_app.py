
from flask import Flask, render_template, url_for, stream_with_context, request, Response
from flask_socketio import SocketIO, emit
from threading import Thread, Event

import os
import time
import pickle

app = Flask(__name__)
app.config['RIVER_IMGs_FOLDER'] = os.path.join("static", "river_imgs")
app.config['DEBUG'] = False
app.config['SECRET_KEY'] = b''

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

#turn the flask app into a socketio app
socketio = SocketIO(app, ping_timeout=300, ping_interval=60, async_mode="threading", reconnection=False, logger=True, cors_allowed_origins='*', engineio_logger=True)

#random number Generator Thread
thread_website = Thread()
thread_stop_event = Event()

def client_connected_website():
    while True:
        with open("./river_level_variables.pickle", "rb") as f:
            [river_A_level_in_meters, river_B_level_in_meters, night_time] = pickle.load(f)

        print([river_A_level_in_meters, river_B_level_in_meters, night_time])
        socketio.emit('river_level_in_meters', {'number': [river_A_level_in_meters,river_B_level_in_meters, night_time]}, namespace='/river_level')
        socketio.sleep(0)
        time.sleep(21)

@app.route('/')
def index():
    return render_template('river_imgs.html')

@app.route('/results')
def results():
    return render_template('river_imgs.html', river_A_level_image = None, river_A_image = None, river_B_level_image = None, river_B_image = None)

@socketio.on('connect', namespace='/river_level')
def test_connect():
    # need visibility of the global thread object
    global thread_website
    print('Client connected')

    #Start the random number generator thread only if the thread has not been started before.
    if not thread_website.is_alive():
        print("Starting Thread")
        thread_website = socketio.start_background_task(client_connected_website)

@socketio.on('disconnect', namespace='/river_level')
def test_disconnect():
    print('Client disconnected')


if __name__ == "__main__":
   #socketio.run(app, host='0.0.0.0', port=5000)
   socketio.run(app)
