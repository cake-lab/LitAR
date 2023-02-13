import os
import numpy as np
import tornado.websocket

counter = 0


class FrameWebSocket(tornado.websocket.WebSocketHandler):
    def open(self):
        global counter

        counter = 0
        os.system('rm ./tmp/session/*.npy')

        print("WebSocket opened")

    def on_message(self, message):
        global counter

        if type(message) is str:
            print(f'< {message}')
            return

        print(f'< {len(message)}')

        data_bytes = np.frombuffer(message, dtype=np.uint8)
        np.save(f'./tmp/session/{counter}.npy', data_bytes)

        counter += 1


    def on_close(self):
        print("WebSocket closed")


frame_websocket_handler = [
    (r"frame/", FrameWebSocket)
]

__all__ = ['frame_websocket_handler']
