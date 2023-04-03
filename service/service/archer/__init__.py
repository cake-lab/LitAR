from .frame import frame_websocket_handler

r = [
    *frame_websocket_handler
]

archer_websocket_routes = [(f'/archer/{v[0]}', v[1]) for v in r]
