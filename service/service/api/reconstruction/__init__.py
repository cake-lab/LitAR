from .keyframe import keyframe_index_ws_routes

r = [
    *keyframe_index_ws_routes
]

reconstruction_http_routes = [(f'reconstruction/{v[0]}', v[1]) for v in r]
