from .reconstruction import reconstruction_http_routes

r = [
    *reconstruction_http_routes
]
api_v1_http_routes = [(f'/api/{v[0]}', v[1]) for v in r]

__all__ = ['api_v1_http_routes']
