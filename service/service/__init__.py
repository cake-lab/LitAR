import os
import tornado
import tornado.web
from tornado.log import enable_pretty_logging

from configs import PORT
from service.api import api_v1_http_routes
from service.archer import archer_websocket_routes


enable_pretty_logging()


def start_service(port=PORT, debug=True):
    """ Holds all the registered HTTP endpoints
    """

    routes = [
        *api_v1_http_routes, # API HTTP routes, marked as version v1.
        *archer_websocket_routes # WebSockets routes for debugging.
    ]

    os.system('clear')

    print('💡 Serving the following routes')
    print('\n'.join(['-> ' + v[0] for v in routes]) + '\n')

    app = tornado.web.Application(
        routes,
        debug=debug,
        autoreload=debug)

    app.listen(port)

    print('Tornado Server Started...')
    tornado.ioloop.IOLoop.current().start()
