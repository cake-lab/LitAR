import json
import tornado.web
from urllib.parse import unquote


class BaseHttpRouter(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header('Content-Type', 'application/json')

    def get_body_json(self):
        body = self.request.body.decode('utf-8')
        body = json.loads(unquote(body))
        return body

    def json(self, data):
        self.write(json.dumps(data))
