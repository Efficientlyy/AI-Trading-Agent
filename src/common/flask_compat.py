"""
Flask compatibility module for Python 3.13

This module provides a minimal implementation of Flask for the dashboard
when the actual Flask framework cannot be properly imported.
"""

import sys
import os
import datetime
import json
import logging
import socket
import threading
import http.server
import socketserver
import urllib.parse
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Any, Callable, Optional, Union, Tuple

logger = logging.getLogger("flask_compat")

# Fix for Python 3.13's handling of module imports
class FlaskModule(ModuleType):
    """Mock Flask module."""
    def __init__(self, name):
        super().__init__(name)

# Mock Request class to simulate Flask's request object
class Request:
    """Simple request object to mimic Flask's request."""
    
    def __init__(self, path='/', method='GET', args=None, form=None, json_data=None):
        self.path = path
        self.method = method
        self.args = args or {}
        self.form = form or {}
        self.json = json_data
        self.headers = {}
        self.cookies = {}
        self.url = f"http://localhost{path}"

# Create a global request object
request = Request()

class Response:
    """Simple response object to mimic Flask's response."""
    
    def __init__(self, content, status=200, content_type='text/html'):
        self.content = content
        self.status = status
        self.content_type = content_type
        self.headers = {'Content-Type': content_type}
    
    def __str__(self):
        return str(self.content)


class Flask:
    """
    Simplified Flask application implementation for compatibility.
    """
    
    def __init__(self, import_name, **options):
        """Initialize a Flask application."""
        self.import_name = import_name
        self.options = options
        self.routes = {}
        self.config = {}
        self.template_folder = options.get('template_folder', 'templates')
        self.static_folder = options.get('static_folder', 'static')
        self.secret_key = None
        
        # Create template and static folders if they don't exist
        self._ensure_dir_exists(self.template_folder)
        self._ensure_dir_exists(self.static_folder)
        
        logger.info(f"Flask app initialized with template_folder={self.template_folder}, static_folder={self.static_folder}")
    
    def _ensure_dir_exists(self, path):
        """Ensure directory exists."""
        os.makedirs(path, exist_ok=True)
        logger.info(f"Ensured directory exists: {path}")
    
    def route(self, rule, **options):
        """Route decorator registration."""
        def decorator(f):
            self.routes[rule] = {'handler': f, 'options': options}
            logger.info(f"Registered route: {rule}")
            return f
        return decorator
    
    def _handle_request(self, path, method, post_data=None):
        """Handle an incoming request."""
        global request
        
        # Parse URL path and query parameters
        parsed_path = urllib.parse.urlparse(path)
        path_only = parsed_path.path
        
        logger.info(f"Handling {method} request for path: {path_only}")
        
        # Get query parameters
        query_params = urllib.parse.parse_qs(parsed_path.query)
        args = {k: v[0] for k, v in query_params.items()}
        
        # Parse post data if available
        form = {}
        json_data = None
        if post_data:
            if post_data.startswith('{') and post_data.endswith('}'):
                try:
                    json_data = json.loads(post_data)
                except:
                    pass
            else:
                try:
                    form = dict(urllib.parse.parse_qsl(post_data))
                except:
                    pass
        
        # Set up request object
        request.path = path_only
        request.method = method
        request.args = args
        request.form = form
        request.json = json_data
        
        # Check for static file
        if path_only.startswith(f"/{self.static_folder}/"):
            return self._serve_static_file(path_only[1:])  # Remove leading slash
        
        # Find matching route
        # First, try exact match
        if path_only in self.routes:
            try:
                response = self.routes[path_only]['handler']()
                if isinstance(response, Response):
                    return response
                elif isinstance(response, str):
                    return Response(response)
                elif isinstance(response, dict):
                    return Response(json.dumps(response), content_type='application/json')
                else:
                    return Response(str(response))
            except Exception as e:
                logger.error(f"Error handling route {path_only}: {e}")
                return Response(f"Error: {e}", status=500)
        
        # Handle root path for index route
        if path_only == "/" and "" in self.routes:
            try:
                response = self.routes[""]["handler"]()
                if isinstance(response, Response):
                    return response
                elif isinstance(response, str):
                    return Response(response)
                elif isinstance(response, dict):
                    return Response(json.dumps(response), content_type='application/json')
                else:
                    return Response(str(response))
            except Exception as e:
                logger.error(f"Error handling route {path_only}: {e}")
                return Response(f"Error: {e}", status=500)
        
        # No route found
        logger.warning(f"No route found for {path_only}")
        return Response("Not Found", status=404)
    
    def _serve_static_file(self, path):
        """Serve a static file."""
        try:
            with open(path, 'rb') as f:
                content = f.read()
            
            # Determine content type
            content_type = 'text/plain'
            if path.endswith('.css'):
                content_type = 'text/css'
            elif path.endswith('.js'):
                content_type = 'application/javascript'
            elif path.endswith('.html'):
                content_type = 'text/html'
            elif path.endswith('.png'):
                content_type = 'image/png'
            elif path.endswith('.jpg') or path.endswith('.jpeg'):
                content_type = 'image/jpeg'
            elif path.endswith('.json'):
                content_type = 'application/json'
            
            logger.info(f"Serving static file: {path} as {content_type}")
            return Response(content, content_type=content_type)
        except FileNotFoundError:
            logger.error(f"Static file not found: {path}")
            return Response("File not found", status=404)
        except Exception as e:
            logger.error(f"Error serving static file: {e}")
            return Response(f"Error: {e}", status=500)
    
    def render_template(self, template_name, **context):
        """Simple template rendering."""
        template_path = Path(self.template_folder) / template_name
        logger.info(f"Rendering template: {template_path}")
        
        try:
            if not template_path.exists():
                logger.error(f"Template not found: {template_path}")
                return Response(f"Template not found: {template_name}", status=404)
                
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            # Very basic template rendering - replace {{ var }} with context values
            for key, value in context.items():
                if isinstance(value, (dict, list)):
                    # For complex objects, use JSON
                    placeholder = f"{{ {key} | tojson }}"
                    if placeholder in template_content:
                        template_content = template_content.replace(placeholder, json.dumps(value))
                
                placeholder = f"{{ {key} }}"
                if placeholder in template_content:
                    template_content = template_content.replace(placeholder, str(value))
            
            # Super simple template rendering
            return Response(template_content, content_type='text/html')
        except Exception as e:
            logger.error(f"Error rendering template: {e}")
            return Response(f"Error rendering template: {e}", status=500)
    
    def run(self, host='127.0.0.1', port=5000, debug=False, **options):
        """Run the Flask application."""
        logger.info(f"Starting Flask server on {host}:{port}")
        
        app = self  # Reference to Flask app
        
        class FlaskHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                response = app._handle_request(self.path, 'GET')
                self._send_response(response)
            
            def do_POST(self):
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length).decode('utf-8')
                response = app._handle_request(self.path, 'POST', post_data)
                self._send_response(response)
            
            def _send_response(self, response):
                self.send_response(response.status)
                for key, value in response.headers.items():
                    self.send_header(key, value)
                self.end_headers()
                
                if isinstance(response.content, str):
                    self.wfile.write(response.content.encode('utf-8'))
                else:
                    self.wfile.write(response.content)
        
        # Create a server
        server = socketserver.TCPServer((host, port), FlaskHandler)
        server.allow_reuse_address = True
        
        # Run server
        try:
            logger.info(f"Server started at http://{host}:{port}/")
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped")
        finally:
            server.server_close()
    
    def jsonify(self, *args, **kwargs):
        """Convert data to JSON response."""
        if len(args) == 1 and not kwargs and isinstance(args[0], (dict, list)):
            data = args[0]
        else:
            data = args[0] if len(args) == 1 else dict(*args, **kwargs)
        
        return Response(json.dumps(data), content_type='application/json')


def render_template(template_name, **context):
    """Standalone render_template function for compatibility."""
    if not hasattr(render_template, '_app'):
        render_template._app = Flask('standalone')
    return render_template._app.render_template(template_name, **context)


def jsonify(*args, **kwargs):
    """Standalone jsonify function for compatibility."""
    if not hasattr(jsonify, '_app'):
        jsonify._app = Flask('standalone')
    return jsonify._app.jsonify(*args, **kwargs)


def redirect(location, code=302, **kwargs):
    """Mock redirect function."""
    response = Response(f"Redirecting to {location}", status=code)
    response.headers['Location'] = location
    return response


def url_for(endpoint, **values):
    """Mock url_for function."""
    if endpoint == 'static':
        return f"/static/{values.get('filename', '')}"
    # Simple mapping for endpoints to routes
    return f"/{endpoint}"


def apply_flask_patches():
    """Apply Flask compatibility patches for Python 3.13."""
    logger.info("Applying Flask compatibility patches...")
    
    # Create a mock Flask module
    flask_module = FlaskModule('flask')
    
    # Add core Flask components to the module
    flask_module.Flask = Flask
    flask_module.request = request
    flask_module.render_template = render_template
    flask_module.jsonify = jsonify
    flask_module.redirect = redirect
    flask_module.url_for = url_for
    flask_module.Response = Response
    
    # Add the module to sys.modules
    sys.modules['flask'] = flask_module
    
    # Add flask components directly to the top level imports
    sys.modules['flask.json'] = flask_module
    
    logger.info("✓ Successfully applied Flask compatibility patches")


if __name__ == "__main__":
    # Apply patches when run directly
    apply_flask_patches()
    
    # Test Flask app
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return "Flask compatibility layer is working!"
    
    app.run(debug=True)
