"""Serves web/ for the threers-ui demo, telling browsers not to cache.

A phone's browser keeps the demo's JavaScript and WebAssembly from an
earlier build otherwise, and a reload runs the old code against the new
page. With a certificate and key it serves https, which a browser needs
before it lets a page read the clipboard.

    python3 serve.py ROOT PORT [CERT KEY]
"""

import functools
import http.server
import ssl
import sys


class NoStore(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        # The page's `/debug` and `/beacon` reports: logged (as every
        # request is) and answered with nothing.
        if self.path.startswith(("/debug?", "/beacon?")):
            self.send_response(204)
            self.end_headers()
            return
        super().do_GET()


def main():
    root, port = sys.argv[1], int(sys.argv[2])
    handler = functools.partial(NoStore, directory=root)
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), handler)
    scheme = "http"
    if len(sys.argv) >= 5:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(sys.argv[3], sys.argv[4])
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"
    print(f"serving {root} on {scheme}://0.0.0.0:{port}/ui-demo/", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
