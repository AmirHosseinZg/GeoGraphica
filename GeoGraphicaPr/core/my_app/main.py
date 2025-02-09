import webview
from GeoGraphicaPr.core.my_app.backend import start_server

if __name__ == '__main__':
    start_server()
    webview.create_window("Plotter", "http://127.0.0.1:5000", width=1200, height=800)
    webview.start()
