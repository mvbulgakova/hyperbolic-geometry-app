import threading

from kivy.app import App
from kivy.uix.label import Label
from kivy.utils import platform


def _start_server():
    from flask_server import app
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)


class HyperbolicApp(App):
    def build(self):
        t = threading.Thread(target=_start_server, daemon=True)
        t.start()

        if platform == "android":
            # Give Flask a moment to start, then show WebView on the UI thread.
            threading.Timer(2.0, self._show_webview).start()

        return Label(text="Запуск сервера…" if platform != "android" else "")

    def _show_webview(self):
        try:
            from android.runnable import run_on_ui_thread

            @run_on_ui_thread
            def _setup():
                from jnius import autoclass

                PA = autoclass("org.kivy.android.PythonActivity")
                WebView = autoclass("android.webkit.WebView")
                WebViewClient = autoclass("android.webkit.WebViewClient")
                LP = autoclass("android.view.ViewGroup$LayoutParams")

                activity = PA.mActivity
                wv = WebView(activity)
                s = wv.getSettings()
                s.setJavaScriptEnabled(True)
                s.setDomStorageEnabled(True)
                s.setLoadWithOverviewMode(True)
                s.setUseWideViewPort(True)
                s.setBuiltInZoomControls(True)
                s.setDisplayZoomControls(False)
                wv.setWebViewClient(WebViewClient())
                wv.loadUrl("http://127.0.0.1:5000")

                activity.addContentView(
                    wv, LP(LP.MATCH_PARENT, LP.MATCH_PARENT)
                )

            _setup()
        except Exception as e:
            print("WebView error:", e)


if __name__ == "__main__":
    HyperbolicApp().run()
