"""REDblocks desktop editor: a native window (pywebview, using Windows' built-in WebView2) around editor.html, with
the compiler and solver in the same process. No server, no browser, no network. Files are plain YAML wherever you
save them; settings (recent files, core count) live in ~/.simblocks/app.json.

Run:  simblocks app            (installed package)
      REDblocks.exe             (folder build from packaging/build_exes.bat, installed by packaging/install.ps1)
"""
import multiprocessing
import os
import sys


def _html_path():
    base = getattr(sys, "_MEIPASS", None)
    if base:                                       # one-file exe: files unpacked next to the package
        p = os.path.join(base, "simblocks", "app", "editor.html")
        if os.path.exists(p):
            return p
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "editor.html")


def main(argv=None):
    multiprocessing.freeze_support()               # study workers in a frozen exe
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".simblocks_numba_cache"))
    try:
        import webview
    except ImportError:
        print("The desktop editor needs pywebview:  pip install pywebview\n"
              "(On Windows 10/11 it draws with the built-in WebView2; nothing else to install.)")
        return 1
    from .api import Api
    api = Api()
    win = webview.create_window("REDblocks", url=_html_path(), js_api=api, width=1500, height=920,
                                min_size=(1000, 640), background_color="#0b1018", text_select=True)
    api._window = win
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and os.path.exists(argv[0]):           # open a model passed on the command line / dropped on the exe
        path = os.path.abspath(argv[0])

        def _load():
            import json
            r = api.open_path_text(path)
            if r.get("ok"):
                win.evaluate_js(f"loadText({json.dumps(r['text'])}, {json.dumps(path)})")
        win.events.loaded += _load
    webview.start(debug=bool(os.environ.get("SIMBLOCKS_DEBUG")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
