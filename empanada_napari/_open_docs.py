import webbrowser
from magicgui import magic_factory

@magic_factory(call_button="Open Empanada Napari Documentation")
def open_documentation():
    """Opens the documentation in default browser"""
    webbrowser.open("https://empanada.readthedocs.io/en/latest/")
