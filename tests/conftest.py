import os

'''Set environmental variables for integration & benchmark tests to run headlessly/without OpenGL'''
os.environ["QT_QPA_PLATFORM"] = "offscreen"
# os.environ.pop("DISPLAY", None)
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")