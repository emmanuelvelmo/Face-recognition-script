from cx_Freeze import setup, Executable

setup(name="Face recognition", executables=[Executable("Face recognition script.py")], options={"build_exe": {"excludes": ["tkinter"]}})