use_tmux = None
session_name = None

def set_use_tmux(b):
    global use_tmux
    use_tmux = b

def set_session_name(name):
    global session_name
    session_name = name