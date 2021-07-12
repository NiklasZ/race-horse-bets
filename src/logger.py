import time


def log(msg):
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(f'{current_time}: {msg}')
