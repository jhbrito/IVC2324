import threading, time


def func_with_delay(n=2):
    while True:
        print("func")
        time.sleep(n)


thread = threading.Thread(target=func_with_delay, daemon=True)
thread.start()

while True:
    print("main")
    time.sleep(1)
