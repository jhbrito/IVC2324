import time
from tqdm import tqdm, trange
import winsound  # this only works on Windows
import numpy as np
from numba import jit

# tqdm
a = 1
for i in tqdm(range(10)):
    a += i
    time.sleep(0.1)
print(a)

a = 1
for i in trange(10):
    a += i
    time.sleep(0.1)
print(a)

# Beeping
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)  # this only works on Windows
time.sleep(2)
winsound.Beep(frequency, duration)  # this only works on Windows

# numba


def go_normal(a):  # Function is not compiled
    trace = 0.0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            trace += np.tanh(a[i, j])
    return a + trace


@jit(nopython=True)
def go_fast(a):  # Function is compiled and runs in machine code
    trace = 0.0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            trace += np.tanh(a[i, j])
    return a + trace


x = np.arange(1000000).reshape(1000, 1000)
print("Starting...")
# Time without compiling
start = time.time()
go_normal(x)
end = time.time()
print("Elapsed (without numba) = %s" % (end - start))

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
go_fast(x)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
go_fast(x)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
