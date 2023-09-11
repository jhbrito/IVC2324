# this is a comment

print("Hello World")

# Builtin types:
# Boolean Type - bool
# Numeric Types - int, float, complex
# Binary Sequence Types - bytes, bytearray, memoryview
# Sequence Types - list, tuple, range
# Text Sequence Type - str
# Set Types - set, frozenset
# Mapping Types — dict

# numeric types — int, float, complex: https://docs.python.org/3/library/stdtypes.html
a = 1
b = 2
c = a + b
print("c:", c)
print(type(c))

d = 0.1
e = a+d
print("e=", e)
print(type(e))

# If
c = 4
if c == 3:
    print("if")
elif c == 4:
    print("elif")
else:
    print("else")

# For
y = 1
for x in range(10):
    print("x:", x)
    y = y + x
print("y:", y)

# While
x = 0
y = 0
while x < 10:
    print("x:", x)
    y = y + x
    x = x + 1
print("y:", y)

# Precision
a = 1
b = 3.1415
c = a + b

a = 0.1
b = 0.2
c = a + b
print(c)

# Strings
m = "hello"
print(m)
print(m[0:1])
o = "goodbye"
p = m + " " + o
print("p:", p)
print("p: {0}. len is {1}".format(p, len(p)))
print("p: {}. len is {:.3f}".format(p, len(p)))


# Functions
def f1(x1, x2):
    yf1 = x1 * x2
    return yf1


q1 = f1(10, 3)
print("q1:", q1)


# variable scope global
n = 0


def f2(x1, x2, x3=1):
    global n
    n = n + 1
    yf2 = x1 * x2 + x3
    return yf2


q2 = f2(x1=10, x2=3)
print("q2:", q2)
q2b = f2(x2=3, x1=10)
print("q2b:", q2b)
print("n:", n)


# variable scope nonlocal
def f_outer():
    x4 = 1

    def f_inner():
        nonlocal x4
        x4 = 2
        print("inner x4:", x4)

    f_inner()
    print("outer x4:", x4)


f_outer()


# Docstring
def f3(a):
    """This function does nothing and only returns zero

    Arguments:
        a: dummy argument
    """
    return 0


help(f3)

# Tuples
a = (1, 3, 5, 7, 9)
print("a:", a)
print(a[3])
print(a[2:4])
print(a[2:])
# a[1]=10
print("len = ", len(a))
for i, value in enumerate(a):
    print("i = ", i, "  a[i] = ", value)


# Lists
b = [2, 4, 6, 8, 10]
print("b:", b)
print(b[3])
print(b[2:4])
print(b[2:])
b[1] = 10

c = list(a)
print("c:", c)
print(c[3])
print(c[2:4])
print(c[2:])
c[1] = 12

# sets - unordered collection with no duplicate elements
d = {1, 2, 3, 4, 3, 5}
print(d)
for i in range(10):
    if i in d:
        print(str(i), "is in d")

e = set((1, 2, 3, 4, 3, 5))
print(e)
for i in range(10):
    if i in e:
        print(str(i), "is in e")

# dictionaries
words = dict()
words["BAIT"] = "Brito Artificial Intelligence Team"
words["M2AI"] = "Master in Applied Artificial Intelligence"
words["EST"] = "Escola Superior de Tecnologia"
words["IPCA"] = "Instituto Politécnico do Cávado e do Ave"

print("BAIT -", words["BAIT"])


# Classes
class MyClass:
    """A simple example class"""
    i = 12345

    """This is the constructor"""

    def __init__(self):
        self.data = [1]

    def f(self):
        return 'hello MyClass'


x = MyClass()
print(x.i)
print(x.data)
print(x.f())
