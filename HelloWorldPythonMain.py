import sys


def test_run():
    a = 1
    b = 2
    c = a+b
    print("a({0}) plus b({1}) is c({2})".format(a, b, c))
    # print("a({2}) plus b({1}) is c({0})".format(c, b, a))


if __name__ == "__main__":
    print(dir())  # names in the module space
    print(dir(sys))  # names in the sys space
    print("Number of arguments:", len(sys.argv))
    print("Arguments:")
    for i in range(len(sys.argv)):
        print(sys.argv[i])
    test_run()
