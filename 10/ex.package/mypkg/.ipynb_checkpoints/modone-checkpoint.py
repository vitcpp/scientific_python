from . import modtwo

class A:
    def __init__(self):
        print(__name__ + "." + A.__name__)

def f():
    print(__name__ + "." + f.__name__)

def g():
    modtwo.g()
