class F:
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return str(self.value)

def f(arg : F):
    print(__name__ + "." + f.__name__ + "(" + f"{arg}" + ")")

a = 10
def p():
    print(a)

