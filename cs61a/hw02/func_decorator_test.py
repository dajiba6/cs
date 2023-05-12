

def de(func):
    def w():
        print("start")
        func()
        print("end")
    return w
@de
def hello():
    print("hello")

hello()