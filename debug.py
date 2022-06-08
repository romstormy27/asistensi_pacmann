class A:
    # def __init__(self):
    #     self.a = 10        

    def func1(self):
        self.a = 10

    def func2(self):
        print(self.a)

a = A()
a.func1()
a.func2()