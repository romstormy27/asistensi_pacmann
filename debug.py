# class A:
#     # def __init__(self):
#     #     self.a = 10        

#     def func1(self):
#         self.a = 10

#     def func2(self):
#         print(self.a)

# class B:
#     def __init__(self, class_a):
#         self.a = class_a.a

#     def func1(self):
#         print(self.a)

# a = A()
# a.func1()
# a.func2()

# b = B(a)
# b.func1()

from sklearn.preprocessing import Normalizer, StandardScaler
import numpy as np

x = [4, 1, 2, 2, 3, 4, 4, 2, 43, 5]
x = np.array(x)
x = x.reshape(-1,1)

scl1 = StandardScaler().fit(x)
print(scl1.transform(x))
scl2 = Normalizer().fit(x)
print(scl2.transform(x))