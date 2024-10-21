# Test matplotlib
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

x = [1, 2, 3, 4]
y = [1, 4, 9, 16]

plt.plot(x, y)
plt.show()