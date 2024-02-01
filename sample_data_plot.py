import numpy as np
import matplotlib.pyplot as plt

l1 = []
l2 = []
l3 = []

with open('sample_data.txt') as file:
    for line in file:
        if line != '\n':
            l, x, y, _, _, _, _ = [i for i in line.split()]
            if l == '00':
                l1.append((float(x), float(y)))
            
            elif l == '01':
                l2.append((float(x), float(y)))

            elif l == '02':
                l3.append((float(x), float(y)))

fig = plt.figure(figsize=(10, 10))
plt.scatter(range(len(l1)), [i[0] for i in l1], label="x")
plt.scatter(range(len(l1)), [i[1] for i in l1], label="y")
plt.xticks(range(len(l1)), fontsize=20)
plt.title('layer1', fontsize=25)
plt.xlabel('Iteration', fontsize=20)
plt.ylabel('Millepde Correction (mm)', fontsize=20)
plt.legend(fontsize=20, loc=4)
fig.savefig('l1.pdf', format='pdf')

fig = plt.figure(figsize=(10, 10))
plt.scatter(range(len(l2)), [i[0] for i in l2], label="x")
plt.scatter(range(len(l2)), [i[1] for i in l2], label="y")
plt.xticks(range(len(l2)), fontsize=20)
plt.title('layer2', fontsize=25)
plt.xlabel('Iteration', fontsize=20)
plt.ylabel('Millepde Correction (mm)', fontsize=20)
plt.legend(fontsize=20, loc=4)
fig.savefig('l2.pdf', format='pdf')

fig = plt.figure(figsize=(10, 10))
plt.scatter(range(len(l3)), [i[0] for i in l3], label="x")
plt.scatter(range(len(l3)), [i[1] for i in l3], label="y")
plt.xticks(range(len(l3)), fontsize=20)
plt.title('layer3', fontsize=25)
plt.xlabel('Iteration', fontsize=20)
plt.ylabel('Millepde Correction (mm)', fontsize=20)
plt.legend(fontsize=20, loc=4)
fig.savefig('l3.pdf', format='pdf')
