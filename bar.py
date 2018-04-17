import numpy as np
import matplotlib.pyplot as plt
n_groups = 5

# heft = (2.05, 1.93, 2.11, 3.98, 6.11)
# cpop = (2.52, 2.23, 2.42, 4.18, 6.12)

# slr
# heft = (2.23, 2.36, 3.16, 3.63, 3.84)
# cpop = (2.43, 2.49, 3.41, 3.91, 4.2)

# speedup
# heft = (1.61, 2.07, 2.14, 2.23, 2.31)
# cpop = (1.56, 1.82, 1.85, 1.98, 1.97)

# beta
heft = [3.95, 2.99, 2.71, 2.56, 3.16]
cpop = [3.8, 3.23, 2.93, 2.63, 3.62]

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4

rects1 = ax.bar(index, heft, bar_width,
                alpha=opacity, color=(0, 0, 0),
                label='HEFT')

rects2 = ax.bar(index + bar_width, cpop, bar_width,
                alpha=opacity, color=(0.5, 0.5, 0.5),
                label='CPOP')

ax.set_xlabel('BETA')
ax.set_ylabel('Average SLR')
# ax.set_title('Scores by group and gender')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('0.1', '0.25', '0.5', '0.75', '1.0'))
# ax.set_xticklabels(('20', '40', '60', '80', '100'))
ax.legend()

fig.tight_layout()
plt.show()
