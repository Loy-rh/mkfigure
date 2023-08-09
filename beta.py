from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


plt.rcParams['font.family'] = 'Times New Roman'
x = np.linspace(0, 1, 100)
alpha = 0.5
y = beta.pdf(x, alpha, alpha)
plt.plot(x, y, linewidth=3)

fig = plt.gcf()
pp = PdfPages("fig/beta_{}.pdf".format(alpha))
pp.savefig(fig)
pp.close()
# pp.clf()