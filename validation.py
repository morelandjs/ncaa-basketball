#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from melo_ncaabb import ncaabb_spreads

# figure style and layout
plt.style.use('clean')
width, height = plt.rcParams['figure.figsize']
fig, (axl, axr) = plt.subplots(
    ncols=2, figsize=(2*width, height)
)

# standard normal distribution
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x)
axl.plot(x, y, color='black')

# standardized residuals
residuals = ncaabb_spreads.residuals(standardize=True)
axl.hist(residuals, bins=40, histtype='step', density=True)

# residual figure attributes
axl.set_xlim(-4, 4)
axl.set_ylim(0, .45)
axl.set_xlabel(r'$(y_\mathrm{obs}-y_\mathrm{pred})/\sigma_\mathrm{pred}$')
axl.set_title('Standardized residuals')

# quantiles
axr.axhline(1, color='black')
axr.hist(ncaabb_spreads.quantiles(), bins=40,
         histtype='step', density=True)

# quantile figure attributes
axr.set_xlim(0, 1)
axr.set_ylim(0, 1.5)
axr.set_xlabel(r'$\int_{-\infty}^{y_\mathrm{obs}} P(y_\mathrm{pred}) dy_\mathrm{pred}$')
axr.set_title('Quantiles')

plt.tight_layout()
plt.savefig('validation.pdf')
