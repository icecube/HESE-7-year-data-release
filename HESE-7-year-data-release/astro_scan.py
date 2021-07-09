import numpy as np
import meander
import scipy.special
import matplotlib.pyplot as plt
import matplotlib.style

matplotlib.style.use("./resources/mpl/paper.mplstyle")

# Each line in the text file corresponds to a fit with fixed astrogamma and astronorm
# The first column is the chosen astrogamma
# The second column is the chosen astronorm
# The third column is the best fit -LLH
params_llhs = np.loadtxt("./resources/scan/astro_scan_data.txt")

# For the Best Fit parameters, we use the overall best fit points and -LLH
# using HESE_fit.py
BF_params_llh = [2.87375956, 6.36488608, 122.95919881139868]

astrogamma = params_llhs[:, 0]
astronorm = params_llhs[:, 1]
llhs = params_llhs[:, 2]

# Set the confidence levels of the contours, in terms of sigmas
sigmas = np.array([1, 2])
print("sigmas:", sigmas)

# Set the number of degrees of freedom
dof = 2

# Convert the sigma value to a percentage
proportions = scipy.special.erf(sigmas / np.sqrt(2.0))
print("proportions:", proportions)

# Calculate the critical delta LLH for each confidence level
levels = scipy.special.gammaincinv(dof / 2.0, np.array(proportions))
print("levels:", levels)

# Pass the data and return the contours for each level
contours_by_level = meander.compute_contours(
    np.array([astrogamma, astronorm]).T, llhs - BF_params_llh[2], levels
)

fig, ax = plt.subplots()

cm = plt.get_cmap("plasma")
plt.xlabel(r"$\gamma_{\texttt{astro}}$")
plt.ylabel(r"$\Phi_{\texttt{astro}}$")
plt.xlim(1.7, 3.8)
plt.ylim(0, 15)

# Plot the best fit point
ax.plot(BF_params_llh[0], BF_params_llh[1], marker="*", color=cm(0.1))

# Plot the confince levels
linestyles = ["solid", "dashed"]
for j, contours in enumerate(contours_by_level):
    for contour in contours:
        ax.plot(
            contour[:, 0],
            contour[:, 1],
            linewidth=2,
            linestyle=linestyles[j],
            color=cm(0.1),
        )

plt.tight_layout()
plt.show()
