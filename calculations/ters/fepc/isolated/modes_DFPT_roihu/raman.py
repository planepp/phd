import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = "fepc.Raman"

# Read file
df = pd.read_csv(
    filename,
    sep=r"\s+",
    comment="#",
    header=None,
    names=[
        "mode",
        "frequency_cm1",
        "zpe_eV",
        "raman_intensity"
    ]
)

freqs = df["frequency_cm1"].values
ints = df["raman_intensity"].values

# Frequency grid for the broadened spectrum
xmin = freqs.min() - 50
xmax = freqs.max() + 50
xmin = 400
xmax = 450
x = np.linspace(xmin, xmax, 5000)

# Gaussian broadening
fwhm = 0.1  # cm^-1
sigma = fwhm / 2.35482

spectrum = np.zeros_like(x)

for f, I in zip(freqs, ints):
    spectrum += I * np.exp(-(x - f)**2 / (2 * sigma**2))

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, spectrum, lw=2, label=f"Gaussian FWHM = {fwhm} cm$^{{-1}}$")

# Optional: show original stick spectrum
plt.vlines(freqs, 0, ints, color="k", alpha=0.3, linewidth=1)

plt.xlabel("Frequency (cm$^{-1}$)")
plt.ylabel("Raman Intensity (a.u.)")
plt.title("FePc Raman Spectrum")
plt.grid(alpha=0.3)
plt.xlim([xmin,xmax])
mask = (x >= xmin) & (x <= xmax)
plt.ylim(0, 1.05 * spectrum[mask].max())
#plt.legend()
plt.tight_layout()
plt.savefig("fepcraman.png")
plt.show()

