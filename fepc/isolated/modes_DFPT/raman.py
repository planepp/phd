import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("fepc.Raman", sep=r"\s+", comment="#", header=None,
                 names=["mode", "frequency_cm1", "zpe_eV", "raman_intensity"])

freqs = df["frequency_cm1"].values
ints = df["raman_intensity"].values

plt.figure(figsize=(8, 5))
plt.vlines(freqs, 0, ints, color="k", linewidth=1.5)

plt.xlim(400, 450)
plt.ylim(0, ints[(freqs >= 400) & (freqs <= 450)].max() * 1.05)

plt.xlabel("Frequency (cm$^{-1}$)")
plt.ylabel("Raman Intensity (a.u.)")
plt.title("FePc Raman Spectrum")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fepcraman.png")
plt.show()
