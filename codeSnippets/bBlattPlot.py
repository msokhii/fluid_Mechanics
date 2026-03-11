import numpy as np
import matplotlib.pyplot as plt

def rho(x, t):
    val = t**(-1/3) * np.maximum(1/12 - x**2/(12*t**(2/3)), 0.0)
    return val

x = np.linspace(-2.2, 2.2, 1200)
times = [0.25, 0.5, 1.0, 2.0, 4.0]

plt.figure(figsize=(7.2, 4.8))
for t in times:
    plt.plot(x, rho(x, t), label=f"t={t}")

plt.xlabel("x")
plt.ylabel(r"$\rho(x,t)$")
plt.title(r"Barenblatt profiles for $u_t=(u^2)_{xx}$ with $c=\frac{1}{12}$")
plt.legend()
plt.tight_layout()

out = "/mnt/data/barenblatt_profiles_m2_c1_12.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.show()

print(out)
