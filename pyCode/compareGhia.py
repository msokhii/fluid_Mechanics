import pandas as pd, numpy as np, matplotlib.pyplot as plt, os
out_dir="/mnt/data"

# Ghia reference data for Re=400 from ivan-pi gists (transcribed from Ghia et al. 1982 Table I/II)
ghia_u_400 = pd.DataFrame({
    "y":[1.0000,0.9766,0.9688,0.9609,0.9531,0.8516,0.7344,0.6172,0.5000,0.4531,0.2813,0.1719,0.1016,0.0703,0.0625,0.0547,0.0000],
    "u_ref":[1.00000,0.75837,0.68439,0.61756,0.55892,0.29093,0.16256,0.02135,-0.11477,-0.17119,-0.32726,-0.24299,-0.14612,-0.10338,-0.09266,-0.08186,0.00000]
})
ghia_v_400 = pd.DataFrame({
    "x":[1.0000,0.9688,0.9609,0.9531,0.9453,0.9063,0.8594,0.8047,0.5000,0.2344,0.2266,0.1563,0.0938,0.0781,0.0703,0.0625,0.0000],
    "v_ref":[0.00000,-0.12146,-0.15663,-0.19254,-0.22847,-0.23827,-0.44993,-0.38598,0.05186,0.30174,0.30203,0.28124,0.22965,0.20920,0.19713,0.18360,0.00000]
})

u_ref_csv = os.path.join(out_dir, "ghia_Re400_u_ref.csv")
v_ref_csv = os.path.join(out_dir, "ghia_Re400_v_ref.csv")
ghia_u_400.to_csv(u_ref_csv, index=False)
ghia_v_400.to_csv(v_ref_csv, index=False)

# Load user's centerline data (SOR)
base = "/mnt/data"
def load_centerline(path, xname, yname):
    df = pd.read_csv(path)
    # use first two columns if names differ
    if xname not in df.columns or yname not in df.columns:
        df = df.iloc[:, :2]
        df.columns = [xname, yname]
    return df.sort_values(xname)

u32 = load_centerline(f"{base}/sor_N32_u_xATp5.csv","y","u")
u64 = load_centerline(f"{base}/sor_N64_u_xATp5.csv","y","u")
u128 = load_centerline(f"{base}/sor_N128_u_xATp5.csv","y","u")

v32 = load_centerline(f"{base}/sor_N32_v_y0p5.csv","x","v")
v64 = load_centerline(f"{base}/sor_N64_v_y0p5.csv","x","v")
v128 = load_centerline(f"{base}/sor_N128_v_y0p5.csv","x","v")

# Plot u centerline with true Ghia reference markers
plt.figure()
plt.plot(u32["y"], u32["u"], label="SOR N=32")
plt.plot(u64["y"], u64["u"], label="SOR N=64")
plt.plot(u128["y"], u128["u"], label="SOR N=128")
plt.plot(ghia_u_400["y"], ghia_u_400["u_ref"], linestyle="None", marker="o", label="Ghia et al. (Re=400)")
plt.xlabel("y")
plt.ylabel(r"$u(x=0.5,y)$")
plt.title(r"Centerline profile $u(x=0.5,y)$ at Re=400")
plt.grid(True)
plt.legend()
u_png = os.path.join(out_dir, "centerline_u_vs_y_with_GhiaRef.png")
u_pdf = os.path.join(out_dir, "centerline_u_vs_y_with_GhiaRef.pdf")
plt.savefig(u_png, bbox_inches="tight", dpi=200)
plt.savefig(u_pdf, bbox_inches="tight")
plt.close()

# Plot v centerline with true Ghia reference markers
plt.figure()
plt.plot(v32["x"], v32["v"], label="SOR N=32")
plt.plot(v64["x"], v64["v"], label="SOR N=64")
plt.plot(v128["x"], v128["v"], label="SOR N=128")
plt.plot(ghia_v_400["x"], ghia_v_400["v_ref"], linestyle="None", marker="o", label="Ghia et al. (Re=400)")
plt.xlabel("x")
plt.ylabel(r"$v(x,y=0.5)$")
plt.title(r"Centerline profile $v(x,y=0.5)$ at Re=400")
plt.grid(True)
plt.legend()
v_png = os.path.join(out_dir, "centerline_v_vs_x_with_GhiaRef.png")
v_pdf = os.path.join(out_dir, "centerline_v_vs_x_with_GhiaRef.pdf")
plt.savefig(v_png, bbox_inches="tight", dpi=200)
plt.savefig(v_pdf, bbox_inches="tight")
plt.close()

(u_ref_csv, v_ref_csv, u_png, u_pdf, v_png, v_pdf)
