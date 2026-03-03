import numpy as np, pandas as pd, os
from caas_jupyter_tools import display_dataframe_to_user

def interp(x, xp, fp):
    return np.interp(x, xp, fp)

def error_at_points(profile_df, xcol, ycol, ref_df, ref_x, ref_y):
    x = profile_df[xcol].to_numpy()
    y = profile_df[ycol].to_numpy()
    # Ensure sorted
    idx = np.argsort(x); x=x[idx]; y=y[idx]
    y_hat = interp(ref_df[ref_x].to_numpy(), x, y)
    diff = y_hat - ref_df[ref_y].to_numpy()
    l2 = np.sqrt(np.mean(diff**2))
    linf = np.max(np.abs(diff))
    # relative by ref max abs (avoid 0)
    denom = np.max(np.abs(ref_df[ref_y].to_numpy()))
    rel_l2 = l2/denom if denom>0 else np.nan
    rel_linf = linf/denom if denom>0 else np.nan
    return l2, linf, rel_l2, rel_linf

rows=[]
for N, uprof, vprof in [(32,u32,v32),(64,u64,v64),(128,u128,v128)]:
    l2, linf, rl2, rlinf = error_at_points(uprof,"y","u",ghia_u_400,"y","u_ref")
    rows.append({"N":N, "profile":"u(x=0.5,y)", "L2":l2, "Linf":linf, "relL2":rl2, "relLinf":rlinf})
    l2, linf, rl2, rlinf = error_at_points(vprof,"x","v",ghia_v_400,"x","v_ref")
    rows.append({"N":N, "profile":"v(x,y=0.5)", "L2":l2, "Linf":linf, "relL2":rl2, "relLinf":rlinf})

err_ref = pd.DataFrame(rows)
display_dataframe_to_user("Errors vs Ghia et al. reference (Re=400) at sample points", err_ref)
err_ref
