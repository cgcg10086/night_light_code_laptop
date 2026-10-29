#!/usr/bin/env python
"""
Sanity checks for the products of transient.tmpl.

    check_transient_cube.py 1476960704_transient [--box 1000] [--hdf5]

Reads  <prefix>-cube.fits   (time, y, x stack of the per-timestep residual images)
       <prefix>-stdev.fits  (per-pixel standard deviation through time)
       <prefix>.hdf5        (optional: cross-check against the make_imstack output)
Prints a report and writes <prefix>_sanity.png.
"""
import argparse
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("prefix", help="e.g. 1476960704_transient")
ap.add_argument("--box", type=int, default=1000, help="side (pixels) of the central box used for noise statistics")
ap.add_argument("--hdf5", action="store_true", help="also cross-check <prefix>.hdf5")
args = ap.parse_args()
P = args.prefix
WARN = []


def mad_rms(a):
    a = a[np.isfinite(a)]
    return 1.4826 * np.median(np.abs(a - np.median(a))) if a.size else np.nan


def flag(msg):
    WARN.append(msg)
    print("  !! " + msg)


# ---------------------------------------------------------------- cube
hdul = fits.open(f"{P}-cube.fits", memmap=True)
hdr, cube = hdul[0].header, hdul[0].data
cube = np.squeeze(cube) if cube.ndim > 3 else cube
nt, ny, nx = cube.shape
b = min(args.box, ny, nx)
y0, x0 = (ny - b) // 2, (nx - b) // 2
box = (slice(y0, y0 + b), slice(x0, x0 + b))
print(f"\n== 1. Shape and time axis\ncube shape (time, y, x) = {cube.shape}")
step = hdr.get("CDELT3")
crval3 = hdr.get("CRVAL3")
print(f"CTYPE3={hdr.get('CTYPE3')}  CRVAL3={crval3}  CDELT3={step} s")
if hdr.get("DATE-OBS"):
    gps_first = Time(hdr["DATE-OBS"], scale="utc").gps
    off = gps_first - crval3
    print(f"first image DATE-OBS = {hdr['DATE-OBS']}  (GPS {gps_first:.1f}); CRVAL3 is {off:+.1f} s from it")
    if abs(off) > 1:
        flag(f"cube time axis (CRVAL3={crval3}) is {off:+.1f} s off the first image's DATE-OBS; "
             "use DATE-OBS / the HDF5 timestamps when you need absolute times")

# --------------------------------------------- per timestep statistics
print("\n== 2. Per-timestep statistics (central box)")
rms = np.zeros(nt); med = np.zeros(nt); nanf = np.zeros(nt); pk = np.zeros(nt)
msum = np.zeros((ny, nx)); mcnt = np.zeros((ny, nx))
hist_edges = np.linspace(-8, 8, 81); hist = np.zeros(len(hist_edges) - 1)
for t in range(nt):
    im = np.asarray(cube[t], dtype=np.float64)
    fin = np.isfinite(im)
    msum += np.where(fin, im, 0.0); mcnt += fin
    c = im[box]
    nanf[t] = 1.0 - np.isfinite(c).mean()
    rms[t] = mad_rms(c)
    med[t] = np.nanmedian(c)
    pk[t] = np.nanmax(np.abs(c - med[t])) / rms[t] if rms[t] > 0 else np.nan
    hist += np.histogram(((c - med[t]) / rms[t])[np.isfinite(c)], bins=hist_edges)[0]
rms_med = np.nanmedian(rms)
print(f"robust rms per timestep: median {rms_med*1e3:.2f} mJy/beam, "
      f"min {np.nanmin(rms)*1e3:.2f}, max {np.nanmax(rms)*1e3:.2f}")
print("(compare the median with wsclean's 'Estimated standard deviation of background noise' in the .o log)")
bad = np.where((rms > 1.5 * rms_med) | (rms < 0.5 * rms_med) | (nanf > 0.01) | (np.abs(med) > 0.5 * rms))[0]
if len(bad):
    flag(f"{len(bad)} odd timestep(s) (rms off by >50%, >1% NaN, or |median| > 0.5 rms): {bad.tolist()}")
else:
    print("all timesteps have similar rms, ~zero median and no NaNs: OK")
if np.nanmax(nanf) > 0:
    print(f"NaN fraction in box: max {np.nanmax(nanf):.3f}")

# ------------------------------------------------------ mean image
print("\n== 3. Mean image (static residuals should average to ~0)")
mean = np.where(mcnt > 0, msum / np.maximum(mcnt, 1), np.nan)
mean_rms = mad_rms(mean[box])
expect = rms_med / np.sqrt(nt)
print(f"rms of mean image {mean_rms*1e3:.2f} mJy; white-noise expectation rms/sqrt(N) = {expect*1e3:.2f} mJy; "
      f"ratio {mean_rms/expect:.2f}")
if mean_rms / expect > 2:
    flag("mean image is much noisier than white noise: residual static structure "
         "(imperfect model subtraction, sidelobes, or correlated noise); look at the PNG")
mabs = np.abs(mean - np.nanmedian(mean[box]))
mabs[~np.isfinite(mabs)] = 0
top = np.dstack(np.unravel_index(np.argsort(mabs.ravel())[::-1][:3], mabs.shape))[0]
print("strongest mean-image pixels (x, y, value in units of rms/sqrt(N)):",
      [(int(x), int(y), round(float(mabs[y, x] / expect), 1)) for y, x in top])
if mabs.max() / expect > 10:
    flag(f"mean image has pixel(s) up to {mabs.max()/expect:.0f}x rms/sqrt(N): a source was NOT removed by the model "
         "subtraction (check MODEL_DATA, the column imaged, and the phase centre)")

# ------------------------------------------------------ stdev image
print("\n== 4. Standard-deviation image (the 'what varies' map)")
std = np.squeeze(fits.getdata(f"{P}-stdev.fits"))
smed = np.nanmedian(std[box])
print(f"median of stdev image {smed*1e3:.2f} mJy; per-timestep rms {rms_med*1e3:.2f} mJy; ratio {smed/rms_med:.2f}")
if not 0.7 < smed / rms_med < 1.3:
    flag("median stdev differs >30% from per-timestep rms: noise is not behaving like independent snapshots")
sflat = np.where(np.isfinite(std), std, 0).ravel()
idx = np.argsort(sflat)[::-1][:5]
w = WCS(fits.getheader(f"{P}-stdev.fits")).celestial
print("5 highest-stdev pixels (x, y, RA, Dec, stdev/median):")
for i in idx:
    y, x = divmod(i, std.shape[1])
    ra, dec = w.pixel_to_world_values(x, y)
    print(f"   {x:5d} {y:5d}  {float(ra):8.3f} {float(dec):8.3f}   {std[y, x]/smed:6.1f}")
print("(bright, known sources with large ratios are normal; check them against a catalogue)")

# ------------------------------------------------------ hdf5
if args.hdf5:
    import h5py
    print("\n== 5. HDF5 cross-check")
    with h5py.File(f"{P}.hdf5", "r") as f:
        print("datasets:", {k: f[k].shape for k in f.keys()}, "| attrs:", dict(f.attrs))
        ts = np.array([Time(s.decode(), scale="utc").gps for s in f["timestamp"][:]])
        dts = np.diff(ts)
        print(f"timestamps: n={len(ts)}, spacing {dts.min():.1f}-{dts.max():.1f} s")
        if len(ts) != nt:
            flag(f"HDF5 has {len(ts)} timesteps but the cube has {nt}")
        if dts.size and (np.ptp(dts) > 0.5):
            flag("uneven timestamp spacing in HDF5")
        for t in sorted({0, nt // 2, nt - 1}):
            h = f["image"][0, :, :, 0, t].astype(np.float64)
            d = np.nanmax(np.abs(h - np.asarray(cube[t], dtype=np.float64)))
            print(f"timestep {t}: max |hdf5 - cube| = {d/rms_med:.4f} rms (float16 rounding expected to be tiny)")
            if d / rms_med > 0.05:
                flag(f"HDF5 and cube disagree at timestep {t}")

# ------------------------------------------------------ figure
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
ax[0, 0].plot(rms * 1e3, "o-", ms=3); ax[0, 0].axhline(rms_med * 1e3, color="k", ls=":")
for t in bad: ax[0, 0].plot(t, rms[t] * 1e3, "ro")
ax[0, 0].set_xlabel("timestep"); ax[0, 0].set_ylabel("robust rms (mJy/beam)"); ax[0, 0].set_title("noise vs time (should be flat)")
im1 = ax[0, 1].imshow(std, origin="lower", vmin=0, vmax=np.nanpercentile(std, 99.5), cmap="viridis")
ax[0, 1].set_title("stdev through time"); plt.colorbar(im1, ax=ax[0, 1])
v = 5 * expect
im2 = ax[1, 0].imshow(mean, origin="lower", vmin=-v, vmax=v, cmap="RdBu_r")
ax[1, 0].set_title("mean over time (+-5 rms/sqrt(N))"); plt.colorbar(im2, ax=ax[1, 0])
cen = 0.5 * (hist_edges[1:] + hist_edges[:-1])
ax[1, 1].semilogy(cen, hist / hist.sum() / (cen[1] - cen[0]), "o", ms=3, label="data")
ax[1, 1].semilogy(cen, np.exp(-cen**2 / 2) / np.sqrt(2 * np.pi), "k-", label="Gaussian")
ax[1, 1].set_ylim(1e-8, 1); ax[1, 1].set_xlabel("pixel value / rms"); ax[1, 1].legend(); ax[1, 1].set_title("pixel histogram (all timesteps)")
fig.tight_layout(); fig.savefig(f"{P}_sanity.png", dpi=110)
print(f"\nfigure written to {P}_sanity.png")
print("\nSUMMARY: " + ("no problems flagged" if not WARN else f"{len(WARN)} thing(s) to look at (marked !! above)"))
