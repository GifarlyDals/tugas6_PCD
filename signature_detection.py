import argparse
import csv
import glob
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROIS = {
    "signature":  (0.63, 0.87, 0.695, 0.825),  
    "blank":      (0.04, 0.26, 0.28, 0.42),    
    "typed_text": (0.64, 0.87, 0.825, 0.875), 
}
GROUND_TRUTH = {"signature": True, "blank": False, "typed_text": False}

NORM_WIDTH = 800       
GLOBAL_T = 110         
ADAPT_BLOCK, ADAPT_C = 51, 12
OTSU_MIN_SEPARATION = 45  

OPEN_K, CLOSE_K = 3, 9  
MIN_COMPONENT_AREA = 30  


RULE = {
    "min_fg_ratio": 0.010,       
    "max_fg_ratio": 0.300,       
    "min_largest_cc_area": 1500,  
    "min_largest_cc_width": 0.35,  
    "min_largest_cc_height": 0.30,  
}



def load_upright(path):
    """Citra scan tersimpan miring 90 derajat -> putar searah jarum jam."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def crop_roi(img, frac):
    h, w = img.shape[:2]
    x0, x1, y0, y1 = frac
    roi = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    s = NORM_WIDTH / roi.shape[1]
    return cv2.resize(roi, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def to_gray(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def threshold_global(gray, t=GLOBAL_T):
    _, m = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)
    return m, t


def threshold_otsu(gray):
    t, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fg, bg = gray[m > 0], gray[m == 0]
    if fg.size == 0 or bg.size == 0 or (bg.mean() - fg.mean()) < OTSU_MIN_SEPARATION:
        m = np.zeros_like(m)
    return m, t


def threshold_adaptive(gray):
    m = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, ADAPT_BLOCK, ADAPT_C)
    return m, None


def morphology(mask):
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kc)
    return closed


def area_features(mask):
    h, w = mask.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    comps = [(stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_WIDTH],
              stats[i, cv2.CC_STAT_HEIGHT]) for i in range(1, n)
             if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA]
    fg = int(sum(c[0] for c in comps))
    if comps:
        a, cw, ch = max(comps, key=lambda c: c[0])
    else:
        a = cw = ch = 0
    return {
        "fg_pixels": fg,
        "fg_ratio": fg / float(h * w),
        "n_components": len(comps),
        "largest_cc_area": int(a),
        "largest_cc_w": cw / float(w),
        "largest_cc_h": ch / float(h),
    }


def decide(f):
    ok = (RULE["min_fg_ratio"] <= f["fg_ratio"] <= RULE["max_fg_ratio"]
          and f["largest_cc_area"] >= RULE["min_largest_cc_area"]
          and f["largest_cc_w"] >= RULE["min_largest_cc_width"]
          and f["largest_cc_h"] >= RULE["min_largest_cc_height"])
    return "SIGNATURE PRESENT" if ok else "SIGNATURE ABSENT"


METHODS = {"global": threshold_global, "otsu": threshold_otsu, "adaptive": threshold_adaptive}


def analyse_roi(roi_bgr):
    gray = to_gray(roi_bgr)
    out = {"gray": gray}
    for name, fn in METHODS.items():
        raw, t = fn(gray)
        clean = morphology(raw)
        feats = area_features(clean)
        out[name] = {"raw": raw, "clean": clean, "t": t, "feat": feats, "decision": decide(feats)}
    return out


# ----------------------------------------------------------------------------
# VISUALISASI
# ----------------------------------------------------------------------------
def save_pipeline_figure(roi, res, title, path):
    fig, ax = plt.subplots(3, 4, figsize=(16, 7))
    ax[0, 0].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)); ax[0, 0].set_title("ROI (crop)")
    ax[1, 0].imshow(res["gray"], cmap="gray", vmin=0, vmax=255); ax[1, 0].set_title("Grayscale")
    ax[2, 0].hist(res["gray"].ravel(), 64, color="gray"); ax[2, 0].set_title("Histogram")
    ax[2, 0].axvline(GLOBAL_T, color="r", ls="--", label="global")
    ax[2, 0].axvline(res["otsu"]["t"], color="b", ls="--", label="Otsu"); ax[2, 0].legend()
    for j, name in enumerate(METHODS, start=1):
        r = res[name]
        ax[0, j].imshow(r["raw"], cmap="gray"); ax[0, j].set_title(f"{name}: threshold")
        ax[1, j].imshow(r["clean"], cmap="gray")
        ax[1, j].set_title(f"{name}: + opening/closing")
        f = r["feat"]
        ax[2, j].axis("off")
        ax[2, j].text(0, 0.9, f"fg_pixels  : {f['fg_pixels']}\nfg_ratio   : {f['fg_ratio']:.4f}\n"
                      f"komponen   : {f['n_components']}\nCC terbesar: {f['largest_cc_area']} px\n"
                      f"lebar CC   : {f['largest_cc_w']:.2f}\ntinggi CC  : {f['largest_cc_h']:.2f}\n\n"
                      f"{r['decision']}", va="top", family="monospace", fontsize=10)
    for a in ax[:2].ravel():
        a.axis("off")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def save_threshold_sweep(roi, path):
    gray = to_gray(roi)
    ts = [30, 70, 110, 170, 215, 235, 250]
    fig, ax = plt.subplots(1, len(ts) + 1, figsize=(20, 3))
    ax[0].imshow(gray, cmap="gray", vmin=0, vmax=255); ax[0].set_title("gray")
    for a, t in zip(ax[1:], ts):
        _, m = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)
        a.imshow(m, cmap="gray"); a.set_title(f"T = {t}\nfg = {np.count_nonzero(m) / m.size:.1%}")
    for a in ax:
        a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="./images")
    ap.add_argument("--output_dir", default="./output")
    ap.add_argument("--method", default="otsu", choices=list(METHODS),
                    help="metode yang dipakai sebagai keputusan utama (default: otsu)")
    args = ap.parse_args()

    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.jpg")) +
                   glob.glob(os.path.join(args.input_dir, "*.png")))
    if not files:
        raise SystemExit(f"Tidak ada citra di {args.input_dir}")

    rows = []
    for fp in files:
        base = os.path.splitext(os.path.basename(fp))[0]
        img = load_upright(fp)
        for roi_name, frac in ROIS.items():
            roi = crop_roi(img, frac)
            res = analyse_roi(roi)
            cv2.imwrite(os.path.join(fig_dir, f"{base}__{roi_name}_roi.png"), roi)
            if base.startswith(("01_", "03_", "06_", "09_")) and roi_name != "blank":
                save_pipeline_figure(roi, res, f"{base} | ROI = {roi_name}",
                                     os.path.join(fig_dir, f"{base}__{roi_name}_pipeline.png"))
            if base.startswith(("01_", "06_")) and roi_name == "signature":
                save_threshold_sweep(roi, os.path.join(fig_dir, f"{base}__threshold_sweep.png"))
            truth = GROUND_TRUTH[roi_name]
            row = {"image": base, "roi": roi_name, "truth": "PRESENT" if truth else "ABSENT"}
            for m in METHODS:
                f = res[m]["feat"]
                row[f"{m}_decision"] = res[m]["decision"].split()[-1]
                row[f"{m}_correct"] = (res[m]["decision"] == "SIGNATURE PRESENT") == truth
                row[f"{m}_fg_pixels"] = f["fg_pixels"]
                row[f"{m}_fg_ratio"] = round(f["fg_ratio"], 4)
                row[f"{m}_largest_cc_w"] = round(f["largest_cc_w"], 3)
            rows.append(row)

    csv_path = os.path.join(args.output_dir, "results.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ringkasan
    print(f"\n{'Citra':34s}{'ROI':12s}{'Truth':9s}" + "".join(f"{m:>11s}" for m in METHODS))
    for r in rows:
        print(f"{r['image']:34s}{r['roi']:12s}{r['truth']:9s}" +
              "".join(f"{r[m + '_decision']:>11s}" for m in METHODS))
    print("\nAkurasi (jumlah benar / total):")
    summary = []
    for m in METHODS:
        c = sum(r[f"{m}_correct"] for r in rows)
        summary.append((m, c, len(rows)))
        print(f"  {m:9s}: {c}/{len(rows)} = {c / len(rows):.1%}")
    with open(os.path.join(args.output_dir, "accuracy.txt"), "w") as fh:
        for m, c, n in summary:
            fh.write(f"{m}\t{c}/{n}\t{c / n:.3f}\n")

    # keputusan akhir per citra (metode utama)
    print(f"\nKeputusan akhir (metode {args.method}) - ROI tanda tangan Dekan:")
    for r in rows:
        if r["roi"] == "signature":
            print(f"  {r['image']:34s} -> SIGNATURE {r[args.method + '_decision']}")
    print(f"\nHasil tersimpan di: {args.output_dir}")


if __name__ == "__main__":
    main()
