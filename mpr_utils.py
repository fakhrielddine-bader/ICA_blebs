"""
Lecture / préparation / export des volumes pour le viewer MPR (app_mpr.py).

- Lecture : NIfTI (.nii, .nii.gz ; nibabel) et NRRD (.nrrd, dont les .seg.nrrd de 3D Slicer ; pynrrd).
- Les volumes sont remis en orientation RAS+ (axe 0 : vers la droite du patient, axe 1 : vers l'avant,
  axe 2 : vers le haut), recadrés autour de la segmentation, puis compressés (gzip + base64) pour le navigateur.
- L'export remet la segmentation corrigée dans l'orientation et l'espace du fichier d'origine : le NIfTI écrit
  se superpose exactement à l'image source.

Ce module n'utilise que numpy ; nibabel et pynrrd ne sont importés qu'au moment de lire / écrire.
"""

import base64
import gzip
from pathlib import Path

import numpy as np



# --------------------------------------------------------------------------- #
# Affine d'un NRRD
# --------------------------------------------------------------------------- #
def affine_from_nrrd_header(hdr) -> np.ndarray:
    """Affine voxel -> mm (repère RAS, convention NIfTI) à partir de l'en-tête d'un fichier NRRD.

    Les NRRD de 3D Slicer sont en LPS : on inverse les axes x et y pour passer en RAS."""
    dirs = hdr.get("space directions")
    if dirs is None:
        raise ValueError("en-tête NRRD sans « space directions »")
    dirs = np.asarray(dirs, dtype=float).reshape(-1, 3)
    dirs = dirs[~np.isnan(dirs).any(axis=1)]  # ignore les axes non spatiaux (« none »)
    if dirs.shape != (3, 3):
        raise ValueError(f"« space directions » inattendu : {dirs.shape}")
    origin = np.asarray(hdr.get("space origin", np.zeros(3)), dtype=float).reshape(3)
    aff = np.eye(4)
    aff[:3, :3] = dirs.T  # colonne i = vecteur de l'axe i
    aff[:3, 3] = origin
    space = str(hdr.get("space", "")).strip().lower().replace("_", "-")
    if space in ("left-posterior-superior", "lps"):
        aff = np.diag([-1.0, -1.0, 1.0, 1.0]) @ aff
    elif space in ("right-anterior-superior", "ras"):
        pass
    else:
        raise ValueError(f"espace NRRD non pris en charge : {space!r} (LPS ou RAS attendus)")
    return aff


# --------------------------------------------------------------------------- #
# Orientation
# --------------------------------------------------------------------------- #
def ras_spec(affine) -> tuple[list[int], list[bool]]:
    """Pour chaque axe RAS m : (axe du tableau d'origine qui lui correspond, retourner ou non)."""
    A = np.asarray(affine, dtype=float)[:3, :3]
    world = np.argmax(np.abs(A), axis=0)  # axe monde dominant de chaque axe du tableau
    if len(set(world.tolist())) != 3:
        raise ValueError("orientation du volume non prise en charge (axes obliques ambigus)")
    sign = np.sign(A[world, np.arange(3)])
    order = [int(np.where(world == m)[0][0]) for m in range(3)]
    flips = [bool(sign[order[m]] < 0) for m in range(3)]
    return order, flips


def to_ras(data, affine):
    """Renvoie (volume en RAS+, pas en mm [x, y, z], spec pour revenir en arrière)."""
    order, flips = ras_spec(affine)
    A = np.asarray(affine, dtype=float)[:3, :3]
    out = np.transpose(data, order)
    for m in range(3):
        if flips[m]:
            out = np.flip(out, axis=m)
    spacing = [float(np.linalg.norm(A[:, order[m]])) for m in range(3)]
    return np.ascontiguousarray(out), spacing, (order, flips)


def from_ras(data_ras, spec):
    """Opération inverse de `to_ras` : retrouve l'ordre des axes et le sens du tableau d'origine."""
    order, flips = spec
    out = data_ras
    for m in range(3):
        if flips[m]:
            out = np.flip(out, axis=m)
    return np.ascontiguousarray(np.transpose(out, np.argsort(order)))


# --------------------------------------------------------------------------- #
# Lecture
# --------------------------------------------------------------------------- #
def read_volume(path):
    """Renvoie (tableau 3D, affine voxel -> mm en RAS, en-tête NIfTI d'origine ou None)."""
    name = str(path).lower()
    if name.endswith((".nii", ".nii.gz")):
        import nibabel as nib

        img = nib.load(str(path))
        raw = np.asarray(img.dataobj)
        if raw.ndim == 4 and raw.shape[3] == 1:
            raw = raw[..., 0]
        if raw.ndim != 3:
            raise ValueError(f"volume à {raw.ndim} dimensions non pris en charge : {Path(path).name}")
        return raw, np.asarray(img.affine, dtype=float), img.header
    if name.endswith(".nrrd"):
        import nrrd  # pynrrd

        raw, hdr = nrrd.read(str(path))
        if raw.ndim != 3:
            raise ValueError(
                f"NRRD à {raw.ndim} dimensions non pris en charge (segmentation à calques multiples ?) : {Path(path).name}"
            )
        return raw, affine_from_nrrd_header(hdr), None
    raise ValueError(f"format non pris en charge : {Path(path).name} (.nii, .nii.gz ou .nrrd attendus)")


def to_label_array(raw) -> np.ndarray:
    lab = np.rint(np.asarray(raw)).astype(np.int64)
    if lab.size and (lab.min() < 0 or lab.max() > 255):
        raise ValueError(f"labels hors de 0..255 (min {lab.min()}, max {lab.max()})")
    return lab.astype(np.uint8)


def normalize_uint8(img, p_lo=0.5, p_hi=99.8) -> np.ndarray:
    """Image en 8 bits pour l'affichage, par écrêtage sur percentiles (robuste aux valeurs extrêmes)."""
    arr = np.asarray(img, dtype=np.float32)
    sample = arr[:: max(1, arr.shape[0] // 64)].ravel()
    lo, hi = np.percentile(sample, [p_lo, p_hi])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def crop_bounds(mask, margin: int):
    """Boîte englobante (début inclus, fin exclue) des voxels non nuls, élargie de `margin` voxels."""
    if not mask.any():
        raise ValueError("segmentation vide (aucun voxel non nul)")
    lo, hi = [], []
    for d in range(3):
        axes = tuple(a for a in range(3) if a != d)
        idx = np.flatnonzero(np.any(mask, axis=axes))
        lo.append(max(0, int(idx[0]) - margin))
        hi.append(min(mask.shape[d], int(idx[-1]) + 1 + margin))
    return lo, hi


def pack(arr) -> str:
    """Tableau 3D uint8 -> gzip -> base64, dans l'ordre mémoire attendu par le viewer (x le plus rapide)."""
    raw = np.asarray(arr, dtype=np.uint8).ravel(order="F").tobytes()
    return base64.b64encode(gzip.compress(raw, compresslevel=4)).decode("ascii")


def unpack(b64: str, dims) -> np.ndarray:
    raw = gzip.decompress(base64.b64decode(b64))
    if len(raw) != int(np.prod(dims)):
        raise ValueError(f"taille inattendue : {len(raw)} octets pour {tuple(dims)}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(tuple(dims), order="F").copy()


def label_counts(labels) -> dict:
    vals, cnt = np.unique(np.asarray(labels), return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnt) if v != 0}


def counts_to_str(counts: dict) -> str:
    return "; ".join(f"{k}:{v}" for k, v in sorted(counts.items()))


# --------------------------------------------------------------------------- #
# Un cas : image + segmentation
# --------------------------------------------------------------------------- #
def load_case(image_path, seg_path, margin=12, max_voxels=80_000_000) -> dict:
    """Charge l'image (facultative) et la segmentation, les met en RAS+ et les recadre."""
    seg_raw, seg_aff, seg_hdr = read_volume(seg_path)
    seg_ras, spacing, spec = to_ras(to_label_array(seg_raw), seg_aff)

    img_ras = None
    if image_path:
        img_raw, img_aff, _ = read_volume(image_path)
        img_ras, _, _ = to_ras(img_raw, img_aff)
        if img_ras.shape != seg_ras.shape:
            raise ValueError(
                f"l'image {tuple(img_ras.shape)} et la segmentation {tuple(seg_ras.shape)} n'ont pas la même "
                "grille (rééchantillonnez-les sur la même grille, par exemple dans 3D Slicer)"
            )
        if not np.allclose(img_aff, seg_aff, rtol=1e-3, atol=1e-2):
            raise ValueError("l'image et la segmentation n'ont pas la même géométrie (origine / orientation / pas)")

    lo, hi = crop_bounds(seg_ras > 0, margin)
    window = tuple(slice(a, b) for a, b in zip(lo, hi))
    labels = np.ascontiguousarray(seg_ras[window])
    if labels.size > max_voxels:
        raise ValueError(f"volume recadré trop grand ({labels.size:,} voxels > {max_voxels:,}) : recadrez-le")
    image = normalize_uint8(img_ras[window]) if img_ras is not None else None

    pts = np.argwhere(labels > 0)
    center = [int(round(x)) for x in pts.mean(axis=0)]
    return {
        "dims": [int(x) for x in labels.shape],
        "spacing": spacing,
        "labels": labels,
        "image": image,
        "center": center,
        "present": sorted(int(v) for v in np.unique(labels) if v != 0),
        "payload": {
            "labels_gz": pack(labels),
            "image_gz": pack(image) if image is not None else None,
        },
        # pour l'export
        "crop": (lo, hi),
        "full_shape": [int(x) for x in seg_ras.shape],
        "spec": spec,
        "seg_path": str(seg_path),
        "seg_affine": seg_aff,
        "seg_header": seg_hdr,
    }


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def restore_native(edited_crop, case) -> np.ndarray:
    """Segmentation éditée (recadrée, RAS+) -> tableau complet dans l'orientation du fichier d'origine."""
    lo, hi = case["crop"]
    full = np.zeros(case["full_shape"], dtype=np.uint8)
    full[tuple(slice(a, b) for a, b in zip(lo, hi))] = edited_crop
    return from_ras(full, case["spec"])


def save_labels_nifti(native_labels, case, out_path) -> Path:
    """Écrit les labels (ordre du fichier d'origine) en NIfTI uint8, avec l'affine de la segmentation source."""
    import nibabel as nib

    affine = case["seg_affine"]
    data = np.ascontiguousarray(native_labels, dtype=np.uint8)
    if case["seg_header"] is not None:  # source NIfTI : on garde son en-tête (qform / sform)
        hdr = case["seg_header"].copy()
        hdr.set_data_dtype(np.uint8)
        hdr.set_slope_inter(1, 0)
        img = nib.Nifti1Image(data, affine, hdr)
    else:  # source NRRD : en-tête neuf, repère « scanner »
        img = nib.Nifti1Image(data, affine)
        img.set_sform(affine, code=1)
        img.set_qform(affine, code=1)
        img.header.set_data_dtype(np.uint8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))
    return out_path
