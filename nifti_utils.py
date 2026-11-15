"""
Conversion d'une correction faite sur la SURFACE (labels par sommet) en masque VOXEL, et export NIfTI.

Principe
--------
Le viewer corrige des labels aux sommets de la surface. Pour retrouver un masque voxel :
  - on part du volume de labels d'origine (rien n'est modifié loin des corrections) ;
  - sommets peints  (non-bleb -> bleb) : les voxels d'origine « parent » situés à moins de
    `depth_mm` d'un sommet peint passent à bleb ;
  - sommets effacés (bleb -> non-bleb) : les voxels bleb situés à moins de `depth_mm` d'un sommet
    effacé redeviennent « parent » (anévrisme) ;
  - un voxel proche des deux types de sommets prend le type du sommet le plus proche.
On ne crée jamais de voxel hors du masque segmenté : on ne fait que changer le label de voxels existants.

Ce module n'utilise que numpy (nibabel n'est importé que pour écrire le fichier).
"""

import itertools
from pathlib import Path

import numpy as np


def physical_to_index(points, origin, spacing, direction=None) -> np.ndarray:
    """Coordonnées physiques (N,3) -> indices de voxel continus (N,3).

    Convention VTK : physique = origine + D @ (spacing * ijk)."""
    pts = np.asarray(points, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    if direction is not None:
        pts = pts @ np.linalg.inv(np.asarray(direction, dtype=np.float64).reshape(3, 3)).T
    return pts / np.asarray(spacing, dtype=np.float64)


def alignment_ok(labels_vol, vertex_ijk, n_check=2000, min_fraction=0.95) -> bool:
    """Contrôle que les sommets tombent bien sur/près du masque dans l'espace voxel.

    Protège contre une géométrie mal interprétée (volume oblique, etc.) : dans ce cas, l'export
    est désactivé plutôt que de produire un masque décalé."""
    n = len(vertex_ijk)
    if n == 0:
        return False
    sel = np.linspace(0, n - 1, min(n, n_check)).astype(int)
    idx = np.rint(vertex_ijk[sel]).astype(int)
    dims = np.asarray(labels_vol.shape)
    inside = np.all((idx >= 0) & (idx < dims), axis=1)
    hit = np.zeros(len(sel), dtype=bool)
    for off in itertools.product((-1, 0, 1), repeat=3):
        c = idx + off
        ok = np.all((c >= 0) & (c < dims), axis=1)
        vals = np.zeros(len(sel), dtype=labels_vol.dtype)
        vals[ok] = labels_vol[c[ok, 0], c[ok, 1], c[ok, 2]]
        hit |= vals != 0
    return float((hit & inside).mean()) >= min_fraction


def apply_surface_corrections(
    labels_vol, vertex_ijk, init_lab, new_lab, spacing, depth_mm, bleb_label, parent_label
):
    """Renvoie (volume corrigé, statistiques). `labels_vol` n'est pas modifié."""
    out = labels_vol.copy()
    stats = {"voxels_added": 0, "voxels_removed": 0}

    init_lab = np.asarray(init_lab)
    new_lab = np.asarray(new_lab)
    add = (new_lab == bleb_label) & (init_lab != bleb_label)
    rem = (init_lab == bleb_label) & (new_lab != bleb_label)
    if not (add.any() or rem.any()):
        return out, stats

    sp = np.asarray(spacing, dtype=float)
    dims = np.asarray(labels_vol.shape)
    rad = np.ceil(depth_mm / sp).astype(int) + 1
    changed_pts = vertex_ijk[add | rem]
    lo = np.maximum(np.floor(changed_pts.min(axis=0)).astype(int) - rad, 0)
    hi = np.minimum(np.ceil(changed_pts.max(axis=0)).astype(int) + rad, dims - 1)
    shape = tuple(hi - lo + 1)
    half_diag = 0.5 * float(np.linalg.norm(sp))

    # décalages utiles (ceux dont la distance minimale possible est <= depth_mm)
    offsets = [
        o
        for o in itertools.product(*(range(-r, r + 1) for r in rad))
        if float(np.linalg.norm(np.asarray(o) * sp)) - half_diag <= depth_mm
    ]

    def dist_field(sel):
        d = np.full(shape, np.inf, dtype=np.float32)
        pts = vertex_ijk[sel]
        if len(pts) == 0:
            return d
        base = np.rint(pts).astype(int)
        for o in offsets:
            cand = base + np.asarray(o)
            dist = np.sqrt((((cand - pts) * sp) ** 2).sum(axis=1))
            ok = (dist <= depth_mm) & np.all((cand >= lo) & (cand <= hi), axis=1)
            if not ok.any():
                continue
            c = cand[ok] - lo
            np.minimum.at(d, (c[:, 0], c[:, 1], c[:, 2]), dist[ok].astype(np.float32))
        return d

    d_add = dist_field(add)
    d_rem = dist_field(rem)

    window = tuple(slice(int(a), int(b) + 1) for a, b in zip(lo, hi))
    orig = labels_vol[window]
    allowed = np.unique(init_lab[add]) if add.any() else np.array([], dtype=init_lab.dtype)
    add_vox = (d_add <= depth_mm) & (d_add < d_rem) & np.isin(orig, allowed)
    rem_vox = (d_rem <= depth_mm) & (d_rem <= d_add) & (orig == bleb_label)

    sub = out[window]  # vue : l'écriture modifie `out`
    sub[add_vox] = bleb_label
    sub[rem_vox] = parent_label
    stats["voxels_added"] = int(add_vox.sum())
    stats["voxels_removed"] = int(rem_vox.sum())
    return out, stats


def save_corrected_nifti(source_path, vtk_labels, corrected, out_path) -> Path:
    """Écrit le masque corrigé en NIfTI, dans le repère du fichier d'origine.

    Le lecteur VTK peut retourner des axes par rapport au tableau du fichier : on retrouve ces
    retournements en comparant le volume VTK au tableau lu par nibabel, puis on les applique au
    volume corrigé. Si aucune correspondance exacte n'est trouvée, on refuse d'écrire un fichier
    potentiellement décalé."""
    import nibabel as nib  # import tardif : n'est requis que pour l'export

    img = nib.load(str(source_path))
    raw = np.asarray(img.dataobj)
    if any(s != 1 for s in raw.shape[3:]):
        raise ValueError("volume 4D non pris en charge")
    native = np.rint(raw.reshape(raw.shape[:3])).astype(vtk_labels.dtype)

    chosen = None
    for flips in itertools.product((False, True), repeat=3):
        axes = tuple(i for i, f in enumerate(flips) if f)
        cand = np.flip(vtk_labels, axis=axes) if axes else vtk_labels
        if cand.shape == native.shape and np.array_equal(cand, native):
            chosen = axes
            break
    if chosen is None:
        raise ValueError("orientation du volume VTK non reconnue par rapport au fichier NIfTI")

    out_native = np.flip(corrected, axis=chosen) if chosen else corrected
    data = np.ascontiguousarray(out_native.reshape(raw.shape)).astype(np.uint8)

    hdr = img.header.copy()
    hdr.set_data_dtype(np.uint8)
    hdr.set_slope_inter(1, 0)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, img.affine, hdr), str(out_path))
    return out_path
