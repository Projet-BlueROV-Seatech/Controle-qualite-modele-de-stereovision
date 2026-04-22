#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════╗
║          Outil de Contrôle Qualité Stéréo 3D             ║
╠══════════════════════════════════════════════════════════╣
║  Simule un système de vision binoculaire pour le CQ.     ║
║                                                          ║
║  Modèles mathématiques :                                 ║
║   • Projection sténopé : P = K @ [R | t]  (3×4)         ║
║   • Triangulation DLT  : x × (P·X) = 0   → SVD          ║
║   • Convention P2 identique à Passage.py                 ║
║     P2 = K2 @ [R_c2_c1 | t_c2_c1]                       ║
║                                                          ║
║  Repère monde = Repère Caméra 1                          ║
║   • X : droite          • Y : bas (convention image)     ║
║   • Z : profondeur      • Caméra 1 à l'origine           ║
║                                                          ║
║  Usage :                                                 ║
║   1. Ajuster position et lacet du pavé (sliders)         ║
║   2. Cliquer-glisser pour encadrer le pavé sur            ║
║      CHACUNE des deux vues caméra                        ║
║   3. Cliquer "▶ Trianguler"                              ║
║   4. Lire les erreurs angulaires et en pixels            ║
╚══════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider, Button

# ══════════════════════════════════════════════════════════
#  1.  PARAMÈTRES DES CAMÉRAS
# ══════════════════════════════════════════════════════════
IMG_W, IMG_H = 640, 480     # résolution virtuelle (pixels)
FOCAL        = 640.0       # distance focale (pixels)
cx0          = IMG_W / 2.   # point principal u
cy0          = IMG_H / 2.   # point principal v

# Matrice intrinsèque (identique pour les deux caméras dans la simulation)
K1 = np.array([[FOCAL, 0.,   cx0],
               [0.,   FOCAL, cy0],
               [0.,   0.,    1. ]])
K2 = K1.copy()

# ── Pose relative caméra 2 / caméra 1 ────────────────────
#  Convention Passage.py :
#    P2 = K2 @ [R_c2_c1 | t_c2_c1]
#    X_cam2 = R_c2_c1 @ X_monde + t_c2_c1
#
#  Géométrie stéréo :
#    Baseline 0.40 m horizontal, légère convergence de 8° (toe-in)
BASELINE = 0.40          # mètres
CONV_DEG = 90.0           # degré de convergence (toe-in)
_c = np.cos(np.radians(CONV_DEG))
_s = np.sin(np.radians(CONV_DEG))

# Rotation de convergence autour de Y (cam2 regarde légèrement à gauche)
R_c2_c1 = np.array([[ _c, 0.,  _s],
                     [ 0., 1.,  0.],
                     [-_s, 0.,  _c]])

# Centre optique de cam2 dans le repère monde : C2 = (BASELINE, 0, 0)
C2_monde  = np.array([1.414, 0., 1.414])
t_c2_c1   = (-R_c2_c1 @ C2_monde).reshape(3, 1)

# Matrices de projection 3×4  (P = K @ [R | t])
P1 = K1 @ np.hstack((np.eye(3),  np.zeros((3, 1))))   # cam1 à l'origine
P2 = K2 @ np.hstack((R_c2_c1,   t_c2_c1))              # cam2 relative






# DOSSIER = r"calibrage direct"
# K1 = np.load(os.path.join(DOSSIER, "K1.npy"))
# K2 = np.load(os.path.join(DOSSIER, "K2.npy"))
# R  = np.load(os.path.join(DOSSIER, "R_c2_c1.npy"))
# T  = np.load(os.path.join(DOSSIER, "t_c2_c1.npy"))



# ══════════════════════════════════════════════════════════
#  2.  GÉOMÉTRIE DU PAVÉ DROIT
# ══════════════════════════════════════════════════════════
BOX_L, BOX_W, BOX_H = 0.50, 0.30, 0.20   # longueur X, largeur Z, hauteur Y (m)

# Numérotation des 8 coins : 0-3 face -Z, 4-7 face +Z
#   0=(-l,-h,-w)  1=(+l,-h,-w)  2=(+l,+h,-w)  3=(-l,+h,-w)
#   4=(-l,-h,+w)  5=(+l,-h,+w)  6=(+l,+h,+w)  7=(-l,+h,+w)
FACES = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[3,2,6,7],[0,3,7,4],[1,2,6,5]]
EDGES = [(0,1),(1,2),(2,3),(3,0),
         (4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]


def box_corners(cx, cy, cz, yaw_deg):
    """Retourne les 8 coins du pavé dans le repère monde (Nx3)."""
    hl, hw, hh = BOX_L/2, BOX_W/2, BOX_H/2
    loc = np.array([[-hl,-hh,-hw],[ hl,-hh,-hw],[ hl, hh,-hw],[-hl, hh,-hw],
                    [-hl,-hh, hw],[ hl,-hh, hw],[ hl, hh, hw],[-hl, hh, hw]])
    yaw = np.radians(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    Ry   = np.array([[c, 0., s],[0., 1., 0.],[-s, 0., c]])
    return (Ry @ loc.T).T + np.array([cx, cy, cz])


# ══════════════════════════════════════════════════════════
#  3.  PROJECTION ET TRIANGULATION
# ══════════════════════════════════════════════════════════

def proj(Xw, P):
    """Projection perspective d'un point 3D monde en pixel.
    Retourne (u, v) ou None si le point est derrière la caméra."""
    ph = P @ np.array([Xw[0], Xw[1], Xw[2], 1.])
    if ph[2] < 1e-6:
        return None
    return ph[:2] / ph[2]


def triangulate_DLT(u1, v1, u2, v2):
    """Triangulation linéaire DLT à partir de deux pixels correspondants.

    Résout le système A·X = 0 par SVD (méthode du projet_sysmer_complet).

         x1[0]·P1[2] - P1[0]   ⎤
         x1[1]·P1[2] - P1[1]   ⎥ · X = 0
         x2[0]·P2[2] - P2[0]   ⎥
         x2[1]·P2[2] - P2[1]   ⎦
    """
    x1 = np.array([u1, v1, 1.])
    x2 = np.array([u2, v2, 1.])
    A = np.array([
        x1[0]*P1[2] - P1[0],
        x1[1]*P1[2] - P1[1],
        x2[0]*P2[2] - P2[0],
        x2[1]*P2[2] - P2[1],
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]     # coordonnées cartésiennes 3D


def angular_error(p_true, p_est):
    """Erreur angulaire (degrés) et distance (pixels) entre deux points image."""
    def ray(p):
        v = np.array([(p[0] - cx0) / FOCAL, (p[1] - cy0) / FOCAL, 1.])
        return v / np.linalg.norm(v)
    cos_a = np.clip(np.dot(ray(p_true), ray(p_est)), -1., 1.)
    ang   = np.degrees(np.arccos(cos_a))
    dpx   = float(np.linalg.norm(np.array(p_true) - np.array(p_est)))
    return ang, dpx


# ══════════════════════════════════════════════════════════
#  4.  ÉTAT GLOBAL
# ══════════════════════════════════════════════════════════
st = dict(
    bx=0.0, by=0.0, bz=2.0, yaw=0.0,   # position et lacet du pavé
    r1=None,   # rectangle cam1 : (xmin, ymin, w, h)
    r2=None,   # rectangle cam2
    c1=None,   # centre du rectangle cam1 : (u, v)
    c2=None,   # centre du rectangle cam2
    pos3d=None # position 3D triangulée
)
_drag  = dict(ax=None, x0=0., y0=0., patch=None, cam=0)
_view  = [22., -50.]    # [elev, azim] sauvegardés entre les mises à jour 3D


# ══════════════════════════════════════════════════════════
#  5.  FIGURE ET AXES
# ══════════════════════════════════════════════════════════
fig = plt.figure(figsize=(17, 10), facecolor='#0b0e18')
try:
    fig.canvas.manager.set_window_title('Contrôle Qualité Stéréo 3D')
except Exception:
    pass

# ── Vue 3D (gauche)
ax3d = fig.add_axes([0.01, 0.33, 0.36, 0.63], projection='3d')
ax3d.set_facecolor('#0b0e18')

# ── Vues caméra (droite, haut = cam1, bas = cam2)
ax_c1 = fig.add_axes([0.39, 0.66, 0.59, 0.30])
ax_c2 = fig.add_axes([0.39, 0.33, 0.59, 0.30])

for ax, col in [(ax_c1, '#ff8855'), (ax_c2, '#5599ff')]:
    ax.set_facecolor('#05080f')
    for sp in ax.spines.values():
        sp.set_edgecolor(col)
        sp.set_linewidth(1.5)

# ── Sliders
ax_sx   = fig.add_axes([0.04, 0.24, 0.30, 0.025])
ax_sy   = fig.add_axes([0.04, 0.19, 0.30, 0.025])
ax_sz   = fig.add_axes([0.04, 0.14, 0.30, 0.025])
ax_syaw = fig.add_axes([0.04, 0.09, 0.30, 0.025])

# ── Bouton triangulation
ax_btn = fig.add_axes([0.50, 0.12, 0.18, 0.065])

# ── Zone d'erreurs
ax_err = fig.add_axes([0.39, 0.02, 0.59, 0.09])
ax_err.axis('off')
ax_err.set_facecolor('#080c16')
err_txt = ax_err.text(
    0.01, 0.95,
    'Aucune triangulation — dessiner un rectangle sur chaque vue puis cliquer "▶ Trianguler".',
    transform=ax_err.transAxes, fontsize=9,
    va='top', color='#99bbff', fontfamily='monospace'
)

# ── Légendes fixes
fig.text(0.20, 0.978, '⚙  Contrôle Qualité — Vision Stéréo 3D',
         fontsize=13, color='white', fontweight='bold', ha='center', va='top')
fig.text(0.02, 0.315, 'Paramètres du pavé :', fontsize=9,
         color='#8899aa', ha='left', va='top')
# fig.text(0.39, 0.975, '● Rouge = centre vrai   △ Vert = position estimée   ■ Cyan = sélection',
#          fontsize=8, color='#8899aa', ha='left', va='top')
fig.text(0.39, 0.975, '',
         fontsize=8, color='#8899aa', ha='left', va='top')


# ══════════════════════════════════════════════════════════
#  6.  AIDE À LA VISUALISATION 3D
# ══════════════════════════════════════════════════════════

# Coordonnées des arêtes de la pyramide caméra (indices sur 5 pts : apex + 4 coins)
_CAM_EDGES = [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)]


def camera_frustum_world(pos, Rcw, scale=0.12):
    """Retourne les 5 sommets de la pyramide d'une caméra dans le repère monde.
    pos  : centre optique dans le monde (3,)
    Rcw  : matrice de rotation monde→caméra (3×3)
    """
    pts_loc = np.array([
        [0.,      0.,        0.       ],   # 0 : apex
        [-scale, -scale*0.75, scale*2.],   # 1
        [ scale, -scale*0.75, scale*2.],   # 2
        [ scale,  scale*0.75, scale*2.],   # 3
        [-scale,  scale*0.75, scale*2.],   # 4
    ])
    Rwc = Rcw.T   # caméra → monde
    return (Rwc @ pts_loc.T).T + pos


def _mpl3(pts):
    """Convertit N points monde (X, Y_bas, Z_profond)
    vers matplotlib 3D (X_mpl, Z_profond, -Y = hauteur vers le haut).
    """
    pts = np.atleast_2d(pts)
    return np.column_stack([pts[:, 0], pts[:, 2], -pts[:, 1]])


# ══════════════════════════════════════════════════════════
#  7.  DESSIN 3D
# ══════════════════════════════════════════════════════════

def draw_3d():
    """Redessine la vue 3D en préservant l'angle de vue courant."""
    try:
        _view[0] = ax3d.elev
        _view[1] = ax3d.azim
    except Exception:
        pass
    ax3d.cla()
    ax3d.set_facecolor('#0b0e18')
    ax3d.xaxis.pane.set_facecolor('#0b0e18')
    ax3d.yaxis.pane.set_facecolor('#0b0e18')
    ax3d.zaxis.pane.set_facecolor('#0b0e18')
    ax3d.xaxis.pane.set_edgecolor('#1a2030')
    ax3d.yaxis.pane.set_edgecolor('#1a2030')
    ax3d.zaxis.pane.set_edgecolor('#1a2030')

    corners = box_corners(st['bx'], st['by'], st['bz'], st['yaw'])
    center  = np.array([st['bx'], st['by'], st['bz']])

    # Convertir vers coordonnées matplotlib 3D
    c3    = _mpl3(corners)
    ctr3  = _mpl3([center])[0]

    # ── Faces semi-transparentes du pavé
    verts = [[c3[i] for i in f] for f in FACES]
    ax3d.add_collection3d(
        Poly3DCollection(verts, alpha=0.13, facecolor='#2255cc',
                         edgecolor='#6688ee', lw=0.7))

    # ── Arêtes du pavé
    for i, j in EDGES:
        ax3d.plot([c3[i,0], c3[j,0]],
                  [c3[i,1], c3[j,1]],
                  [c3[i,2], c3[j,2]],
                  color='#7799ee', lw=1.2, alpha=0.9)

    # ── Centre rouge (toujours visible, depthshade=False)
    ax3d.scatter(ctr3[0], ctr3[1], ctr3[2],
                 color='red', s=100, zorder=10, depthshade=False)

    # ── Caméra 1 (à l'origine)
    fr1   = _mpl3(camera_frustum_world(np.zeros(3), np.eye(3)))
    ctr_c1 = _mpl3([np.zeros(3)])[0]
    for i, j in _CAM_EDGES:
        ax3d.plot([fr1[i,0], fr1[j,0]], [fr1[i,1], fr1[j,1]],
                  [fr1[i,2], fr1[j,2]], color='#ff7755', lw=1.5)
    ax3d.text(ctr_c1[0] - 0.05, ctr_c1[1], ctr_c1[2] + 0.08,
              'C1', color='#ff8866', fontsize=8, fontweight='bold')

    # ── Caméra 2
    fr2   = _mpl3(camera_frustum_world(C2_monde, R_c2_c1))
    ctr_c2 = _mpl3([C2_monde])[0]
    for i, j in _CAM_EDGES:
        ax3d.plot([fr2[i,0], fr2[j,0]], [fr2[i,1], fr2[j,1]],
                  [fr2[i,2], fr2[j,2]], color='#5588ff', lw=1.5)
    ax3d.text(ctr_c2[0] + 0.05, ctr_c2[1], ctr_c2[2] + 0.08,
              'C2', color='#6699ff', fontsize=8, fontweight='bold')

    # ── Baseline entre les deux caméras
    ax3d.plot([ctr_c1[0], ctr_c2[0]], [ctr_c1[1], ctr_c2[1]],
              [ctr_c1[2], ctr_c2[2]], '--', color='#445566', lw=1.)

    # ── Rayons cam→centre pavé (lignes de visée)
    ax3d.plot([ctr_c1[0], ctr3[0]], [ctr_c1[1], ctr3[1]],
              [ctr_c1[2], ctr3[2]], ':', color='#ff5533', lw=0.9, alpha=0.5)
    ax3d.plot([ctr_c2[0], ctr3[0]], [ctr_c2[1], ctr3[1]],
              [ctr_c2[2], ctr3[2]], ':', color='#3366ff', lw=0.9, alpha=0.5)

    # ── Point vert = position triangulée
    if st['pos3d'] is not None:
        est3 = _mpl3([st['pos3d']])[0]
        ax3d.scatter(est3[0], est3[1], est3[2],
                     color='#00ff66', s=120, marker='^',
                     zorder=10, depthshade=False)
        # Segment vrai → estimé (jaune)
        ax3d.plot([ctr3[0], est3[0]], [ctr3[1], est3[1]], [ctr3[2], est3[2]],
                  '--', color='#ffee33', lw=1.2, alpha=0.9)

    # ── Mise en forme des axes
    bz  = st['bz']
    bx  = st['bx']
    ax3d.set_xlim(min(-0.1, bx - 0.4), max(1.6, bx + 0.4))  # Étend X jusqu'à 1.6m minimum
    ax3d.set_ylim(-0.3, max(1.6, bz + 0.5))                 # Étend la profondeur (Z monde / Y Matplotlib)
    ax3d.set_zlim(-0.35, 0.35)

    ax3d.set_xlabel('X (m)',            color='#8899aa', fontsize=7, labelpad=1)
    ax3d.set_ylabel('Z  profondeur (m)', color='#8899aa', fontsize=7, labelpad=1)
    ax3d.set_zlabel('Hauteur (m)',       color='#8899aa', fontsize=7, labelpad=1)
    ax3d.set_title('Vue 3D monde', color='white', fontsize=10, pad=6)
    ax3d.tick_params(colors='#556677', labelsize=6)
    ax3d.view_init(elev=_view[0], azim=_view[1])
    x_range = ax3d.get_xlim()[1] - ax3d.get_xlim()[0]
    y_range = ax3d.get_ylim()[1] - ax3d.get_ylim()[0]
    z_range = ax3d.get_zlim()[1] - ax3d.get_zlim()[0]
    ax3d.set_box_aspect((x_range, y_range, z_range))


# ══════════════════════════════════════════════════════════
#  8.  DESSIN DES VUES CAMÉRA
# ══════════════════════════════════════════════════════════

def _draw_one_cam(ax, P, label, col, rect_key, center_key):
    """Redessine une vue caméra 2D et retourne la projection du centre vrai."""
    ax.cla()
    ax.set_facecolor('#05080f')
    for sp in ax.spines.values():
        sp.set_edgecolor(col)
        sp.set_linewidth(1.5)
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)    # Y image : 0 en haut
    ax.set_title(
        f'{label}  ({IMG_W}×{IMG_H} px) — Cliquer-glisser pour encadrer le pavé',
        color=col, fontsize=8.5, pad=3)
    ax.set_xlabel('u (pixels)', color='#667788', fontsize=7)
    ax.set_ylabel('v (pixels)', color='#667788', fontsize=7)
    ax.tick_params(colors='#445566', labelsize=6)

    # Grille légère
    ax.grid(True, color='#111820', lw=0.6, alpha=0.6)

    corners = box_corners(st['bx'], st['by'], st['bz'], st['yaw'])
    center  = np.array([st['bx'], st['by'], st['bz']])

    # ── Faces (remplissage transparent)
    for fi in FACES:
        pts2d = [proj(corners[k], P) for k in fi]
        if all(p is not None for p in pts2d):
            ax.fill([p[0] for p in pts2d], [p[1] for p in pts2d],
                    color='#1a3388', alpha=0.20, zorder=1)

    # ── Arêtes du pavé
    for i, j in EDGES:
        p_i = proj(corners[i], P)
        p_j = proj(corners[j], P)
        if p_i is not None and p_j is not None:
            ax.plot([p_i[0], p_j[0]], [p_i[1], p_j[1]],
                    color='#6688dd', lw=1.3, alpha=0.9, zorder=2)

    # ── Point rouge = vrai centre du pavé
    rp = proj(center, P)
    if rp is not None:
        ax.plot(*rp, 'o', color='#ff3322', ms=8, zorder=6)
        ax.plot(*rp, '+', color='#ff5544', ms=15, mew=2.0, zorder=6)

    # ── Rectangle sélectionné par l'utilisateur
    r = st[rect_key]
    if r is not None:
        ax.add_patch(mpatches.Rectangle(
            (r[0], r[1]), r[2], r[3],
            lw=2., edgecolor='cyan', facecolor='#00ffff15', ls='--', zorder=7))

    # ── Centre du rectangle (cyan)
    c = st[center_key]
    if c is not None:
        ax.plot(*c, 'o', color='#00ddee', ms=8, zorder=8, fillstyle='none', mec='cyan', mew=2)
        ax.plot(*c, '+', color='cyan', ms=16, mew=2.5, zorder=8)

    # ── Point vert = position estimée projetée
    if st['pos3d'] is not None:
        gp = proj(st['pos3d'], P)
        if gp is not None:
            ax.plot(*gp, '^', color='#00ff66', ms=10, zorder=9,
                    markeredgecolor='#00cc55', markeredgewidth=1.)

    # ── Légende
    legend_elems = []
    if rp is not None:
        legend_elems.append(
            mpatches.Patch(color='#ff5544', label='● Centre vrai'))
    if c is not None:
        legend_elems.append(
            mpatches.Patch(color='cyan', label='+ Centre sélection'))
    if st['pos3d'] is not None:
        legend_elems.append(
            mpatches.Patch(color='#00ff66', label='▲ Position estimée'))
    if legend_elems:
        ax.legend(handles=legend_elems, fontsize=7, loc='upper right',
                  facecolor='#0a1020', edgecolor='#334455', labelcolor='white')

    return rp


def draw_cams():
    """Redessine les deux vues caméra. Retourne (rp1, rp2) = projections du centre vrai."""
    rp1 = _draw_one_cam(ax_c1, P1, 'Caméra 1', '#ff8855', 'r1', 'c1')
    rp2 = _draw_one_cam(ax_c2, P2, 'Caméra 2', '#5599ff', 'r2', 'c2')
    return rp1, rp2


# ══════════════════════════════════════════════════════════
#  9.  AFFICHAGE DES ERREURS
# ══════════════════════════════════════════════════════════

def update_errors(rp1, rp2):
    """Met à jour la zone texte des erreurs."""
    if st['pos3d'] is None:
        err_txt.set_text(
            'Aucune triangulation — dessiner un rectangle sur CHACUNE des deux '
            'vues caméra puis cliquer "▶ Trianguler".')
        return

    ctr   = np.array([st['bx'], st['by'], st['bz']])
    e3d   = float(np.linalg.norm(st['pos3d'] - ctr))

    lines = [
        f"Position vraie       : "
        f"X={ctr[0]:+.4f}  Y={ctr[1]:+.4f}  Z={ctr[2]:.4f}  (m)",

        f"Position estimée : "
        f"X={st['pos3d'][0]:+.4f}  Y={st['pos3d'][1]:+.4f}  Z={st['pos3d'][2]:.4f}  (m)",

        f"Erreur 3D euclidienne : {e3d * 1000:.3f} mm",
    ]

    for cam_lbl, P_cam, rp in [('Cam1', P1, rp1), ('Cam2', P2, rp2)]:
        gp = proj(st['pos3d'], P_cam)
        if rp is not None and gp is not None:
            ang, dpx = angular_error(rp, gp)
            lines.append(
                f"{cam_lbl} — erreur angulaire : {ang:.5f}°   |   "
                f"distance : {dpx:.2f} px")

    err_txt.set_text('\n'.join(lines))


# ══════════════════════════════════════════════════════════
#  10.  MISE À JOUR GLOBALE
# ══════════════════════════════════════════════════════════

def redraw():
    draw_3d()
    rp1, rp2 = draw_cams()
    update_errors(rp1, rp2)
    fig.canvas.draw_idle()


# ══════════════════════════════════════════════════════════
#  11.  SLIDERS
# ══════════════════════════════════════════════════════════
_SLIDER_STYLE = dict(color='#1a3050')
s_x   = Slider(ax_sx,   'X  position (m)',  -1.5, 1.5,  valinit=0.0, **_SLIDER_STYLE)
s_y   = Slider(ax_sy,   'Y  hauteur  (m)',  -0.5, 0.5,  valinit=0.0, **_SLIDER_STYLE)
s_z   = Slider(ax_sz,   'Z  profond. (m)',   0.5, 6.0,  valinit=2.0, **_SLIDER_STYLE)
s_yaw = Slider(ax_syaw, 'Lacet       (°)',  -90., 90.,  valinit=0.0, **_SLIDER_STYLE)

for _s in [s_x, s_y, s_z, s_yaw]:
    _s.label.set_color('#aabbcc')
    _s.label.set_fontsize(9)
    _s.valtext.set_color('#88aadd')
    _s.valtext.set_fontsize(9)


def on_slider(_val):
    """Callback commun pour les 4 sliders."""
    st.update(bx=s_x.val, by=s_y.val, bz=s_z.val, yaw=s_yaw.val,
              r1=None, c1=None, r2=None, c2=None, pos3d=None)
    redraw()


for _s in [s_x, s_y, s_z, s_yaw]:
    _s.on_changed(on_slider)


# ══════════════════════════════════════════════════════════
#  12.  BOUTON TRIANGULER
# ══════════════════════════════════════════════════════════
btn = Button(ax_btn, '▶  Trianguler', color='#0c2040', hovercolor='#1a4070')
btn.label.set_color('white')
btn.label.set_fontsize(11)
btn.label.set_fontweight('bold')


def on_triangulate(_event):
    if st['c1'] is None or st['c2'] is None:
        err_txt.set_text(
            '⚠  Veuillez d\'abord dessiner un rectangle sur CHACUNE des deux vues caméra.')
        fig.canvas.draw_idle()
        return
    u1, v1 = st['c1']
    u2, v2 = st['c2']
    st['pos3d'] = triangulate_DLT(u1, v1, u2, v2)
    redraw()


btn.on_clicked(on_triangulate)


# ══════════════════════════════════════════════════════════
#  13.  INTERACTION SOURIS — DESSIN DES RECTANGLES
# ══════════════════════════════════════════════════════════

def on_press(ev):
    """Début du tracé d'un rectangle sur une vue caméra."""
    if ev.inaxes not in (ax_c1, ax_c2):
        return
    if ev.xdata is None or ev.ydata is None:
        return
    cam = 1 if ev.inaxes is ax_c1 else 2
    _drag.update(ax=ev.inaxes, x0=ev.xdata, y0=ev.ydata, cam=cam)
    # Créer un patch de rectangle en cours de dessin
    patch = mpatches.Rectangle(
        (ev.xdata, ev.ydata), 0., 0.,
        lw=2., edgecolor='cyan', facecolor='#00ffff12', ls='--', zorder=20)
    ev.inaxes.add_patch(patch)
    _drag['patch'] = patch
    fig.canvas.draw_idle()


def on_motion(ev):
    """Mise à jour du rectangle pendant le glissement."""
    if _drag['ax'] is None or ev.inaxes is not _drag['ax']:
        return
    if ev.xdata is None or ev.ydata is None:
        return
    x0, y0 = _drag['x0'], _drag['y0']
    xm = min(x0, ev.xdata);  ym = min(y0, ev.ydata)
    _drag['patch'].set_xy((xm, ym))
    _drag['patch'].set_width(abs(ev.xdata - x0))
    _drag['patch'].set_height(abs(ev.ydata - y0))
    fig.canvas.draw_idle()


def on_release(ev):
    """Fin du tracé : enregistre rectangle et centre."""
    if _drag['ax'] is None:
        return
    cam = _drag['cam']
    if ev.xdata is None or ev.ydata is None:
        _drag['ax'] = None
        return
    x0, y0  = _drag['x0'],   _drag['y0']
    x1, y1  = ev.xdata,      ev.ydata
    xm, ym  = min(x0, x1),   min(y0, y1)
    width   = abs(x1 - x0);  height = abs(y1 - y0)
    cx_r    = (x0 + x1) / 2.
    cy_r    = (y0 + y1) / 2.

    st[f'r{cam}'] = (xm, ym, width, height)
    st[f'c{cam}'] = (cx_r, cy_r)

    _drag.update(ax=None, patch=None)
    rp1, rp2 = draw_cams()
    update_errors(rp1, rp2)
    fig.canvas.draw_idle()


fig.canvas.mpl_connect('button_press_event',   on_press)
fig.canvas.mpl_connect('motion_notify_event',  on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)


# ══════════════════════════════════════════════════════════
#  14.  LÉGENDE 3D (texte fixe)
# ══════════════════════════════════════════════════════════
fig.text(0.03, 0.325,
         '● Rouge = centre vrai   △ Vert = estimé   C1/C2 = caméras',
         fontsize=8, color='#667788', ha='left', va='top')

fig.text(0.03, 0.302,
         f'Pavé : L={BOX_L}m × l={BOX_W}m × h={BOX_H}m  |  '
         f'Baseline : {BASELINE}m  |  Convergence : {CONV_DEG}°',
         fontsize=7.5, color='#556677', ha='left', va='top')


# ══════════════════════════════════════════════════════════
#  15.  LANCEMENT
# ══════════════════════════════════════════════════════════
redraw()
plt.show()
