# =============================================================================
# Fuzzy Washing Machine Controller  ·  PA-02 – ECyT UNSAM
# Spyder version  —  uses matplotlib.widgets (no ipywidgets needed)
#
# ► BEFORE RUNNING: set the graphics backend to 'Automatic' or 'Qt5'
#     Tools ▸ Preferences ▸ IPython console ▸ Graphics ▸ Backend → Qt5
#   Then restart the kernel (Consoles ▸ Restart kernel).
#   The default 'Inline' backend renders static images — sliders won't work.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
from matplotlib.patches import FancyBboxPatch


# ── Membership functions ──────────────────────────────────────────────────────

def trapezoid(x, a, b, c, d):
    """Trapezoidal MF --- handles scalars and NumPy arrays."""
    x = np.atleast_1d(np.asarray(x, float))
    y = np.zeros_like(x)
    if b > a:
        m = (x > a) & (x < b)
        y[m] = (x[m] - a) / (b - a)
    y[(x >= b) & (x <= c)] = 1.0
    if d > c:
        m = (x > c) & (x < d)
        y[m] = (d - x[m]) / (d - c)
    return float(y[0]) if y.size == 1 else y


def triangle(x, a, b, c):
    """Triangular MF — shorthand for a peak trapezoid."""
    return trapezoid(x, a, b, b, c)


# ── Fuzzy set definitions ─────────────────────────────────────────────────────

SETS = {
    'weight': {
        'Empty':      (trapezoid, (0,   0,   0.2, 0.6)),
        'Light':      (triangle,  (0.2, 1.0, 2.2     )),
        'Medium':     (triangle,  (1.5, 2.5, 3.8     )),
        'Heavy':      (triangle,  (3.0, 4.2, 5.0     )),
        'Overloaded': (trapezoid, (4.5, 5.0, 6,   6  )),
    },
    'color': {
        'Dark':     (trapezoid, (0,  0,  20,  40 )),
        'Mid-tone': (triangle,  (20, 50, 80      )),
        'White':    (trapezoid, (60, 80, 100, 100)),
    },
    'fiber': {
        'Natural':   (trapezoid, (0,  0,  30,  60 )),
        'Synthetic': (trapezoid, (40, 70, 100, 100)),
    },
}

DOMAINS   = {'weight': (0, 6),    'color': (0, 100),   'fiber': (0, 100)}
TITLES    = {'weight': 'Weight membership',
             'color':  'Color membership',
             'fiber':  'Fiber membership'}
XLABELS   = {'weight': 'Load  (kg)',
             'color':  'Brightness  (0 = dark → 100 = white)',
             'fiber':  'Fiber index  (0 = cotton → 100 = polyester)'}


# ── Fuzzy inference ───────────────────────────────────────────────────────────

def compute(w, br, fc):
    """
    Mamdani min-inference.

    Parameters
    ----------
    w   load weight     [0–6] kg
    br  color brightness    [0–100]   0=dark, 100=white
    fc  fiber conductivity index [0–100]  0=cotton, 100=polyester
    """
    wM = {k: fn(w,  *a) for k, (fn, a) in SETS['weight'].items()}
    bM = {k: fn(br, *a) for k, (fn, a) in SETS['color'].items()}
    fM = {k: fn(fc, *a) for k, (fn, a) in SETS['fiber'].items()}

    active = min(1 - wM['Overloaded'], 1 - wM['Empty'])

    wc  = min(bM['White'],                         fM['Natural'],  active)   # R3
    cc  = min(max(bM['Mid-tone'], bM['Dark']),      fM['Natural'],  active)   # R4+R5
    sm  = min(fM['Synthetic'],                                      active)   # R6
    err = wM['Overloaded']                                                    # R1

    wl_num = wM['Light'] * 30 + wM['Medium'] * 60 + wM['Heavy'] * 100
    wl_den = wM['Light'] + wM['Medium'] + wM['Heavy']
    wl_pct = round(wl_num / wl_den) if wl_den > 0.01 else 0

    return dict(wM=wM, bM=bM, fM=fM,
                wc=wc, cc=cc, sm=sm, err=err,
                wl_pct=wl_pct, empty=wM['Empty'])


def classify(r):
    """Winner-takes-all defuzzification → (label, detail, hex_color)."""
    mx = max(r['wc'], r['cc'], r['sm'], r['err'])

    if r['err'] >= 0.5 and r['err'] == mx:
        return 'Overloaded', 'Reduce load to below 5 kgf', '#c0392b'

    if r['empty'] > 0.8:
        return 'Waiting NO load', 'Place garments and press Run', '#7f8c8d'

    wl = 'Low' if r['wl_pct'] < 40 else ('Medium' if r['wl_pct'] < 75 else 'High')

    if r['wc'] >= r['cc'] and r['wc'] >= r['sm']:
        return ('WC --- White cotton',
                f'Temperature: 90 °C\nWater level: {wl}  ({r["wl_pct"]} %)',
                '#185FA5')
    if r['cc'] >= r['sm']:
        return ('CC --- Colored cotton',
                f'Temperature: 40 °C\nWater level: {wl}  ({r["wl_pct"]} %)',
                '#27651a')
    return ('SM --- Synthetic material',
            f'Temperature: 30 °C\nWater level: {wl}  ({r["wl_pct"]} %)',
            '#534AB7')


# ── Axes drawing helpers ──────────────────────────────────────────────────────

def _draw_mf(ax, var, current_val):
    """Redraw one membership-function axes."""
    ax.cla()
    xs = np.linspace(*DOMAINS[var], 500)

    for label, (fn, args) in SETS[var].items():
        ys  = fn(xs, *args)
        mu  = float(fn(current_val, *args))
        ln, = ax.plot(xs, ys, lw=1.8, label=f'{label}  μ = {mu:.2f}')
        if mu > 0.005:
            ax.plot(current_val, mu, 'o', ms=5, color=ln.get_color(), zorder=5)

    ax.axvline(current_val, color='crimson', lw=1.5, ls='--', alpha=0.75,
               label=f'x = {current_val:.2g}')
    ax.set_ylim(-0.05, 1.18)
    ax.set_title(TITLES[var],  fontsize=10, fontweight='medium', pad=5)
    ax.set_xlabel(XLABELS[var], fontsize=8.5)
    ax.set_ylabel('μ', fontsize=10)
    ax.legend(fontsize=7.5, ncol=2, loc='upper right', framealpha=0.85,
              handlelength=1.2)
    ax.grid(True, alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)


def _draw_rules(ax, r):
    """Redraw rule-activation bar chart."""
    ax.cla()
    names  = ['WC -- white cotton', 'CC -- colored cotton',
              'SM -- synthetic material', 'Error -- overloaded']
    vals   = [r['wc'], r['cc'], r['sm'], r['err']]
    colors = ['#185FA5', '#27651a', '#534AB7', '#c0392b']

    bars = ax.barh(names, vals, color=colors, alpha=0.78, height=0.45)
    for bar, val in zip(bars, vals):
        ax.text(min(val + 0.025, 1.08),
                bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', ha='left',
                fontsize=10, fontweight='medium')

    ax.set_xlim(0, 1.25)
    ax.set_xlabel('Activation strength  (α)', fontsize=9)
    ax.set_title('Rule firing strengths', fontsize=10, fontweight='medium', pad=5)
    ax.axvline(0.5, color='gray', ls=':', lw=1, alpha=0.45)
    ax.grid(True, axis='x', alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)


def _draw_output(ax, r):
    """Redraw controller output panel."""
    ax.cla()
    ax.axis('off')
    prog, detail, color = classify(r)

    rect = FancyBboxPatch(
        (0.05, 0.08), 0.90, 0.84,
        boxstyle='round,pad=0.02',
        linewidth=2.5, edgecolor=color,
        facecolor=color + '18',       # ~10 % opacity tint
        transform=ax.transAxes,
    )
    ax.add_patch(rect)

    ax.text(0.5, 0.70, prog,
            ha='center', va='center', fontsize=12, fontweight='bold',
            color=color, transform=ax.transAxes)

    ax.text(0.5, 0.34, detail,
            ha='center', va='center', fontsize=9.5, color='#333',
            transform=ax.transAxes, linespacing=1.8)

    ax.set_title('Controller output', fontsize=10, fontweight='medium', pad=5)


# ── Build the figure ──────────────────────────────────────────────────────────

plt.close('all')
fig = plt.figure('Fuzzy Washing Machine Controller', figsize=(15, 10))
fig.suptitle('Fuzzy Washing Machine Controller  ·  PA-02',
             fontsize=13, fontweight='medium', y=0.97)

# Leave bottom 22 % of the figure for the three sliders
fig.subplots_adjust(left=0.15, right=0.97, top=0.92, bottom=0.25,
                    hspace=0.48, wspace=0.33)

gs = gridspec.GridSpec(2, 3, figure=fig,
                        top=0.91, bottom=0.26,
                        hspace=0.50, wspace=0.33)

ax_w     = fig.add_subplot(gs[0, 0])
ax_b     = fig.add_subplot(gs[0, 1])
ax_f     = fig.add_subplot(gs[0, 2])
ax_rules = fig.add_subplot(gs[1, :2])
ax_out   = fig.add_subplot(gs[1,  2])

# ── Slider axes (manually placed below the GridSpec) ─────────────────────────
#   [left, bottom, width, height]  — all in figure-fraction coordinates

ax_sw  = fig.add_axes([0.15, 0.16, 0.72, 0.022])
ax_sbr = fig.add_axes([0.15, 0.10, 0.72, 0.022])
ax_sfc = fig.add_axes([0.15, 0.04, 0.72, 0.022])

s_w  = Slider(ax_sw,  'Weight (kg)',           0,   6,   valinit=2.5,
              valstep=0.1, color='#1a6eb5')
s_br = Slider(ax_sbr, 'Brightness (0=dark)',   0, 100,   valinit=85,
              valstep=1,   color='#c47c10')
s_fc = Slider(ax_sfc, 'Fiber (0=cotton)',      0, 100,   valinit=10,
              valstep=1,   color='#534AB7')

# Style slider labels
for slider in (s_w, s_br, s_fc):
    slider.label.set_fontsize(9)
    slider.valtext.set_fontsize(9)

# ── Initial draw ──────────────────────────────────────────────────────────────

def _full_redraw(w, br, fc):
    r = compute(w, br, fc)
    _draw_mf(ax_w,  'weight', w)
    _draw_mf(ax_b,  'color',  br)
    _draw_mf(ax_f,  'fiber',  fc)
    _draw_rules(ax_rules, r)
    _draw_output(ax_out,  r)
    fig.canvas.draw_idle()

_full_redraw(s_w.val, s_br.val, s_fc.val)

# ── Slider callbacks ──────────────────────────────────────────────────────────

def _on_change(_):
    _full_redraw(s_w.val, s_br.val, s_fc.val)

s_w .on_changed(_on_change)
s_br.on_changed(_on_change)
s_fc.on_changed(_on_change)

plt.show()