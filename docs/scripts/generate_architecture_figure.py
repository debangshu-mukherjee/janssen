#!/usr/bin/env python3
"""Generate the Janssen library architecture figure.

This script creates a professional architecture diagram showing the four-tier
module structure, composable design, and data flow patterns in Janssen.

The figure is saved as both PDF (for LaTeX) and PNG (for README).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

# Enable LaTeX rendering for mathematical symbols
plt.rcParams['text.usetex'] = False  # Use mathtext instead of full LaTeX
plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern for math

# Set up the figure with three panels
fig = plt.figure(figsize=(16, 10))

# Color scheme - professional blues and grays
COLORS = {
    'foundation': '#E3F2FD',      # Light blue
    'foundation_border': '#1976D2',
    'physics': '#E8F5E9',          # Light green
    'physics_border': '#388E3C',
    'coherence': '#FFF3E0',        # Light orange
    'coherence_border': '#F57C00',
    'application': '#F3E5F5',      # Light purple
    'application_border': '#7B1FA2',
    'wavefront': '#BBDEFB',        # Medium blue
    'wavefront_border': '#1565C0',
    'arrow': '#455A64',            # Dark gray
    'text': '#212121',             # Near black
    'subtext': '#212121',          # Black (same as text for readability)
    'gradient_start': '#42A5F5',   # Gradient blue
    'gradient_end': '#1565C0',
}


def draw_rounded_box(ax, x, y, width, height, label, sublabel=None,
                     facecolor='white', edgecolor='black', fontsize=10,
                     sublabel_fontsize=8, linewidth=2, alpha=1.0):
    """Draw a rounded rectangle with label."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        transform=ax.transAxes,
        zorder=2
    )
    ax.add_patch(box)

    # Main label
    if sublabel:
        label_y = y + height * 0.6
    else:
        label_y = y + height / 2

    ax.text(x + width / 2, label_y, label,
            ha='center', va='center',
            fontsize=fontsize, fontweight='bold',
            color=COLORS['text'],
            transform=ax.transAxes, zorder=3)

    # Sublabel
    if sublabel:
        ax.text(x + width / 2, y + height * 0.3, sublabel,
                ha='center', va='center',
                fontsize=sublabel_fontsize,
                color=COLORS['subtext'],
                transform=ax.transAxes, zorder=3)

    return box


def draw_arrow(ax, start, end, color=None, style='->', linewidth=2,
               connectionstyle="arc3,rad=0"):
    """Draw an arrow between two points."""
    if color is None:
        color = COLORS['arrow']
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        color=color,
        linewidth=linewidth,
        connectionstyle=connectionstyle,
        transform=ax.transAxes,
        zorder=1,
        mutation_scale=15
    )
    ax.add_patch(arrow)
    return arrow


def draw_panel_a(ax):
    """Panel A: Composable Design - wavefront transformations."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, '(a)', fontsize=16, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)
    ax.text(0.5, 0.88, 'Composable Design', fontsize=12, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)

    # Layout parameters - align with panel (c)
    # Panel (c) has: OpticalWavefront at x=0.05, CoherentModeSet ends at x=0.95
    x_start = 0.05  # align with panel (c) OpticalWavefront
    x_end = 0.95    # align with panel (c) CoherentModeSet end

    wf_height = 0.12
    wf_y = 0.65
    op_height = 0.10
    op_y = wf_y + 0.01

    # Box widths
    wf1_width = 0.15  # OpticalWavefront
    op_width = 0.11   # operation boxes (Aperture, Lens, Propagator)
    wf2_width = 0.12  # final Wavefront

    # Calculate spacing
    # Total box width: wf1 + 3*op + wf2 = 0.15 + 0.33 + 0.12 = 0.60
    # Available width: x_end - x_start = 0.90
    # Space for 4 gaps: 0.90 - 0.60 = 0.30, so each gap = 0.075
    total_box_width = wf1_width + 3 * op_width + wf2_width
    total_width = x_end - x_start
    gap = (total_width - total_box_width) / 4  # 4 gaps between 5 boxes

    # Positions (x coordinates for left edge of each box)
    x1 = x_start                          # OpticalWavefront
    x2 = x1 + wf1_width + gap             # Aperture
    x3 = x2 + op_width + gap              # Lens
    x4 = x3 + op_width + gap              # Propagator
    x5 = x4 + op_width + gap              # Wavefront (should end at x_end)

    # Starting wavefront
    draw_rounded_box(ax, x1, wf_y, wf1_width, wf_height,
                     'OpticalWavefront', 'E(x,y), λ, Δx, z',
                     facecolor=COLORS['wavefront'],
                     edgecolor=COLORS['wavefront_border'],
                     fontsize=9, sublabel_fontsize=9)

    # Operation boxes
    ops = [
        ('Aperture', x2),
        ('Lens', x3),
        ('Propagator', x4),
    ]

    for op_name, op_x in ops:
        draw_rounded_box(ax, op_x, op_y, op_width, op_height,
                         op_name, None,
                         facecolor=COLORS['physics'],
                         edgecolor=COLORS['physics_border'],
                         fontsize=9)

    # Final wavefront
    draw_rounded_box(ax, x5, wf_y, wf2_width, wf_height,
                     'Wavefront', "E'(x,y)",
                     facecolor=COLORS['wavefront'],
                     edgecolor=COLORS['wavefront_border'],
                     fontsize=9, sublabel_fontsize=9)

    # Arrows between elements
    arrow_y = wf_y + wf_height / 2
    arrows = [
        (x1 + wf1_width, x2),              # after OpticalWavefront -> Aperture
        (x2 + op_width, x3),               # after Aperture -> Lens
        (x3 + op_width, x4),               # after Lens -> Propagator
        (x4 + op_width, x5),               # after Propagator -> Wavefront
    ]
    for start_x, end_x in arrows:
        draw_arrow(ax, (start_x, arrow_y), (end_x, arrow_y))

    # Gradient arrows going back
    grad_y = 0.45
    ax.text(0.5, grad_y + 0.08, 'Automatic Differentiation (gradients flow back)',
            ha='center', va='center', fontsize=9, style='italic',
            color=COLORS['subtext'], transform=ax.transAxes)

    # Dashed arrow showing gradient flow
    draw_arrow(ax, (0.85, grad_y), (0.15, grad_y),
               color='#E65100', style='<-', linewidth=1.5)

    ax.text(0.5, grad_y - 0.05, r'$\partial\mathcal{L}/\partial$params',
            ha='center', va='center', fontsize=9,
            color='#E65100', transform=ax.transAxes)

    # Key message box
    msg_box = FancyBboxPatch(
        (0.08, 0.18), 0.84, 0.15,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor='#FAFAFA',
        edgecolor='#BDBDBD',
        linewidth=1,
        transform=ax.transAxes,
        zorder=1
    )
    ax.add_patch(msg_box)
    ax.text(0.5, 0.255, 'Each element is a pure function: Wavefront → Wavefront',
            ha='center', va='center', fontsize=12,
            color=COLORS['text'], transform=ax.transAxes)
    ax.text(0.5, 0.21, 'JAX traces computation graph → gradients computed automatically',
            ha='center', va='center', fontsize=10,
            color=COLORS['text'], transform=ax.transAxes)


def draw_panel_b(ax):
    """Panel B: Four-tier module structure."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.98, '(c)', fontsize=16, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)
    ax.text(0.5, 0.93, 'Module Architecture', fontsize=13, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)

    # Layer dimensions
    margin = 0.04
    layer_width = 1 - 2 * margin
    layer_height = 0.18
    gap = 0.025

    # Application Layer (top)
    app_y = 0.70
    app_box = FancyBboxPatch(
        (margin, app_y), layer_width, layer_height,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=COLORS['application'],
        edgecolor=COLORS['application_border'],
        linewidth=2,
        transform=ax.transAxes,
        zorder=1
    )
    ax.add_patch(app_box)
    ax.text(margin + 0.02, app_y + layer_height - 0.01, 'Application Layer',
            fontsize=13, fontweight='bold', color=COLORS['application_border'],
            va='top', transform=ax.transAxes)

    # Application modules
    app_modules = [
        ('scopes', 'forward models'),
        ('invert', 'ePIE, mixed-state'),
        ('plots', 'visualization'),
    ]
    mod_width = 0.28
    mod_height = 0.10
    mod_y = app_y + 0.025
    mod_gap = 0.02  # gap between modules
    n_mods = len(app_modules)
    total_mods_width = n_mods * mod_width + (n_mods - 1) * mod_gap
    mod_start_x = margin + (layer_width - total_mods_width) / 2  # center modules
    for i, (name, desc) in enumerate(app_modules):
        mod_x = mod_start_x + i * (mod_width + mod_gap)
        draw_rounded_box(ax, mod_x, mod_y, mod_width, mod_height,
                         name, desc,
                         facecolor='white',
                         edgecolor=COLORS['application_border'],
                         fontsize=12, sublabel_fontsize=10, linewidth=1.5)

    # Coherence Layer
    coh_y = app_y - layer_height - gap
    coh_box = FancyBboxPatch(
        (margin, coh_y), layer_width, layer_height,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=COLORS['coherence'],
        edgecolor=COLORS['coherence_border'],
        linewidth=2,
        transform=ax.transAxes,
        zorder=1
    )
    ax.add_patch(coh_box)
    ax.text(margin + 0.02, coh_y + layer_height - 0.01, 'Coherence Layer',
            fontsize=13, fontweight='bold', color=COLORS['coherence_border'],
            va='top', transform=ax.transAxes)

    # Coherence module (single wide box) - centered
    coh_mod_width = layer_width - 0.06
    coh_mod_x = margin + (layer_width - coh_mod_width) / 2
    draw_rounded_box(ax, coh_mod_x, coh_y + 0.025, coh_mod_width, mod_height,
                     'cohere', 'spatial  |  temporal  |  modes  |  sources  |  propagation',
                     facecolor='white',
                     edgecolor=COLORS['coherence_border'],
                     fontsize=12, sublabel_fontsize=10, linewidth=1.5)

    # Physics Core Layer
    phys_y = coh_y - layer_height - gap
    phys_box = FancyBboxPatch(
        (margin, phys_y), layer_width, layer_height,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=COLORS['physics'],
        edgecolor=COLORS['physics_border'],
        linewidth=2,
        transform=ax.transAxes,
        zorder=1
    )
    ax.add_patch(phys_box)
    ax.text(margin + 0.02, phys_y + layer_height - 0.01, 'Physics Core',
            fontsize=13, fontweight='bold', color=COLORS['physics_border'],
            va='top', transform=ax.transAxes)

    # Physics modules
    phys_modules = [
        ('optics', 'Zernike, Jones'),
        ('prop', 'ASM, Fresnel'),
        ('models', 'beams, materials'),
        ('lenses', 'lens profiles'),
    ]
    mod_width = 0.21
    mod_gap = 0.01  # gap between modules
    n_mods = len(phys_modules)
    total_mods_width = n_mods * mod_width + (n_mods - 1) * mod_gap
    mod_start_x = margin + (layer_width - total_mods_width) / 2  # center modules
    for i, (name, desc) in enumerate(phys_modules):
        mod_x = mod_start_x + i * (mod_width + mod_gap)
        draw_rounded_box(ax, mod_x, phys_y + 0.025, mod_width, mod_height,
                         name, desc,
                         facecolor='white',
                         edgecolor=COLORS['physics_border'],
                         fontsize=12, sublabel_fontsize=10, linewidth=1.5)

    # Foundation Layer
    found_y = phys_y - layer_height - gap
    found_box = FancyBboxPatch(
        (margin, found_y), layer_width, layer_height,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=COLORS['foundation'],
        edgecolor=COLORS['foundation_border'],
        linewidth=2,
        transform=ax.transAxes,
        zorder=1
    )
    ax.add_patch(found_box)
    ax.text(margin + 0.02, found_y + layer_height - 0.01, 'Foundation Layer',
            fontsize=13, fontweight='bold', color=COLORS['foundation_border'],
            va='top', transform=ax.transAxes)

    # Foundation modules
    found_modules = [
        ('types', 'OpticalWavefront, CoherentModeSet,\nSampleFunction, factory functions'),
        ('utils', 'distributed computing,\nBessel functions, Wirtinger grad'),
    ]
    mod_width = 0.44
    mod_gap = 0.02  # gap between modules
    n_mods = len(found_modules)
    total_mods_width = n_mods * mod_width + (n_mods - 1) * mod_gap
    mod_start_x = margin + (layer_width - total_mods_width) / 2  # center modules
    for i, (name, desc) in enumerate(found_modules):
        mod_x = mod_start_x + i * (mod_width + mod_gap)
        # Draw box manually for multiline
        box = FancyBboxPatch(
            (mod_x, found_y + 0.02), mod_width, mod_height + 0.02,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            facecolor='white',
            edgecolor=COLORS['foundation_border'],
            linewidth=1.5,
            transform=ax.transAxes,
            zorder=2
        )
        ax.add_patch(box)
        ax.text(mod_x + mod_width / 2, found_y + 0.095, name,
                ha='center', va='center', fontsize=12, fontweight='bold',
                color=COLORS['text'], transform=ax.transAxes)
        ax.text(mod_x + mod_width / 2, found_y + 0.055, desc,
                ha='center', va='center', fontsize=10,
                color=COLORS['subtext'], transform=ax.transAxes)


def draw_panel_c(ax):
    """Panel C: Data flow in inverse problem."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title
    ax.text(0.5, 1.02, '(b)', fontsize=16, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)
    ax.text(0.5, 0.95, 'Inverse Problem Data Flow', fontsize=12, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)

    # Input PyTrees (top row)
    input_width = 0.26
    input_height = 0.10
    input_y = 0.76

    inputs = [
        ('OpticalWavefront', 'probe E(x,y)', 0.05),
        ('SampleFunction', 'object φ(x,y)', 0.37),
        ('CoherentModeSet', 'coherence modes', 0.69),
    ]

    for name, desc, x in inputs:
        draw_rounded_box(ax, x, input_y, input_width, input_height,
                         name, desc,
                         facecolor=COLORS['wavefront'],
                         edgecolor=COLORS['wavefront_border'],
                         fontsize=9, sublabel_fontsize=9)

    # Arrows down to forward model
    for x in [0.18, 0.50, 0.82]:
        draw_arrow(ax, (x, input_y), (x, 0.66))

    # Forward model box
    fwd_y = 0.54
    draw_rounded_box(ax, 0.15, fwd_y, 0.70, 0.12,
                     r'Forward Model  $\mathcal{T}$', 'scopes.simple_microscope',
                     facecolor=COLORS['physics'],
                     edgecolor=COLORS['physics_border'],
                     fontsize=11, sublabel_fontsize=9)

    # Arrow to output
    draw_arrow(ax, (0.50, fwd_y), (0.50, 0.46))

    # Output box
    out_y = 0.36
    draw_rounded_box(ax, 0.25, out_y, 0.50, 0.10,
                     'MicroscopeData', 'predicted diffraction patterns',
                     facecolor='#E0E0E0',
                     edgecolor='#757575',
                     fontsize=10, sublabel_fontsize=9)

    # Arrow to loss
    draw_arrow(ax, (0.50, out_y), (0.50, 0.28))

    # Loss box
    loss_y = 0.19
    loss_box = FancyBboxPatch(
        (0.20, loss_y), 0.60, 0.09,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor='#FFEBEE',
        edgecolor='#C62828',
        linewidth=2,
        transform=ax.transAxes,
        zorder=2
    )
    ax.add_patch(loss_box)
    ax.text(0.50, loss_y + 0.045, r'Loss  $\mathcal{L} = \|I_{\mathrm{exp}} - \mathcal{T}(\mathrm{params})\|^2$',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='#C62828', transform=ax.transAxes)

    # Gradient arrows back up
    grad_y = 0.11
    grad_labels = [
        (r'$\partial\mathcal{L}/\partial$probe', 0.18),
        (r'$\partial\mathcal{L}/\partial$object', 0.50),
        (r'$\partial\mathcal{L}/\partial$coh.', 0.82),
    ]

    for label, x in grad_labels:
        draw_arrow(ax, (x, loss_y), (x, grad_y + 0.04),
                   color='#E65100', style='<-', linewidth=1.5)
        ax.text(x, grad_y, label, ha='center', va='center',
                fontsize=9, color='#E65100', transform=ax.transAxes)

    # Optimizer note
    ax.text(0.50, 0.05, 'Adam / Adagrad optimizer updates PyTree leaves',
            ha='center', va='center', fontsize=9, style='italic',
            color=COLORS['subtext'], transform=ax.transAxes)


def main():
    """Generate the architecture figure."""
    # Create figure with GridSpec for panel layout
    fig = plt.figure(figsize=(16, 8))

    # Panel A takes top-left, Panel B takes right side, Panel C takes bottom-left
    # Using GridSpec for flexible layout
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.2], height_ratios=[1, 1],
                  wspace=0.08, hspace=0.02,
                  left=0.02, right=0.98, top=0.96, bottom=0.02)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[:, 1])  # Panel B spans both rows

    # Draw panels
    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)

    # Save figures
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.dirname(script_dir)  # docs/

    # Save to docs/source/tutorials/Figures directory
    tutorials_fig_dir = os.path.join(docs_dir, 'source', 'tutorials', 'Figures')
    os.makedirs(tutorials_fig_dir, exist_ok=True)

    pdf_path = os.path.join(tutorials_fig_dir, 'architecture_figure.pdf')
    png_path = os.path.join(tutorials_fig_dir, 'architecture_figure.png')

    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

    # Also save to the paper's Figures directory
    paper_fig_dir = os.path.expanduser('~/Papers/Optical Ptychography/Figures')
    if os.path.exists(paper_fig_dir):
        paper_pdf = os.path.join(paper_fig_dir, 'Outline.pdf')
        paper_png = os.path.join(paper_fig_dir, 'Outline.png')
        fig.savefig(paper_pdf, format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(paper_png, format='png', bbox_inches='tight', dpi=300)
        print(f"Saved: {paper_pdf}")
        print(f"Saved: {paper_png}")

    plt.close(fig)


if __name__ == '__main__':
    main()
