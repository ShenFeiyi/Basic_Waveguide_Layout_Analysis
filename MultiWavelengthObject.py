# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D as LegendLine

from ReflectorObject import Reflector
from functions import convex_hull, polygon_area, wavelength_to_rgb

# Per-wavelength line styles, so wavelength identity never rests on color alone.
LINESTYLES = ['-', '--', ':', '-.', (0, (5, 1, 1, 1)), (0, (1, 4))]


class MultiWavelengthWaveguide:
    """Run the single-wavelength Waveguide analysis at several wavelengths.

    Each wavelength gets its own Waveguide.at_wavelength() copy: grating periods
    stay fixed, so grating vectors scale as wavelength/period, and dispersive
    indices are re-evaluated. All copies share the input FOV grid, so per-field
    results line up across wavelengths.
    """

    def __init__(self, waveguide, wavelengths):
        wavelengths = [float(w) for w in np.atleast_1d(wavelengths)]
        if not wavelengths:
            raise ValueError("At least one wavelength is required.")
        if len(set(wavelengths)) != len(wavelengths):
            raise ValueError(f"Duplicate wavelengths in {wavelengths}.")
        self.reference = waveguide
        self.wavelengths = wavelengths
        self.waveguides = {w: waveguide.at_wavelength(w) for w in wavelengths} # wavelength (nm) -> Waveguide
        self.layout_errors = {} # wavelength -> reason its coupler layout is undefined

    def __repr__(self):
        return f"MultiWavelengthWaveguide(wavelengths={self.wavelengths}, reference={self.reference!r})"

    def __getitem__(self, wavelength):
        return self.waveguides[float(wavelength)]

    def linestyle(self, wavelength):
        return LINESTYLES[self.wavelengths.index(wavelength) % len(LINESTYLES)]

    def analyze(self):
        """Trace and lay out every wavelength; layout failures go to layout_errors instead of raising."""
        self.layout_errors = {}
        for wavelength, wg in self.waveguides.items():
            wg.trace_rays()
            try:
                wg.calc_coupler_dimensions()
            except ValueError as error:
                wg._clear_layout_results()
                self.layout_errors[wavelength] = str(error)
        return self

    @property
    def laid_out(self):
        """Wavelengths with a valid coupler layout (after analyze())."""
        return [w for w in self.wavelengths if self.waveguides[w].EG_corners is not None]

    @property
    def common_transmitted_mask(self):
        """Mask over the shared input fields: True where a field reaches the output at every wavelength."""
        return np.logical_and.reduce([wg.transmitted_mask for wg in self.waveguides.values()])

    @property
    def combined_EG_footprint(self):
        """Convex hull (counter-clockwise vertices) of every wavelength's EG region, or None."""
        corners = [self.waveguides[w].EG_corners for w in self.laid_out]
        return convex_hull(np.vstack(corners)) if corners else None

    @property
    def combined_OG_size(self):
        """Output footprint that covers every wavelength's output FOV, or None."""
        sizes = [self.waveguides[w].OG_size for w in self.laid_out]
        return [float(size) for size in np.max(sizes, axis=0)] if sizes else None

    def _kind(self):
        return 'Geometric' if self.reference.has_reflectors else 'Diffractive'

    def _wavelength_list(self):
        return ', '.join(f'{w:g}' for w in self.wavelengths) + ' nm'

    def print_summary(self):
        """Refresh the analysis and print a per-wavelength comparison."""
        self.analyze()
        ref = self.reference
        print(f"{self._kind()} Waveguide Multi-Wavelength Summary:")
        if not ref.has_reflectors:
            print(
                f"Grating period for IG: {ref.IG.grating_period:.2f} nm, "
                f"EG: {ref.EG.grating_period:.2f} nm, OG: {ref.OG.grating_period:.2f} nm"
            )
        elif getattr(ref, 'alpha_im', None) is not None:
            print(f"Input mirror rotation alpha_IM: {ref.alpha_im:.6f} deg")
        print(f"Substrate index: {ref.substrate_material!r}")
        print(f"Input FOV: {ref.FOV[0]:.2f} deg x {ref.FOV[1]:.2f} deg, {ref.number_of_rays} sampled fields")

        lost_headers = "  ".join(f"{'Lost ' + ref.stage_label(stage):>8}" for stage in ref.STAGES)
        print(f"{'Wavelength':>10}  {'n_sub':>8}  {lost_headers}  {'Transmitted':>11}")
        for wavelength, wg in self.waveguides.items():
            lost = "  ".join(f"{wg.rejected_percentages[stage]:>7.2f}%" for stage in ref.STAGES)
            print(f"{wavelength:>7.1f} nm  {wg.N1:>8.5f}  {lost}  {wg.transmitted_percentage:>10.2f}%")
        common = self.common_transmitted_mask
        print(f"Fields transmitted at every wavelength: {common.sum()}/{common.size} ({100 * common.mean():.2f}%)")

        first = self.waveguides[self.wavelengths[0]]
        print(f"Input footprint diameter: {first.calc_IG_diameter():.4f} mm (all wavelengths)")
        for wavelength, wg in self.waveguides.items():
            if wavelength in self.layout_errors:
                print(f"{wavelength:.1f} nm: layout unavailable. {self.layout_errors[wavelength]}")
                continue
            print(
                f"{wavelength:.1f} nm: output FOV {wg.output_FOV[0]:.2f} deg x {wg.output_FOV[1]:.2f} deg, "
                f"output footprint {wg.OG_size[0]:.4f} mm x {wg.OG_size[1]:.4f} mm, "
                f"expansion area {polygon_area(wg.EG_corners[[0, 1, 3, 2]]):.2f} mm^2"
            )
            corners = ", ".join(f"[{x:.4f}, {y:.4f}]" for x, y in wg.EG_corners)
            print(f"    Expansion Coupler Corners: {corners}")

        hull = self.combined_EG_footprint
        if hull is not None:
            OG_size = self.combined_OG_size
            print(f"Output footprint, all wavelengths: {OG_size[0]:.4f} mm x {OG_size[1]:.4f} mm")
            print(
                f"Expansion footprint, all wavelengths (convex hull): area {polygon_area(hull):.2f} mm^2, "
                f"x [{hull[:, 0].min():.4f}, {hull[:, 0].max():.4f}] mm, "
                f"y [{hull[:, 1].min():.4f}, {hull[:, 1].max():.4f}] mm"
            )
            print("    Hull vertices: " + ", ".join(f"[{x:.4f}, {y:.4f}]" for x, y in hull))

    def plot_k_space(self, filename=None, dpi=600, ax=None):
        """Overlay every wavelength in k-space and return (figure, axes).

        Marker shape gives the stage; color, the line style of each n1*sin(theta_max)
        circle, and a direct label (when there is room) give the wavelength.
        """
        for wg in self.waveguides.values():
            wg.trace_rays()
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))
        else:
            fig = ax.figure
        ax.grid(True, alpha=0.5)
        ref = self.reference
        waveguides = list(self.waveguides.values())

        points = waveguides[0].k_points['input']
        ax.scatter(points[:, 0], points[:, 1], color='0.6', s=12, marker='o', zorder=2)
        for n0 in sorted({wg.N0 for wg in waveguides}):
            ax.add_artist(plt.Circle((0, 0), n0, color='black', fill=False, linestyle='--', linewidth=1.0))

        outputs = [wg.k_points['OG'] for wg in waveguides]
        shared_output = all(
            output.shape == outputs[0].shape and np.allclose(output[:, :2], outputs[0][:, :2])
            for output in outputs
        )
        labels = [] # (x, y, text) beside each after-IG cluster
        for wavelength, wg in self.waveguides.items():
            color, linestyle = wavelength_to_rgb(wavelength), self.linestyle(wavelength)
            ax.add_artist(plt.Circle(
                (0, 0), wg.N1 * np.sin(np.radians(wg.MAX_ALLOWED_ANGLE)),
                color=color, fill=False, linestyle=linestyle, linewidth=1.2,
            ))
            for stage, marker in (('IG', 'x'), ('EG', '^')):
                points = wg.k_points[stage]
                if len(points):
                    ax.scatter(points[:, 0], points[:, 1], color=color, s=6, marker=marker, alpha=0.5, zorder=3)
                if len(points) >= 3:
                    # Outline each cluster so overlapping wavelengths keep their full extent visible.
                    ax.add_artist(plt.Polygon(
                        convex_hull(points[:, :2]), fill=False, edgecolor=color,
                        linestyle=linestyle, linewidth=1.0, zorder=5,
                    ))
            points = wg.k_points['OG']
            if not shared_output and len(points):
                ax.scatter(
                    points[:, 0], points[:, 1], s=16, marker='s', facecolors='none',
                    edgecolors=color, linewidths=0.6, zorder=4,
                )
            points = wg.k_points['IG']
            if len(points):
                labels.append((points[:, 0].max() + 0.08, points[:, 1].mean(), f'{wavelength:g} nm'))
        if shared_output and len(outputs[0]):
            ax.scatter(
                outputs[0][:, 0], outputs[0][:, 1], s=16, marker='s', facecolors='none',
                edgecolors='0.25', linewidths=0.6, zorder=4,
            )
        label_heights = np.sort([y for _, y, _ in labels])
        if len(labels) < 2 or np.diff(label_heights).min() >= 0.12: # skip labels that would collide
            for x, y, text in labels:
                ax.text(x, y, text, fontsize=8, va='center', color='0.15')

        output_label = 'Output rays (air)' if ref.has_reflectors else f'After {ref.stage_label("OG")}'
        handles = [
            LegendLine([], [], color='0.6', marker='o', linestyle='none', label='Input rays'),
            LegendLine([], [], color='0.3', marker='x', linestyle='none', label=f'After {ref.stage_label("IG")}'),
            LegendLine([], [], color='0.3', marker='^', linestyle='none', label=f'After {ref.stage_label("EG")}'),
            LegendLine(
                [], [], color='0.25', marker='s', markerfacecolor='none', linestyle='none',
                label=output_label + (', same at every wavelength' if shared_output else ''),
            ),
            LegendLine([], [], color='black', linestyle='--', label=r'$n_0$'),
        ] + [
            LegendLine(
                [], [], color=wavelength_to_rgb(wavelength), linestyle=self.linestyle(wavelength), linewidth=1.5,
                label=rf'{wavelength:g} nm: $n_1$={wg.N1:.4f}, circle $n_1\sin(\theta_{{max}})$',
            )
            for wavelength, wg in self.waveguides.items()
        ]
        k_radius = max(max(wg.N0, wg.N1 * np.sin(np.radians(wg.MAX_ALLOWED_ANGLE))) for wg in waveguides)
        k_limit = max(2.0, np.ceil(10 * (k_radius + 0.02)) / 10)
        ax.set_xlim(-k_limit, k_limit)
        ax.set_ylim(-k_limit, k_limit)
        ax.set_xlabel('kx')
        ax.set_ylabel('ky')
        ax.set_title(f'{self._kind()} Waveguide K-Space, {self._wavelength_list()}')
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8)
        ax.set_aspect('equal')
        if filename is not None:
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        return fig, ax

    def plot_waveguide_layout(self, filename=None, dpi=600, ax=None, mirror_counts=None,
                              mirror_fill=0.5, xlim=None, ylim=None):
        """Overlay every wavelength's EG layout on the shared IG and OG; return (figure, axes).

        Each wavelength's EG outline and boundary rays use its color and line style.
        The shaded region is their convex hull: one EG footprint serving all wavelengths.
        The view fits the footprints unless xlim/ylim are given.
        """
        self.analyze()
        if not self.laid_out:
            raise ValueError("No wavelength has a coupler layout. " + " ".join(self.layout_errors.values()))
        wg0 = self.waveguides[self.laid_out[0]]
        mirror_counts = {'IG': 1, 'EG': 16, 'OG': 12} if mirror_counts is None else mirror_counts
        if ax is None:
            fig, ax = plt.subplots(figsize=(11, 6.5))
        else:
            fig = ax.figure

        input_region = wg0._draw_pupils_and_input(ax)
        for wavelength in self.laid_out:
            wg = self.waveguides[wavelength]
            color, linestyle = wavelength_to_rgb(wavelength), self.linestyle(wavelength)
            for ray in wg.key_rays1 + wg.key_rays2:
                ax.axline(
                    [ray.x, ray.y], [ray.x + ray.K[0], ray.y + ray.K[1]],
                    color=color, linestyle=linestyle, linewidth=0.8, alpha=0.45,
                )
            outline = wg.EG_corners[[0, 1, 3, 2]]
            ax.add_artist(plt.Polygon(
                outline, fill=False, edgecolor=color, linestyle=linestyle, linewidth=1.6, zorder=3,
                label=f'{wg.stage_label("EG")} region at {wavelength:g} nm ({polygon_area(outline):.1f} mm²)',
            ))
        for wavelength in self.layout_errors:
            ax.plot([], [], linestyle='none', label=(
                f'{wavelength:g} nm: no layout '
                f'({self.waveguides[wavelength].transmitted_percentage:.1f}% transmitted)'
            ))

        hull = self.combined_EG_footprint
        kind_EG = 'Reflector' if isinstance(wg0.EG, Reflector) else 'Grating'
        expansion_region = plt.Polygon(
            hull, facecolor='purple', edgecolor='none', alpha=0.15,
            label=f'Expansion {kind_EG} ({wg0.stage_label("EG")}), all wavelengths ({polygon_area(hull):.1f} mm²)',
        )
        ax.add_artist(expansion_region)
        ax.scatter(hull[:, 0], hull[:, 1], color='purple', s=30, marker='^', zorder=3)

        OG_size = self.combined_OG_size
        output_region = wg0._draw_output_region(ax, OG_size)
        if wg0.has_reflectors:
            wg0._draw_layout_mirrors(
                ax, {'IG': input_region, 'EG': expansion_region, 'OG': output_region},
                hull, OG_size, mirror_counts, mirror_fill,
            )
            ax.text(
                0.5, -0.1, 'Mirror count and face width are schematic.',
                transform=ax.transAxes, ha='center', va='top', fontsize=8, color='0.35',
            )

        IG_radius = wg0.IG_diameter / 2
        extent = np.vstack([
            np.array(wg0.IG_origin) + IG_radius * np.array([[-1, -1], [1, 1]]),
            np.array(wg0.OG_origin) + np.array(OG_size) / 2 * np.array([[-1, -1], [1, 1]]),
            hull,
        ])
        low, high = extent.min(axis=0) - 2, extent.max(axis=0) + 2
        ax.set_xlim(*((low[0], high[0]) if xlim is None else xlim))
        ax.set_ylim(*((low[1], high[1]) if ylim is None else ylim))
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(f'{self._kind()} Waveguide Layout (XY View), {self._wavelength_list()}')
        ax.grid(True, alpha=0.5)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8)
        ax.set_aspect('equal')
        if filename is not None:
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        return fig, ax

    def plot_fov_coverage(self, filename=None, dpi=600):
        """Small multiples of input-field coverage, one panel per wavelength plus
        the fields that reach the output at every wavelength. Returns (figure, axes array).
        """
        for wg in self.waveguides.values():
            wg.trace_rays()
        ref = self.reference
        panels = len(self.wavelengths) + 1
        ncols = min(panels, 4)
        nrows = int(np.ceil(panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)
        for ax, (wavelength, wg) in zip(axes.flat, self.waveguides.items()):
            wg._draw_fov_status(ax, wavelength_to_rgb(wavelength))
            ax.set_title(f'{wavelength:g} nm: {wg.transmitted_percentage:.1f}% transmitted', fontsize=10)

        ax = axes.flat[len(self.wavelengths)]
        wg0 = self.waveguides[self.wavelengths[0]]
        common = self.common_transmitted_mask
        if common.any():
            ax.scatter(wg0.field_angles[common, 0], wg0.field_angles[common, 1], s=20, marker='o', color='0.1')
        if (~common).any():
            ax.scatter(
                wg0.field_angles[~common, 0], wg0.field_angles[~common, 1], s=20, marker='o',
                facecolors='none', edgecolors='0.6',
            )
        wg0._format_fov_axes(ax)
        ax.set_title(f'All wavelengths: {100 * common.mean():.1f}% transmitted', fontsize=10)
        for ax in axes.flat[panels:]:
            ax.set_visible(False)

        lost = {status for wg in self.waveguides.values() for status in wg.ray_status} - {None}
        handles = [LegendLine([], [], color='0.45', marker='o', linestyle='none', label="Transmitted (panel's wavelength color)")]
        for stage, marker, color in (('IG', 'x', '0.2'), ('EG', '^', '0.3'), ('OG', 's', '0.5')):
            if stage in lost:
                handles.append(LegendLine(
                    [], [], color=color, marker=marker, markerfacecolor='none' if marker != 'x' else color,
                    linestyle='none', label=f'Lost at {ref.stage_label(stage)}',
                ))
        handles.append(LegendLine([], [], color='0.1', marker='o', linestyle='none', label='Transmitted at every wavelength'))
        if (~common).any():
            handles.append(LegendLine(
                [], [], color='0.6', marker='o', markerfacecolor='none', linestyle='none',
                label='Lost at one or more wavelengths',
            ))
        fig.suptitle(f'{self._kind()} Waveguide FOV Coverage, {self._wavelength_list()}')
        legend_rows = int(np.ceil(len(handles) / (2 * ncols)))
        fig.tight_layout(rect=(0, 0.05 + 0.05 * legend_rows / nrows, 1, 1))
        fig.legend(handles=handles, loc='lower center', ncol=int(np.ceil(len(handles) / legend_rows)), frameon=False, fontsize=8)
        if filename is not None:
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        return fig, axes
