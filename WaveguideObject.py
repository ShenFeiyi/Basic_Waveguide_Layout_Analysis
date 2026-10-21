# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection

from RayObject import Ray
from GratingObject import Grating
from ReflectorObject import Reflector
from functions import unit_normal, intersection, Line2D

class Waveguide:
    def __init__(self, **kwargs):
        self.IG = kwargs.get('IG', None) # ipnut grating
        self.EG = kwargs.get('EG', None) # expansion grating
        self.OG = kwargs.get('OG', None) # output grating
        assert self.IG is not None, "Input grating (IG) must be provided."
        assert self.EG is not None, "Expansion grating (EG) must be provided."
        assert self.OG is not None, "Output grating (OG) must be provided."

        self.input_surface_normal = unit_normal(kwargs.get('input_surface_normal', [0, 0, -1]))
        self.output_surface_normal = unit_normal(kwargs.get('output_surface_normal', [0, 0, 1]))

        self.FOV = kwargs.get('FOV', [1, 1]) # field of view, degrees
        self.wavelength = kwargs.get('wavelength', None) # design wavelength, nm
        self.index_environment = kwargs.get('index_environment', 1.0) # refractive index of the environment
        self.index_substrate = kwargs.get('index_substrate', 1.5) # refractive index of the substrate
        self.MAX_ALLOWED_ANGLE = kwargs.get('max_allowed_angle', 90) # maximum allowed angle, degrees
        self.ray_position_samples = kwargs.get('ray_position_samples', [21, 11]) # number of samples for ray positions in the input grating plane
        self.EPD_projector = kwargs.get('EPD_projector', 3.0) # exit pupil diameter of the projector, mm
        self.gap_projector = kwargs.get('gap_projector', 2.0) # gap between the projector and the input grating, mm
        self.IG_origin = kwargs.get('IG_origin', [-10, 5]) # origin of the input grating, mm
        self.OG_origin = kwargs.get('OG_origin', [6, -11]) # origin of the output grating, mm
        self.eye_relief = kwargs.get('eye_relief', 20.0) # eye relief, mm
        self.eyebox_size = kwargs.get('eyebox_size', [12, 10]) # size of the eyebox, mm

        self.incident_rays = kwargs.get('incident_rays', None) # incident rays, list of Ray objects
        if self.incident_rays is None:
            self.generate_incident_rays()

        self._clear_trace_results()
        self._clear_layout_results()

    def __repr__(self):
        return (
            f"Waveguide(FOV={self.FOV},"
            f"index_environment={self.index_environment},"
            f"index_substrate={self.index_substrate},"
            f"MAX_ALLOWED_ANGLE={self.MAX_ALLOWED_ANGLE},"
            f"IG={self.IG}, EG={self.EG}, OG={self.OG})"
            )

    @property
    def N0(self):
        return self.index_environment

    @property
    def N1(self):
        return self.index_substrate

    def _gen_single_incident_ray(self, theta_x, theta_y):
        ray = Ray(field_angle=np.array([theta_x, theta_y]), n=self.N0)
        if isinstance(self.IG, Reflector) and self.input_surface_normal[2] < 0:
            K = ray.K.copy()
            K[2] *= -1
            ray = Ray(K=K, n=self.N0)
        return ray

    def generate_incident_rays(self, ray_position_samples=None):
        """Generate the rays for rectangular FOV grid.
        """
        samples = self.ray_position_samples if ray_position_samples is None else ray_position_samples
        self.ray_position_samples = list(samples)
        self.incident_rays = []
        for theta_x in np.linspace(-self.FOV[0]/2, self.FOV[0]/2, samples[0]):
            for theta_y in np.linspace(-self.FOV[1]/2, self.FOV[1]/2, samples[1]):
                self.incident_rays.append(self._gen_single_incident_ray(theta_x, theta_y))

    @property
    def rays0(self):
        """Computational rays before IG; refract once for an internal mirror.
        """
        if not isinstance(self.IG, Reflector):
            return self.incident_rays
        refracted_rays = []
        for ray in self.incident_rays:
            internal_ray = self._interface_ray(ray, 'IG', to_internal=True)
            if internal_ray is not None:
                refracted_rays.append(internal_ray)
        return refracted_rays

    @property
    def rays0_air(self):
        """Incident rays in the environment, before the entrance interface."""
        return self.incident_rays

    @property
    def rays3_air(self):
        """Accepted output rays in the environment, after the exit interface."""
        return self.output_rays

    def _clear_trace_results(self):
        self.rays1 = []
        self.rays2 = []
        self.rays3 = []
        self.output_rays = []
        self.k_points = {}
        self.k_points_internal = {}
        self.rejected_counts = {'IG': 0, 'EG': 0}

    def _clear_layout_results(self):
        self.FOV_diagonal = None
        self.IG_diameter = None
        self.OG_size = None
        self.EG_corners = None
        self.key_rays0 = []
        self.key_rays1 = []
        self.key_rays2 = []
        self.key_rays3 = []
        self.key_rays0_air = []
        self.key_rays3_air = []
        self.key_rays0_LR = [[], []]
        self.key_rays0_air_LR = [[], []]
        self.key_rays1_LR = [[], []]

    @property
    def number_of_rays(self):
        return len(self.incident_rays)

    @property
    def rejected_percentages(self):
        """Stage-specific rejections, all divided by the incident count."""
        if self.number_of_rays == 0:
            raise ValueError("At least one incident ray is required.")
        return {
            stage: count / self.number_of_rays * 100
            for stage, count in self.rejected_counts.items()
        } # IG: 0.xx, EG: 0.xx

    @property
    def has_reflectors(self):
        """Return True if any of the gratings are reflectors."""
        return any(isinstance(coupler, Reflector) for coupler in [self.IG, self.EG, self.OG])

    def is_within_k_boundary(self, kx, ky):
        """
        Check if the wave vector (kx, ky) is within the k-space boundary defined by the waveguide.
        :param kx: x-component of the wave vector
        :param ky: y-component of the wave vector
        :param waveguide: Waveguide object containing the refractive indices and maximum allowed angle
        :return: True if the wave vector is within the boundary, False otherwise
        """
        k_magnitude = np.sqrt(kx**2 + ky**2)
        k_boundary1 = self.N0
        k_boundary2 = self.N1 * np.sin(np.radians(self.MAX_ALLOWED_ANGLE))
        satisfy = k_boundary1 <= k_magnitude <= k_boundary2
        return satisfy

    def _refraction_candidate(self, K, n_out, normal=(0, 0, 1)):
        """Return (transmitted K candidate, propagating) for a flat interface.
        K is one normalized wave vector, with magnitude equal to the incident refractive index.
        """
        normal = unit_normal(normal)
        normal_component = np.dot(K, normal)
        tangent = K - normal_component * normal
        normal_squared = n_out**2 - np.dot(tangent, tangent)
        if normal_squared <= 0 or normal_component == 0:
            return tangent, False
        transmitted = tangent + np.copysign(np.sqrt(normal_squared), normal_component) * normal
        return transmitted, True

    def refract_ray(self, ray, n_out, normal=(0, 0, 1)):
        """Apply flat-interface Snell refraction, returning None for no transmission.
        """
        K, propagating = self._refraction_candidate(ray.K, n_out, normal)
        return ray.copy_with(K=K, n=n_out) if propagating else None

    def _interface_ray(self, ray, stage, to_internal):
        """Convert an endpoint between the environment and its coupler medium.
        Only a reflector endpoint needs this extra interface.
        """
        if stage not in ('IG', 'OG'):
            raise ValueError('Only IG and OG have external interfaces.')
        if not isinstance(getattr(self, stage), Reflector):
            return ray
        normal = self.input_surface_normal if stage == 'IG' else self.output_surface_normal
        if np.dot(ray.K, normal) <= 0:
            return None
        return self.refract_ray(ray, self.N1 if to_internal else self.N0, normal)

    def _coupler_candidate(self, ray, stage, n_out, reverse=False):
        """Return (candidate K, propagating)"""
        coupler = getattr(self, stage)
        if isinstance(coupler, Grating):
            K = coupler.calc_diffract_k(ray, n_out, reverse=reverse)
            return K, not K[2] == 0
        # Below considers a Reflector
        return coupler.calc_reflect_k(ray), True

    def couple_ray(self, ray, stage, reverse=False):
        """Apply one local coupler to a computational ray, without refraction.
        """
        external_output = (stage == 'IG' and reverse) or (stage == 'OG' and not reverse)
        n_out = self.N0 if external_output and not isinstance(getattr(self, stage), Reflector) else self.N1
        K, propagating = self._coupler_candidate(ray, stage, n_out, reverse=reverse)
        return ray.copy_with(K=K, n=n_out) if propagating else None

    def _trace_coupler(self, rays, coupler, n_out, stage):
        """
        Trace rays through either grating/mirror coupler.
        :param rays: List of Ray objects to be traced
        :param coupler: Grating or mirror object representing the coupler
        :param n_out: Refractive index of the output medium
        :param stage: String indicating the stage ('IG', 'EG', or 'OG')
        :return: List of Ray objects after tracing through the grating
        """
        output_rays = []
        attempted_k = []
        rejected_count = 0
        for ray in rays:
            K, propagating = self._coupler_candidate(ray, stage, n_out)
            if isinstance(coupler, Reflector) and not propagating:
                rejected_count += 1
                continue
            if not stage == 'OG' and not self.is_within_k_boundary(K[0], K[1]):
                rejected_count += 1
                continue
            if isinstance(coupler, Grating):
                coupler.validate_candidate(K)
            output_rays.append(
                Ray(K=K, n=n_out, x=ray.x, y=ray.y, z=ray.z)
            )
            attempted_k.append(K)
        return output_rays, attempted_k, rejected_count

    def trace_rays(self):
        """Trace the incident rays through IG -> EG -> OG
        Returns the output rays list, self.rays3
        """
        self._clear_trace_results()
        rays0 = self.rays0
        entry_rejected = len(self.incident_rays) - len(rays0)
        self.rays1, _, self.rejected_counts['IG'] = self._trace_coupler(rays0, self.IG, self.N1, 'IG')
        self.rejected_counts['IG'] += entry_rejected
        self.rays2, _, self.rejected_counts['EG'] = self._trace_coupler(self.rays1, self.EG, self.N1, 'EG')
        output_index = self.N1 if isinstance(self.OG, Reflector) else self.N0
        rays3, _, rejected_OG = self._trace_coupler(self.rays2, self.OG, output_index, 'OG')
        for ray in rays3:
            external = self._interface_ray(ray, 'OG', to_internal=False)
            if external is None:
                rejected_OG += 1
            else:
                self.rays3.append(ray)
                self.output_rays.append(external)
        for name, rays in (('input', rays0), ('IG', self.rays1), ('EG', self.rays2), ('OG', self.rays3)):
            self.k_points_internal[name] = np.asarray([ray.K for ray in rays]).reshape(-1, 3)
        self.k_points = dict(self.k_points_internal)
        self.k_points['input'] = np.asarray([ray.K for ray in self.rays0_air]).reshape(-1, 3)
        self.k_points['OG'] = np.asarray([ray.K for ray in self.rays3_air]).reshape(-1, 3)
        return self.output_rays

    def calc_IG_diameter(self):
        """Calculate the IG diameter from projector EPD, gap, and FOV."""
        half_fov_rad = np.radians(np.array(self.FOV) / 2)
        tan_half_fov = np.sqrt(np.tan(half_fov_rad[0])**2 + np.tan(half_fov_rad[1])**2)
        self.FOV_diagonal = np.degrees(2 * np.arctan(tan_half_fov))
        self.IG_diameter = self.EPD_projector + 2 * self.gap_projector * np.tan(np.radians(self.FOV_diagonal/2))
        return self.IG_diameter

    def calc_OG_size(self):
        """Calculate the output footprint."""
        output_FOV = self.FOV[::-1] if self.has_reflectors else self.FOV
        self.OG_size = [
            self.eyebox_size[0] + 2 * self.eye_relief * np.tan(np.radians(output_FOV[0]/2)),
            self.eyebox_size[1] + 2 * self.eye_relief * np.tan(np.radians(output_FOV[1]/2)),
        ]
        return self.OG_size

    def calc_coupler_dimensions(self):
        """Calculate IG/OG sizes and the original four-corner EG layout."""
        self._clear_layout_results()
        self.calc_IG_diameter()
        self.calc_OG_size()

        # Exact FOV corners before IG, grouped as left side and right side.
        self.key_rays0_air_LR = [
            [
                self._gen_single_incident_ray(-self.FOV[0]/2, self.FOV[1]/2),
                self._gen_single_incident_ray(-self.FOV[0]/2, -self.FOV[1]/2)
            ],
            [
                self._gen_single_incident_ray(self.FOV[0]/2, self.FOV[1]/2),
                self._gen_single_incident_ray(self.FOV[0]/2, -self.FOV[1]/2),
            ]
            ] # rays before IG, left and right side
        self.key_rays0_LR = []
        for LR in range(2):
            rays = []
            for ray in self.key_rays0_air_LR[LR]:
                internal_ray = self._interface_ray(ray, 'IG', to_internal=True)
                if internal_ray is not None:
                    rays.append(internal_ray)
            self.key_rays0_LR.append(rays)
        self.key_rays1_LR = [[], []]
        for LR in range(2):
            for ray in self.key_rays0_LR[LR]:
                coupled_ray = self.couple_ray(ray, 'IG')
                if coupled_ray is not None and self.is_within_k_boundary(coupled_ray.K[0], coupled_ray.K[1]):
                    self.key_rays1_LR[LR].append(coupled_ray)

        num_ray_left = np.argmin([ray.K[1]/ray.K[0] for ray in self.key_rays1_LR[0]])
        num_ray_right = np.argmax([ray.K[1]/ray.K[0] for ray in self.key_rays1_LR[1]])
        bound_rays_left = self.key_rays1_LR[0][num_ray_left]
        bound_rays_right = self.key_rays1_LR[1][num_ray_right]

        # Position the two bounding rays tangent to the IG circle.
        slope_left = bound_rays_left.K[1] / bound_rays_left.K[0]
        left_direction_vector = np.array([1, -1 / slope_left])
        left_direction_vector /= np.linalg.norm(left_direction_vector)
        left_point_on_IG = np.array(self.IG_origin) - left_direction_vector * self.IG_diameter/2

        slope_right = bound_rays_right.K[1] / bound_rays_right.K[0]
        right_direction_vector = np.array([1, -1 / slope_right])
        right_direction_vector /= np.linalg.norm(right_direction_vector)
        right_point_on_IG = np.array(self.IG_origin) + right_direction_vector * self.IG_diameter/2

        bound_rays_left.x, bound_rays_left.y = left_point_on_IG
        bound_rays_right.x, bound_rays_right.y = right_point_on_IG
        self.key_rays1 = [bound_rays_left, bound_rays_right]
        self.key_rays0 = [
            self.IG.interact(ray, self.N0, reverse=True) for ray in self.key_rays1
        ]

        # using the output rays, calculate the rays after EG, before OG
        top_output_point = [
            self.OG_origin[0] + self.OG_size[0]/2,
            self.OG_origin[1] + self.OG_size[1]/2,
        ]
        bottom_output_point = [
            self.OG_origin[0] + self.OG_size[0]/2,
            self.OG_origin[1] - self.OG_size[1]/2,
        ]
        self.key_rays3_air = [
            Ray(field_angle=np.array([self.FOV[0]/2, -self.FOV[1]/2]), n=self.N0, x=top_output_point[0], y=top_output_point[1],),
            Ray(field_angle=np.array([-self.FOV[0]/2, self.FOV[1]/2]), n=self.N0, x=bottom_output_point[0], y=bottom_output_point[1],),
        ]
        # Reverse layout starts in air; enter the substrate before inverse OM.
        self.key_rays3 = [
            self._interface_ray(ray, 'OG', to_internal=True) for ray in self.key_rays3_air
        ]
        self.key_rays2 = [self.OG.interact(ray, self.N1, reverse=True) for ray in self.key_rays3]

        lines = [
            Line2D([ray.x, ray.y], [ray.x + ray.K[0], ray.y + ray.K[1]])
            for ray in self.key_rays1 + self.key_rays2
        ]
        line1, line2, line3, line4 = lines
        self.EG_corners = np.array([
            intersection(line1, line3),
            intersection(line2, line3),
            intersection(line1, line4),
            intersection(line2, line4),
        ])
        return {
            'IG_diameter': self.IG_diameter,
            'OG_size': self.OG_size,
            'EG_corners': self.EG_corners,
        }

    def analyze(self):
        """Refresh ray tracing and grating dimensions."""
        self.calc_coupler_dimensions()
        self.trace_rays()

    @classmethod
    def from_grating_design(cls, wavelength=650, grating_pitch=380, **overrides):
        """Build the diffractive waveguide."""
        periods = [grating_pitch, grating_pitch / np.sqrt(2), grating_pitch]
        couplers = [
            Grating(wavelength=wavelength, grating_period=period,
                    diffraction_order=1, grating_orientation=orientation)
            for period, orientation in zip(periods, [-90, 45, 180])
        ]
        default_params = dict(
            FOV=[20, 10], index_environment=1.0, index_substrate=1.9107,
            max_allowed_angle=85, wavelength=wavelength,
            IG=couplers[0], EG=couplers[1], OG=couplers[2],
            ray_position_samples=[21, 11],
            EPD_projector=3, gap_projector=2,
            IG_origin=[-10, 5], eyebox_origin=[6, -11],
            eye_relief=20, eyebox_size=[12, 10],
        )
        default_params.update(overrides)
        return cls(**default_params)

    @classmethod
    def from_geometric_design(cls, alpha_im=35.0, **overrides):
        """Build the geometric waveguide.
        Normals start at +z. Input travels toward -z and output toward +z,
        """
        if not 0 < alpha_im < 90:
            raise ValueError("alpha_im must lie strictly between 0 and 90 degrees.")
        default_params = dict(
            FOV=[10, 20], index_environment=1.0, index_substrate=1.9710,
            max_allowed_angle=85, wavelength=450.0,
            IG=Reflector.from_rotation([alpha_im, 0, 0]),
            EG=Reflector.from_rotation([-45, 90, 0]),
            OG=Reflector.from_rotation([0, alpha_im - 90, 0]),
            input_interface_normal=[0, 0, -1],
            output_interface_normal=[0, 0, 1],
            ray_position_samples=[11, 21],
            EPD_projector=3, gap_projector=2, IG_origin=[-10, 5],
            eyebox_origin=[6, -11], eye_relief=20, eyebox_size=[12, 10],
        )
        default_params.update(overrides)
        wg = cls(**default_params)
        wg.alpha_im = alpha_im
        return wg

    ##################################################################################
    ##################################################################################
    #                                                                                #
    # Below are plotting functions for k-space, waveguide layout and system summary. #
    #                                                                                #
    ##################################################################################
    ##################################################################################

    def stage_label(self, stage):
        """Use mirror labels for Reflector objects while keeping IG/EG/OG API keys."""
        if isinstance(getattr(self, stage), Reflector):
            return {'IG': 'IM', 'EG': 'EM', 'OG': 'OM'}[stage]
        return stage


    def plot_k_space(self, filename=None, dpi=600, ax=None):
        """Refresh the trace, plot k-space, and return (figure, axes).
        Set filename=None to skip saving or supply ax to use an existing axes.
        """
        self.trace_rays()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure
        ax.grid(True, alpha=0.5)

        points = self.k_points['input']
        ax.scatter(
            points[:, 0], points[:, 1], color='blue', s=12,
            marker='o', label='Input Rays',
        )
        ax.add_artist(plt.Circle(
            (0, 0), self.N0, color='black', fill=False,
            linestyle='--', label=r'$n_0$',
        ))
        ax.add_artist(plt.Circle(
            (0, 0), self.N1 * np.sin(np.radians(self.MAX_ALLOWED_ANGLE)),
            color='red', fill=False, linestyle='--',
            label=r'$n_1\sin(\theta_{max})$',
        ))
        for stage, color, size, marker in [
            ('IG', 'green', 6, 'x'),
            ('EG', 'orange', 6, '^'),
            ('OG', 'red', 2, 's'),
        ]:
            points = self.k_points[stage]
            if len(points):
                ax.scatter(
                    points[:, 0], points[:, 1], color=color,
                    s=size, marker=marker, label=f'After {self.stage_label(stage)}',
                )
            if stage == 'OG' and self.has_reflectors:
                ax.scatter(
                    points[:, 0], points[:, 1], s=18, marker='s',
                    facecolors='none', edgecolors=color, linewidths=0.6,
                    label='Output rays (air)', zorder=4,
                )

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xlabel('kx')
        ax.set_ylabel('ky')
        ax.set_title(('Geometric' if self.has_reflectors else 'Diffractive') + ' Waveguide K-Space')
        ax.legend()
        ax.set_aspect('equal')
        if filename is not None:
            fig.savefig(filename, dpi=dpi)
        return fig, ax

    def draw_mirrors(self, ax, stage, region, vertices, **kwargs):
        """Draw schematic mirror faces, clipped to their placement region.

        The surface trace follows (-ny, nx), perpendicular to the projected mirror normal.
        A finite face's other in-plane dimension projects by abs(nz); a vertical mirror therefore appears as a single edge.
        Counts and face widths are for demo only, not solved apertures or physical mirror pitch.
        Neither ray tracing nor footprint sizing uses them.
        """
        count = kwargs.get('count', 1)
        fill = kwargs.get('fill', 0.5)
        color = kwargs.get('color', 'gray')
        label = kwargs.get('label', f'{self.stage_label(stage)} Mirrors')
        if count == 0:
            return None
        normal = getattr(self, stage).normal # surface normal, 3D
        transverse = np.linalg.norm(normal[:2])
        if transverse <= 1e-12:
            # A face parallel to xy cannot trace rays in the waveguide.
            projected_normal = np.array([1.0, 0.0])
            count = 1
        else:
            projected_normal = normal[:2] / transverse
            # Choose one unoriented axis so n and -n draw identical artists.
            pivot = np.flatnonzero(np.abs(projected_normal) > 1e-12)[0]
            if projected_normal[pivot] < 0:
                projected_normal = -projected_normal
        tangent = np.array([-projected_normal[1], projected_normal[0]])

        vertices = np.asarray(vertices, dtype=float)
        across = vertices @ projected_normal
        along = vertices @ tangent
        pitch = np.ptp(across) / count
        offsets = across.min() + (np.arange(count) + 0.5) * pitch
        padding = max(float(np.ptp(along)), 1.0)
        ends = np.array([along.min() - padding, along.max() + padding])

        if abs(normal[2]) <= 1e-12:
            segments = [
                offset * projected_normal + ends[:, None] * tangent
                for offset in offsets
            ]
            artist = LineCollection(
                segments, colors=color, linewidths=1.5, alpha=0.95,
                label=label, zorder=3,
            )
        else:
            width = pitch if transverse <= 1e-12 else fill * pitch
            faces = [
                np.array([
                    (offset - width/2) * projected_normal + ends[0] * tangent,
                    (offset + width/2) * projected_normal + ends[0] * tangent,
                    (offset + width/2) * projected_normal + ends[1] * tangent,
                    (offset - width/2) * projected_normal + ends[1] * tangent,
                ])
                for offset in offsets
            ]
            artist = PolyCollection(
                faces, facecolors=color, alpha=0.5,
                edgecolors=color, linewidths=1.0,
                label=label, zorder=2.5,
            )
        artist.set_gid(f'{stage}_mirror_surfaces')
        ax.add_collection(artist, autolim=False)
        artist.set_clip_path(region.get_path(), region.get_transform())
        return artist

    def plot_waveguide_layout(self, filename=None, dpi=600, ax=None, **kwargs):
        """Plot waveguide layouts with boundary rays, gratings or mirror surfaces, etc.

        mirror_counts draws schematic counts by IG/EG/OG (defaults 1/16/12; zero hides a stage).
        mirror_fill is the projected surface-width/pitch ratio.
        Schematic drawings only; not calculated mirror properties.
        """
        self.calc_coupler_dimensions()

        if self.has_reflectors:
            mirror_counts = kwargs.get('mirror_counts', {'IG': 1, 'EG': 16, 'OG': 12})
            mirror_fill = kwargs.get('mirror_fill', 0.5)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        kind_IG = 'Reflector' if isinstance(self.IG, Reflector) else 'Grating'
        kind_EG = 'Reflector' if isinstance(self.EG, Reflector) else 'Grating'
        kind_OG = 'Reflector' if isinstance(self.OG, Reflector) else 'Grating'

        ax.add_artist(plt.Circle(
            self.IG_origin, self.EPD_projector/2, color='gray',
            fill=False, linestyle='--', linewidth=3,
            label=f'Projector (EPD={self.EPD_projector:.2f} mm)',
        ))
        ax.scatter(self.IG_origin[0], self.IG_origin[1], color='gray', s=50, marker='o')

        plot_origin_eyebox = [
            self.OG_origin[0] - self.eyebox_size[0]/2,
            self.OG_origin[1] - self.eyebox_size[1]/2,
        ]
        ax.add_artist(plt.Rectangle(
            plot_origin_eyebox, self.eyebox_size[0], self.eyebox_size[1],
            color='orange', fill=False, linestyle='--', linewidth=3,
            label=f'Eyebox (size={self.eyebox_size[0]:.2f} mm x {self.eyebox_size[1]:.2f} mm)',
        ))
        ax.scatter(self.OG_origin[0], self.OG_origin[1], color='orange', s=50, marker='s')

        input_region = plt.Circle(
            self.IG_origin, self.IG_diameter/2, color='blue',
            fill=True, alpha=0.15 if self.has_reflectors else 0.5,
            label=f'Input {kind_IG} ({self.stage_label("IG")}) (diameter={self.IG_diameter:.2f} mm)',
        )
        ax.add_artist(input_region)
        ax.scatter(self.IG_origin[0], self.IG_origin[1], color='blue', s=50, marker='o')

        for ray in self.key_rays1:
            ax.axline(
                [ray.x, ray.y], [ray.x + ray.K[0], ray.y + ray.K[1]],
                color='green', linestyle='--', alpha=0.3,
            )
        for ray in self.key_rays2:
            ax.axline(
                [ray.x, ray.y], [ray.x + ray.K[0], ray.y + ray.K[1]],
                color='orange', linestyle='--', alpha=0.3,
            )

        for x, y in self.EG_corners:
            ax.text(
                x, y, f"({x:.2f}, {y:.2f})", color='black',
                fontsize=8, ha='center',
            )

        if isinstance(self.EG, Reflector):
            expansion_region = plt.Polygon(
                self.EG_corners[[0, 1, 3, 2]], edgecolor='purple', facecolor='none',
                linestyle='--', linewidth=1.0, alpha=0.6,
                label=f'Expansion {kind_EG} ({self.stage_label("EG")})',
            )
        else:
            expansion_region = plt.Polygon(
                self.EG_corners[[0, 1, 3, 2]] if self.has_reflectors else self.EG_corners[[0, 1, 3, 2]],
                color='purple', fill=True,
                alpha=0.3, label=f'Expansion {kind_EG} ({self.stage_label("EG")})',
            )
        ax.add_artist(expansion_region)
        ax.scatter(
            self.EG_corners[:, 0], self.EG_corners[:, 1],
            color='purple', s=50, marker='^',
        )

        output_region = plt.Rectangle(
            (self.OG_origin[0] - self.OG_size[0]/2,
             self.OG_origin[1] - self.OG_size[1]/2),
            self.OG_size[0], self.OG_size[1], color='red', fill=True, alpha=0.18 if self.has_reflectors else 0.5,
            label=f'Output {kind_OG} ({self.stage_label("OG")}) (size={self.OG_size[0]:.2f} mm x {self.OG_size[1]:.2f} mm)',
        )
        ax.add_artist(output_region)
        ax.scatter(self.OG_origin[0], self.OG_origin[1], color='red', s=50, marker='s')

        if self.has_reflectors:
            box_corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
            corners = {
                'IG': self.IG_origin + self.IG_diameter/2 * box_corners,
                'EG': self.EG_corners,
                'OG': self.OG_origin + np.asarray(self.OG_size)/2 * box_corners,
            }
            self.draw_mirrors(
                ax, 'IG', input_region, corners['IG'],
                count=mirror_counts.get('IG', 1), fill=mirror_fill, color='blue',
            )
            self.draw_mirrors(
                ax, 'EG', expansion_region, corners['EG'],
                count=mirror_counts.get('EG', 16), fill=mirror_fill, color='purple',
            )
            self.draw_mirrors(
                ax, 'OG', output_region, corners['OG'],
                count=mirror_counts.get('OG', 12), fill=mirror_fill, color='red',
            )

        ax.set_xlim(-20, 25)
        ax.set_ylim(-22, 12)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(('Geometric' if self.has_reflectors else 'Diffractive') + ' Waveguide Layout (XY View)')
        ax.grid(True, alpha=0.5)
        if self.has_reflectors:
            ax.text(
                0.5, -0.14,
                'Mirror count and face width are schematic.',
                transform=ax.transAxes, ha='center', va='top', fontsize=8, color='0.35',
            )
        ax.legend(loc='upper right')
        ax.set_aspect('equal')
        if filename is not None:
            fig.savefig(filename, dpi=dpi)
        return fig, ax

    def print_summary(self):
        """Refresh the complete analysis and print the original reports."""
        self.analyze()
        if self.has_reflectors:
            print("Geometric Waveguide Summary:")
            print(f"Input mirror rotation alpha_IM: {self.alpha_im:.6f} deg")
            print(f"Design wavelength: {self.wavelength:.1f} nm")
            print(f"Substrate refractive index: {self.N1:.6f}")
        else:
            print("Diffractive Waveguide Summary:")
            print(
                f"Grating period for IG: {self.IG.grating_period:.2f} nm, "
                f"EG: {self.EG.grating_period:.2f} nm, OG: {self.OG.grating_period:.2f} nm"
            )

        for stage in ('IG', 'EG'):
            print(f'{self.stage_label(stage)}: {getattr(self, stage)!r}')
            print(f'Rejected rays at {self.stage_label(stage)}: {self.rejected_percentages[stage]:.2f}%')
        print(f'Input footprint diameter: {self.IG_diameter:.4f} mm')
        print(f'Input footprint center: ({self.IG_origin[0]:.4f}, {self.IG_origin[1]:.4f}) mm')
        print(f'Output footprint size: {self.OG_size[0]:.4f} mm x {self.OG_size[1]:.4f} mm')
        print(f'Output footprint center: ({self.OG_origin[0]:.4f}, {self.OG_origin[1]:.4f}) mm')
        corners = ", ".join(f"[{x:.4f}, {y:.4f}]" for x, y in self.EG_corners)
        print(f"Expansion Coupler Corners: {corners}")





