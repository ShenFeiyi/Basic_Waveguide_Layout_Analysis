# -*- coding:utf-8 -*-
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection

from RayObject import Ray
from GratingObject import Grating
from ReflectorObject import Reflector
from MaterialObject import Material
from functions import unit_normal, intersection, Line2D

class Waveguide:
    STAGES = ('IG', 'EG', 'OG')
    # Accepted constructor keywords; anything else raises instead of being silently ignored.
    PARAMETERS = (
        'IG', 'EG', 'OG', 'input_surface_normal', 'output_surface_normal',
        'FOV', 'wavelength', 'index_environment', 'index_substrate',
        'max_allowed_angle', 'ray_position_samples', 'EPD_projector',
        'gap_projector', 'IG_origin', 'OG_origin', 'eye_relief', 'eyebox_size',
        'incident_rays',
    )

    def __init__(self, **kwargs):
        unknown = sorted(set(kwargs) - set(self.PARAMETERS))
        if unknown:
            raise TypeError(
                f"Unknown Waveguide parameter(s): {', '.join(unknown)}. "
                f"Valid parameters: {', '.join(self.PARAMETERS)}."
            )
        self.IG = kwargs.get('IG', None) # ipnut grating
        self.EG = kwargs.get('EG', None) # expansion grating
        self.OG = kwargs.get('OG', None) # output grating
        assert self.IG is not None, "Input grating (IG) must be provided."
        assert self.EG is not None, "Expansion grating (EG) must be provided."
        assert self.OG is not None, "Output grating (OG) must be provided."

        self.input_surface_normal = unit_normal(kwargs.get('input_surface_normal', [0, 0, -1]))
        self.output_surface_normal = unit_normal(kwargs.get('output_surface_normal', [0, 0, 1]))

        self.FOV = kwargs.get('FOV', [1, 1]) # field of view, degrees
        self.wavelength = kwargs.get('wavelength', None) # analysis wavelength, nm
        self._check_grating_wavelengths()
        # Indices may be numbers or dispersion models (see MaterialObject); the
        # index_* values below are those models evaluated at self.wavelength.
        self.environment_material = Material.create(kwargs.get('index_environment', 1.0))
        self.substrate_material = Material.create(kwargs.get('index_substrate', 1.5))
        self.index_environment = self.environment_material(self.wavelength) # refractive index of the environment
        self.index_substrate = self.substrate_material(self.wavelength) # refractive index of the substrate
        self.MAX_ALLOWED_ANGLE = kwargs.get('max_allowed_angle', 90) # maximum allowed angle, degrees
        self.ray_position_samples = kwargs.get('ray_position_samples', [21, 11]) # number of samples for ray positions in the input grating plane
        self.EPD_projector = kwargs.get('EPD_projector', 3.0) # exit pupil diameter of the projector, mm
        self.gap_projector = kwargs.get('gap_projector', 2.0) # gap between the projector and the input grating, mm
        self.IG_origin = kwargs.get('IG_origin', [-10, 5]) # origin of the input grating, mm
        self.OG_origin = kwargs.get('OG_origin', [6, -11]) # origin of the output grating, mm
        self.eye_relief = kwargs.get('eye_relief', 20.0) # eye relief, mm
        self.eyebox_size = kwargs.get('eyebox_size', [12, 10]) # size of the eyebox, mm

        self.incident_rays = kwargs.get('incident_rays', None) # incident rays, list of Ray objects
        self._user_incident_rays = self.incident_rays is not None
        if self.incident_rays is None:
            self.generate_incident_rays()
        else:
            self.field_angles = np.array([ray.field_angle for ray in self.incident_rays]).reshape(-1, 2)

        self._clear_trace_results()
        self._clear_layout_results()

    def __repr__(self):
        return (
            f"Waveguide(FOV={self.FOV},"
            f"wavelength={self.wavelength},"
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

    def _check_grating_wavelengths(self):
        """Gratings must diffract the analysis wavelength; adopt theirs if none is given."""
        wavelengths = [c.wavelength for c in (self.IG, self.EG, self.OG) if isinstance(c, Grating)]
        if not wavelengths:
            return
        reference = wavelengths[0] if self.wavelength is None else self.wavelength
        if not np.allclose(wavelengths, reference):
            raise ValueError(
                f"Grating wavelengths {wavelengths} nm differ from the analysis wavelength "
                f"{reference} nm; build matching gratings or use Waveguide.at_wavelength()."
            )
        self.wavelength = reference

    def _wavelength_text(self):
        return '' if self.wavelength is None else f' at {self.wavelength:g} nm'

    def at_wavelength(self, wavelength):
        """Return a copy of this waveguide analyzed at another vacuum wavelength (nm).

        Grating periods are kept, so grating vectors scale as wavelength/period;
        dispersive indices are re-evaluated and the incident FOV grid is regenerated.
        """
        wg = copy.deepcopy(self)
        wg.wavelength = float(wavelength)
        wg.IG, wg.EG, wg.OG = (coupler.at_wavelength(wg.wavelength) for coupler in (wg.IG, wg.EG, wg.OG))
        wg.index_environment = wg.environment_material(wg.wavelength)
        wg.index_substrate = wg.substrate_material(wg.wavelength)
        if wg._user_incident_rays:
            wg.incident_rays = [ray.copy_with(n=wg.N0) for ray in wg.incident_rays]
        else:
            wg.generate_incident_rays()
        wg._clear_trace_results()
        wg._clear_layout_results()
        return wg

    def _gen_single_incident_ray(self, theta_x, theta_y):
        ray = Ray(field_angle=np.array([theta_x, theta_y]), n=self.N0)
        if isinstance(self.IG, Reflector) and self.input_surface_normal[2] < 0:
            K = ray.K.copy()
            K[2] *= -1
            ray = Ray(K=K, n=self.N0)
        return ray

    def _corner_fields(self):
        """Exact FOV corners: left side (top, bottom), then right side (top, bottom)."""
        half_x, half_y = self.FOV[0]/2, self.FOV[1]/2
        return [(-half_x, half_y), (-half_x, -half_y), (half_x, half_y), (half_x, -half_y)]

    def _field_grid(self, samples=None):
        """Rectangular FOV grid of (theta_x, theta_y) in degrees."""
        samples = self.ray_position_samples if samples is None else samples
        return [
            (theta_x, theta_y)
            for theta_x in np.linspace(-self.FOV[0]/2, self.FOV[0]/2, samples[0])
            for theta_y in np.linspace(-self.FOV[1]/2, self.FOV[1]/2, samples[1])
        ]

    def generate_incident_rays(self, ray_position_samples=None):
        """Generate the rays for rectangular FOV grid.
        """
        samples = self.ray_position_samples if ray_position_samples is None else ray_position_samples
        self.ray_position_samples = list(samples)
        self.field_angles = np.array(self._field_grid(samples)).reshape(-1, 2) # input field angles, degrees
        self.incident_rays = [
            self._gen_single_incident_ray(theta_x, theta_y) for theta_x, theta_y in self.field_angles
        ]
        self._user_incident_rays = False

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
        self.rejected_counts = {stage: 0 for stage in self.STAGES}
        self.ray_status = [] # per incident ray: stage where it was lost, None if it reaches the output
        self.ray_indices = {} # per stage: incident-ray index of each traced ray

    def _clear_layout_results(self):
        self.FOV_diagonal = None
        self.output_FOV = None
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
        } # IG: 0.xx, EG: 0.xx, OG: 0.xx

    @property
    def transmitted_percentage(self):
        """Share of incident rays that reach the output, in percent."""
        if self.number_of_rays == 0:
            raise ValueError("At least one incident ray is required.")
        return len(self.output_rays) / self.number_of_rays * 100

    @property
    def transmitted_mask(self):
        """Boolean mask over incident rays (and field_angles): True where the ray reaches the output."""
        return np.array([status is None for status in self.ray_status], dtype=bool)

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

    def _trace_coupler(self, rays, n_out, stage):
        """
        Trace rays through either grating/mirror coupler.
        :param rays: List of Ray objects to be traced
        :param n_out: Refractive index of the output medium
        :param stage: String indicating the stage ('IG', 'EG', or 'OG')
        :return: (rays after the coupler, positions in `rays` of the kept rays, rejected count)
        """
        output_rays = []
        kept = []
        rejected_count = 0
        for i, ray in enumerate(rays):
            K, propagating = self._coupler_candidate(ray, stage, n_out)
            if not propagating:
                # Evanescent diffraction order, e.g. an OG order outside the environment's light cone.
                rejected_count += 1
                continue
            if not stage == 'OG' and not self.is_within_k_boundary(K[0], K[1]):
                rejected_count += 1
                continue
            output_rays.append(
                Ray(K=K, n=n_out, x=ray.x, y=ray.y, z=ray.z)
            )
            kept.append(i)
        return output_rays, kept, rejected_count

    def _propagate(self, incident_rays):
        """Trace incident rays through the entrance, IG, EG, OG, and exit.

        Returns {'rays': per-stage ray lists, 'index': incident-ray index of each
        traced ray per stage, 'status': per incident ray, the stage where it was
        lost (None if it reaches the output)}. Entrance losses count against IG.
        """
        status = ['IG'] * len(incident_rays)
        rays0, index0 = [], []
        for i, ray in enumerate(incident_rays):
            internal_ray = self._interface_ray(ray, 'IG', to_internal=True)
            if internal_ray is not None:
                rays0.append(internal_ray)
                index0.append(i)
        rays1, kept, _ = self._trace_coupler(rays0, self.N1, 'IG')
        index1 = [index0[k] for k in kept]
        for i in index1:
            status[i] = 'EG'
        rays2, kept, _ = self._trace_coupler(rays1, self.N1, 'EG')
        index2 = [index1[k] for k in kept]
        for i in index2:
            status[i] = 'OG'
        output_index = self.N1 if isinstance(self.OG, Reflector) else self.N0
        candidates, kept, _ = self._trace_coupler(rays2, output_index, 'OG')
        rays3, output_rays, index3 = [], [], []
        for k, ray in zip(kept, candidates):
            external = self._interface_ray(ray, 'OG', to_internal=False)
            if external is not None:
                rays3.append(ray)
                output_rays.append(external)
                index3.append(index2[k])
                status[index2[k]] = None
        return {
            'rays': {'input': rays0, 'IG': rays1, 'EG': rays2, 'OG': rays3, 'output': output_rays},
            'index': {'input': index0, 'IG': index1, 'EG': index2, 'OG': index3},
            'status': status,
        }

    def trace_rays(self):
        """Trace the incident rays through IG -> EG -> OG
        Returns the output rays list, self.output_rays
        """
        self._clear_trace_results()
        result = self._propagate(self.incident_rays)
        rays = result['rays']
        self.rays1, self.rays2, self.rays3 = rays['IG'], rays['EG'], rays['OG']
        self.output_rays = rays['output']
        self.ray_status = result['status']
        self.ray_indices = result['index']
        for stage in self.STAGES:
            self.rejected_counts[stage] = self.ray_status.count(stage)
        for name in ('input', 'IG', 'EG', 'OG'):
            self.k_points_internal[name] = np.asarray([ray.K for ray in rays[name]]).reshape(-1, 3)
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

    def _layout_trace(self):
        """Trace the exact FOV corners plus the FOV grid for footprint sizing.
        Only fields that reach the output are used, so the layout follows the
        part of the FOV that is guided at this wavelength.
        """
        air_rays = [self._gen_single_incident_ray(*field) for field in self._corner_fields() + self._field_grid()]
        trace = self._propagate(air_rays)
        trace['air'] = air_rays
        trace['transmitted'] = [i for i, status in enumerate(trace['status']) if status is None]
        if not trace['transmitted']:
            raise ValueError(f"No field reaches the output{self._wavelength_text()}; the coupler layout is undefined.")
        return trace

    def calc_output_FOV(self, trace=None):
        """Full output FOV [horizontal, vertical] in degrees, from forward-traced fields.
        For the geometric preset the mirrors rotate the field, so this is FOV[::-1].
        """
        trace = self._layout_trace() if trace is None else trace
        output = dict(zip(trace['index']['OG'], trace['rays']['output']))
        angles = np.abs([output[i].field_angle for i in trace['transmitted']])
        self.output_FOV = [float(angle) for angle in 2 * angles.max(axis=0)]
        return self.output_FOV

    def calc_OG_size(self, trace=None):
        """Calculate the output footprint from the forward-traced output FOV."""
        output_FOV = self.calc_output_FOV(trace)
        self.OG_size = [
            self.eyebox_size[0] + 2 * self.eye_relief * np.tan(np.radians(output_FOV[0]/2)),
            self.eyebox_size[1] + 2 * self.eye_relief * np.tan(np.radians(output_FOV[1]/2)),
        ]
        return self.OG_size

    def calc_coupler_dimensions(self):
        """Calculate IG/OG sizes and the four-corner EG layout.

        Left/right EG edges: the most divergent guided rays after IG, tangent to the IG circle.
        Top/bottom EG edges: each output field only needs the eyebox shifted back by
        eye_relief*tan(field) on the OG, so the key rays are the after-EG lines through
        those points that pass highest and lowest at the IG x position.
        """
        self._clear_layout_results()
        trace = self._layout_trace()
        self.calc_IG_diameter()
        self.calc_OG_size(trace)

        # Guided rays after IG travel toward -y; tilt = kx/|ky| is negative for left-going rays.
        after_IG = dict(zip(trace['index']['IG'], trace['rays']['IG']))
        internal = dict(zip(trace['index']['input'], trace['rays']['input']))
        if any(after_IG[i].K[1] >= 0 for i in trace['transmitted']):
            raise ValueError(f"The layout model expects light to leave {self.stage_label('IG')} toward -y.")
        tilt = {i: after_IG[i].K[0] / -after_IG[i].K[1] for i in trace['transmitted']}
        sides = [
            [i for i in trace['transmitted'] if tilt[i] < 0],
            [i for i in trace['transmitted'] if tilt[i] >= 0],
        ]
        self.key_rays0_air_LR = [[trace['air'][i] for i in side] for side in sides]
        self.key_rays0_LR = [[internal[i] for i in side] for side in sides]
        self.key_rays1_LR = [[after_IG[i] for i in side] for side in sides]

        # The most divergent guided rays, positioned tangent to the left and right of the IG circle.
        bound_rays_left = after_IG[min(tilt, key=tilt.get)].copy_with()
        bound_rays_right = after_IG[max(tilt, key=tilt.get)].copy_with()
        for ray, side in ((bound_rays_left, -1), (bound_rays_right, 1)):
            outward = np.array([-ray.K[1], ray.K[0]]) / np.hypot(ray.K[0], ray.K[1]) # unit normal, +x side
            ray.x, ray.y = np.array(self.IG_origin) + side * outward * self.IG_diameter/2
        self.key_rays1 = [bound_rays_left, bound_rays_right]
        self.key_rays0 = [
            self.IG.interact(ray, self.N0, reverse=True) for ray in self.key_rays1
        ]

        # Rays after EG that feed the highest and lowest needed OG points.
        after_EG = dict(zip(trace['index']['EG'], trace['rays']['EG']))
        after_OG = dict(zip(trace['index']['OG'], trace['rays']['OG']))
        output = dict(zip(trace['index']['OG'], trace['rays']['output']))
        eyebox_corners = np.array(self.OG_origin) + np.array(self.eyebox_size)/2 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        x_ref = self.IG_origin[0]
        candidates = [] # (y of the after-EG line at x_ref, incident index, OG point)
        for i in trace['transmitted']:
            K_out, direction = output[i].K, after_EG[i].K
            if abs(direction[0]) < 1e-12:
                continue
            for point in eyebox_corners - self.eye_relief * K_out[:2] / abs(K_out[2]):
                candidates.append((point[1] + direction[1]/direction[0] * (x_ref - point[0]), i, point))
        if not candidates:
            raise ValueError(f"Rays after {self.stage_label('EG')} do not travel toward the output; the EG layout is undefined.")
        key_points = [max(candidates, key=lambda c: c[0]), min(candidates, key=lambda c: c[0])] # top, bottom
        self.key_rays3_air = [output[i].copy_with(position=(*point, 0)) for _, i, point in key_points]
        self.key_rays3 = [after_OG[i].copy_with(position=(*point, 0)) for _, i, point in key_points]
        self.key_rays2 = [after_EG[i].copy_with(position=(*point, 0)) for _, i, point in key_points]

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
            'output_FOV': self.output_FOV,
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
            IG_origin=[-10, 5], OG_origin=[6, -11],
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
            input_surface_normal=[0, 0, -1],
            output_surface_normal=[0, 0, 1],
            ray_position_samples=[11, 21],
            EPD_projector=3, gap_projector=2, IG_origin=[-10, 5],
            OG_origin=[6, -11], eye_relief=20, eyebox_size=[12, 10],
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
            if stage == 'OG' and self.has_reflectors and len(points):
                ax.scatter(
                    points[:, 0], points[:, 1], s=18, marker='s',
                    facecolors='none', edgecolors=color, linewidths=0.6,
                    label='Output rays (air)', zorder=4,
                )

        k_radius = max(self.N0, self.N1 * np.sin(np.radians(self.MAX_ALLOWED_ANGLE)))
        k_limit = max(2.0, np.ceil(10 * (k_radius + 0.02)) / 10)
        ax.set_xlim(-k_limit, k_limit)
        ax.set_ylim(-k_limit, k_limit)
        ax.set_xlabel('kx')
        ax.set_ylabel('ky')
        ax.set_title(('Geometric' if self.has_reflectors else 'Diffractive') + ' Waveguide K-Space' + self._wavelength_text())
        ax.legend()
        ax.set_aspect('equal')
        if filename is not None:
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
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

    def _draw_pupils_and_input(self, ax):
        """Draw the projector pupil, eyebox, and input coupler; return the input patch."""
        kind_IG = 'Reflector' if isinstance(self.IG, Reflector) else 'Grating'
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
        return input_region

    def _draw_output_region(self, ax, OG_size=None):
        """Draw the output coupler footprint (default self.OG_size); return its patch."""
        OG_size = self.OG_size if OG_size is None else OG_size
        kind_OG = 'Reflector' if isinstance(self.OG, Reflector) else 'Grating'
        output_region = plt.Rectangle(
            (self.OG_origin[0] - OG_size[0]/2,
             self.OG_origin[1] - OG_size[1]/2),
            OG_size[0], OG_size[1], color='red', fill=True, alpha=0.18 if self.has_reflectors else 0.5,
            label=f'Output {kind_OG} ({self.stage_label("OG")}) (size={OG_size[0]:.2f} mm x {OG_size[1]:.2f} mm)',
        )
        ax.add_artist(output_region)
        ax.scatter(self.OG_origin[0], self.OG_origin[1], color='red', s=50, marker='s')
        return output_region

    def _draw_layout_mirrors(self, ax, regions, EG_corners, OG_size, mirror_counts, mirror_fill):
        """Draw schematic IM/EM/OM faces clipped to their footprint patches."""
        box_corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        corners = {
            'IG': self.IG_origin + self.IG_diameter/2 * box_corners,
            'EG': EG_corners,
            'OG': self.OG_origin + np.asarray(OG_size)/2 * box_corners,
        }
        for stage, color, default_count in (('IG', 'blue', 1), ('EG', 'purple', 16), ('OG', 'red', 12)):
            self.draw_mirrors(
                ax, stage, regions[stage], corners[stage],
                count=mirror_counts.get(stage, default_count), fill=mirror_fill, color=color,
            )

    def plot_waveguide_layout(self, filename=None, dpi=600, ax=None, mirror_counts=None,
                              mirror_fill=0.5, xlim=(-20, 25), ylim=(-22, 12)):
        """Plot waveguide layouts with boundary rays, gratings or mirror surfaces, etc.

        mirror_counts draws schematic counts by IG/EG/OG (defaults 1/16/12; zero hides a stage).
        mirror_fill is the projected surface-width/pitch ratio.
        Schematic drawings only; not calculated mirror properties.
        """
        self.calc_coupler_dimensions()
        mirror_counts = {'IG': 1, 'EG': 16, 'OG': 12} if mirror_counts is None else mirror_counts

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        kind_EG = 'Reflector' if isinstance(self.EG, Reflector) else 'Grating'
        input_region = self._draw_pupils_and_input(ax)

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
                self.EG_corners[[0, 1, 3, 2]],
                color='purple', fill=True,
                alpha=0.3, label=f'Expansion {kind_EG} ({self.stage_label("EG")})',
            )
        ax.add_artist(expansion_region)
        ax.scatter(
            self.EG_corners[:, 0], self.EG_corners[:, 1],
            color='purple', s=50, marker='^',
        )

        output_region = self._draw_output_region(ax)

        if self.has_reflectors:
            self._draw_layout_mirrors(
                ax, {'IG': input_region, 'EG': expansion_region, 'OG': output_region},
                self.EG_corners, self.OG_size, mirror_counts, mirror_fill,
            )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(('Geometric' if self.has_reflectors else 'Diffractive') + ' Waveguide Layout (XY View)' + self._wavelength_text())
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
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        return fig, ax

    def _draw_fov_status(self, ax, color='tab:blue'):
        """Scatter the input field grid by fate: transmitted, or the stage where it was lost."""
        status = self.ray_status
        styles = [
            (None, 'Transmitted', dict(marker='o', color=color)),
            ('IG', f'Lost at {self.stage_label("IG")}', dict(marker='x', color='0.2')),
            ('EG', f'Lost at {self.stage_label("EG")}', dict(marker='^', facecolors='none', edgecolors='0.3')),
            ('OG', f'Lost at {self.stage_label("OG")}', dict(marker='s', facecolors='none', edgecolors='0.5')),
        ]
        for stage, label, style in styles:
            mask = np.array([s is None if stage is None else s == stage for s in status], dtype=bool)
            if mask.any():
                ax.scatter(self.field_angles[mask, 0], self.field_angles[mask, 1], s=20, label=label, **style)
        self._format_fov_axes(ax)

    def _format_fov_axes(self, ax):
        """Axis limits and labels for an input-field (FOV) map."""
        margin = 0.06 * max(self.FOV)
        ax.set_xlim(-self.FOV[0]/2 - margin, self.FOV[0]/2 + margin)
        ax.set_ylim(-self.FOV[1]/2 - margin, self.FOV[1]/2 + margin)
        ax.set_xlabel(r'Input field $\theta_x$ (deg)')
        ax.set_ylabel(r'Input field $\theta_y$ (deg)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.5)

    def plot_fov_coverage(self, filename=None, dpi=600, ax=None, color='tab:blue'):
        """Refresh the trace and plot which input fields reach the output; return (figure, axes).
        Lost fields are marked by the stage that rejected them.
        """
        self.trace_rays()
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        else:
            fig = ax.figure
        self._draw_fov_status(ax, color)
        ax.set_title(
            ('Geometric' if self.has_reflectors else 'Diffractive') + ' Waveguide FOV Coverage'
            + self._wavelength_text() + f'\n{self.transmitted_percentage:.1f}% of sampled fields transmitted'
        )
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
        if filename is not None:
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        return fig, ax

    def print_summary(self):
        """Refresh the complete analysis and print the original reports."""
        self.analyze()
        if self.has_reflectors:
            print("Geometric Waveguide Summary:")
            if getattr(self, 'alpha_im', None) is not None:
                print(f"Input mirror rotation alpha_IM: {self.alpha_im:.6f} deg")
        else:
            print("Diffractive Waveguide Summary:")
            print(
                f"Grating period for IG: {self.IG.grating_period:.2f} nm, "
                f"EG: {self.EG.grating_period:.2f} nm, OG: {self.OG.grating_period:.2f} nm"
            )
        if self.wavelength is not None:
            print(f"Wavelength: {self.wavelength:.1f} nm")
        print(f"Substrate refractive index: {self.N1:.6f}")

        for stage in self.STAGES:
            print(f'{self.stage_label(stage)}: {getattr(self, stage)!r}')
            print(f'Rejected rays at {self.stage_label(stage)}: {self.rejected_percentages[stage]:.2f}%')
        print(f'Transmitted rays: {self.transmitted_percentage:.2f}%')
        print(f'Output FOV: {self.output_FOV[0]:.2f} deg x {self.output_FOV[1]:.2f} deg')
        print(f'Input footprint diameter: {self.IG_diameter:.4f} mm')
        print(f'Input footprint center: ({self.IG_origin[0]:.4f}, {self.IG_origin[1]:.4f}) mm')
        print(f'Output footprint size: {self.OG_size[0]:.4f} mm x {self.OG_size[1]:.4f} mm')
        print(f'Output footprint center: ({self.OG_origin[0]:.4f}, {self.OG_origin[1]:.4f}) mm')
        corners = ", ".join(f"[{x:.4f}, {y:.4f}]" for x, y in self.EG_corners)
        print(f"Expansion Coupler Corners: {corners}")
