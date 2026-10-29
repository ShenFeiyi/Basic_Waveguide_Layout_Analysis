# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from WaveguideObject import Waveguide
from MultiWavelengthObject import MultiWavelengthWaveguide

def main(parameters):
    """Build, summarize, and plot one design.

    parameters["waveguide"]["wavelength"] is one wavelength or a list (nm). A list
    runs MultiWavelengthWaveguide and overlays the results. Plots are made for the
    "layout_plot", "k_space_plot", and "fov_plot" entries that are present.
    Returns the Waveguide or MultiWavelengthWaveguide.
    """
    waveguide_type = parameters.get("waveguide_type")
    waveguide_parameters = dict(parameters["waveguide"])
    wavelengths = np.atleast_1d(waveguide_parameters.get("wavelength", [])).astype(float)
    if wavelengths.size:
        waveguide_parameters["wavelength"] = float(wavelengths[0])
    if waveguide_type == "geometric":
        waveguide = Waveguide.from_geometric_design(**waveguide_parameters)
    elif waveguide_type == "diffractive":
        waveguide = Waveguide.from_grating_design(**waveguide_parameters)
    else:
        raise ValueError("Invalid waveguide type")

    analysis = MultiWavelengthWaveguide(waveguide, wavelengths) if wavelengths.size > 1 else waveguide
    analysis.print_summary()
    for key, plot in (
        ("layout_plot", analysis.plot_waveguide_layout),
        ("k_space_plot", analysis.plot_k_space),
        ("fov_plot", analysis.plot_fov_coverage),
    ):
        if key in parameters:
            fig, _ = plot(**parameters[key])
            plt.close(fig)
    return analysis



if __name__ == "__main__":
    parameters_geometric = {
        "waveguide_type": "geometric",
        "waveguide": {
            # Optical parameters
            "wavelength": 450.0, # nm
            "index_environment": 1.0,
            "index_substrate": 1.9710,
            "FOV": [10.0, 20.0], # full horizontal/vertical FOV, degrees
            "max_allowed_angle": 85.0,

            # Mirror configuration
            "alpha_im": 30.0, # degrees
            "input_surface_normal": [0, 0, -1],
            "output_surface_normal": [0, 0, 1],

            # Sampling
            "ray_position_samples": [11, 21],

            # Projector and input region
            "EPD_projector": 3.0,        # mm
            "gap_projector": 2.0,        # mm
            "IG_origin": [-10.0, 5.0],   # mm

            # Eyebox and output geometry
            "OG_origin": [6.0, -11.0],   # mm
            "eyebox_size": [12.0, 10.0], # mm
            "eye_relief": 20.0,          # mm
        },

        "layout_plot": {
            "filename": "geometric_layout.png",
            "dpi": 600,
            "mirror_counts": {
                "IG": 1,
                "EG": 8,
                "OG": 6,
            },
            "mirror_fill": 1.0,
        },

        "k_space_plot": {
            "filename": "geometric_k_space.png",
            "dpi": 600,
        },
    }

    main(parameters=parameters_geometric)
    print('')

    parameters_diffractive = {
        "waveguide_type": "diffractive",
        "waveguide": {
            "wavelength": 650.0, # nm
            "grating_pitch": 380.0, # nm

            "index_environment": 1.0,
            "index_substrate": 1.9107,
            "FOV": [20.0, 10.0],
            "max_allowed_angle": 85.0,

            "ray_position_samples": [21, 11],

            "EPD_projector": 3.0,
            "gap_projector": 2.0,
            "IG_origin": [-10.0, 5.0],

            "OG_origin": [6.0, -11.0],
            "eyebox_size": [12.0, 10.0],
            "eye_relief": 20.0,
        },

        "layout_plot": {
            "filename": "grating_layout.png",
            "dpi": 600,
        },

        "k_space_plot": {
            "filename": "grating_k_space.png",
            "dpi": 600,
        },
    }
    main(parameters=parameters_diffractive)
    print('')

    # Multi-wavelength (RGB) versions of the two designs. 
    substrate_dispersion = {450.0: 1.9710, 550.0: 1.9310, 650.0: 1.9107}
    rgb_wavelengths = [450.0, 550.0, 650.0] # nm

    parameters_geometric_rgb = {
        "waveguide_type": "geometric",
        "waveguide": {
            **parameters_geometric["waveguide"],
            "wavelength": rgb_wavelengths,
            "index_substrate": substrate_dispersion,
        },
        "layout_plot": {
            **parameters_geometric["layout_plot"],
            "filename": "geometric_layout_rgb.png",
        },
        "k_space_plot": {"filename": "geometric_k_space_rgb.png", "dpi": 600},
        "fov_plot": {"filename": "geometric_fov_rgb.png", "dpi": 600},
    }
    main(parameters=parameters_geometric_rgb)
    print('')

    parameters_diffractive_rgb = {
        "waveguide_type": "diffractive",
        "waveguide": {
            **parameters_diffractive["waveguide"],
            "wavelength": rgb_wavelengths,
            "index_substrate": substrate_dispersion,
        },
        "layout_plot": {"filename": "grating_layout_rgb.png", "dpi": 600},
        "k_space_plot": {"filename": "grating_k_space_rgb.png", "dpi": 600},
        "fov_plot": {"filename": "grating_fov_rgb.png", "dpi": 600},
    }
    main(parameters=parameters_diffractive_rgb)
