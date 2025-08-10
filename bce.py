import numpy as np
from manim import *

class bce_surface(ThreeDScene):
    """
    """
    def construct(self):
        ax = ThreeDAxes(
            x_range = [0, 1.1, 0.5],
            y_range = [0, 1.1, 0.5],
            z_range = [0, 5.5, 1],
            axis_config={"include_numbers": True}
            )
        
        x_label = ax.get_x_axis_label(r'y')
        y_label = ax.get_y_axis_label(r'\hat y')
        z_label = ax.get_z_axis_label(r'BCE(y,\hat y)')
        axis_labels = VGroup(x_label, y_label, z_label)
        
        COLOR_RAMP = [
            rgb_to_color([57/255, 0.0, 153/255]),
            rgb_to_color([158/255, 0.0, 89/255]),
            rgb_to_color([1.0, 0.0, 84/255]),
            rgb_to_color([1.0, 84/255, 0.0]),
            rgb_to_color([1.0, 189/255, 0.0])
        ]
        
        bce_surface = Surface(
            lambda u, v: ax.c2p(
                u, v, -(u*np.log(v) + (1-u)*np.log(1-v))
            ),
            u_range=[0.01, 0.99],
            v_range=[0.01, 0.99],
            resolution=42,
            fill_color=BLUE,
            fill_opacity=0.5
        ).set_fill_by_value(
            axes = ax,
            # Utilize color ramp colors, higher values are "warmer"
            colors = [(COLOR_RAMP[0], 0),
                    (COLOR_RAMP[1], 1),
                    (COLOR_RAMP[2], 2),
                    (COLOR_RAMP[3], 3),
                    (COLOR_RAMP[4], 4)]
        )

        self.set_camera_orientation(
            phi=70*DEGREES,
            theta=-70*DEGREES,
            frame_center=[0, 0, 2],
            zoom=0.5
        )
        
        self.add(ax, axis_labels)#, bce_surface)

        self.play(Create(bce_surface), run_time=2)
        self.wait()
        
        self.move_camera(
            theta=110*DEGREES,
            run_time=2
        )
        self.wait()

        self.move_camera(
            theta=-70*DEGREES,
            run_time=2
        )
        self.wait()