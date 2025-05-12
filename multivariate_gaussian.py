import numpy as np
from manim import *

class gaussian_1d(Scene):
    """
    A visual depiction of the Gaussian distribution in one dimension, 
    showing how it changes with its parameters mu and sigma.
    """
    def construct(self):
        def gaussian_pdf(x, mu, sigma):
            return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-mu)/sigma)**2)

        ax = Axes(
            x_range=[-5,5,1], 
            y_range=[0,1.1,0.5], 
            axis_config={"include_numbers": True}
            )
        labels = ax.get_axis_labels(x_label="x", y_label=MathTex("f_X(x)"))

        mu = ValueTracker(0)
        sigma = ValueTracker(1)

        f = always_redraw(
            lambda: ax.plot(
                lambda x: gaussian_pdf(x, mu.get_value(), sigma.get_value()),
                color=BLUE
            )
        )

        text_mu = MathTex("\mu = ", color=YELLOW_B)
        text_sigma = MathTex("\sigma = ", color=GOLD_B)

        vgroup_text = VGroup(text_mu, text_sigma).shift(RIGHT*4, UP*2)
        vgroup_text.arrange(DOWN, center=False, aligned_edge=LEFT, buff=0.5)

        mu_arr = [0, 2, -1]
        mu_values = [MathTex(val, color=YELLOW_B).next_to(text_mu, RIGHT).shift(0.08*UP) for val in mu_arr]
        sigma_arr = [1, 2, 0.5]
        sigma_values = [MathTex(val, color=GOLD_B).next_to(text_sigma, RIGHT).shift(0.05*UP) for val in sigma_arr]

        self.add(ax, labels, vgroup_text, mu_values[0], sigma_values[0])

        self.play(Create(f))
        self.wait()

        self.play(mu.animate.set_value(mu_arr[1]), ReplacementTransform(mu_values[0], mu_values[1]), 
                  run_time=1.5)
        self.wait(1.5)

        self.play(mu.animate.set_value(mu_arr[2]), ReplacementTransform(mu_values[1], mu_values[2]), 
                  run_time=1.5)
        self.wait(1.5)

        self.play(sigma.animate.set_value(sigma_arr[1]), ReplacementTransform(sigma_values[0], sigma_values[1]), 
                  run_time=1.5)
        self.wait(1.5)

        self.play(sigma.animate.set_value(sigma_arr[2]), ReplacementTransform(sigma_values[1], sigma_values[2]), 
                  run_time=1.5)
        self.wait(1.5)

        self.wait(2)


class mvn(ThreeDScene):
    """
    A visual depiction of the 2d Multivariate Normal,
    showing how it changes with the vector mu and covariance matrix Sigma.
    The corresponding pdf warrants a surface plot in 3D. 
    Link to GitHub repo that helped me figure out how to code this:
    https://github.com/t-ott/manim-normal-distributions/blob/master/bivariate.py
    """
    def construct(self):
        def mvn_pdf(x, mu, sigma):
            """
            Probability density function of the multivariate normal distribution.
            """
            d = len(x)
            prefactor = 1/np.sqrt(((2*np.pi)**d*np.abs(np.linalg.det(sigma))))
            exponent1 = np.dot((np.linalg.inv(sigma)), (x-mu))
            exponent2 = np.dot(np.transpose((x-mu)), exponent1)
            exponent = -0.5*exponent2

            return prefactor * np.exp(exponent)
        
        COLOR_RAMP = [
            rgb_to_color([57/255, 0.0, 153/255]),
            rgb_to_color([158/255, 0.0, 89/255]),
            rgb_to_color([1.0, 0.0, 84/255]),
            rgb_to_color([1.0, 84/255, 0.0]),
            rgb_to_color([1.0, 189/255, 0.0])
        ]

        ax = ThreeDAxes(
            x_range = [-4, 4, 1],
            y_range = [-4, 4, 1],
            z_range = [0, 0.2, 0.1],
            axis_config={"include_numbers": True}
            )
        
        x_label = ax.get_x_axis_label(r'x_1')
        y_label = ax.get_y_axis_label(r'x_2', edge=UP, buff=0.2)
        z_label = ax.get_z_axis_label(r'f_{X_1,X_2}(x_1,x_2)', buff=0.2)
        axis_labels = VGroup(x_label, y_label, z_label)

        mu_1 = ValueTracker(0)
        mu_2 = ValueTracker(0)
        sigma_11 = ValueTracker(1)
        sigma_12 = ValueTracker(0)
        sigma_21 = ValueTracker(0)
        sigma_22 = ValueTracker(1)

        # scaling size for brackets
        bracket_scale = 2.5

        text_mu = MathTex(r"\mu = ", color=YELLOW_B).shift(LEFT*6, UP*3)
        text_mu_left = Tex("(", color=YELLOW_B).next_to(text_mu, RIGHT).shift(0.1*RIGHT).scale(bracket_scale)
        text_mu_right = Tex(")", color=YELLOW_B).next_to(text_mu, RIGHT).shift(2*RIGHT).scale(bracket_scale)
        text_mu_1_val = always_redraw(
            lambda: DecimalNumber(num_decimal_places=2, include_sign=True, color=YELLOW_B)
            .set_value(mu_1.get_value())
            .next_to(text_mu, RIGHT).shift(0.5*RIGHT, 0.5*UP)
        )
        text_mu_2_val = always_redraw(
            lambda: DecimalNumber(num_decimal_places=2, include_sign=True, color=YELLOW_B)
            .set_value(mu_2.get_value())
            .next_to(text_mu, RIGHT).shift(0.5*RIGHT, 0.5*DOWN)
        )

        text_sigma = MathTex(r"\Sigma = ", color=GOLD_B).shift(RIGHT*3.5, UP*3)
        text_sigma_left = Tex("(", color=GOLD_B).next_to(text_sigma, RIGHT).scale(bracket_scale)
        text_sigma_right = Tex(")", color=GOLD_B).next_to(text_sigma, RIGHT).shift(2.7*RIGHT).scale(bracket_scale)
        text_sigma_11_val = always_redraw(
            lambda: DecimalNumber(num_decimal_places=2, color=GOLD_B)
            .set_value(sigma_11.get_value())
            .next_to(text_sigma, RIGHT).shift(0.5*RIGHT, 0.5*UP)
        )
        text_sigma_12_val = always_redraw(
            lambda: DecimalNumber(num_decimal_places=2, color=GOLD_B)
            .set_value(sigma_12.get_value())
            .next_to(text_sigma_11_val, RIGHT)
        )
        text_sigma_21_val = always_redraw(
            lambda: DecimalNumber(num_decimal_places=2, color=GOLD_B)
            .set_value(sigma_21.get_value())
            .next_to(text_sigma, RIGHT).shift(0.5*RIGHT, 0.5*DOWN)
        )
        text_sigma_22_val = always_redraw(
            lambda: DecimalNumber(num_decimal_places=2, color=GOLD_B)
            .set_value(sigma_22.get_value())
            .next_to(text_sigma_21_val, RIGHT)
        )
        vgroup_text_mu = VGroup(text_mu, text_mu_1_val, text_mu_2_val, text_mu_left, text_mu_right)
        vgroup_text_sigma = VGroup(text_sigma, text_sigma_left, text_sigma_right,
                                   text_sigma_11_val, text_sigma_12_val, 
                                   text_sigma_21_val, text_sigma_22_val)
        
        vgroup_text_all = VGroup(*vgroup_text_mu, *vgroup_text_sigma)

        res = 42 # try 42 for the final render
        surface = always_redraw(
            lambda: Surface(
                lambda u, v: ax.c2p(
                    u, v, mvn_pdf(
                        x = np.array([u, v]),
                        mu = np.array([mu_1.get_value(), mu_2.get_value()]),
                        sigma = np.array([[sigma_11.get_value(), sigma_12.get_value()],
                        [sigma_21.get_value(), sigma_22.get_value()]])
                        )
                    ),
                    resolution=(res, res),
                    u_range=[-3.5, 3.5],
                    v_range=[-3.5, 3.5],
                    fill_opacity=0.7
                ).set_fill_by_value(
                    axes = ax,
                    # Utilize color ramp colors, higher values are "warmer"
                    colors = [(COLOR_RAMP[0], 0),
                            (COLOR_RAMP[1], 0.05),
                            (COLOR_RAMP[2], 0.1),
                            (COLOR_RAMP[3], 0.15),
                            (COLOR_RAMP[4], 0.2)]
                )
            )

        # setup
        # fix all text on the screen, so that they don't move with the camera
        self.add_fixed_in_frame_mobjects(vgroup_text_all)

        self.add(ax, axis_labels, *vgroup_text_all)#, surface)
        self.set_camera_orientation(
            phi=75*DEGREES,
            theta=-70*DEGREES,
            frame_center=[0, 0, 2],
            zoom=0.75)
        
        self.play(Create(surface))
        self.wait()

        # changing the value of mu
        self.play(mu_1.animate.set_value(2), run_time=1.5)
        self.wait()

        self.play(mu_1.animate.set_value(0))
        self.play(mu_2.animate.set_value(2), run_time=1.5)
        self.wait()

        # set camera to a plan view
        self.move_camera(
            theta=-90*DEGREES,
            phi=0,
            frame_center=[0, 0, 0],
            zoom=0.5
        )
        self.play(mu_1.animate.set_value(-1), mu_2.animate.set_value(-2.5), run_time=1.5)
        self.wait()

        self.play(mu_1.animate.set_value(0), mu_2.animate.set_value(0))
        # Return camera to original position
        self.move_camera(
            theta=-70*DEGREES,
            phi=70*DEGREES,
            frame_center=[0, 0, 2],
            zoom=0.6
        )
        self.wait()

        # changing the value of sigma
        # (don't forget that the covariance matrix is symmetric!)
        self.play(sigma_11.animate.set_value(2), run_time=1.5)
        self.wait()

        self.move_camera(theta=70*DEGREES, run_time=2)
        self.move_camera(theta=-70*DEGREES, run_time=2)

        # reset value
        self.play(sigma_11.animate.set_value(1))

        self.play(sigma_22.animate.set_value(2), run_time=1.5)
        self.wait()

        self.move_camera(theta=70*DEGREES, run_time=2)
        self.move_camera(theta=-70*DEGREES, run_time=2)

        # reset value
        self.play(sigma_22.animate.set_value(1))

        # set camera to a plan view
        self.move_camera(
            theta=-90*DEGREES,
            phi=0,
            frame_center=[0, 0, 0],
            zoom=0.5
        )
        self.wait()

        self.play(sigma_12.animate.set_value(0.5), sigma_21.animate.set_value(0.5), run_time=1.5)
        self.wait()

        # Return camera to original position
        self.move_camera(
            theta=-70*DEGREES,
            phi=70*DEGREES,
            frame_center=[0, 0, 2],
            zoom=0.6
        )

        self.wait(2)