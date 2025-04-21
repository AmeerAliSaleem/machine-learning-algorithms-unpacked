import random
import numpy as np
import pandas as pd
from manim import *

class hyperplane_visual(Scene):
    def construct(self):

        number_plane = NumberPlane(
            x_range=(-4,4),
            y_range=(-4,4),
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.4
            }
        )

        straight_line = Line(start=[-8,-8,1], end=[8,8,1], color=PURPLE_A)

        line_formula = MathTex("y=x", color=PURPLE_A)
        line_formula.rotate(PI/4)
        line_formula.move_to((1.5,2,0))

        vector = Vector([1,-1]).set_color(YELLOW_C)
        vector_label = vector.coordinate_label(color=YELLOW)

        right_angle = VGroup(
            Line(start=(0.25,0.25,0), end=(0.5,0,0), color=ORANGE),
            Line(start=(0.5,0,0), end=(0.25,-0.25,0), color=ORANGE)
        )
        self.add(number_plane, straight_line, line_formula, vector, vector_label, right_angle)


class svm_two_d(Scene):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.highlights = []  # Initialize highlights as an empty list

    def show_highlight(self, dots, color=YELLOW, scale_factor=1.5):
        """Creates and displays highlight circles around dots."""
        self.highlights = [Circle(radius=dot.radius * scale_factor, color=color).move_to(dot) for dot in dots]
        self.play(*[Create(highlight) for highlight in self.highlights])

    def remove_highlight(self):
        """Fades out all highlight circles."""
        if self.highlights:  # Ensure highlights exist before trying to remove them
            self.play(*[FadeOut(highlight) for highlight in self.highlights])
            self.highlights = []  # Clear the list after fading out
    
    def construct(self):
        number_plane = NumberPlane(
            # x_range=(-4,4),
            # y_range=(-4,4),
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.4
            }
        )

        dots_green_coords = [[4,-2,0], [4,3,0], [3.5,1,0], [5,-0.5,0], [1,3.5,0], [1.5,2.5,0], [5.5,2,0]]
        dots_green = [Dot(dot) for dot in dots_green_coords]
        dots_red_coords = [[-3,1,0], [0,-3,0], [-4,-2,0], [-3,-2.5,0], [-6,-3.5,0], [-5,0.5,0], [-6,-3,0], [-6.5,2,0]]
        dots_red = [Dot(dot) for dot in dots_red_coords]
        
        arrow_green = DoubleArrow(start=dots_green[0].get_center(), end=[3,-3,0], color=GREEN_C, buff=0.2)
        arrow_red = DoubleArrow(start=[-2,2,0], end=dots_red[0].get_center(), color=RED_C, buff=0.2)

        arrow_green_text = MathTex(r"\frac{1}{\|w\|}", color=GREEN_C, 
                                   font_size=35).next_to(arrow_green, RIGHT).shift(0.5*DOWN, 0.5*LEFT)
        arrow_red_text = MathTex(r"\frac{1}{\|w\|}", color=RED_C, 
                                 font_size=35).next_to(arrow_red, LEFT).shift(0.5*UP, 0.5*RIGHT)

        line = Line(start=[-8,8,1], end=[8,-8,1], color=PURPLE_A)
        line_formula = MathTex(f"w \cdot x + b = 0", color=PURPLE_A)
        line_formula.rotate(-PI/4)
        line_formula.move_to((-1.5,2,0))

        line_above = DashedLine(start=[-8,10,1], end=[8,-6,1], color=GREEN_C)
        line_above_formula = MathTex(f"w \cdot x + b = 1", color=GREEN_C)
        line_above_formula.rotate(-PI/4)
        line_above_formula.move_to((1.25,1.25,0))

        line_below = DashedLine(start=[-8,6,1], end=[8,-10,1], color=RED_C)
        line_below_formula = MathTex(f"w \cdot x + b = -1", color=RED_C)
        line_below_formula.rotate(-PI/4)
        line_below_formula.move_to((-1.25,-1.25,0))

        self.add(number_plane)

        # add dots to the screen
        self.play(LaggedStart(*[FadeIn(dot) for dot in dots_green], lag_ratio=.05))
        self.play(LaggedStart(*[FadeIn(dot) for dot in dots_red], lag_ratio=.05))
        self.wait()
        self.play(Create(line_above), Create(line_below))
        self.wait()
        self.play(FadeIn(line_above_formula), FadeIn(line_below_formula))
        self.wait(2)
        self.play(Create(line))
        self.wait()
        self.play(FadeIn(line_formula))
        self.wait()
        self.play(GrowArrow(arrow_green), GrowArrow(arrow_red))
        self.wait()
        self.play(FadeIn(arrow_green_text), FadeIn(arrow_red_text))
        self.wait()

        # colour changes for dots
        animations_green = []
        for i in range(1,len(dots_green)):
            animations_green.append(dots_green[i].animate.set_color(GREEN_C))

        animations_red = []
        for i in range(1,len(dots_red)):
            animations_red.append(dots_red[i].animate.set_color(RED_C))
        
        self.wait()
        circle = Circle(radius=0.1, color=YELLOW).move_to(dots_green[0])
        self.play(GrowFromCenter(circle))
        self.wait()
        self.play(dots_green[0].animate.set_color(PURE_GREEN))
        self.wait(0.5)
        self.play(FadeOut(circle))
        self.wait(0.5)
        self.show_highlight(dots_green[1:])
        self.play(AnimationGroup(*animations_green, lag_ratio=0))
        self.wait()
        self.remove_highlight()
        self.wait()
        circle.move_to(dots_red[0])
        self.play(GrowFromCenter(circle))
        self.wait()
        self.play(dots_red[0].animate.set_color(PURE_RED))
        self.wait(0.5)
        self.play(FadeOut(circle))
        self.wait(0.5)

        self.show_highlight(dots_red[1:])
        self.wait()
        self.play(AnimationGroup(*animations_red, lag_ratio=0))
        self.remove_highlight()

        self.wait(2)


class trig(Scene):
    def construct(self):

        plane = NumberPlane(
            x_range=(-2,5),
            y_range=(-2,4),
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.4
            }
        )

        vector_text = MathTex(r"\underline{v}", color=WHITE).move_to(plane.coords_to_point(1.5,2))
        vector_text1 = MathTex(r"\underline{v} \cos \theta", color=YELLOW).move_to(plane.coords_to_point(1.5,-0.5))
        vector_text2 = MathTex(r"\underline{v} \sin \theta", color=YELLOW).move_to(plane.coords_to_point(4,1.5))
        theta_text = MathTex(r"\theta", color=ORANGE, font_size=40).move_to(plane.coords_to_point(0.6,0.25))

        vector = Arrow(start=plane.coords_to_point(0,0), end=plane.coords_to_point(3,3), 
                        buff=0).set_color(WHITE)
        vector_cos = Arrow(start=plane.coords_to_point(0,0), end=plane.coords_to_point(3,0), 
                        buff=0, color=YELLOW_C, tip_length=0.15)
        vector_sin = Arrow(start=plane.coords_to_point(3,0), end=plane.coords_to_point(3,3), 
                        buff=0, color=YELLOW_C, tip_length=0.15)
        
        arc = Arc(angle=PI/4, arc_center=plane.coords_to_point(0,0), color=ORANGE)

        right_angle = VGroup(
            Line(start=plane.coords_to_point(2.6,0), end=plane.coords_to_point(2.6,0.4), color=YELLOW_C),
            Line(start=plane.coords_to_point(2.6,0.4), end=plane.coords_to_point(3,0.4), color=YELLOW_C)
        )
        self.add(plane, vector_text, vector_text1, vector_text2, vector, 
                    vector_cos, vector_sin, arc, theta_text, right_angle)


class dot_product(Scene):
    def construct(self):
        plane = NumberPlane(
            x_range=(-5,9),
            y_range=(-3,5),
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.4
            }
        )
        plane.set_opacity(0.2)

        n_end_point = (4,2)

        v = Arrow(start=plane.coords_to_point(0,0), end=plane.coords_to_point(2,4), color=BLUE_C, buff=0)
        n = Arrow(start=plane.coords_to_point(0,0), 
                    end=plane.coords_to_point(n_end_point[0], n_end_point[1]), buff=0, color=YELLOW_C)
        
        v_projection = DashedLine(start=plane.coords_to_point(2,4), 
                    end=plane.coords_to_point(3.2,1.6), color=PURPLE)

        v_text = MathTex(r"\underline{v}", color=BLUE_C).move_to(plane.coords_to_point(0.75,2.5))
        n_text = MathTex(r"\underline{n}", color=YELLOW_C).move_to(plane.coords_to_point(2.15,0.6))
        n_hat_text = MathTex(r"\hat{\underline{n}}", color=GOLD_C).move_to(plane.coords_to_point(0.5,-0.2))
        
        n_norm = np.sqrt(n_end_point[0]**2 + n_end_point[1]**2)
        # print(n_end_point)
        # print(n_norm)
        # print(n_end_point[0]/n_norm)
        # print(n_end_point[1]/n_norm)
        n_hat = Arrow(start=plane.coords_to_point(0,0), 
                      end=plane.coords_to_point(n_end_point[0]/n_norm, n_end_point[1]/n_norm), 
                      buff=0, color=GOLD_C)
        
        d_arrow = DoubleArrow(start=plane.coords_to_point(0,0), end=plane.coords_to_point(3.2,1.6),
                                buff=0, color=GREEN_C)

        dot_product_circle = Dot(point=plane.coords_to_point(2.8,-1.7), radius=0.06)

        self.add(plane)
        self.play(GrowArrow(v), GrowArrow(n))
        self.wait(0.5)
        self.play(FadeIn(v_text), FadeIn(n_text))
        self.wait()
        self.play(FadeIn(v_projection))
        self.wait()
        self.play(n.animate.set_opacity(0.5), n_text.animate.set_opacity(0.5))
        self.wait()
        self.play(GrowArrow(n_hat), FadeIn(n_hat_text))
        self.wait(2)
        self.play(GrowArrow(d_arrow))
        self.wait()
        self.play(d_arrow.animate.shift(2*DOWN, 1*RIGHT))
        self.wait()
        self.play(v_text.animate.move_to(plane.coords_to_point(2.5,-1.75)))
        self.wait()
        self.play(FadeIn(dot_product_circle), n_hat_text.animate.move_to(plane.coords_to_point(3.1,-1.7)))

        self.wait(3)


class margin_derivation(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.highlights = []  # Initialize highlights as an empty list

    def show_highlight(self, dots, color=YELLOW, scale_factor=1.5):
        """Creates and displays highlight circles around dots."""
        self.highlights = [Circle(radius=dot.radius * scale_factor, color=color).move_to(dot) for dot in dots]
        self.play(*[Create(highlight) for highlight in self.highlights])

    def remove_highlight(self):
        """Fades out all highlight circles."""
        if self.highlights:  # Ensure highlights exist before trying to remove them
            self.play(*[FadeOut(highlight) for highlight in self.highlights])
            self.highlights = []  # Clear the list after fading out


    def construct(self):
        plane = NumberPlane(
            x_range=(-4,10),
            y_range=(-2,6),
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.4
            }
        )
        plane.set_opacity(0.2)

        lower_line = DashedLine(start=plane.coords_to_point(-3,6), end=plane.coords_to_point(5,-2), color=RED_C)
        upper_line = DashedLine(start=plane.coords_to_point(-2,9), end=plane.coords_to_point(9,-2), color=GREEN_C)

        lower_support = Dot(point=plane.coords_to_point(1,2), color=RED_A)
        upper_support = Dot(point=plane.coords_to_point(4,3), color=GREEN_A)

        lower_support_vector = Arrow(start=plane.coords_to_point(0,0), end=plane.coords_to_point(1,2), 
                                     buff=0.1, color=GOLD_B)
        lower_support_vector_text = MathTex(r"\underline{x}_+", color=GOLD_B).move_to(plane.coords_to_point(0,1.5))
        upper_support_vector = Arrow(start=plane.coords_to_point(0,0), end=plane.coords_to_point(4,3), 
                                     buff=0.1, color=TEAL_E)
        upper_support_vector_text = MathTex(r"\underline{x}_-", color=TEAL_E).move_to(plane.coords_to_point(2.5,1.25))

        lower_to_upper1 = Arrow(start=plane.coords_to_point(1,2), end=plane.coords_to_point(5,5), 
                               buff=0.1, color=TEAL_E)
        lower_to_upper2 = Arrow(start=plane.coords_to_point(5,5), end=plane.coords_to_point(4,3), 
                               buff=0.1, color=GOLD_B)
        lower_to_upper = Arrow(start=plane.coords_to_point(1,2), end=plane.coords_to_point(4,3), 
                               buff=0.1, color=MAROON_B)
        lower_to_upper_text = MathTex(r"(\underline{x}_+) - (\underline{x}_-)", 
                                      color=MAROON_B).move_to(plane.coords_to_point(2.2,2.8)).rotate(lower_to_upper.get_angle())
        
        v_text = MathTex(r"\underline{v}", color=MAROON_B).move_to(plane.coords_to_point(2.5,2.1))
        w_text = MathTex(r"\underline{w}", color=PURPLE_C).move_to(plane.coords_to_point(2,3.6))
        w_hat_text = MathTex(r"\hat{\underline{w}}", color=PURPLE_C).move_to(plane.coords_to_point(1.1,2.7))
        w_vector = Arrow(start=plane.coords_to_point(1,2), end=plane.coords_to_point(4,5), 
                               buff=0.1, color=PURPLE_C)
        w_hat_vector = Arrow(start=plane.coords_to_point(1,2), 
                             end=plane.coords_to_point(1+np.sqrt(1/2),2+np.sqrt(1/2)), buff=0.1, color=PURPLE_C)
        
        margin_distance = DoubleArrow(start=plane.coords_to_point(1,2), end=plane.coords_to_point(3,4),
                                buff=0, color=GREEN_C)
        
        dot_product_circle = Dot(point=plane.coords_to_point(7.55,3.55), radius=0.06)

        self.add(plane)
        self.play(Create(lower_line), Create(upper_line))
        self.play(FadeIn(lower_support), FadeIn(upper_support))
        self.wait()
        self.play(GrowArrow(lower_support_vector), FadeIn(lower_support_vector_text))
        self.play(GrowArrow(upper_support_vector), FadeIn(upper_support_vector_text))
        self.wait()
        self.show_highlight([lower_support, upper_support])
        self.wait()
        self.play(GrowArrow(lower_to_upper1))
        self.play(GrowArrow(lower_to_upper2))
        self.wait(2)
        self.play(GrowArrow(lower_to_upper), FadeIn(lower_to_upper_text), 
                  lower_to_upper1.animate.set_opacity(0.4), lower_to_upper2.animate.set_opacity(0.4))
        self.wait()
        self.remove_highlight()
        self.play(lower_support_vector.animate.set_opacity(0.2), 
                  upper_support_vector.animate.set_opacity(0.2),
                  lower_support_vector_text.animate.set_opacity(0.2),
                  upper_support_vector_text.animate.set_opacity(0.2),
                  FadeOut(lower_to_upper_text))
        self.play(FadeIn(v_text))
        self.wait(2)
        # part 2 starts here
        self.play(GrowArrow(w_vector), FadeIn(w_text))
        self.wait(2)
        self.play(w_vector.animate.set_opacity(0.2), w_text.animate.set_opacity(0.2))
        self.wait()
        self.play(GrowArrow(w_hat_vector), FadeIn(w_hat_text))
        self.wait(2)
        self.play(GrowArrow(margin_distance))
        self.wait(2)
        self.play(margin_distance.animate.shift(1*UP, 5*RIGHT))
        self.wait()
        self.play(v_text.animate.move_to(plane.coords_to_point(7.25,3.5)))
        self.wait()
        self.play(FadeIn(dot_product_circle), w_hat_text.animate.move_to(plane.coords_to_point(7.9,3.55)))

        self.wait(2)

class lagrange(ThreeDScene):
    # troubleshooting for Arrow3D (source: https://github.com/ManimCommunity/manim/issues/3481)
    def custom_create_starting_mobject(self) -> Mobject:
        start_arrow = self.mobject.copy()
        if isinstance(self.point, np.ndarray) and self.point.shape[-1] == 3:
            start_arrow.scale(0, about_point=self.point)
        else:
            start_arrow.scale(0, scale_tips=True, about_point=self.point)

        if self.point_color:
            start_arrow.set_color(self.point_color)
        return start_arrow
    GrowArrow.create_starting_mobject = custom_create_starting_mobject
    
    def f(self, x, y):
        return np.array([x, y, x**2+y**2])

    def construct(self):
        # # Function to create gradient arrows
        # def get_gradient_arrow(u, v):
        #     x, y, z = v * np.cos(u), v * np.sin(u), v ** 2
        #     grad_x, grad_y = 2 * x, 2 * y  # Compute gradient
        #     arrow = Arrow3D(
        #         start=ax.c2p(x, y, z),
        #         end=ax.c2p(x + 0.3 * grad_x, y + 0.3 * grad_y, z),  # Scale the arrow size
        #         color=RED,
        #         thickness=2
        #     )
        #     return arrow


        # zoom out so we see the axes
        self.set_camera_orientation(zoom=0.5)
        # self.set_camera_orientation(phi=60*DEGREES, theta=-45*DEGREES)
        ax = ThreeDAxes()

        x_label = ax.get_x_axis_label(Tex("x"))
        y_label = ax.get_y_axis_label(Tex("y")).shift(UP * 1.8)

        

        # paraboloid f(x,y)=x^2+y^2
        surface1 = Surface(
            # lambda u, v: ax.c2p(*self.f(u, v)),
            lambda u, v: ax.c2p(v * np.cos(u), v * np.sin(u), v ** 2),
            # u_range=[-6,6],
            # v_range=[-5,5],
            u_range=[0, 2*PI],
            v_range=[0, 3],
            # resolution=8,
            fill_opacity=0.6
        )

        # plane z=x+1
        surface2 = Surface(
            lambda u, v: ax.c2p(u, v, u + 1),
            u_range=[-2, 2],
            v_range=[-2, 2],
            fill_opacity=0.5,
            # resolution=8,
            checkerboard_colors=[GREEN_C, GREEN_E]
        )

        # Function to create level sets (circles at different heights)
        def get_level_set(c, colour):
            """ Returns a Parametric Function representing the level set x^2 + y^2 = c """
            return ParametricFunction(
                lambda t: ax.c2p(np.sqrt(c) * np.cos(t), np.sqrt(c) * np.sin(t), c),
                t_range=[0, 2 * PI],
                color=colour
            )

        # Level sets to display (including the intersection level)
        level_values = [1, 2, 3, 4]  # Choose different z-values for contours
        level_sets = VGroup(*[get_level_set(c, colour=YELLOW) for c in level_values])


        # Intersection circle (where plane meets paraboloid)
        # this is a slanted circle with radius sqrt{5}/2 centred at (0.5,0)
        intersection_circle = ParametricFunction(
            lambda t: ax.c2p((1/2) + (np.sqrt(5)/2) * np.cos(t), (np.sqrt(5)/2) * np.sin(t), (1/2) + (np.sqrt(5)/2) * np.cos(t) + 1),
            t_range=[0, 2 * PI],
            color=ORANGE,
            stroke_width=4
        )

        # solution is at (0.5-sqrt(5)/2, 0, 1.5-sqrt(5)/2)
        intersection_level_set = get_level_set(1.5 - np.sqrt(5)/2, RED)
        intersection_point = Dot3D(ax.c2p(0.5-np.sqrt(5)/2, 0, 1.5-np.sqrt(5)/2), color=ORANGE)

        # gradient arrows
        surface1_gradient_arrow = Arrow3D(start=ax.c2p(0.5-np.sqrt(5)/2, 0, 1.5-np.sqrt(5)/2), 
                                          end=ax.c2p(-0.5-np.sqrt(5)/2, 0, 2.5-np.sqrt(5)/2), color=PURE_BLUE)
        surface2_gradient_arrow = Arrow3D(start=ax.c2p(0.5-np.sqrt(5)/2, 0, 1.5-np.sqrt(5)/2), 
                                          end=ax.c2p(1.5-np.sqrt(5)/2, 0, 0.5-np.sqrt(5)/2), color=PURE_GREEN)

        # intersection level set occurs at

        # # Generate gradient vectors at different points
        # gradient_arrows = VGroup(
        #     *[get_gradient_arrow(u, v) for u in np.linspace(0, 2 * PI, 12)
        #       for v in np.linspace(0.5, 3, 6)]
        # )


        self.play(FadeIn(ax), FadeIn(x_label), FadeIn(y_label))
        self.wait(0.5)
        # animate the move of the camera to properly see the axes
        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES, zoom=1, run_time=1.5)
        self.wait()
        self.play(Create(surface1))
        self.wait()
        self.begin_ambient_camera_rotation(rate=PI/12)
        # self.play(LaggedStart(*[GrowArrow(arrow) for arrow in gradient_arrows], lag_ratio=0.05))
        self.wait(1)
        self.play(Create(surface2))
        self.wait(3)
        self.stop_ambient_camera_rotation()
        self.wait(2)
        self.play(LaggedStart(*[Create(ls) for ls in level_sets], lag_ratio=0.2))
        self.wait(2)
        # self.play(Create(intersection_circle))
        self.play(Create(intersection_level_set))
        self.wait()
        self.play(Create(intersection_point))
        self.wait(2)
        # self.play(intersection_level_set.animate.set_opacity(0.5), level_sets.animate.set_opacity(0.5))
        # self.wait()
        self.play(GrowArrow(surface1_gradient_arrow))
        self.wait()
        self.play(GrowArrow(surface2_gradient_arrow))

        self.wait()
        self.begin_ambient_camera_rotation(rate=PI/12)
        self.wait(8)
        # self.stop_ambient_camera_rotation()
        
        self.wait(2)


class kernel(ThreeDScene):
    def feature_map(self, x1, x2):
        return np.array([x1**2, x2**2, np.sqrt(2) * x1 * x2])
    
    def construct(self):
        # zoom out so we see the axes
        self.set_camera_orientation(zoom=0.6)

        # self.set_camera_orientation(phi=60*DEGREES, theta=-45*DEGREES)
        ax = ThreeDAxes()

        x_label = ax.get_x_axis_label(MathTex("x_1"))
        y_label = ax.get_y_axis_label(MathTex("x_2")).shift(UP * 1.8)

        x_label_new = ax.get_x_axis_label(MathTex("x_1^2"))
        y_label_new = ax.get_y_axis_label(MathTex("x_2^2")).shift(UP * 1.8)

        # Generate random points (set to 40 in the final video)
        num_points = 40
        points = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(num_points)]
        
        random.seed(8)
        # Assign colors based on a simple rule (e.g., x1 + x2 > 0 is green, otherwise red)
        colors = [GREEN if x1**2 + x2**2 > 2 else RED for x1, x2 in points]

        dots = [Dot3D(point=ax.c2p(x1, x2, 0), color=color) for (x1, x2), color in zip(points, colors)]

        projection_text = MathTex(r"\varphi (x_1,x_2) = (x_1^2, x_2^2,\sqrt{2} x_1 x_2)", color=YELLOW).scale(0.8).to_corner(UR)

        projection_surface = Surface(
            lambda u, v: ax.c2p(u**2, v**2, np.sqrt(2)*u*v),
            u_range=[-2, 2],
            v_range=[-2, 2],
            fill_opacity=0.2,
            # resolution=8,
            checkerboard_colors=[BLUE_C, BLUE_E]
        )

        # Add separating plane
        plane = Polygon(
            ax.c2p(2, 0, -4),
            ax.c2p(0, 2, -4),
            ax.c2p(0, 2, 4),
            ax.c2p(2, 0, 4),
            color=ORANGE, fill_opacity=0.5
        )


        # on-screen
        self.play(FadeIn(ax), FadeIn(x_label), FadeIn(y_label))
        self.play(LaggedStart(*[FadeIn(dot) for dot in dots], lag_ratio=.05))
        self.wait()
        self.move_camera(phi=75 * DEGREES, theta=-60 * DEGREES, zoom=0.8, run_time=1.5)
        self.wait()
        # self.play(FadeIn(projection_text))
        self.add_fixed_in_frame_mobjects(projection_text)
        self.wait(0.5)
        self.play(FadeOut(x_label), FadeOut(y_label), FadeIn(x_label_new), FadeIn(y_label_new))
        self.wait(0.5)

        self.play(Create(projection_surface))
        self.wait()

        # move dots to their new positions under feature_map
        self.play(
            *[dot.animate.move_to(ax.c2p(*self.feature_map(x1, x2))) for dot, (x1, x2) in zip(dots, points)],
            run_time=3
        )
        self.wait(0.5)
        self.play(FadeIn(plane))

        self.begin_ambient_camera_rotation(rate=PI/8)
        self.wait(10)

        self.wait(2)


class svm(ThreeDScene):
    """ This class was not used in any articles. It was used only for experimenting with ThreeDScene."""
    def construct(self):
        self.set_camera_orientation(phi=60*DEGREES, theta=-45*DEGREES)
        ax = ThreeDAxes()

        j = 5
        vertex_coords = [[j,0,j],[0,j,-j],[-j,0,-j],[0,-j,j]]
        faces_list = [[0,1,2,3]]
        hyper1 = Polyhedron(vertex_coords, faces_list)

        vertex_coords = [[j,0,j+2],[0,j,-j+2],[-j,0,-j+2],[0,-j,j+2]]
        hyper2 = Polyhedron(vertex_coords, faces_list).set_color(GREEN)
        
        vertex_coords = [[j,0,j-2],[0,j,-j-2],[-j,0,-j-2],[0,-j,j-2]]
        hyper3 = Polyhedron(vertex_coords, faces_list).set_color(RED)

        self.add(ax)

        self.wait(2)
        self.play(FadeIn(hyper1), runtime=2)
        self.wait()
        self.begin_ambient_camera_rotation(rate=PI/12)
        self.wait(3)
        self.stop_ambient_camera_rotation()
        self.wait()
        self.play(FadeIn(hyper2), FadeIn(hyper3))
        self.wait()

        self.wait(2)