import random
import numpy as np
from manim import *

# set randomness for all classes
random.seed(8)
np.random.seed(8)

# num_points = 10
# dots_red_one = VGroup(*[Dot(point=np.array([np.random.uniform(-4,-2.1), np.random.uniform(1.1,3), 0]), color=RED_C) for _ in range(num_points)])
# dots_green = VGroup(*[Dot(point=np.array([np.random.uniform(-1, 1), np.random.uniform(-0.9, 0.9), 0]), color=GREEN_C) for _ in range(num_points)])
# dots_red_two = VGroup(*[Dot(point=np.array([np.random.uniform(2.1,4), np.random.uniform(-3,-0.1), 0]), color=RED_C) for _ in range(num_points)])

class linearly_inseparable_points(Scene):
    """
    jpeg of three clusters of points. Two groups of red points and one group of green points.
    """
    def construct(self):
        plane = NumberPlane(
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.2
            }
        )

        num_points = 10
        dots_red_one = VGroup(*[Dot(point=np.array([np.random.uniform(-4,-2.1), np.random.uniform(1.1,3), 0]), color=RED_C) for _ in range(num_points)])
        dots_green = VGroup(*[Dot(point=np.array([np.random.uniform(-1, 1), np.random.uniform(-0.9, 0.9), 0]), color=GREEN_C) for _ in range(num_points)])
        dots_red_two = VGroup(*[Dot(point=np.array([np.random.uniform(2.1,4), np.random.uniform(-3,-0.1), 0]), color=RED_C) for _ in range(num_points)])
        

        self.add(plane, dots_red_one, dots_green, dots_red_two)

class decision_tree(Scene):
    """
    Depiction of the decision boundaries made on the data points in the above class.
    """
    def construct(self):
        plane = NumberPlane(
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.2
            }
        )

        num_points = 10
        dots_red_one = VGroup(*[Dot(point=np.array([np.random.uniform(-4,-2.1), np.random.uniform(1.1,3), 0])) for _ in range(num_points)])
        dots_green = VGroup(*[Dot(point=np.array([np.random.uniform(-1, 1), np.random.uniform(-0.9, 0.9), 0])) for _ in range(num_points)])
        dots_red_two = VGroup(*[Dot(point=np.array([np.random.uniform(2.1,4), np.random.uniform(-3,-0.1), 0])) for _ in range(num_points)])

        dashed_1 = DashedLine(start=[-8,1,0], end=[8,1,0], color=YELLOW_C)
        region_1 = Polygon(
            [-8,1.01,0],
            [-8,4,0],
            [8,4,0],
            [8,1.01,0],
            color=None, fill_opacity=0.5, fill_color=RED
        )
        dashed_2 = DashedLine(start=[1.5,1,0], end=[1.5,-4,0], color=YELLOW_C)
        region_2 = Polygon(
            [-8,-4,0],
            [-8,0.99,0],
            [1.49,0.99,0],
            [1.49,-4,0],
            color=None, fill_opacity=0.5, fill_color=GREEN
        )
        region_3 = Polygon(
            [1.51,-4,0],
            [1.51,0.99,0],
            [8,0.99,0],
            [8,-4,0],
            color=None, fill_opacity=0.5, fill_color=RED
        )

        self.add(plane)
        self.play(LaggedStart(*[Create(dot) for dot in dots_green], lag_ratio=.05))
        self.play(LaggedStart(*[Create(dot) for dot in dots_red_one], lag_ratio=.05))
        self.play(LaggedStart(*[Create(dot) for dot in dots_red_two], lag_ratio=.05))
        self.wait()
        self.play(Create(dashed_1))
        self.wait(0.5)

        # shade upper region red
        self.play(FadeIn(region_1))
        self.wait(0.5)
        # colour in first set of red dots
        self.play(dots_red_one.animate.set_color(RED_C))
        self.wait(0.5)
        self.play(FadeOut(region_1))
        self.wait()

        self.play(Create(dashed_2))
        self.wait(0.5)
        # shade in bottom-left region green and colour in green dots
        self.play(FadeIn(region_2))
        self.wait(0.5)
        self.play(dots_green.animate.set_color(GREEN_C))
        self.wait(0.5)
        self.play(FadeOut(region_2))
        # shade in bottom-right region green and colour in red dots
        self.play(FadeIn(region_3))
        self.wait(0.5)
        self.play(dots_red_two.animate.set_color(RED_C))
        self.wait(0.5)
        self.play(FadeOut(region_3))
        self.wait()
        self.play(FadeOut(dashed_1), FadeOut(dashed_2))

        self.wait(2)


class decision_tree_diagram(Scene):
    """
    Correspondence between the previous animation and how it looks as a visual tree.
    """
    def construct(self):
        plane = NumberPlane(
            x_length=6,
            y_length=3.5,
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.2
            }
        ).move_to([-4,0,0])

        # plane_xlabel = MathTex(f"x_1").scale(0.6).move_to(plane.x_axis.get_right())
        # plane_ylabel = MathTex(f"x_2").scale(0.6).move_to(plane.y_axis.get_top())
        plane_xlabel = MathTex(f"x_1").scale(0.6).next_to(plane.x_axis, RIGHT)
        plane_ylabel = MathTex(f"x_2").scale(0.6).next_to(plane.y_axis, UP)

        num_points = 10
        dots_red_one = VGroup(*[Dot(point=plane.coords_to_point(np.random.uniform(-4,-2.1), np.random.uniform(1.1,3)), radius=0.06) for _ in range(num_points)])
        dots_green = VGroup(*[Dot(point=plane.coords_to_point(np.random.uniform(-1,1), np.random.uniform(-0.9,0.9)), radius=0.06) for _ in range(num_points)])
        dots_red_two = VGroup(*[Dot(point=plane.coords_to_point(np.random.uniform(2.1,4), np.random.uniform(-3,-0.1)), radius=0.06) for _ in range(num_points)])

        dashed_1 = DashedLine(start=plane.coords_to_point(-7.111,1), end=plane.coords_to_point(7.111,1), color=YELLOW_C)
        region_1 = Polygon(
            plane.coords_to_point(-7.111,1.01),
            plane.coords_to_point(-7.111,4),
            plane.coords_to_point(7.111,4),
            plane.coords_to_point(7.111,1.01),
            color=None, fill_opacity=0.5, fill_color=RED
        )
        dashed_2 = DashedLine(start=plane.coords_to_point(1.5,1), end=plane.coords_to_point(1.5,-4), color=YELLOW_C)
        region_2 = Polygon(
            plane.coords_to_point(-7.111,-4),
            plane.coords_to_point(-7.111,0.99),
            plane.coords_to_point(1.49,0.99),
            plane.coords_to_point(1.49,-4),
            color=None, fill_opacity=0.5, fill_color=GREEN
        )
        region_3 = Polygon(
            plane.coords_to_point(1.51,-4),
            plane.coords_to_point(1.51,0.99),
            plane.coords_to_point(7.111,0.99),
            plane.coords_to_point(7.111,-4),
            color=None, fill_opacity=0.5, fill_color=RED
        )

        # don't forget to put the text in each node
        root = Rectangle(width=2, height=1, color='YELLOW_C').move_to([3,2,0])
        root_label = MathTex(f"x_2 \geq 1?", color='YELLOW_C').scale(0.6).move_to(root.get_center())
        leaf1 = Rectangle(width=2, height=1, color='RED_C').move_to([1.5,0,0])
        leaf1_label = Text("Colour=RED", color='RED_C').scale(0.4).move_to(leaf1.get_center())
        arrow1 = Arrow(start=root.get_bottom(), end=leaf1.get_top())
        yes1 = Text("Yes").scale(0.4).next_to(arrow1, LEFT)
        leaf2 = Rectangle(width=2, height=1, color='YELLOW_C').move_to([4.5,0,0])
        leaf2_label = MathTex(f"x_1 \leq 1.5?", color='YELLOW_C').scale(0.6).move_to(leaf2.get_center())
        arrow2 = Arrow(start=root.get_bottom(), end=leaf2.get_top())
        no2 = Text("No").scale(0.4).next_to(arrow2, RIGHT)
        leaf3 = Rectangle(width=2, height=1, color='GREEN_C').move_to([3,-2,0])
        leaf3_label = Text("Colour=GREEN", color='GREEN_C').scale(0.4).move_to(leaf3.get_center())
        arrow3 = Arrow(start=leaf2.get_bottom(), end=leaf3.get_top())
        yes3 = Text("Yes").scale(0.4).next_to(arrow3, LEFT)
        leaf4 = Rectangle(width=2, height=1, color='RED_C').move_to([6,-2,0])
        leaf4_label = Text("Colour=RED", color='RED_C').scale(0.4).move_to(leaf4.get_center())
        arrow4 = Arrow(start=leaf2.get_bottom(), end=leaf4.get_top())
        no4 = Text("No").scale(0.4).next_to(arrow4, RIGHT)

        self.add(plane, plane_xlabel, plane_ylabel)
        self.play(LaggedStart(*[Create(dot) for dot in dots_green], lag_ratio=.05))
        self.play(LaggedStart(*[Create(dot) for dot in dots_red_one], lag_ratio=.05))
        self.play(LaggedStart(*[Create(dot) for dot in dots_red_two], lag_ratio=.05))
        self.wait()
        self.play(Create(dashed_1), Create(root), Write(root_label))
        self.wait()

        # shade upper region red
        self.play(FadeIn(region_1), GrowArrow(arrow1), Write(yes1), Create(leaf1), Write(leaf1_label))
        self.wait(0.5)
        # colour in first set of red dots
        self.play(dots_red_one.animate.set_color(RED_C))
        self.wait(0.5)
        self.play(FadeOut(region_1))
        self.wait()

        self.play(GrowArrow(arrow2), Write(no2))
        self.play(Create(dashed_2), Create(leaf2), Write(leaf2_label))
        self.wait(1.5)
        # shade in bottom-left region green and colour in green dots
        self.play(FadeIn(region_2), GrowArrow(arrow3), Write(yes3), Create(leaf3), Write(leaf3_label))
        self.wait(0.5)
        self.play(dots_green.animate.set_color(GREEN_C))
        self.wait(0.5)
        self.play(FadeOut(region_2))
        # shade in bottom-right region green and colour in red dots
        self.play(FadeIn(region_3), GrowArrow(arrow4), Write(no4), Create(leaf4), Write(leaf4_label))
        self.wait(0.5)
        self.play(dots_red_two.animate.set_color(RED_C))
        self.wait(0.5)
        self.play(FadeOut(region_3))


        self.wait(2)

class decision_tree_overfitting_example(Scene):
    """
    Start of second decision tree article. Jpeg of one red point included in the cluster of green points.
    """
    def construct(self):
        plane = NumberPlane(
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.2
            }
        )
        
        num_points = 10
        dots_red_one = VGroup(*[Dot(point=plane.coords_to_point(np.random.uniform(-4,-2.1), np.random.uniform(1.1,3)), color='RED_C') for _ in range(num_points)])
        dots_green = VGroup(*[Dot(point=plane.coords_to_point(np.random.uniform(-1,1), np.random.uniform(-0.9,0.9)), color='GREEN_C') for _ in range(num_points)])
        dots_red_two = VGroup(*[Dot(point=plane.coords_to_point(np.random.uniform(2.1,4), np.random.uniform(-3,-0.1)), color='RED_C') for _ in range(num_points)])
        outlier = Dot(plane.coords_to_point(0.4,0.8), color='RED_C')

        self.add(plane, dots_red_one, dots_green, dots_red_two, outlier)

class decision_tree_overfitting_example_animated(Scene):
    """
    Animation of an example decision tree constructed from the data in the above jpeg.
    """
    def construct(self):
        plane = NumberPlane(
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.2
            }
        )
        
        num_points = 10
        dots_red_one = VGroup(*[Dot(point=plane.coords_to_point(np.random.uniform(-4,-2.1), np.random.uniform(1.1,3)), color='RED_C') for _ in range(num_points)])
        dots_green = VGroup(*[Dot(point=plane.coords_to_point(np.random.uniform(-1,1), np.random.uniform(-0.9,0.9)), color='GREEN_C') for _ in range(num_points)])
        dots_red_two = VGroup(*[Dot(point=plane.coords_to_point(np.random.uniform(2.1,4), np.random.uniform(-3,-0.1)), color='RED_C') for _ in range(num_points)])
        outlier = Dot(plane.coords_to_point(0.4,0.8), color=RED_C)

        dashed_1 = DashedLine(start=plane.coords_to_point(-7.111,1), end=plane.coords_to_point(7.111,1), color=YELLOW_C)
        region_1 = Polygon(
            plane.coords_to_point(-7.111,1),
            plane.coords_to_point(-7.111,4),
            plane.coords_to_point(7.111,4),
            plane.coords_to_point(7.111,1),
            color=None, fill_opacity=0.5, fill_color=RED
        )
        dashed_2 = DashedLine(start=plane.coords_to_point(1.5,1), end=plane.coords_to_point(1.5,-4), color=YELLOW_C)
        region_2 = Polygon(
            plane.coords_to_point(1.5,-4),
            plane.coords_to_point(1.5,1),
            plane.coords_to_point(7.111,1),
            plane.coords_to_point(7.111,-4),
            color=None, fill_opacity=0.5, fill_color=RED
        )
        dashed_3 = DashedLine(start=plane.coords_to_point(0.2,1), end=plane.coords_to_point(0.2,-4), color=YELLOW_C)
        region_3 = Polygon(
            plane.coords_to_point(-7.111,-4),
            plane.coords_to_point(-7.111,1),
            plane.coords_to_point(0.2,1),
            plane.coords_to_point(0.2,-4),
            color=None, fill_opacity=0.5, fill_color=GREEN
        )
        dashed_4 = DashedLine(start=plane.coords_to_point(0.2,0.5), end=plane.coords_to_point(1.5,0.5), color=YELLOW_C)
        region_4 = Polygon(
            plane.coords_to_point(0.2,0.5),
            plane.coords_to_point(0.2,1),
            plane.coords_to_point(1.5,1),
            plane.coords_to_point(1.5,0.5),
            color=None, fill_opacity=0.5, fill_color=RED
        )
        region_5 = Polygon(
            plane.coords_to_point(0.2,-4),
            plane.coords_to_point(0.2,0.5),
            plane.coords_to_point(1.5,0.5),
            plane.coords_to_point(1.5,-4),
            color=None, fill_opacity=0.5, fill_color=GREEN
        )

        self.add(plane, dots_red_one, dots_green, dots_red_two, outlier)

        self.play(Create(dashed_1))
        self.wait(0.5)

        # shade upper region red
        self.play(FadeIn(region_1))
        self.wait()

        self.play(Create(dashed_2))
        self.wait(0.5)
        # shade in bottom-right region red
        self.play(FadeIn(region_2))
        self.wait()

        self.play(Create(dashed_3))
        self.wait(0.5)
        # shade in bottom-left region green
        self.play(FadeIn(region_3))
        self.wait()

        self.play(Create(dashed_4))
        self.wait()
        # shade in outlier section red
        self.play(FadeIn(region_4))
        self.wait()

        # shade in remaining green section
        self.play(FadeIn(region_5))
        self.wait()

        self.wait(2)


class decision_tree_overfitting_example2(Scene):
    """
    Jpeg for the second overfitting example.
    """
    def construct(self):
        plane = NumberPlane(
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.2
            }
        )
        
        num_points = 10
        dots_red_one = VGroup(*[Dot(point=plane.coords_to_point(np.random.uniform(-4,-2.1), np.random.uniform(1.1,3)), color='RED_C') for _ in range(num_points)])
        dots_green = VGroup(*[Dot(point=plane.coords_to_point(np.random.uniform(-1,1), np.random.uniform(-0.9,0.9)), color='GREEN_C') for _ in range(num_points)])
        dots_red_two = VGroup(*[Dot(point=plane.coords_to_point(np.random.uniform(2.1,4), np.random.uniform(-3,-0.1)), color='RED_C') for _ in range(num_points)])
        outlier = Dot(plane.coords_to_point(2.7,-2.2), color=GREEN_C)

        self.add(plane, dots_red_one, dots_green, dots_red_two, outlier)

class decision_tree_overfitting_example_animated2(Scene):
    """
    Second tree overfitting example.
    """
    def construct(self):
        plane = NumberPlane(
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.2
            }
        )
        
        num_points = 10
        dots_red_one = VGroup(*[Dot(point=plane.coords_to_point(np.random.uniform(-4,-2.1), np.random.uniform(1.1,3)), color='RED_C') for _ in range(num_points)])
        dots_green = VGroup(*[Dot(point=plane.coords_to_point(np.random.uniform(-1,1), np.random.uniform(-0.9,0.9)), color='GREEN_C') for _ in range(num_points)])
        dots_red_two = VGroup(*[Dot(point=plane.coords_to_point(np.random.uniform(2.1,4), np.random.uniform(-3,-0.1)), color='RED_C') for _ in range(num_points)])
        outlier = Dot(plane.coords_to_point(2.7,-2.2), color=GREEN_C)

        dashed_1 = DashedLine(start=plane.coords_to_point(-7.111,1), end=plane.coords_to_point(7.111,1), color=YELLOW_C)
        region_1 = Polygon(
            plane.coords_to_point(-7.111,1),
            plane.coords_to_point(-7.111,4),
            plane.coords_to_point(7.111,4),
            plane.coords_to_point(7.111,1),
            color=None, fill_opacity=0.5, fill_color=RED
        )
        dashed_2 = DashedLine(start=plane.coords_to_point(1.5,1), end=plane.coords_to_point(1.5,-4), color=YELLOW_C)
        region_2 = Polygon(
            plane.coords_to_point(-7.111,-4),
            plane.coords_to_point(-7.111,1),
            plane.coords_to_point(1.5,1),
            plane.coords_to_point(1.5,-4),
            color=None, fill_opacity=0.5, fill_color=GREEN
        )
        dashed_3 = DashedLine(start=plane.coords_to_point(1.5,-2), end=plane.coords_to_point(7.111,-2), color=YELLOW_C)
        region_3 = Polygon(
            plane.coords_to_point(1.5,-2),
            plane.coords_to_point(1.5,1),
            plane.coords_to_point(7.111,1),
            plane.coords_to_point(7.111,-2),
            color=None, fill_opacity=0.5, fill_color=RED
        )
        dashed_4 = DashedLine(start=plane.coords_to_point(1.5,-2.4), end=plane.coords_to_point(7.111,-2.4), color=YELLOW_C)
        region_4 = Polygon(
            plane.coords_to_point(1.5,-2.4),
            plane.coords_to_point(1.5,-2),
            plane.coords_to_point(7.111,-2),
            plane.coords_to_point(7.111,-2.4),
            color=None, fill_opacity=0.5, fill_color=GREEN
        )
        region_5 = Polygon(
            plane.coords_to_point(1.5,-4),
            plane.coords_to_point(1.5,-2.4),
            plane.coords_to_point(7.111,-2.4),
            plane.coords_to_point(7.111,-4),
            color=None, fill_opacity=0.5, fill_color=RED
        )

        self.add(plane, dots_red_one, dots_green, dots_red_two, outlier)

        self.play(Create(dashed_1))
        self.wait(0.5)

        # shade upper region red
        self.play(FadeIn(region_1))
        self.wait()

        self.play(Create(dashed_2))
        self.wait(0.5)
        # shade in bottom-right region red
        self.play(FadeIn(region_2))
        self.wait()

        self.play(Create(dashed_3))
        self.wait(0.5)
        # shade in bottom-left region green
        self.play(FadeIn(region_3))
        self.wait()

        self.play(Create(dashed_4))
        self.wait()
        # shade in outlier section red
        self.play(FadeIn(region_4))
        self.wait()

        # shade in remaining green section
        self.play(FadeIn(region_5))
        self.wait()

        self.wait(2)


class decision_tree_overfitting_examples_side_by_side(Scene):
    """
    Showcasing the two aforementioned decision tree decision boundaries side-by-side in the same vid.

    The suffix '_1' is used for all elements on plane1, '_2' used for those on plane2.
    """
    def construct(self):
        random.seed(8)
        np.random.seed(8)

        plane1 = NumberPlane(
            x_length=6,
            y_length=3.5,
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.2
            }
        ).move_to([-3.5,0,0])

        plane2 = NumberPlane(
            x_length=6,
            y_length=3.5,
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.2
            }
        ).move_to([3.5,0,0])
        
        num_points = 10
        red_points = [(np.random.uniform(-4,-2.1), np.random.uniform(1.1,3)) for _ in range(num_points)]
        green_points = [(np.random.uniform(-1,1), np.random.uniform(-0.9,0.9)) for _ in range(num_points)]
        red_points_again = [(np.random.uniform(2.1,4), np.random.uniform(-3,-0.1)) for _ in range(num_points)]

        dots_red_one_1 = VGroup(*[Dot(point=plane1.coords_to_point(red_points[i][0], red_points[i][1]), radius=0.05, color='RED_C') for i in range(num_points)])
        dots_green_1 = VGroup(*[Dot(point=plane1.coords_to_point(green_points[i][0], green_points[i][1]), radius=0.05, color='GREEN_C') for i in range(num_points)])
        dots_red_two_1 = VGroup(*[Dot(point=plane1.coords_to_point(red_points_again[i][0], red_points_again[i][1]), radius=0.05, color='RED_C') for i in range(num_points)])
        outlier_1 = Dot(plane1.coords_to_point(0.4,0.8), radius=0.05, color=RED_C)

        dots_red_one_2 = VGroup(*[Dot(point=plane2.coords_to_point(red_points[i][0], red_points[i][1]), radius=0.05, color='RED_C') for i in range(num_points)])
        dots_green_2 = VGroup(*[Dot(point=plane2.coords_to_point(green_points[i][0], green_points[i][1]), radius=0.05, color='GREEN_C') for i in range(num_points)])
        dots_red_two_2 = VGroup(*[Dot(point=plane2.coords_to_point(red_points_again[i][0], red_points_again[i][1]), radius=0.05, color='RED_C') for i in range(num_points)])
        outlier_2 = Dot(plane2.coords_to_point(2.7,-2.2), radius=0.05, color=GREEN_C)

        dashed_1_1 = DashedLine(start=plane1.coords_to_point(-7.111,1), end=plane1.coords_to_point(7.111,1), color=YELLOW_C)
        region_1_1 = Polygon(
            plane1.coords_to_point(-7.111,1),
            plane1.coords_to_point(-7.111,4),
            plane1.coords_to_point(7.111,4),
            plane1.coords_to_point(7.111,1),
            color=None, fill_opacity=0.5, fill_color=RED
        )
        dashed_2_1 = DashedLine(start=plane1.coords_to_point(1.5,1), end=plane1.coords_to_point(1.5,-4), color=YELLOW_C)
        region_2_1 = Polygon(
            plane1.coords_to_point(1.5,-4),
            plane1.coords_to_point(1.5,1),
            plane1.coords_to_point(7.111,1),
            plane1.coords_to_point(7.111,-4),
            color=None, fill_opacity=0.5, fill_color=RED
        )
        dashed_3_1 = DashedLine(start=plane1.coords_to_point(0.2,1), end=plane1.coords_to_point(0.2,-4), color=YELLOW_C)
        region_3_1 = Polygon(
            plane1.coords_to_point(-7.111,-4),
            plane1.coords_to_point(-7.111,1),
            plane1.coords_to_point(0.2,1),
            plane1.coords_to_point(0.2,-4),
            color=None, fill_opacity=0.5, fill_color=GREEN
        )
        dashed_4_1 = DashedLine(start=plane1.coords_to_point(0.2,0.5), end=plane1.coords_to_point(1.5,0.5), color=YELLOW_C)
        region_4_1 = Polygon(
            plane1.coords_to_point(0.2,0.5),
            plane1.coords_to_point(0.2,1),
            plane1.coords_to_point(1.5,1),
            plane1.coords_to_point(1.5,0.5),
            color=None, fill_opacity=0.5, fill_color=RED
        )
        region_5_1 = Polygon(
            plane1.coords_to_point(0.2,-4),
            plane1.coords_to_point(0.2,0.5),
            plane1.coords_to_point(1.5,0.5),
            plane1.coords_to_point(1.5,-4),
            color=None, fill_opacity=0.5, fill_color=GREEN
        )

        dashed_1_2 = DashedLine(start=plane2.coords_to_point(-7.111,1), end=plane2.coords_to_point(7.111,1), color=YELLOW_C)
        region_1_2 = Polygon(
            plane2.coords_to_point(-7.111,1),
            plane2.coords_to_point(-7.111,4),
            plane2.coords_to_point(7.111,4),
            plane2.coords_to_point(7.111,1),
            color=None, fill_opacity=0.5, fill_color=RED
        )
        dashed_2_2 = DashedLine(start=plane2.coords_to_point(1.5,1), end=plane2.coords_to_point(1.5,-4), color=YELLOW_C)
        region_2_2 = Polygon(
            plane2.coords_to_point(-7.111,-4),
            plane2.coords_to_point(-7.111,1),
            plane2.coords_to_point(1.5,1),
            plane2.coords_to_point(1.5,-4),
            color=None, fill_opacity=0.5, fill_color=GREEN
        )
        dashed_3_2 = DashedLine(start=plane2.coords_to_point(1.5,-2), end=plane2.coords_to_point(7.111,-2), color=YELLOW_C)
        region_3_2 = Polygon(
            plane2.coords_to_point(1.5,-2),
            plane2.coords_to_point(1.5,1),
            plane2.coords_to_point(7.111,1),
            plane2.coords_to_point(7.111,-2),
            color=None, fill_opacity=0.5, fill_color=RED
        )
        dashed_4_2 = DashedLine(start=plane2.coords_to_point(1.5,-2.4), end=plane2.coords_to_point(7.111,-2.4), color=YELLOW_C)
        region_4_2 = Polygon(
            plane2.coords_to_point(1.5,-2.4),
            plane2.coords_to_point(1.5,-2),
            plane2.coords_to_point(7.111,-2),
            plane2.coords_to_point(7.111,-2.4),
            color=None, fill_opacity=0.5, fill_color=GREEN
        )
        region_5_2 = Polygon(
            plane2.coords_to_point(1.5,-4),
            plane2.coords_to_point(1.5,-2.4),
            plane2.coords_to_point(7.111,-2.4),
            plane2.coords_to_point(7.111,-4),
            color=None, fill_opacity=0.5, fill_color=RED
        )




        self.add(plane1, dots_red_one_1, dots_green_1, dots_red_two_1, 
                    plane2, dots_red_one_2, dots_green_2, dots_red_two_2)

        # explicitly animate the outliers
        self.wait()
        self.play(Write(outlier_1), Write(outlier_2))
        self.play(Flash(outlier_1), Flash(outlier_2))
        self.wait()

        self.play(Create(dashed_1_1), Create(dashed_1_2))
        self.wait(0.5)

        # shade upper region red
        self.play(FadeIn(region_1_1), FadeIn(region_1_2))
        self.wait()

        self.play(Create(dashed_2_1), Create(dashed_2_2))
        self.wait(0.5)
        # shade in bottom-right region red
        self.play(FadeIn(region_2_1), FadeIn(region_2_2))
        self.wait()

        self.play(Create(dashed_3_1), Create(dashed_3_2))
        self.wait(0.5)
        # shade in bottom-left region green
        self.play(FadeIn(region_3_1), FadeIn(region_3_2))
        self.wait()

        self.play(Create(dashed_4_1), Create(dashed_4_2))
        self.wait()
        # shade in outlier section red
        self.play(FadeIn(region_4_1), FadeIn(region_4_2))
        self.wait()

        # shade in remaining green section
        self.play(FadeIn(region_5_1), FadeIn(region_5_2))

        self.wait(2)