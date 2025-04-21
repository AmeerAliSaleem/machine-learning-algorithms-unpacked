import numpy as np
from manim import *

class man_vs_euc(Scene):
    def construct(self):
        dot1 = Dot([-3,-3,0], color=WHITE)
        dot2 = Dot([3,3,0], color=WHITE)
        euc_arrow = Arrow(dot1.get_center(), dot2.get_center(), color=ORANGE, buff=0)

        self.add(NumberPlane())
        self.play(FadeIn(dot1, dot2))
        self.wait()
        self.play(GrowArrow(euc_arrow))
        self.wait()

        path = VMobject(color=YELLOW_C)
        dot = Dot([-3,-3,0], color=YELLOW_C)
        path.set_points_as_corners([dot.get_center(), dot.get_center()])
        def update_path(path):
            previous_path = path.copy()
            previous_path.add_points_as_corners([dot.get_center()])
            path.become(previous_path)
        path.add_updater(update_path)

        # route 1
        self.add(path, dot)
        self.play(dot.animate.shift(RIGHT), run_time=0.5)
        self.play(dot.animate.shift(UP), run_time=0.2)
        self.play(dot.animate.shift(RIGHT), run_time=0.2)
        self.play(dot.animate.shift(UP), run_time=0.2)
        self.play(dot.animate.shift(RIGHT), run_time=0.2)
        self.play(dot.animate.shift(UP), run_time=0.2)
        self.play(dot.animate.shift(RIGHT), run_time=0.2)
        self.play(dot.animate.shift(UP), run_time=0.2)
        self.play(dot.animate.shift(RIGHT), run_time=0.2)
        self.play(dot.animate.shift(UP), run_time=0.2)
        self.play(dot.animate.shift(RIGHT), run_time=0.2)
        self.play(dot.animate.shift(UP), run_time=0.5)
        # self.play(Rotating(dot, radians=PI, about_point=RIGHT, run_time=2))
        self.wait()
        self.play(FadeOut(dot), FadeOut(path))
        self.wait()

        # route 2
        path = VMobject(color=YELLOW_C)
        dot = Dot([-3,-3,0], color=YELLOW_C)
        path.set_points_as_corners([dot.get_center(), dot.get_center()])
        path.add_updater(update_path)

        self.add(path, dot)
        self.play(dot.animate.shift(2*RIGHT), run_time=0.5)
        self.play(dot.animate.shift(UP), run_time=0.5)
        self.play(dot.animate.shift(2*RIGHT), run_time=0.5)
        self.play(dot.animate.shift(3*UP), run_time=0.5)
        self.play(dot.animate.shift(2*RIGHT), run_time=0.5)
        self.play(dot.animate.shift(2*UP), run_time=0.5)
        self.wait()
        self.play(FadeOut(dot), FadeOut(path))
        self.wait()

        # route 3
        path = VMobject(color=YELLOW_C)
        dot = Dot([-3,-3,0], color=YELLOW_C)
        path.set_points_as_corners([dot.get_center(), dot.get_center()])
        path.add_updater(update_path)

        self.add(path, dot)
        self.play(dot.animate.shift(4*UP), run_time=0.5)
        self.play(dot.animate.shift(RIGHT), run_time=0.5)
        self.play(dot.animate.shift(UP), run_time=0.5)
        self.play(dot.animate.shift(5*RIGHT), run_time=0.5)
        self.play(dot.animate.shift(UP), run_time=0.5)
        self.wait()
        self.play(FadeOut(dot), FadeOut(path))
        self.wait()

        # route 4
        path = VMobject(color=YELLOW_C)
        dot = Dot([-3,-3,0], color=YELLOW_C)
        path.set_points_as_corners([dot.get_center(), dot.get_center()])
        path.add_updater(update_path)

        self.add(path, dot)
        self.play(dot.animate.shift(6*RIGHT))
        self.play(dot.animate.shift(6*UP))
        self.wait()
        self.play(FadeOut(dot), FadeOut(path))

        self.wait(2)

class metrics(Scene):
    def construct(self):
        index_list = [0.5, 1, 2, 4, 8]
        p_string = MathTex('p =')
        num_string = [MathTex(f'{i}') for i in index_list]

        p_string.move_to([3,2.5,0])
        for i in range(len(num_string)):
            if i == 0:
                num_string[i].move_to([3.8,2.6,0])
            else:
                num_string[i].move_to([3.6,2.6,0])

        self.add(NumberPlane())
        self.wait()
        
        arr_of_graphs = [ImplicitFunction(
                    lambda x_1, x_2: (np.abs(x_1)**p + np.abs(x_2)**p)**(1/p)-2, 
                    color=YELLOW
                    ) for p in index_list]
        
        initial = arr_of_graphs[0]

        final_val = 100
        final = ImplicitFunction(
                    lambda x_1, x_2: (np.abs(x_1)**final_val + np.abs(x_2)**final_val)**(1/final_val)-2, 
                    color=YELLOW
                    )

        self.play(Create(initial))
        self.play(Write(p_string), Write(num_string[0]))
        self.wait()
        for i in range(len(arr_of_graphs)-1):
            self.play(Transform(initial, arr_of_graphs[i+1]), Transform(num_string[0], num_string[i+1]))
            self.wait()

        # t1 = ValueTracker(index_list[-1])
        # number = always_redraw(lambda: DecimalNumber(t1.get_value(), num_decimal_places = 0).move_to([3.8, 2.6, 0]))
        self.wait()

        # self.play(num_string[0].animate.move_to([3.8,2.6,0]))
        # self.play(FadeOut(num_string[0]), FadeIn(number), run_time=0.5)
        # self.play(Transform(initial, final), t1.animate.set_value(100))
        self.play(Transform(initial, final), FadeOut(p_string), FadeOut(num_string[0]), FadeIn(MathTex(r'p \to \infty').move_to([3.4,2.5,0])))
        # self.play(Transform(initial, final))
        self.wait(3)