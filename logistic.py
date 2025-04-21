from manim import *
import numpy as np

class logistic(Scene):
    def construct(self):
        w_and_b = [(1,0), (1,-1), (0.7,-0.5), (0.3,2), (-0.5,1.5)]
        k = ValueTracker(-8)

        line_formulas = [MathTex(f"y={w_and_b[i][0]}x+{w_and_b[i][1]}", color=RED_C) for i in range(len(w_and_b))]
        #sigmoid_formulas = [MathTex(fr"\sigma(y)=\frac{1}{{1+e^{-(w_and_b[i][0]+w_and_b[i][1])}}}") for i in range(len(w_and_b))]
        sigmoid_formulas = [MathTex(fr"\sigma(y)=\frac{1}{{1+e^{{-({w_and_b[i][0]}x+{w_and_b[i][1]})}}}}", color=YELLOW_C) for i in range(len(w_and_b))]

        plane0 = NumberPlane(
            x_range=[-4,4], y_range=[-4,4],
            x_length=4, y_length=4
        ).add_coordinates().move_to([-4.5,0,0])

        plane1 = NumberPlane(
            x_range=[-8,8], y_range=[0, 1], x_length=8, y_length=4
        ).add_coordinates().move_to([3,0,0])

        lines = [
            plane0.plot(
            lambda x : wand[0]*x + wand[1], x_range=[-4,4], color=RED_C
            ) for wand in w_and_b
        ]

        sigmoids = [
             plane1.plot(
            lambda x : 1/(1+np.exp(-(wand[0]*x + wand[1]))), x_range=[-8,8], color=YELLOW_C
        ) for wand in w_and_b
        ]


        sigmoid = plane1.plot(
            lambda x : 1/(1+np.exp(-x)), x_range=[-8,8], color=PURPLE_E
        )

        dot = always_redraw(
            lambda : Dot(color=YELLOW).move_to(
                plane1.c2p(k.get_value(), sigmoid.underlying_function(k.get_value()))
            )
        )

        self.play(FadeIn(plane0), FadeIn(plane1))
        self.wait()

        current_line_formula = line_formulas[0].next_to(plane0, UP)
        current_sigmoid_formula = sigmoid_formulas[0].next_to(plane1, UP)

        self.play(Create(lines[0]), Create(current_line_formula))
        self.wait()
        self.play(Create(sigmoids[0]), Create(current_sigmoid_formula))
        self.wait()

        for i in range(1, len(w_and_b)):
            self.play(Transform(lines[0], lines[i]), Transform(current_line_formula, line_formulas[i].next_to(plane0, UP)),
                      Transform(sigmoids[0], sigmoids[i]), Transform(current_sigmoid_formula, sigmoid_formulas[i].next_to(plane1, UP)))

            self.wait(1.5)
        
        self.wait(2)


class threshold(Scene):
    def construct(self):
        k = ValueTracker(0.5)
        ax = Axes(x_range=[-5,5], y_range=[0,1.2], axis_config={"include_numbers": True}).add_coordinates()
        
        sig_graph = ax.plot(lambda x : 1 / (1+np.exp(-x)), color=MAROON)
        logit_graph = ax.plot(lambda x : np.log(x/(1-x)), x_range=[0.01,0.99])

        random_numbers = [-4.5, -3.8, -2.3, -0.8, 0.4, 1.2, 2.1, 3.2, 4.8]
        dots = [Dot(ax.c2p(i, sig_graph.underlying_function(i), 0), color=WHITE) for i in random_numbers]

        # dynamic updater for the position of the threshold on the axes
        threshold_dot = always_redraw(
            lambda : Dot(color=YELLOW).move_to(
                ax.c2p(logit_graph.underlying_function(k.get_value()), k.get_value())
            )
        )

        horizontal_line = always_redraw(
            lambda : ax.get_horizontal_line(ax.input_to_graph_point(logit_graph.underlying_function(k.get_value()), sig_graph)
            , color=YELLOW)
        )


        self.play(FadeIn(ax))
        self.wait()
        self.play(Create(sig_graph))
        self.play(LaggedStart(*[FadeIn(dot) for dot in dots], lag_ratio=.05))
        self.wait()
        self.play(Create(threshold_dot))
        self.add(horizontal_line)
        self.wait()

        def check_threshold(dots, threshold):
            animations = []
            for dot in dots:
                # need to convert points to coordinates to get the correct y values
                dot_y = ax.p2c(dot.get_center())[1]

                if dot_y > threshold:
                    animations.append(dot.animate.set_color(GREEN))
                else:
                    animations.append(dot.animate.set_color(RED))

            # animate all colour changes at the same time
            self.play(AnimationGroup(*animations, lag_ratio=0))
        
        check_threshold(dots, k.get_value())

        self.wait()
        self.play(k.animate.set_value(0.05), runtime=2)
        self.wait()

        check_threshold(dots, k.get_value())

        self.wait()
        self.play(k.animate.set_value(0.85), runtime=2)
        self.wait()
        check_threshold(dots, k.get_value())

        self.wait(2)