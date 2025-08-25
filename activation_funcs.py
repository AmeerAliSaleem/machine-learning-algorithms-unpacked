
import numpy as np
from manim import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class sigmoid_plot(Scene):
    def construct(self):
        ax = Axes(
            x_range=[-6, 6, 1],
            y_range=[0, 1.1, 0.5],
        ).add_coordinates()

        upperLine = DashedLine(ax.c2p(-6,1), ax.c2p(6,1), color=YELLOW).set_opacity(0.5)
        sigmoid_text = MathTex(r"\sigma(x) = \frac{1}{1 + e^{-x}}", color='#5ce1e6').move_to([-4,1,0])

        sigmoid_curve = ax.plot(
            lambda x: 1 / (1 + np.exp(-x)),
            x_range=[-6, 6],
            color='#5ce1e6'
        )

        self.add(ax, sigmoid_curve, upperLine, sigmoid_text)

class tanh_plot(Scene):
    def construct(self):
        ax = Axes(
            x_range=[-6, 6, 1],
            y_range=[-1, 1.1, 0.5],
        ).add_coordinates()

        upperLine1 = DashedLine(ax.c2p(-6,1), ax.c2p(6,1), color=YELLOW).set_opacity(0.5)
        upperLine2 = DashedLine(ax.c2p(-6,-1), ax.c2p(6,-1), color=YELLOW).set_opacity(0.5)
        tanh_text = MathTex(r"\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}", color='#ff5757').move_to([-3.75,1.5,0])

        tanh_curve = ax.plot(
            lambda x: np.tanh(x),
            x_range=[-6, 6],
            color='#ff5757'
        )

        self.add(ax, upperLine1, upperLine2, tanh_curve, tanh_text)

class gradient_tanh_sigmoid(Scene):
    def construct(self):
        ax = Axes(
            x_range=[-8, 8, 1],
            y_range=[0, 1.1, 0.5],
        ).add_coordinates()

        gradient_sigmoid = ax.plot(
            lambda x: sigmoid(x) * (1 - sigmoid(x)),
            x_range=[-8, 8],
            color='#5ce1e6',
        )

        gradient_tanh = ax.plot(
            lambda x: 1 - tanh(x)**2,
            x_range=[-8, 8],
            color='#ff5757',
        )

        gradient_tanh_text = MathTex(r"\tanh'(x) = 1 - \tanh^2(x)", color='#ff5757').move_to([4,2,0])
        gradient_sigmoid_text = MathTex(r"\sigma'(x) = \sigma(x)\cdot(1 - \sigma(x))", color='#5ce1e6').move_to([4,1,0])

        self.add(ax, gradient_sigmoid, gradient_tanh, gradient_sigmoid_text, gradient_tanh_text)

class relu_plot(Scene):
    def construct(self):
        ax = Axes(
            x_range=[-6, 6, 1],
            y_range=[0, 6, 1],
        ).add_coordinates()

        relu_text = MathTex(r"ReLU(x) = \max(0, x)", color='#ffbd59').move_to([-3.5, 1, 0])

        relu_curve = ax.plot(
            lambda x: np.maximum(0, x),
            x_range=[-6, 6],
            color='#ffbd59',
        )

        self.add(ax, relu_text, relu_curve)

class gradient_relu(Scene):
    def construct(self):
        ax = Axes(
            x_range=[-6, 6, 1],
            y_range=[0, 1.1, 0.5],
        ).add_coordinates()

        gradient_relu_pt1 = Line(ax.c2p(-6,0), ax.c2p(0,0), color='#ffbd59')
        gradient_relu_pt2 = Line(ax.c2p(0,1), ax.c2p(6,1), color='#ffbd59')
        dot1 = Dot(point=ax.c2p(0,0), color='#ffbd59')
        dot2 = Circle(radius=0.08, color='#ffbd59').move_to(ax.c2p(0,1))

        gradient_relu_text = MathTex(r"ReLU'(x) = \begin{cases} 0 & x \leq 0 \\ 1 & x > 0 \end{cases}", color='#ffbd59').move_to([-3.5,1,0])

        self.add(ax, gradient_relu_pt1, gradient_relu_pt2, dot1, dot2, gradient_relu_text)

class leaky_relu_plot(Scene):
    def construct(self):
        alpha = ValueTracker(0.05)
        ax = Axes(
            x_range=[-6, 6, 1],
            y_range=[-6, 6, 1],
        ).add_coordinates()

        leaky_relu_text = MathTex(r"LReLU(x) = \max(\alpha x, x)", color='#ff66c4').move_to([-3.5, 2, 0])
        alpha_text = MathTex(r"\alpha = ", color='#ff66c4').move_to([-3.75,1,0])
        alpha_value = always_redraw(
            lambda: MathTex(f"{alpha.get_value():.2f}", color='#ff66c4').next_to(alpha_text, RIGHT)
        )

        relu_curve = always_redraw(
            lambda: ax.plot(
                lambda x: np.maximum(alpha.get_value()*x, x),
                x_range=[-6, 6],
                color='#ff66c4',
            )
        )

        self.add(ax, leaky_relu_text, relu_curve, alpha_text, alpha_value)
        self.wait()
        self.play(alpha.animate.set_value(0.5), run_time=2)
        self.wait()
        self.play(alpha.animate.set_value(1), run_time=2)
        self.wait()
        self.play(alpha.animate.set_value(0.01), run_time=2)
        self.wait(2)

class gradient_leaky_relu(Scene):
    def construct(self):
        alpha = ValueTracker(0.05)
        ax = Axes(
            x_range=[-6, 6, 1],
            y_range=[0, 1.1, 0.5],
        ).add_coordinates()

        gradient_leaky_relu_text = MathTex(r"LReLU'(x) = \begin{cases} \alpha & x \leq 0 \\ 1 & x > 0 \end{cases}", color='#ff66c4').move_to([3.5,1,0])
        alpha_text = MathTex(r"\alpha = ", color='#ff66c4').move_to([3.3,-0.5,0])
        alpha_value = always_redraw(
            lambda: MathTex(f"{alpha.get_value():.2f}", color='#ff66c4').next_to(alpha_text, RIGHT)
        )

        gradient_relu_pt1 = always_redraw(
            lambda: Line(
                ax.c2p(-6, alpha.get_value()), 
                ax.c2p(0, alpha.get_value()), 
                color='#ff66c4'
            )
        )
        gradient_relu_pt2 = Line(ax.c2p(0,1), ax.c2p(6,1), color='#ff66c4')
        dot1 = always_redraw(
            lambda: Dot(point=ax.c2p(0,alpha.get_value()), color='#ff66c4')
        )
        dot2 = Circle(radius=0.08, color='#ff66c4').move_to(ax.c2p(0,1))

        self.add(ax, gradient_leaky_relu_text, alpha_text, alpha_value, 
                 gradient_relu_pt1, gradient_relu_pt2, dot1, dot2)
        self.wait()
        self.play(alpha.animate.set_value(0.5), run_time=2)
        self.wait()
        self.play(alpha.animate.set_value(1), run_time=2)
        self.wait()
        self.play(alpha.animate.set_value(0.01), run_time=2)
        self.wait(2)

class elu_plot(Scene):
    def construct(self):
        alpha = ValueTracker(0.05)
        ax = Axes(
            x_range=[-6, 6, 1],
            y_range=[-2, 6, 1],
        ).add_coordinates()

        elu_text = MathTex(r"ELU(x) = \begin{cases} \alpha(e^x - 1) & x \leq 0 \\ x & x > 0 \end{cases}", color='#00bf63').scale(0.8).move_to([-3.5, 2, 0])
        alpha_text = MathTex(r"\alpha = ", color='#00bf63').scale(0.8).move_to([-4.6,0.5,0])
        alpha_value = always_redraw(
            lambda: MathTex(f"{alpha.get_value():.2f}", color='#00bf63').next_to(alpha_text, RIGHT)
        )

        elu_curve = always_redraw(
            lambda: ax.plot(
                lambda x: alpha.get_value()*(np.exp(x)-1) if x<=0 else x,
                x_range=[-6, 6],
                color='#00bf63',
            )
        )

        self.add(ax, elu_text, elu_curve, alpha_text, alpha_value)
        self.wait()
        self.play(alpha.animate.set_value(1), run_time=2)
        self.wait()
        self.play(alpha.animate.set_value(2), run_time=2)
        self.wait()
        self.play(alpha.animate.set_value(0.3), run_time=2)
        self.wait(2)

class gradient_elu(Scene):
    def construct(self):
        alpha = ValueTracker(0.05)
        ax = Axes(
            x_range=[-6, 6, 1],
            y_range=[0, 1.1, 0.5],
        ).add_coordinates()

        gradient_elu_text = MathTex(r"ELU'(x) = \begin{cases} \alpha e^x & x \leq 0 \\ 1 & x > 0 \end{cases}", color='#00bf63').scale(0.8).move_to([3.5,0,0])
        alpha_text = MathTex(r"\alpha = ", color='#00bf63').scale(0.8).move_to([3,-1.5,0])
        alpha_value = always_redraw(
            lambda: MathTex(f"{alpha.get_value():.2f}", color='#00bf63').next_to(alpha_text, RIGHT)
        )

        gradient_elu_pt1 = always_redraw(
            lambda: ax.plot(
                lambda x: alpha.get_value()*np.exp(x),
                x_range=[-6, 0],
                color='#00bf63'
            )
        )
        gradient_elu_pt2 = Line(ax.c2p(0,1), ax.c2p(6,1), color='#00bf63')
        dot1 = always_redraw(
            lambda: Dot(point=ax.c2p(0,alpha.get_value()), color='#00bf63')
        )
        dot2 = Circle(radius=0.08, color='#00bf63').move_to(ax.c2p(0,1))

        self.add(ax, gradient_elu_text, alpha_text, alpha_value, 
                 gradient_elu_pt1, gradient_elu_pt2, dot1, dot2)
        self.wait()
        self.play(alpha.animate.set_value(0.5), run_time=2)
        self.wait()
        self.play(alpha.animate.set_value(1), run_time=2)
        self.wait()
        self.play(alpha.animate.set_value(0.01), run_time=2)
        self.wait(2)