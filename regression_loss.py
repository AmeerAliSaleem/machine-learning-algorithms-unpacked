import numpy as np
from manim import *

np.random.seed(8)
x_vals  = np.linspace(1, 5, 20)
y_vals = x_vals + np.random.normal(0, 0.5, size=len(x_vals))
dot_coords = np.column_stack((x_vals, y_vals))

class regression_line(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0,6,1],
            y_range=[0,6,1],
            x_length=6,
            tips=False
        )

        dots = VGroup(*[Dot(ax.c2p(x,y), color='#5CD0B3'
) for x, y in dot_coords])

        self.add(ax, dots)


class mse_mae(Scene):
    """
    A comparison between the MSE and MAE loss functions.
    """
    def construct(self):
        ax = Axes(
            x_range=[-2,2,1],
            y_range=[0,4,1],
            x_length=6,
            tips=False
        ).add_coordinates()

        # def predicted_labels(data, m, b):
        #     return m*data[:,0] + b

        # def mse_loss(data, m, b):
        #     y_hat = predicted_labels(data, m, b)
        #     return np.mean((data[:,1] - y_hat) ** 2)
        
        # def mae_loss(data, m, b):
        #     y_hat = predicted_labels(data, m, b)
        #     return np.mean(np.abs(data[:,1] - y_hat))
        
        # m_vals = np.linspace(-1, 3, 101)
        # mse_vals = [mse_loss(dot_coords, m, 0) for m in m_vals]
        # mae_vals = [mae_loss(dot_coords, m, 0) for m in m_vals]

        mse_curve = ax.plot(
            lambda x: x**2,
            x_range=[-2, 2],
            color='#FF8080'
        )
        mae_curve = ax.plot(
            lambda x: np.abs(x),
            x_range=[-2, 2],
            color='#9CDCEB'
        )

        mse_text = Text("MSE", color='#FF8080').move_to([5,3,0])
        mae_text = Text("MAE", color='#9CDCEB').move_to([5,2,0])

        self.add(ax, mse_curve, mae_curve, mse_text, mae_text)


class huber_loss(Scene):
    """
    A visualization of the Huber loss function.
    """
    def construct(self):
        ax = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 4, 1],
            x_length=6,
            tips=False
        ).add_coordinates()

        delta = ValueTracker(1)

        def huber_loss(x, delta):
            return np.where(np.abs(x) <= delta, 0.5 * x**2, delta * (np.abs(x) - 0.5 * delta))

        huber_curve = always_redraw(
            lambda :ax.plot(
                lambda x: huber_loss(
                    x, 
                    delta=delta.get_value()
                ),
            x_range=[-3,3],
            color='#FFFF00'
            )
        )

        # Red curve (MSE): |x| <= delta
        huber_curve_red = always_redraw(
            lambda: ax.plot(
                lambda x: huber_loss(x, delta.get_value()),
                x_range=[-delta.get_value(), delta.get_value()],
                color='#FF8080'
            )
        )

        # Blue curve (MAE): |x| < delta (left)
        huber_curve_blue_left = always_redraw(
            lambda: ax.plot(
                lambda x: huber_loss(x, delta.get_value()),
                x_range=[-3, -delta.get_value()],
                color='#9CDCEB'
            )
        )

        # Blue curve (MAE): |x| > delta (right)
        huber_curve_blue_right = always_redraw(
            lambda: ax.plot(
                lambda x: huber_loss(x, delta.get_value()),
                x_range=[delta.get_value(), 3],
                color='#9CDCEB'
            )
        )

        mse_text = Text("MSE", color='#FF8080').move_to([5,3,0]).scale(0.8)
        mae_text = Text("MAE", color='#9CDCEB').move_to([5,2,0]).scale(0.8)

        huber_text = Text("Huber loss:", color='#FFFF00').move_to([-5,3,0]).scale(0.8)
        delta_label = MathTex(r"\delta=", color='#FFFF00').move_to([-5.5,2,0])
        delta_value = always_redraw(
            lambda: MathTex(f"{delta.get_value():.2f}", color='#FFFF00').next_to(delta_label, RIGHT)
        )

        self.add(ax, mse_text, mae_text, huber_text, delta_label, delta_value, huber_curve_red, huber_curve_blue_left, huber_curve_blue_right)
        self.wait()

        self.play(delta.animate.set_value(0.5), run_time=2)
        self.wait(2)

        self.play(delta.animate.set_value(0.1), run_time=2)
        self.wait(2)

        self.play(delta.animate.set_value(2), run_time=2)
        self.wait(2)

        self.play(delta.animate.set_value(5), run_time=2)
        self.wait(2)

        self.wait(2)