import numpy as np
from manim import *

# A collection of functions to play around with
def square_func(x):
    return x**2

def derivative_square_func(x):
    return 2*x

def multiple_local_minima(x):
    # return x**4 - 2*x**2 + x
    return (x**2-2*x)**2 + 0.5*np.sin(12*x)

def derivative_multiple_local_minima(x):
    # return 4*x**3 - 4*x + 1
    return 4*x**3 - 12*x**2 + 8*x + 6*np.cos(12*x)

def oscillating_func(x):
    return (x**2-2*x)**2 + 0.5*np.sin(12*x)

def derivative_oscillating_func(x):
    return 4*x**3 - 12*x**2 + 8*x + 6*np.cos(12*x)

class graph(Scene):
    """
    A simple graph of a function of the user's choice.
    """
    def construct(self):
        ax = Axes(
            x_range=[-1,3, 1],
            y_range=[-1, 2, 1],
        ).set_opacity(0.7)

        func = ax.plot(
            lambda x: oscillating_func(x),
            x_range=[-3,3],
            color="#5ce1e6"
        )

        self.add(ax, func)

class fixed_learning_rate(Scene):
    """
    An animation depicting a fixed learning rate in gradient descent.
    """
    def construct(self):
        ax = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 5, 1],
        ).set_opacity(0.7)

        func = ax.plot(
            lambda x: multiple_local_minima(x),
            x_range=[-2.5,2.5],
            color="#5ce1e6"
        )

        # Number of desired gradient descent steps
        NUM_STEPS = 8
        # Learning rates
        alpha = [0.5, 0.8]
        # Starting point
        x_yellow = ValueTracker(0.3)
        x_orange = ValueTracker(0.1)
        iter_step = ValueTracker(0)

        text_yellow = MathTex(fr"\alpha = {alpha[0]}", color=YELLOW).to_corner(UL)
        text_orange = MathTex(fr"\alpha = {alpha[1]}", color=ORANGE).next_to(text_yellow, DOWN)
        text_step = always_redraw(
            lambda: MathTex(
                rf"\text{{Step }} {int(iter_step.get_value())}", 
                color=WHITE
            ).to_corner(UR)
        )

        yellow_dot = always_redraw(
            lambda: Dot(point=ax.c2p(x_yellow.get_value(), multiple_local_minima(x_yellow.get_value())), color=YELLOW)
        )

        orange_dot = always_redraw(
            lambda: Dot(point=ax.c2p(x_orange.get_value(), multiple_local_minima(x_orange.get_value())), color=ORANGE)
        )

        self.add(ax, func, yellow_dot, orange_dot, text_yellow, text_orange, text_step)
        self.wait()

        # Apply gradient descent
        for i in range(NUM_STEPS):
            iter_step.set_value(i+1)
            self.play(Indicate(text_step, color=BLUE))
            next_yellow = x_yellow.get_value() - alpha[0] * derivative_multiple_local_minima(x_yellow.get_value())
            next_orange = x_orange.get_value() - alpha[1] * derivative_multiple_local_minima(x_orange.get_value())
            self.play(
                x_yellow.animate.set_value(next_yellow), 
                x_orange.animate.set_value(next_orange), 
                run_time=2
            )
            self.wait()
        self.wait()


class adagrad(Scene):
    def construct(self):
        ax = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 5, 1],
        ).set_opacity(0.7)

        func = ax.plot(
            lambda x: multiple_local_minima(x),
            x_range=[-2.5,2.5],
            color="#5ce1e6"
        )

        # Number of desired gradient descent steps
        NUM_STEPS = 8
        # Learning rates
        alpha = 0.1
        sum_squared_gradients = 0
        # Starting point
        x_yellow = ValueTracker(0.1)
        x_orange = ValueTracker(0.1)
        iter_step = ValueTracker(0)

        text_yellow = MathTex(fr"\alpha = {alpha}", color=YELLOW).move_to([-6,3.5,0])
        text_orange = MathTex(r"Adagrad", color=ORANGE).move_to([-5.9,2.75,0])
        text_step = always_redraw(
            lambda: MathTex(
                rf"\text{{Step }} {int(iter_step.get_value())}", 
                color=WHITE
            ).to_corner(UR)
        )

        yellow_dot = always_redraw(
            lambda: Dot(point=ax.c2p(x_yellow.get_value(), multiple_local_minima(x_yellow.get_value())), color=YELLOW)
        )

        orange_dot = always_redraw(
            lambda: Dot(point=ax.c2p(x_orange.get_value(), multiple_local_minima(x_orange.get_value())), color=ORANGE)
        )

        self.add(ax, func, yellow_dot, orange_dot, text_yellow, text_orange, text_step)
        self.wait()

        # Apply gradient descent
        for i in range(NUM_STEPS):
            iter_step.set_value(i+1)
            self.play(Indicate(text_step, color=BLUE))
            next_yellow = x_yellow.get_value() - alpha * derivative_multiple_local_minima(x_yellow.get_value())

            # Implement Adagrad learning rate
            if i == 0:
                sum_squared_gradients = derivative_multiple_local_minima(x_orange.get_value())**2
            else:
                sum_squared_gradients += derivative_multiple_local_minima(x_orange.get_value())**2
            
            learning_rate = alpha / np.sqrt(1e-8 + sum_squared_gradients)

            next_orange = x_orange.get_value() - learning_rate * derivative_multiple_local_minima(x_orange.get_value())

            # Animate changes
            self.play(
                x_yellow.animate.set_value(next_yellow), 
                x_orange.animate.set_value(next_orange), 
                run_time=2
            )
            self.wait(0.5)
        self.wait()


class rmsprop(Scene):
    def construct(self):
        ax = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 5, 1],
        ).set_opacity(0.7)

        func = ax.plot(
            lambda x: multiple_local_minima(x),
            x_range=[-2.5,2.5],
            color="#5ce1e6"
        )

        # Number of desired gradient descent steps
        NUM_STEPS = 8
        
        alpha = 0.1
        sum_squared_gradients_yellow = 0
        # Decay rate for RMSProp
        BETA = 0.9
        
        exponential_moving_average = 0
        # Starting point
        x_yellow = ValueTracker(0.1)
        x_orange = ValueTracker(0.1)
        iter_step = ValueTracker(0)

        text_alpha = MathTex(fr"\alpha = {alpha}", color=WHITE).move_to([-6,3.5,0])
        text_yellow = MathTex(r"Adagrad", color=YELLOW).move_to([-5.9,2.75,0])
        text_orange = MathTex(r"RMSProp", color=ORANGE).move_to([-5.7,2,0])
        text_step = always_redraw(
            lambda: MathTex(
                rf"\text{{Step }} {int(iter_step.get_value())}", 
                color=WHITE
            ).to_corner(UR)
        )

        yellow_dot = always_redraw(
            lambda: Dot(point=ax.c2p(x_yellow.get_value(), multiple_local_minima(x_yellow.get_value())), color=YELLOW)
        )

        orange_dot = always_redraw(
            lambda: Dot(point=ax.c2p(x_orange.get_value(), multiple_local_minima(x_orange.get_value())), color=ORANGE)
        )

        self.add(ax, func, yellow_dot, orange_dot, text_alpha, text_yellow, text_orange, text_step)
        self.wait()

        # Apply gradient descent
        for i in range(NUM_STEPS):
            iter_step.set_value(i+1)
            self.play(Indicate(text_step, color=BLUE))

            # Implement Adagrad for yellow dot and RMSProp for orange dot
            if i == 0:
                sum_squared_gradients_yellow = derivative_multiple_local_minima(x_yellow.get_value())**2
                exponential_moving_average = (1-BETA) * derivative_multiple_local_minima(x_orange.get_value())**2
            else:
                sum_squared_gradients_yellow += derivative_multiple_local_minima(x_yellow.get_value())**2
                exponential_moving_average = BETA*exponential_moving_average + (1 - BETA)*derivative_multiple_local_minima(x_orange.get_value())**2
            
            learning_rate_yellow = alpha / np.sqrt(1e-8 + sum_squared_gradients_yellow)
            learning_rate = alpha / np.sqrt(1e-8 + exponential_moving_average)

            next_yellow = x_yellow.get_value() - learning_rate_yellow * derivative_multiple_local_minima(x_yellow.get_value())
            next_orange = x_orange.get_value() - learning_rate * derivative_multiple_local_minima(x_orange.get_value())

            # Animate changes
            self.play(
                x_yellow.animate.set_value(next_yellow), 
                x_orange.animate.set_value(next_orange), 
                run_time=2
            )
            self.wait(0.5)
        self.wait()


class adam(Scene):
    def construct(self):
        ax = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 5, 1],
        ).set_opacity(0.7)

        func = ax.plot(
            lambda x: multiple_local_minima(x),
            x_range=[-2.5,2.5],
            color="#5CD0B3"
        )

        # Number of desired gradient descent steps
        NUM_STEPS = 8
        
        alpha = 0.05
        exponential_moving_average = 0
        # Decay rate for RMSProp
        BETA = 0.9
        # Decay rates for Adam
        BETA_1 = 0.9
        BETA_2 = 0.999
        
        ema_gradients = 0
        ema_squared_gradients = 0
        # Starting point
        x_yellow = ValueTracker(0.5)
        x_orange = ValueTracker(0.5)
        iter_step = ValueTracker(0)

        text_alpha = MathTex(fr"\alpha = {alpha}", color=WHITE).move_to([-6,3.5,0])
        text_yellow = MathTex(r"RMSProp", color=YELLOW).move_to([-5.75,2.7,0])
        text_orange = MathTex(r"Adam", color=ORANGE).move_to([-6.3,2,0])
        text_step = always_redraw(
            lambda: MathTex(
                rf"\text{{Step }} {int(iter_step.get_value())}", 
                color=WHITE
            ).to_corner(UR)
        )

        yellow_dot = always_redraw(
            lambda: Dot(point=ax.c2p(x_yellow.get_value(), multiple_local_minima(x_yellow.get_value())), color=YELLOW)
        )

        orange_dot = always_redraw(
            lambda: Dot(point=ax.c2p(x_orange.get_value(), multiple_local_minima(x_orange.get_value())), color=ORANGE)
        )

        self.add(ax, func, yellow_dot, orange_dot, text_alpha, text_yellow, text_orange, text_step)
        self.wait()

        # Apply gradient descent
        for i in range(1, NUM_STEPS+1):
            iter_step.set_value(i)
            self.play(Indicate(text_step, color=BLUE))

            # Implement RMSProp for yellow dot and Adam for orange dot
            if i == 1:
                exponential_moving_average = (1-BETA) * derivative_multiple_local_minima(x_yellow.get_value())**2
                # sum_squared_gradients_yellow = derivative_multiple_local_minima(x_yellow.get_value())**2
                ema_gradients = (1-BETA_1) * derivative_multiple_local_minima(x_orange.get_value())
                ema_squared_gradients = (1-BETA_2) * derivative_multiple_local_minima(x_orange.get_value())**2
            else:
                exponential_moving_average = BETA*exponential_moving_average + (1 - BETA)*derivative_multiple_local_minima(x_yellow.get_value())**2
                # sum_squared_gradients_yellow += derivative_multiple_local_minima(x_yellow.get_value())**2
                ema_gradients = BETA_1*ema_gradients + (1-BETA_1)*derivative_multiple_local_minima(x_orange.get_value())
                ema_squared_gradients = BETA_2*ema_squared_gradients + (1-BETA_2)*derivative_multiple_local_minima(x_orange.get_value())**2
            
            # Bias corrections for Adam
            ema_gradients_corrected = ema_gradients / (1-BETA_1**i)
            ema_squared_gradients_corrected = ema_squared_gradients / (1-BETA_2**i)

            learning_rate_yellow = alpha / np.sqrt(1e-8 + exponential_moving_average)
            learning_rate = (alpha*ema_gradients_corrected) / np.sqrt(1e-8 + ema_squared_gradients_corrected)

            next_yellow = x_yellow.get_value() - learning_rate_yellow * derivative_multiple_local_minima(x_yellow.get_value())
            next_orange = x_orange.get_value() - learning_rate * derivative_multiple_local_minima(x_orange.get_value())

            # Animate changes
            self.play(
                x_yellow.animate.set_value(next_yellow), 
                x_orange.animate.set_value(next_orange), 
                run_time=2
            )
            self.wait(0.5)
        self.wait()