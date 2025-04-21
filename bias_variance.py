import numpy as np
from manim import *
from sklearn.linear_model import LinearRegression

"""
Note of warning: I was low on time for this one, and so the software engineering best practices
have gone out the window (not that I was great with best practice in the first place lmao).

"""

n = 21
eps = 0.3
np.random.seed(8)

def true_function(x):
        """
        The true function f(X) that we want our models to learn.
        """
        return 0.3 * (x - 5) ** 2 + 1

def generate_polynomial_fit(ax, x_train, y_train, p, color):
        """
        Generate a polynomial fit of degree p and return the model and line.
        """
        # Fit a polynomial of degree p
        poly_model = np.polyfit(x_train, y_train, p)

        # Create the corresponding line
        poly_line = ax.plot(
            lambda x: sum([poly_model[i] * x**(p - i) for i in range(p + 1)]),
            x_range=[0, 10],
            color=color
        )

        return poly_model, poly_line

def compute_bias_variance(coefficients, x, y_true):
        """
        Compute the bias and variance of a model given the model coefficients, the input data, and true values.
        """
        # Compute predicted values using the polynomial model
        y_pred = np.polyval(coefficients, x)

        variance = np.var(y_pred)
        # MSE = bias^2 + variance
        bias_squared = np.mean((y_true - y_pred) ** 2)

        return bias_squared, variance

x_train = np.linspace(0, 10, n)
y_train = 0.3*(x_train-5)**2 + np.random.normal(0, eps, n) + 1

x_test = np.linspace(0.25, 9.25, 10)
y_test = 0.3*(x_test-5)**2 + np.random.normal(0, 0.3, 10) + 1

# values of f(X), the 'true function' that we're trying to learn
y_true = true_function(x_train)

# fitting a linear model to the data
model = LinearRegression()
model.fit(x_train.reshape(-1,1), y_train)

# fitting a quadratic model to the data
quadratic_model = np.polyfit(x_train, y_train, 2)

class bias_variance(Scene):
    """
    Used to create images of the three models against the training and testing data.
    """
    def construct(self):
        ax = Axes(
            x_range=[0, 10],
            y_range=[0, 10],
            tips=False
        )

        data = VGroup(*[Dot(point=[ax.c2p(x, y, 0)], color=BLUE) for x, y in zip(x_train, y_train)])
        test_data = VGroup(*[Dot(point=[ax.c2p(x, y, 0)], color=RED) for x, y in zip(x_test, y_test)])

        # print(model.coef_)
        # print(model.intercept_)

        line = ax.plot(lambda x: model.intercept_ + model.coef_[0]*x, x_range=[0, 10], color=YELLOW)

        quadratic_line = ax.plot(
            lambda x: quadratic_model[0]*x**2 + quadratic_model[1]*x + quadratic_model[2],
            x_range = [0,10],
            color = GREEN
        )

        overfit_model, overfit_line = generate_polynomial_fit(ax, x_train, y_train, 
                                                                   p=20, color=TEAL)

        # line.set_stroke(opacity=0.2)
        quadratic_line.set_stroke(opacity=0.4)

        # self.add(ax, data, test_data, line, quadratic_line, overfit_line)
        self.add(ax, test_data, quadratic_line, overfit_line)


class model_comparisons(Scene):
    """
    Used for the animation comparing the variances and (square) biases of the models.
    """
    def construct(self):
        ax = Axes(
                    x_range=[0, 10, 5],
                    y_range=[0, 10, 5],
                    axis_config={"include_numbers": False},
                    x_length=6,
                    y_length=4
                ).move_to([-3.5,0,0])
        

        ax2 = Axes(
                    x_range=[0, 10, 5],
                    y_range=[0, 10, 5],
                    axis_config={"include_numbers": False},
                    x_length=6,
                    y_length=4
                ).move_to([3.5,0,0])
        
        x_label_ax2 = ax2.get_x_axis_label(Tex("Model Complexity"), 
                                           edge=DOWN, direction=DOWN).scale(0.5)
        y_label_ax2 = ax2.get_y_axis_label(Tex("Error"),
                                           edge=UP, direction=UP).scale(0.5)

        k = ValueTracker(2)
        
        data = VGroup(*[Dot(point=[ax.c2p(x, y, 0)], radius=0.04, color=BLUE) for x, y in zip(x_train, y_train)])
        
        line = ax.plot(lambda x: model.intercept_ + model.coef_[0]*x, x_range=[0, 10], color=YELLOW_E)

        total_error = ax2.plot(lambda x: 0.2*((x-5)**2 + 25), x_range=[1, 9], color=RED)
        # total_error = ax2.plot(lambda x: 0.3*np.cosh(x-5)+4, x_range=[1, 9], color=RED)

        error_dot = always_redraw(
            lambda : Dot(color=YELLOW).move_to(
                ax2.c2p(k.get_value(), total_error.underlying_function(k.get_value()))
            )
        )

        vertical_line = always_redraw(
            lambda : ax2.get_vertical_line(error_dot.get_center(), color=YELLOW)
        )

        line_bias = ax2.plot(lambda x: 1.25**(10-x), x_range=[1, 9], color=PURPLE_B)
        line_variance = ax2.plot(lambda x: 1.25**x, x_range=[1, 9], color=MAROON_B)

        text_total_error = Tex("Total Error", color=RED).move_to([6,3,0]).scale(0.5)
        text_bias_squared = Tex("Bias$^2$", color=PURPLE_B).move_to([6,2.5,0]).scale(0.5)
        text_variance = Tex("Variance", color=MAROON_B).move_to([6,2,0]).scale(0.5)

        quadratic_line = ax.plot(
            lambda x: quadratic_model[0]*x**2 + quadratic_model[1]*x + quadratic_model[2],
            x_range = [0,10],
            color = GREEN
        )
        
        overfit_model, overfit_line = generate_polynomial_fit(ax, x_train, y_train, 
                                                                   p=20, color=TEAL)

        self.add(ax, data, ax2, x_label_ax2, y_label_ax2, total_error, 
                 line_bias, line_variance, text_total_error,
                 text_bias_squared, text_variance)
        self.wait()

        self.play(Create(line))
        self.play(Create(error_dot), Create(vertical_line))
        self.wait(2)

        self.play(Transform(line, overfit_line), k.animate.set_value(8), run_time=3)
        self.wait(2)

        self.play(Transform(line, quadratic_line), k.animate.set_value(5), run_time=3)

        self.wait(2)



class old_model_comparisons(Scene):
    """
    I tried using the empirical values for the bias and variance,
      but I think I messed it up + ran out of time.
    """
    def construct(self):
        ax = Axes(
                    x_range=[0, 10],
                    y_range=[0, 10],
                    axis_config={"include_numbers": True},
                    x_length=6,
                    y_length=4
                ).move_to([-3,0,0])
        

        data = VGroup(*[Dot(point=[ax.c2p(x, y, 0)], radius=0.04, color=BLUE) for x, y in zip(x_train, y_train)])
        # test_data = VGroup(*[Dot(point=[ax.c2p(x, y, 0)], color=RED) for x, y in zip(x_test, y_test)])
        
        line = ax.plot(lambda x: model.intercept_ + model.coef_[0]*x, x_range=[0, 10], color=YELLOW)

        quadratic_line = ax.plot(
            lambda x: quadratic_model[0]*x**2 + quadratic_model[1]*x + quadratic_model[2],
            x_range = [0,10],
            color = GREEN
        )
        
        overfit_model, overfit_line = generate_polynomial_fit(ax, x_train, y_train, 
                                                                   p=20, color=TEAL)

        bias_var = []
            
        # compute the bias and variance of the models for the training data
        bias_var.append(compute_bias_variance([model.coef_[0], model.intercept_], x_train, y_true))
        bias_var.append(compute_bias_variance(quadratic_model, x_train, y_true))
        bias_var.append(compute_bias_variance(overfit_model, x_train, y_true))

        # check (square) bias and variance values
        print(bias_var)

        p = ValueTracker(bias_var[0][0])
        q = ValueTracker(bias_var[0][1])

        def get_bar_chart():
            return BarChart(
                values=[p.get_value(), q.get_value()],
                bar_names=[],
                # y_range=[0, 7, 1],
                y_range=[-4, 1, 1],
                y_axis_config={"scaling": LogBase(custom_labels=True)},
                x_length=4,
                y_length=4,
                x_axis_config={"font_size": 36},
                # axis_config={'include_numbers': False},
                bar_colors=[MAROON_B, PURPLE_B]
            ).move_to([4,0,0])
        
        def get_labels(bar_chart):
            return VGroup(
                MathTex(f"bias^2").next_to(bar_chart.bars[0], DOWN),      # Label for first bar
                MathTex(f"variance").next_to(bar_chart.bars[1], DOWN).shift(0.1*DOWN)   # Label for second bar
            )
        
        bar_chart = always_redraw(get_bar_chart)
        bar_chart_labels = always_redraw(lambda: get_labels(bar_chart))

        self.add(ax, data)
        self.wait()

        # linear model
        self.play(Create(line))
        self.wait()

        self.play(DrawBorderThenFill(bar_chart), FadeIn(bar_chart_labels))
        self.wait()

        # overfit model
        self.play(Transform(line, overfit_line), p.animate.set_value(bias_var[2][0]), 
                  q.animate.set_value(bias_var[2][1]), run_time=3)
        self.wait()

        # quadratic model
        self.play(Transform(line, quadratic_line), p.animate.set_value(bias_var[1][0]), 
                  q.animate.set_value(bias_var[1][1]), run_time=3)

        self.wait(2)