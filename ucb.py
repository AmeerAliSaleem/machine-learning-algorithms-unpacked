import numpy as np
from scipy.stats import norm
from manim import *

class ucb_algorithm:
    """
    The UCB algorithm for the MAB problem.
    """
    def __init__(self, delta=0.1, K=4):
        """
        Attributes
        ----------
        delta : float, optional
            Parameter for controlling the size of the confidence intervals.
        K : int, optional
            The number of arms in the current MAB setup.

        arm_pull_counts : ndarray
            The number of times each of the K arms has been played.
        empirical_mean_rewards : ndarray
            The current empirical average reward observed for each arm.
        upper_confidence_bounds : ndarray
            The current UCB value for each arm.
        """
        self.gaussians = [
            norm(loc=3.5, scale=0.8),
            norm(loc=4, scale=1.2),
            norm(loc=5.5, scale=0.9),
            norm(loc=6.5, scale=0.5)
        ]

        self.delta = delta
        self.K = K
        self.arm_pull_counts = np.zeros(K)
        self.empirical_mean_rewards = np.zeros(K)
        self.upper_confidence_bounds = np.zeros(K)

    def select_arm(self):
        """
        A method that makes the arm selection for the next round using Upper Confidence Bounds.
        """

        # Choose the arm with the highest UCB
        arm_choice = np.argmax(self.upper_confidence_bounds)

        return arm_choice

    def update_rewards(self, arm_index):
        """
        Sample a reward from the Gaussian distribution of the chosen arm and update the observed rewards.
        """
        
        self.arm_pull_counts[arm_index] += 1

        # Sample reward from corresponding Gaussian distribution
        current_reward = self.gaussians[arm_index].rvs()

        n = self.arm_pull_counts[arm_index]
        emp_reward = self.empirical_mean_rewards[arm_index]
        # Update the observed reward for the arm
        self.empirical_mean_rewards[arm_index] = (n-1)/n * emp_reward + (1/n) * current_reward

        # Update upper confidence bounds
        self.upper_confidence_bounds = self.empirical_mean_rewards + np.sqrt(2/self.arm_pull_counts * np.log(1/self.delta))

        print(f"Upper confidence bounds: {self.upper_confidence_bounds}.")

        return current_reward
    
class ucb(Scene):
    """
    An animation depicting the UCB algorithm for a set of 4 Gaussian arms.
    """
    def construct(self):
        def generate_lever(handle_colour = RED):
            """
            Creates a lever graphic.
            """

            lever = Rectangle(height=0.1, width=1, color=GRAY, fill_opacity=1)
            bottom = Rectangle(height=0.1, width=1.5, color=WHITE, fill_opacity=1)

            # Add a handle at the end of the lever
            handle = Circle(radius=0.3, color=handle_colour, fill_opacity=1)
            handle.move_to(lever.get_right())

            lever_group = VGroup(lever, handle).rotate(PI/2)

            bottom.move_to(lever.get_bottom()).shift(RIGHT*0.5)

            return VGroup(lever, handle, bottom)
        
        def pull_lever(lever_group):
            """
            Animates the pulling of a lever.
            """

            handle = VGroup(lever_group[0], lever_group[1])

            self.play(handle.animate.shift(RIGHT), rate_func=rush_into, run_time=0.5)
            self.play(handle.animate.shift(LEFT), run_time=0.5)

        def generate_data(ax, arm_index, current_reward, ucb_bandit):
            """
            Generate a data point from the given arm and plot the point on the 
            corresponding Gaussian curve.
            """

            dot = Dot(ax.c2p(current_reward, ucb_bandit.gaussians[arm_index].pdf(current_reward), 0), 
                      color=colours[arm_index], fill_opacity=0.7)

            self.play(FadeIn(dot), Flash(dot, color=dot.get_color()))
        
        def empirical_average_dots(ucb_bandit):
            """
            Creates the Dots for the UCB plots.
            The UCB arrows are handled in a separate funtion.

            Parameters
            ----------
            ucb_bandit : ucb_algorithm
                The bandit object that's currently in play.

            Returns
            ----------
            dots : list
                The list of dots plotted for the empirical confidence values.
            """

            dots = []
            for index, value in enumerate(ucb_bandit.empirical_mean_rewards):
                dots.append(Dot(ax_ci.c2p(index+1, value, 0), color=colours[index]))
            
            return dots
        
        def confidence_arrows(emp_dots, ucb_bandit):
            """
            Creates the confidence arrows on the input empirical mean Dot objects.

            Parameters
            ----------
            emp_dots : list
                The list of the Dot objects plotted on ax_ci.
            ucb_bandit : ucb_algorithm
                The bandit object that's currently in play.

            Returns
            ----------
            confidence_arrows : list
                A list of the confidence interval arrows to draw on each Dot.

            """
            confidence_arrows = []

            for index, val in enumerate(ucb_bandit.empirical_mean_rewards):
                arrow = Arrow(
                    start=ax_ci.c2p(index+1, val, 0),
                    end=ax_ci.c2p(index+1, ucb_bandit.upper_confidence_bounds[index], 0),
                    color=emp_dots[index].get_color(),
                    tip_length=0.1,
                    stroke_width=3,
                    buff=0
                    )
                
                confidence_arrows.append(arrow)

            return confidence_arrows
        
        lever1 = generate_lever(handle_colour=BLUE).move_to([0, -2, 0])
        lever2 = generate_lever(handle_colour=GREEN).move_to([2, -2, 0])
        lever3 = generate_lever(handle_colour=RED).move_to([4, -2, 0])
        lever4 = generate_lever(handle_colour=PURPLE).move_to([6, -2, 0])

        levers = VGroup(lever1, lever2, lever3, lever4)

        np.random.seed(8)

        ucb_bandit = ucb_algorithm()

        ax = Axes(
            x_range=[0, 12, 1],
            y_range=[0, 1.01, 0.5],
            axis_config={'tip_shape': StealthTip, 'include_numbers': True}
            )
        ax_labels = ax.get_axis_labels(
            Tex("Rewards"), Tex("Probability")
        )
        
        ax_ci_y_min = 2
        ax_ci_y_max = 10
        ax_ci = Axes(
            x_range=[0, 4, 1],
            y_range = [ax_ci_y_min, ax_ci_y_max, 1],
            x_length = 4,
            y_length = 5,
            x_axis_config={'include_tip': False},
            y_axis_config={'include_numbers': True, 'include_tip': False}
        ).move_to([-4.5, 0, 0])
        ax_ci_y_label = ax_ci.get_y_axis_label(Tex("Empirical mean rewards", font_size=24)).move_to([-5.5, 3, 0])

        # add dashed horizontal lines to ax_ci
        for y in range(ax_ci_y_min, ax_ci_y_max+1):
            dashed = DashedLine(
                ax_ci.c2p(0, y, 0),
                ax_ci.c2p(4, y, 0),
                color=GREY_B,
                stroke_width=1,
                dash_length=0.1,
                dashed_ratio=0.5,
            )
            self.add(dashed)
        
        colours = [BLUE_B, GREEN_B, RED_B, PURPLE_B]

        gaussian1 = ucb_bandit.gaussians[0]
        gaussian2 = ucb_bandit.gaussians[1]
        gaussian3 = ucb_bandit.gaussians[2]
        gaussian4 = ucb_bandit.gaussians[3]

        f1_opacity = ValueTracker(1.0)
        f2_opacity = ValueTracker(1.0)
        f3_opacity = ValueTracker(1.0)
        f4_opacity = ValueTracker(1.0)

        pdf_opacities = [f1_opacity, f2_opacity, f3_opacity, f4_opacity]

        f1 = always_redraw(
            lambda: ax.plot(
                lambda x: gaussian1.pdf(x),
                color=colours[0],
                stroke_opacity=f1_opacity.get_value()
            )
        )
        
        f2 = always_redraw(
            lambda: ax.plot(
                lambda x: gaussian2.pdf(x),
                color=colours[1],
                stroke_opacity=f2_opacity.get_value()
            )
        )
        
        f3 = always_redraw(
            lambda: ax.plot(
                lambda x: gaussian3.pdf(x),
                color=colours[2],
                stroke_opacity=f3_opacity.get_value()
            )
        )

        f4 = always_redraw(
            lambda: ax.plot(
                lambda x: gaussian4.pdf(x),
                color=colours[3],
                stroke_opacity=f4_opacity.get_value()
            )
        )

        vgroup_plot= VGroup(ax, ax_labels, f1, f2, f3, f4).scale(0.6).move_to([3, 1.25, 0])

        self.add(*vgroup_plot, *levers, ax_ci, ax_ci_y_label)

        # step 1: play each arm once
        for i in range(ucb_bandit.K):
            pull_lever(levers[i])

            current_reward = ucb_bandit.update_rewards(i)
            generate_data(ax, i, current_reward, ucb_bandit)

        emp_dots = empirical_average_dots(ucb_bandit)
        self.play(*[FadeIn(dot) for dot in emp_dots])

        ucb_arrows = confidence_arrows(emp_dots, ucb_bandit)
        self.play(*[GrowArrow(arrow) for arrow in ucb_arrows])

        self.wait(1.5)

        # Indicate the arm with the largest UCB
        largest_ucb_index = np.argmax(ucb_bandit.upper_confidence_bounds)
        largest_ucb_group = VGroup(emp_dots[largest_ucb_index], ucb_arrows[largest_ucb_index])
        self.play(Indicate(largest_ucb_group), run_time=2)

        self.wait()

        # Fade out all the confidence interval information
        self.play(
            *[FadeOut(dot) for dot in emp_dots],
            *[FadeOut(arrow) for arrow in ucb_arrows]
            )

        self.wait()
        
        # step 2: from now on, select the arm with the highest UCB
        NUM_ROUNDS = 8

        for i in range(NUM_ROUNDS):
            arm_index = ucb_bandit.select_arm()

            pull_lever(levers[arm_index])

            current_reward = ucb_bandit.update_rewards(arm_index)
            generate_data(ax, arm_index, current_reward, ucb_bandit)

            emp_dots = empirical_average_dots(ucb_bandit)
            self.play(*[FadeIn(dot) for dot in emp_dots])

            ucb_arrows = confidence_arrows(emp_dots, ucb_bandit)
            self.play(*[GrowArrow(arrow) for arrow in ucb_arrows])

            self.wait(1.5)

            # Indicate the arm with the largest UCB
            largest_ucb_index = np.argmax(ucb_bandit.upper_confidence_bounds)
            largest_ucb_group = VGroup(emp_dots[largest_ucb_index], ucb_arrows[largest_ucb_index])
            self.play(Indicate(largest_ucb_group), run_time=2)

            self.wait()

            # Fade out all the confidence interval information
            self.play(
                *[FadeOut(dot) for dot in emp_dots],
                *[FadeOut(arrow) for arrow in ucb_arrows]
                )
            
            self.wait()
        
        self.wait()