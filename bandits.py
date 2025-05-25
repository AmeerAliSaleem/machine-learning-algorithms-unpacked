import numpy as np
from scipy.stats import norm
from manim import *

class epsilon_greedy_bandit:
    """
    Code adapted from: https://www.geeksforgeeks.org/multi-armed-bandit-problem-in-reinforcement-learning/
    """
    def __init__(self, eps=0.5, K=3):
        self.gaussians = [
            norm(loc=2, scale=0.8),
            norm(loc=5, scale=1.2),
            norm(loc=7, scale=0.5)
        ]

        self.eps = eps
        self.K = K
        self.arm_pull_counts = np.zeros(K)
        self.observed_rewards = np.zeros(K)

    def select_arm(self):
        roll = np.random.rand()
        if roll < self.eps:
            # Explore: choose a random arm
            arm_index = np.random.randint(0, self.K)
        else:
            # Exploit: choose the arm with the highest observed reward
            arm_index = np.argmax(self.observed_rewards)

        return arm_index, roll

    def update_rewards(self, arm_index):
        """
        Sample a reward from the Gaussian distribution of the given arm and update the observed rewards.
        """
        
        self.arm_pull_counts[arm_index] += 1

        # Sample reward from corresponding Gaussian distribution
        current_reward = self.gaussians[arm_index].rvs()

        n = self.arm_pull_counts[arm_index]
        emp_reward = self.observed_rewards[arm_index]
        # Update the observed reward for the arm
        self.observed_rewards[arm_index] = (n-1)/n * emp_reward + (1/n) * current_reward

        return current_reward


class bandits(Scene):
    """
    An animation depicting the use of the epsilon-greedy algorithm for three Gaussian bandits.
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

        lever1 = generate_lever(handle_colour=BLUE).move_to([0, -2, 0])
        lever2 = generate_lever(handle_colour=GREEN).move_to([2, -2, 0])
        lever3 = generate_lever(handle_colour=RED).move_to([4, -2, 0])

        levers = VGroup(lever1, lever2, lever3)

        np.random.seed(42)

        EPS = 0.6

        ax = Axes(
            x_range=[-1, 10, 1],
            y_range=[0, 1.01, 0.5],
            axis_config={'tip_shape': StealthTip, 'include_numbers': True}
            )
        
        colours = [BLUE_B, GREEN_B, RED_B]

        gaussian1 = norm(loc=2, scale=0.8)
        gaussian2 = norm(loc=5, scale=1.2)
        gaussian3 = norm(loc=7, scale=0.5)

        f1 = always_redraw(
            lambda: ax.plot(
                lambda x: gaussian1.pdf(x),
                color=colours[0]
            )
        )
        
        f2 = always_redraw(
            lambda: ax.plot(
                lambda x: gaussian2.pdf(x),
                color=colours[1]
            )
        )
        
        f3 = always_redraw(
            lambda: ax.plot(
                lambda x: gaussian3.pdf(x),
                color=colours[2]
            )
        )

        gaussians = [gaussian1, gaussian2, gaussian3]
        gaussian_curves = [f1, f2, f3]

        def generate_data(ax, arm_index, current_reward):
            """
            Generate a data point from the given arm and plot the point on the 
            corresponding Gaussian curve.
            """

            # x = gaussians[arm_index].rvs()# size=1)
            dot = Dot(ax.c2p(current_reward, gaussians[arm_index].pdf(current_reward), 0), 
                      color=colours[arm_index], fill_opacity=0.7)

            self.play(FadeIn(dot), Flash(dot, color=dot.get_color()))

        vgroup_plot= VGroup(ax, f1, f2, f3).scale(0.5).move_to([2, 1.5, 0])

        # Explore-exploit graphic
        explore_exploit_line = Line(start=[-6, 0, 0], end=[-3, 0, 0], color=GRAY_C)
        eps_threshold_line = Line(start=[-6 + 3*EPS, 0.2, 0], end=[-6 + 3*EPS, -0.2, 0], color=PURE_RED)
        text_explore = Text("Explore", font_size=20).move_to([-5.1, 0.25, 0])
        text_exploit = Text("Exploit", font_size=20).move_to([-3.6, 0.25, 0])
        text_epsilon_title = MathTex(r"\varepsilon \text{-greedy algorithm:}", font_size=45).move_to([-4.5, 1.5, 0])
       
        text_epsilon = MathTex(r"\varepsilon = " + str(EPS), color=YELLOW, font_size=30).move_to([-5.9 + 3*EPS, -0.4, 0])
        
        # Start the scene
        self.add(ax, f1, f2, f3, *levers, explore_exploit_line, eps_threshold_line, 
                 text_epsilon_title, text_explore, text_exploit, text_epsilon)

        eps_greedy_bandit = epsilon_greedy_bandit(eps=0.6, K=3)
        for i in range(10):
            arm_index, roll = eps_greedy_bandit.select_arm()

            explore_exploit_line = Line(start=[-6, 0, 0], end=[-6 + 3*roll, 0, 0], color=PURE_RED)
            self.play(GrowFromPoint(explore_exploit_line, [-6,0,0]))
            if roll < EPS:
                self.play(Wiggle(text_explore))
            else:
                self.play(Wiggle(text_exploit))
            
            self.play(FadeOut(explore_exploit_line))

            pull_lever(levers[arm_index])

            current_reward = eps_greedy_bandit.update_rewards(arm_index)
            generate_data(ax, arm_index, current_reward)

        self.wait(2)

        print(f"Arm pull counts: {eps_greedy_bandit.arm_pull_counts}.")
        print(f"Empirical arm rewards: {eps_greedy_bandit.observed_rewards}.")