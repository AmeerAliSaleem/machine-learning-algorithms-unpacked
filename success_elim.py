import numpy as np
from scipy.stats import norm
from manim import *

class successive_elimination(Scene):
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
        
        def pull_levers(lever_group):
            """
            Animates the pulling of ALL levers in the input VGroup.

            Returns a list of the animations for the pulling of all the levers.
            """
            animations = []
            for lever in lever_group:
                handle = VGroup(lever[0], lever[1])

                animations.append(handle.animate.shift(RIGHT))

            return animations
        
        def return_levers(lever_group):
            """
            Resets all the pulled levers in the lever group to their neutral positions.

            Returns a list of the animations.
            """
            animations = []
            for lever in lever_group:
                handle = VGroup(lever[0], lever[1])

                animations.append(handle.animate.shift(LEFT))

            return animations
        
        def generate_data(ax, arm_set=[0,1,2,3], num_points=1):
            """
            Generates data for the set of input arms.

            Parameters
            ----------
            ax : Axes
                The set of axes to plot the data on
            arm_set : list
                The indices of the arms currently in play.

            Returns
            ----------
            full_dots_list : list
                A list containing the Dot objects corresponding to all the generated data points.
            data_list : list
                A list containing four lists, each of which stores the rewards from this set of data generation.
            """

            full_dots_list = []
            data_list = [[], [], [], []]
            for i in range(num_points):
                dots = []
                for arm_index in arm_set:
                    current_reward = gaussians[arm_index].rvs()
                    dots.append(Dot(ax.c2p(current_reward, gaussians[arm_index].pdf(current_reward), 0), 
                        color=colours[arm_index], fill_opacity=0.7))
                    data_list[arm_index].append(current_reward)
                
                full_dots_list.extend(dots)
            return full_dots_list, data_list
        
        def empirical_averages(rewards_for_each_arm, arm_indices):
            """
            Computes and returns the new empirical rewards for the arms still in play.

            Parameters
            ----------
            rewards_for_each_arm : list
                The list of lists containing the observed rewards for each arm.
            arm_indices : list
                The list of indices of arms still in play.

            Returns
            ----------
            empirical_avg_dict : dict
                The dictionary whose key-value pairs correspond to each arm index and the corresponding
                empirical mean reward.
            """

            # empirical_avg_list = [[], [], [], []]
            empirical_avg_dict = {}

            for arm_index in arm_indices:
                empirical_avg_dict[arm_index] = np.mean(np.array(rewards_for_each_arm[arm_index]))

            return empirical_avg_dict
        
        def empirical_average_plots(emp_avg_dict):
            """
            Takes the dictionary of current empirical averages and returns the animations for the 
            corresponding confidence interval plots.
            """
            plots = []
            for key, val in emp_avg_dict.items():
                # plots.append(Dot(point=[key-3, val/10, 0], color=colours[key]))
                plots.append(Dot(ax_ci.c2p(key, val, 0), color=colours[key]))
            
            return plots
        
        def confidence_intervals(emp_dots, delta=1, t=1):
            """
            Creates the confidence intervals on the input empirical Dot objects.

            Parameters
            ----------
            emp_dots : list
                The list of the Dot objects plotted on ax_ci.
            delta : float
                The value of delta to use for the confidence interval.
                (Look for the appearance of the variables in the confidence interval 
                derived from Hoeffding's inequality.)
            t : float
                The value of t to use for the confidence interval.
                (Look for the appearance of the variables in the confidence interval 
                derived from Hoeffding's inequality.)

            Returns
            ----------
            confidence_arrows : list
                A list of the confidence interval lines to draw on each Dot.
                (The idea is to animate these line to grow from their centres.)
            """
            confidence_arrows = []

            for index, dot in enumerate(emp_dots):
                coords = ax_ci.p2c(dot.get_center())
                print(f"Coords are {coords}.")
                arrow = DoubleArrow(
                    start=ax_ci.c2p(coords[0], coords[1] - 2, 0), 
                    end=ax_ci.c2p(coords[0], coords[1] + 2, 0), 
                    color=dot.get_color(),
                    # tip_shape_start=ArrowCircleTip,
                    # tip_shape_end=ArrowCircleTip,
                    tip_length=0.1,
                    stroke_width=3
                    )
                
                confidence_arrows.append(arrow)

            return confidence_arrows

        lever1 = generate_lever(handle_colour=BLUE).move_to([-1, -2, 0])
        lever2 = generate_lever(handle_colour=GREEN).move_to([1, -2, 0])
        lever3 = generate_lever(handle_colour=RED).move_to([3, -2, 0])
        lever4 = generate_lever(handle_colour=YELLOW).move_to([5, -2, 0])

        levers = VGroup(lever1, lever2, lever3, lever4)

        np.random.seed(8)

        ax = Axes(
            x_range=[-1, 12, 1],
            y_range=[0, 1.01, 0.5],
            axis_config={'tip_shape': StealthTip, 'include_numbers': True}
            )
        
        ax_ci = Axes(
            x_range=[0, 4, 1],
            y_range = [0, 10, 1],
            x_length = 4,
            y_length = 5
        ).move_to([-4, 0, 0])
        
        colours = [BLUE_B, GREEN_B, RED_B, YELLOW_B]

        gaussian1 = norm(loc=2, scale=0.8)
        gaussian2 = norm(loc=5, scale=1.2)
        gaussian3 = norm(loc=7, scale=0.5)
        gaussian4 = norm(loc=8, scale=0.9)

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

        f4 = always_redraw(
            lambda: ax.plot(
                lambda x: gaussian4.pdf(x),
                color=colours[3]
            )
        )

        gaussians = [gaussian1, gaussian2, gaussian3, gaussian4]
        gaussian_curves = [f1, f2, f3, f4]

        # global count for the rewards drawn from each arm
        rewards_for_each_arm = [[], [], [], []]

        vgroup_plot= VGroup(ax, f1, f2, f3, f4).scale(0.6).move_to([2, 1.25, 0])

        self.add(*vgroup_plot, *levers)#, ax_ci)

        lever_pull_animations = pull_levers(levers)
        lever_pull_animations_final = [anim for anim in lever_pull_animations]
        self.play(*lever_pull_animations_final, rate_func=rush_into, run_time=0.5)

        lever_return_animations = return_levers(levers)
        lever_return_animations_final = [anim for anim in lever_return_animations]
        self.play(*lever_return_animations_final, run_time=0.5)

        new_points, new_data = generate_data(ax, arm_set=[0,1,2,3], num_points=5)
        self.play(*[(FadeIn(dot), Flash(dot, color=dot.get_color())) for dot in new_points])

        # store results of previous generate_data in the full rewards list(s)
        for i, data in enumerate(new_data):
            print(f"Arm {i}: rewards are {data}.")
            rewards_for_each_arm[i].extend(data)

        print(rewards_for_each_arm)

        emp_avg_dict = empirical_averages(rewards_for_each_arm, arm_indices=[0,1,2,3])
        print(emp_avg_dict)

        emp_dots = empirical_average_plots(emp_avg_dict)
        self.play(*[FadeIn(dot) for dot in emp_dots])

        confidence_arrows = confidence_intervals(emp_dots)
        self.play(*[GrowFromPoint(arrow, arrow.get_center()) for arrow in confidence_arrows])

        self.wait(2)