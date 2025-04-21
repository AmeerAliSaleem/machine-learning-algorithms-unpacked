import random
import numpy as np
import pandas as pd
from manim import *

# import the data
df = pd.read_csv('knn_points.csv')
points = [np.array([df['Points_x'].iloc[i], df['Points_y'].iloc[i], 0]) for i in range(len(df))]
colours = [df['Colour'].iloc[i] for i in range(len(df))]

class knn(Scene):
    def construct(self):
        ax = Axes(
            x_range=[-6,6],
            y_range=[-3,3],
            axis_config={"include_numbers": True},
        )

        self.add(ax)
        self.wait()

        dots = [Dot(ax.c2p(points[i][0], points[i][1], points[i][2]), color=colours[i]) for i in range(len(df))]

        # uncomment the line below for n randomly generated points instead
        # dots = self.random_coloured_dots(ax=ax, n=10, colour_list=['RED', 'GREEN'])
        
        self.play(LaggedStart(*[FadeIn(dot) for dot in dots], lag_ratio=.05))

        self.wait()

        self.classify_new_dot(axes=ax, ref_dot=np.array([3, 1, 0]), all_current_points=dots, k=3)
        self.classify_new_dot(axes=ax, ref_dot=np.array([-2, 0.5, 0]), all_current_points=dots, k=3)
        self.classify_new_dot(axes=ax, ref_dot=np.array([1, -2, 0]), all_current_points=dots, k=3)

    def classify_new_dot(self, axes, ref_dot, all_current_points, k):
        """
        Applies the knn algorithm to the input data point.
        """
        new_point = Dot(axes.c2p(ref_dot[0], ref_dot[1], ref_dot[2]), color=WHITE)
        self.play(GrowFromCenter(new_point))
        self.wait(1)

        distances = [
            (np.linalg.norm(new_point.get_center() - dot.get_center()), dot) 
            for dot in all_current_points if not np.array_equal(dot.get_center(), new_point.get_center())
        ]
        distances.sort(key=lambda x: x[0])

        k_nearest_neighbours = [distances[i][1] for i in range(k)]

        # draw arrows between the new point and its k nearest neighbours
        arrows = [
            Arrow(start=new_point, end=point, buff=0.2, color=YELLOW)
            for point in k_nearest_neighbours
        ]
        
        self.play(*[GrowArrow(arrow) for arrow in arrows])
        self.wait(1)

        # decide what colour the new point ought to take
        red_count = sum(1 for dot in k_nearest_neighbours if dot.color == RED)
        green_count = sum(1 for dot in k_nearest_neighbours if dot.color == GREEN)

        # majority voting favours GREEN
        if red_count > green_count:
            self.play(new_point.animate.set_color(RED))
        else:
            self.play(new_point.animate.set_color(GREEN))
        
        all_current_points.append(new_point)

        self.wait(2)
        self.play(*[FadeOut(arrow) for arrow in arrows])
        self.wait()
    
    def random_coloured_dots(self, ax, n, colour_list):
        """
        Use this class if you want to generate random coloured dots.
        """
        dots = []

        for _ in range(n):
            x = random.uniform(-6, 6)
            y = random.uniform(-3, 3)

            color = random.choice(colour_list)
            
            # Create the dot at the random position with the random color
            dot = Dot(ax.c2p(x, y, 0), color=color)
            dots.append(dot)
        
        return dots