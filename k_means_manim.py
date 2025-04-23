import numpy as np
from manim import *
from k_means import KMeans

class kMeansVisual(Scene):
    """
    Visualisation of the K-Means algorithm in Manim.
    """
    def construct(self):
        # set randomness seed for reproducibility
        np.random.seed(8)

        # number of clusters to use in algorithm (K=4, NUM_OF_POINTS=100 is a good fit)
        K = 4
        MAX_ITER = 5
        NUM_OF_POINTS = 100

        ax = Axes(
            x_range=[-6, 6, 1],
            y_range=[-3, 3, 1],
            axis_config={'tip_shape': StealthTip}
            )
        
        # colour palette currently supports up to K=6 clusters. If you want more clusters, add more colours to the two palettes below:
        colours = [RED_B, GREEN_B, BLUE_B, GOLD_B, PURPLE_B, TEAL_B]
        centroid_colours = [RED_E, GREEN_E, BLUE_E, GOLD_E, PURPLE_E, TEAL_E]

        x_vals = np.random.uniform(-6, 6, NUM_OF_POINTS)
        y_vals = np.random.uniform(-3, 3, NUM_OF_POINTS)

        data_raw = [np.array([x_vals[i], y_vals[i], 0]) for i in range(NUM_OF_POINTS)]

        data = [Cross().set_color(WHITE).set_opacity(0.5).scale(0.1).move_to(point) for point in data_raw]

        # apply the K-Means algorithm
        kmeans = KMeans(k=K, max_iter=MAX_ITER)
        kmeans.fit(data_raw)

        labels = kmeans.labels
        centroids_raw = kmeans.centroids

        print(f"Number of iterations: {len(centroids_raw) - 1}")

        # final centroid locations: good to see where centroids end up
        centroids_final = [Dot(centroid, color=GRAY, radius=0.12) for centroid in centroids_raw[-1]]

        # storing history of centroid allocations in a list of Dots
        centroids_history = [
            [Dot(centroid, radius=0.12) for centroid in current_step] for current_step in centroids_raw
        ]

        # change centroid colours according to colour palette
        for centroid_set in centroids_history:
            _ = [centroid.set_color(centroid_colours[i]) for i, centroid in enumerate(centroid_set)]

        centroid_traced_paths = [TracedPath(centroid.get_center, stroke_color=centroid.get_color(), dissipating_time=0.8) for centroid in centroids_history[0]]

        # group and move all the plotted objects
        diagram_vgroup = VGroup(ax, *data, *centroids_history, *centroid_traced_paths, *centroids_final).scale(0.7).shift(RIGHT*2.5)

        # text elements
        iteration_text = Text("Iteration", font_size=30)
        colour_update_text = MathTex(fr"c_i := \arg \min_j \| \boldsymbol x_i - \boldsymbol \mu_j \|_2")
        centroid_update_text = MathTex(r"\boldsymbol{\mu}_j := \frac{\sum_{i=1}^n \mathbf{1}_{\{c_i=j\}} \boldsymbol{x}_i}{\sum_{i=1}^n \mathbf{1}_{\{c_i=j\}}}")

        text_vgroup = VGroup(iteration_text, colour_update_text, centroid_update_text).scale(0.7).shift(LEFT*6, UP*1.5)
        text_vgroup.arrange(DOWN, center=False, aligned_edge=LEFT, buff=1)

        iteration_text_numbers = [MathTex(f'{i}').scale(0.7).next_to(iteration_text, RIGHT, buff=0.2) for i in range(len(centroids_history))]

        self.add(ax, *data, *centroids_final, *centroid_traced_paths, iteration_text, colour_update_text, centroid_update_text, centroid_update_text)

        self.wait()

        # algorithm initialisation
        self.play(Write(iteration_text_numbers[0]))
        self.play(LaggedStart(*[(FadeIn(dot), Flash(dot, color=dot.get_color())) for dot in centroids_history[0]], lag_ratio=0.5))

        for i in range(0, len(centroids_history)-1):
            self.play(ReplacementTransform(iteration_text_numbers[i], iteration_text_numbers[i+1]))
            self.play(Indicate(iteration_text_numbers[i+1]))
            self.play(Indicate(colour_update_text))

            if i == 0:
                self.play(*[Circumscribe(data_point, Circle) for data_point in data])
                self.play(*[data_point.animate.set_color(colours[labels[0][j]]) for j, data_point in enumerate(data)])
            else:
                previous_labels = labels[i-1]
                current_labels = labels[i]

                data_points_to_update = []
                colours_to_update = []

                # store animations for data points that have changed colour from the previous step
                for j, data_point in enumerate(data):
                    if previous_labels[j] != current_labels[j]:
                        data_points_to_update.append(Circumscribe(data_point, Circle))
                        colours_to_update.append(data_point.animate.set_color(colours[current_labels[j]]))

                # animate colour changes (if any)
                if colours_to_update:
                    self.play(*data_points_to_update)
                    self.play(*colours_to_update)
                
                self.wait()

            # animate centroid position changes
            self.play(Indicate(centroid_update_text))
            self.play(*[Transform(centroids_history[0][j], centroids_history[i+1][j]) for j in range(K)], run_time=2)
            self.wait()
        
        self.wait()


class kMeansVisual3D(ThreeDScene):
    """
    3D visualisation of K-Means if I have time for it.
    """
    def construct(self):

        # zoom out so we see the axes
        self.set_camera_orientation(zoom=0.5)
        # self.set_camera_orientation(phi=60*DEGREES, theta=-45*DEGREES)

        ax = ThreeDAxes()

        self.add(ax)