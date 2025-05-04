import random
import numpy as np
from sklearn.datasets import make_moons
from manim import *
from k_means import KMeans
from dbscan import dbscan

class moons(Scene):
    """
    An animation depicting how K-Means fails on the make_moons dataset.
    """
    def construct(self):
        ax = Axes(x_range=[-3,3,1])

        # set seeds for reproducibility
        np.random.seed(8)
        random.seed(8)

        X, y = make_moons(noise=0.05, random_state=8)

        # translating and enlargening the data to make it fit better on ax
        cluster1 = [-1+2*np.array([X[i][0], X[i][1], 0]) for i in range(len(y)) if y[i] == 0]
        cluster2 = [-1+2*np.array([X[i][0], X[i][1], 0]) for i in range(len(y)) if y[i] == 1]

        data_raw = [*cluster1, *cluster2]

        colours = [RED_B, GREEN_B, BLUE_B, GOLD_B, PURPLE_B, TEAL_B]

        moon1 = [Cross().set_color(WHITE).set_opacity(0.5).scale(0.05).move_to(point) for point in cluster1]
        moon2 = [Cross().set_color(WHITE).set_opacity(0.5).scale(0.05).move_to(point) for point in cluster2]
        data = [*moon1, *moon2]

        # K-Means stuff
        K=2
        MAX_ITER = 5

        centroid_colours = [RED_E, GREEN_E, BLUE_E, GOLD_E, PURPLE_E, TEAL_E]

        kmeans = KMeans(k=K, max_iter=MAX_ITER)
        kmeans.fit(data_raw)

        labels = kmeans.labels
        centroids_raw = kmeans.centroids

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

        iteration_text = Text("Iteration", font_size=30).move_to([-5,3,0])
        iteration_text_numbers = [MathTex(f'{i}').next_to(iteration_text, RIGHT, buff=0.2) for i in range(len(centroids_history))]

        self.add(ax, *moon1, *moon2, *centroids_final, *centroid_traced_paths, iteration_text)

        self.play(Write(iteration_text_numbers[0]))
        self.play(LaggedStart(*[(FadeIn(dot), Flash(dot, color=dot.get_color())) for dot in centroids_history[0]], lag_ratio=0.5))

        for i in range(0, len(centroids_history)-1):
            self.play(ReplacementTransform(iteration_text_numbers[i], iteration_text_numbers[i+1]))
            self.play(Indicate(iteration_text_numbers[i+1]))
            # self.play(Indicate(colour_update_text))

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
            # self.play(Indicate(centroid_update_text))
            self.play(*[Transform(centroids_history[0][j], centroids_history[i+1][j]) for j in range(K)], run_time=2)
            self.wait()
        
        self.wait()

class dbscan_animated(Scene):
    """
    Animation depicting how the DBSCAN algorithm works on a dataset.
    """
    def construct(self):
        ax = Axes(x_range=[-6,6,1], y_range=[-3,3,1])
        colours = [RED_B, GREEN_B, BLUE_B, GOLD_B, PURPLE_B, TEAL_B]

        # set seeds for reproducibility
        np.random.seed(8)
        random.seed(8)

        X, y = make_moons(n_samples=200, noise=0.05, random_state=8)

        # translating and enlargening the data to make it fit better on ax
        cluster1 = [-1+2*np.array([X[i][0], X[i][1], 0]) for i in range(len(y)) if y[i] == 0]
        cluster2 = [-1+2*np.array([X[i][0], X[i][1], 0]) for i in range(len(y)) if y[i] == 1]

        data_raw = [*cluster1, *cluster2]

        moon1 = [Dot(ax.c2p(point[0], point[1], 0)).set_opacity(0.5) for point in cluster1]
        moon2 = [Dot(ax.c2p(point[0], point[1], 0)).set_opacity(0.5) for point in cluster2]
        data = [*moon1, *moon2]

        EPS = 0.2
        MINPOINTS = 4

        dbs = dbscan(eps=EPS, minPoints=MINPOINTS)
        dbs.fit(X)

        cluster_text = Text("Cluster", font_size=30).move_to(np.array([-5,3,0]))
        cluster_text_numbers = [MathTex('0').next_to(cluster_text, RIGHT)]
        extra_numbers = [MathTex(f'{cluster_num}').next_to(cluster_text, RIGHT) for cluster_num in list(dbs.history.keys())]

        cluster_text_numbers.extend(extra_numbers)

        self.add(ax, cluster_text, *data)

        self.play(Write(cluster_text_numbers[0]))
        self.wait()

        for cluster_num, point_index_list in dbs.history.items():
            # reset colour animating list for next pass
            colour_animations = []

            # update which cluster we're currently expanding
            self.play(ReplacementTransform(cluster_text_numbers[cluster_num-1], cluster_text_numbers[cluster_num]))
            self.play(Indicate(cluster_text_numbers[cluster_num]))

            for point_index in point_index_list:
                colour_animations.append(data[point_index].animate.set_color(colours[cluster_num]))

            # animate the expansion of the current cluster
            self.play(LaggedStart(*colour_animations), lag_ratio=0.1)

            self.wait()

        self.wait()