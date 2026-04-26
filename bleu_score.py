from manim import *

class brevity_penalty_plot(Scene):
    """
    Plots the brevity penalty curve, which is used in the BLEU score to penalize short translations.
    """
    def construct(self):
        self.camera.background_color = '#161718'

        ax = Axes(
            x_range=[-2, 6, 1],
            y_range=[0, 1.5, 0.5],
        ).add_coordinates()

        labels = ax.get_axis_labels(x_label='r/c', y_label='BP')

        bp_curve = ax.plot(
            lambda x: np.exp(1-x) if x > 1 else 1,
            x_range=[-2, 6],
            use_smoothing=False,
            color='#5ce1e6'
        )

        self.add(ax, labels, bp_curve)