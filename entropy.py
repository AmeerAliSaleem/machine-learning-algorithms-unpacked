import math
import numpy as np
import pandas as pd
from manim import *

class self_information(Scene):
    def construct(self):
        self.camera.background_color = '#111111'

        log_tol = 0.001
        y_max = -1*np.log2(log_tol)

        ax = Axes(
            x_range=[0, 1.1, 0.5],
            y_range=[-1,y_max,1],
            tips=False,
            axis_config={"include_numbers": True}
        )

        labels = ax.get_axis_labels(x_label="p", y_label="I(p)")

        information_plot = ax.plot(
            lambda x : -1*np.log2(x),
            x_range=[log_tol, 1, 0.001],
            use_smoothing=True,
            color='#E8C547'
        ).set_stroke(width=7)


        self.add(ax, labels, information_plot)


class entropy_plot(Scene):
    def construct(self):
        self.camera.background_color = '#111111'
        p = ValueTracker(0)

        ax = Axes(
                x_range=[0, 1, 0.5],
                y_range=[0,1,0.5],
                x_length=6,
                y_length=4,
                tips=False,
                axis_config={"include_numbers": True}
            ).move_to([3,-0.2,0])
        
        labels = ax.get_axis_labels(x_label="p", y_label="H(p)")

        def get_bar_chart():
            return BarChart(
                values=[p.get_value(), 1-p.get_value()],
                # bar_names=["p", "1-p"],
                bar_names=[],
                y_range=[0, 1, 0.5],
                x_length=4,
                y_length=4,
                x_axis_config={"font_size": 36},
                bar_colors=['#d81e5b', '#6290c3']
            ).move_to([-4,0,0])
        
        def get_labels(bar_chart):
            return VGroup(
                MathTex(r"p").next_to(bar_chart.bars[0], DOWN),      # Label for first bar
                MathTex(r"1 - p").next_to(bar_chart.bars[1], DOWN)   # Label for second bar
            )

        bar_chart = always_redraw(get_bar_chart)
        bar_chart_labels = always_redraw(lambda: get_labels(bar_chart))
        
        entropy = ax.plot(
            lambda x : -x*np.log2(x) - (1-x)*np.log2(1-x) if (0<x and x<1) else 0,
            x_range=[0,1],
            color='#E8C547'
        )

        dot = always_redraw(
            lambda : Dot(color='#d81e5b').move_to(
                ax.c2p(p.get_value(), entropy.underlying_function(p.get_value()))
            )
        )

        self.play(FadeIn(ax), FadeIn(labels), DrawBorderThenFill(bar_chart), FadeIn(bar_chart_labels))
        self.wait(1.5)
        self.play(Create(entropy))
        self.play(Create(dot))
        self.wait()
        self.play(p.animate.set_value(0.5),  run_time=4)
        self.wait()
        self.play(p.animate.set_value(1),  run_time=4)

        self.wait(2)