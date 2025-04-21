import numpy as np
from manim import *

class confusion_matrix_numbers(Scene):
    """
    Confusion matrix with numerical values.
    """
    def construct(self):
        table = Table(
            [['5', '156'],
             ['452', '387']],
            col_labels=[Text('Positive'), Text('Negative')],
            row_labels=[Text('Positive'), Text('Negative')],
        )
        # ent = table.get_entries_without_labels()
        # ent[0].set_color(GREEN)  # True Positive
        # ent[1].set_color(RED)    # False Positive
        # ent[2].set_color(RED)    # False Negative
        # ent[3].set_color(GREEN)  # True Negative

        text_actual = Text('Actual', color=YELLOW_B).next_to(table, UP).shift(RIGHT*1.75)
        text_predicted = Text('Predicted', color=YELLOW_B).rotate(PI/2).next_to(table, LEFT)

        matrix = VGroup(table, text_actual, text_predicted)

        self.add(matrix)

class confusion_matrix(Scene):
    """
    Confusion matrix labelled with TP, FP, TN and FN.
    """
    def construct(self):
        table = Table(
            [['TP', 'FP'],
             ['FN', 'TN']],
            col_labels=[Text('Positive'), Text('Negative')],
            row_labels=[Text('Positive'), Text('Negative')],
        )
        ent = table.get_entries_without_labels()
        ent[0].set_color(GREEN)  # True Positive
        ent[1].set_color(RED)    # False Positive
        ent[2].set_color(RED)    # False Negative
        ent[3].set_color(GREEN)  # True Negative

        text_actual = Text('Actual', color=YELLOW_B).next_to(table, UP).shift(RIGHT*1.75)
        text_predicted = Text('Predicted', color=YELLOW_B).rotate(PI/2).next_to(table, LEFT)

        matrix = VGroup(table, text_actual, text_predicted)

        self.add(matrix)

class performance_metric_formulas(Scene):
    """
    Animation depicting the formuas for three common performance metrics.
    """
    def construct(self):
        table = Table(
            [['TP', 'FP'],
             ['FN', 'TN']],
            col_labels=[Text('Positive'), Text('Negative')],
            row_labels=[Text('Positive'), Text('Negative')],
        )

        text_actual = Text('Actual', color=YELLOW_B).next_to(table, UP).shift(RIGHT*1.75)
        text_predicted = Text('Predicted', color=YELLOW_B).rotate(PI/2).next_to(table, LEFT)

        # rect_22 = BackgroundRectangle(table.get_cell((2,2)), color=GREEN)
        rect_green_22 = table.get_highlighted_cell((2,2), color=GREEN)
        rect_red_22 = table.get_highlighted_cell((2,2), color=RED)
        rect_red_23 = table.get_highlighted_cell((2,3), color=RED)
        rect_red_32 = table.get_highlighted_cell((3,2), color=RED)
        rect_green_33 = table.get_highlighted_cell((3,3), color=GREEN)
        rect_red_33 = table.get_highlighted_cell((3,3), color=RED)

        matrix = VGroup(table, text_actual, text_predicted,
                        rect_green_22, rect_green_33,
                        rect_red_22, rect_red_23, 
                        rect_red_32, rect_red_33).scale(0.5).move_to([-3, 0, 0])

        text_accuracy = Text("Accuracy = ").scale(0.7).move_to([2, 2, 0])
        text_accuracy_math = MathTex(r"TP+TN", r"\over", r"TP+FP+TN+FN").scale(0.7).next_to(text_accuracy, RIGHT)
        text_accuracy_total = VGroup(text_accuracy, text_accuracy_math)

        text_precision = Text("Precision = ").scale(0.7).next_to(text_accuracy, DOWN*8)
        text_precision_math = MathTex(r"TP", r"\over", r"TP+FP").scale(0.7).next_to(text_precision, RIGHT)
        text_precision_total = VGroup(text_precision, text_precision_math)

        text_recall = Text("Recall = ").scale(0.7).next_to(text_precision, DOWN*8)
        text_recall_math = MathTex(r"TP", r"\over", r"TP+FN").scale(0.7).next_to(text_recall, RIGHT)
        text_recall_total = VGroup(text_recall, text_recall_math)

        self.add(table, text_actual, text_predicted, text_accuracy_total, text_precision_total, text_recall_total)
        self.wait()

        # highlighting accuracy
        self.play(Indicate(text_accuracy[:-1], color=YELLOW_B))
        self.play(Create(rect_green_22), Create(rect_green_33), text_accuracy_math[0].animate.set_color(GREEN))
        self.play(Wiggle(text_accuracy_math[0]))

        self.play(FadeOut(rect_green_22), FadeOut(rect_green_33), text_accuracy_math[0].animate.set_color(WHITE))

        self.play(Create(rect_red_22), Create(rect_red_23), 
                  Create(rect_red_32), Create(rect_red_33), 
                  text_accuracy_math[2].animate.set_color(RED))
        self.play(Wiggle(text_accuracy_math[2]))

        self.play(FadeOut(rect_red_22), FadeOut(rect_red_23), 
                  FadeOut(rect_red_32), FadeOut(rect_red_33), text_accuracy_math[2].animate.set_color(WHITE))
        self.wait()
        
        # highlighting precision
        self.play(Indicate(text_precision[:-1], color=YELLOW_B))
        self.play(Create(rect_green_22), text_precision_math[0].animate.set_color(GREEN))
        self.play(Wiggle(text_precision_math[0]))

        self.play(FadeOut(rect_green_22), text_precision_math[0].animate.set_color(WHITE))

        self.play(Create(rect_red_22), Create(rect_red_23),
                  text_precision_math[2].animate.set_color(RED))
        self.play(Wiggle(text_precision_math[2]))

        self.play(FadeOut(rect_red_22), FadeOut(rect_red_23),
                  text_precision_math[2].animate.set_color(WHITE))
        self.wait()

        # highlighting precision
        self.play(Indicate(text_recall[:-1], color=YELLOW_B))
        self.play(Create(rect_green_22), text_recall_math[0].animate.set_color(GREEN))
        self.play(Wiggle(text_recall_math[0]))

        self.play(FadeOut(rect_green_22), text_recall_math[0].animate.set_color(WHITE))

        self.play(Create(rect_red_22), Create(rect_red_32),
                  text_recall_math[2].animate.set_color(RED))
        self.play(Wiggle(text_recall_math[2]))

        self.play(FadeOut(rect_red_22), FadeOut(rect_red_32),
                  text_recall_math[2].animate.set_color(WHITE))
        

        self.wait(2)


class logistic_regression(Scene):
    """
    Animation showing the logistic regression function with a dynamic threshold.
    """
    def construct(self):
        k = ValueTracker(0.5)
        ax = Axes(
            x_range=[-5,5], 
            y_range=[0,1.2], 
            axis_config={"include_numbers": True}).add_coordinates()
        
        sig_graph = ax.plot(lambda x : 1 / (1+np.exp(-x)), color=PURPLE_B)
        sig_text = MathTex(r"\sigma(x) = \frac{1}{1+e^{-x}}").move_to([-4, 2, 0])
        sig_text.set_color(PURPLE_B)
        logit_graph = ax.plot(lambda x : np.log(x/(1-x)), x_range=[0.01,0.99])

        random_numbers = [-4.5, -3.8, -2.3, -0.8, 0.4, 1.2, 2.1, 3.2, 4.8]
        dots = [Dot(ax.c2p(i, sig_graph.underlying_function(i), 0), color=WHITE) for i in random_numbers]

        # dynamic updater for the position of the threshold on the axes
        threshold_dot = always_redraw(
            lambda : Dot(color=YELLOW).move_to(
                ax.c2p(logit_graph.underlying_function(k.get_value()), k.get_value())
            )
        )

        horizontal_line = always_redraw(
            lambda : ax.get_horizontal_line(
                ax.input_to_graph_point(logit_graph.underlying_function(k.get_value()), sig_graph), 
                color=YELLOW)
        )


        self.add(ax, sig_text)
        self.wait()
        self.play(Create(sig_graph))
        self.play(LaggedStart(*[FadeIn(dot) for dot in dots], lag_ratio=.05))
        self.wait()
        self.play(Create(threshold_dot), Flash(threshold_dot, color=YELLOW))
        self.add(horizontal_line)
        self.wait()

        def check_threshold(dots, threshold):
            animations = []
            for dot in dots:
                # need to convert points to coordinates to get the correct y values
                dot_y = ax.p2c(dot.get_center())[1]

                if dot_y > threshold and dot.get_color() != GREEN:
                    animations.append(Flash(dot, color=GREEN))
                    animations.append(dot.animate.set_color(GREEN))
                elif dot_y < threshold and dot.get_color() != RED:
                    animations.append(Flash(dot, color=RED))
                    animations.append(dot.animate.set_color(RED))

            # animate all colour changes at the same time
            self.play(AnimationGroup(*animations, lag_ratio=0))
        
        check_threshold(dots, k.get_value())

        self.wait()
        self.play(k.animate.set_value(0.05), runtime=2)
        self.wait()

        check_threshold(dots, k.get_value())

        self.wait()
        self.play(k.animate.set_value(0.85), runtime=2)
        self.wait()
        check_threshold(dots, k.get_value())

        self.wait(2)

class logistic_regression_data(Scene):
    n = 50
    np.random.seed(8)
    
    def construct(self):

        def generate_coloured_points(n):
            """
            Function to generate n points in the interval (-5,5), with most 
            points on the left being red and most points on the right being green.
            """
            x = np.random.uniform(-5, 5, n)

            # probability of being green based on x value
            k = 0.2  # slope parameter
            green_prob = 1 / (1 + np.exp(-k * x))

            # generate color labels based on probability
            colors = np.where(np.random.rand(n) < green_prob, 'GREEN', 'RED')

            return x, colors
        
        ax = Axes(
            x_range=[-5,5], 
            y_range=[0,1.2], 
            )
        
        sig_graph = ax.plot(lambda x : 1 / (1+np.exp(-x)), color=PURPLE_B)

        self.add(sig_graph)
        
        x_vals, colors = generate_coloured_points(self.n)
        dots = [Dot(ax.c2p(x, sig_graph.underlying_function(x)), color=color)#, radius=0.04)
                for x, color in zip(x_vals, colors)]
        
        self.add(ax, *dots)


class roc(Scene):
    """
    Animation depicting how the ROC curve is constructed for the setup in the
    `logistic_regression_data` scene.
    """    
    def construct(self):

        # components of the first plot:
        def generate_coloured_points(n, k):
            """
            Function to generate n points in the interval (-1,1), with most 
            points on the left being red and most points on the right being green.

            k is the slope parameter
            """
            x = np.random.uniform(0, 1, n)

            # probability of being green based on x value
            threshold = 1/(1 + np.exp(-k * (x-0.5)))

            # generate color labels based on probability
            # Gaussian noise added to mix up colours
            colours = np.where(x+np.random.normal(scale=0.2, size=n) > threshold, 'GREEN', 'RED')

            return x, colours
        
        ax = Axes(
            x_range=[0,1,0.25], 
            y_range=[0,3],
            x_axis_config={'include_numbers': True},
            y_axis_config={'include_ticks': False, 'include_numbers': False})
        
        x_label = ax.get_x_axis_label(Tex("Threshold")).shift(DOWN*1.5)
        
        n = 50
        np.random.seed(8)

        spread_y_vals = np.linspace(0, 3, n)

        x_vals, colours = generate_coloured_points(n, 2)
        dots = [
            Dot(ax.c2p(x, y), color=color)
            for x, color, y in zip(x_vals, colours, spread_y_vals)
        ]

        k = ValueTracker(0)

        threshold_line = always_redraw(
            lambda : Line(ax.c2p(k.get_value(), -0.25), ax.c2p(k.get_value(), 3), color=YELLOW)
        )

        # components of the second plot:
        def performance_metrics(threshold_value, x_vals, colours):
            """
            Calculate the current performance metrics with respect to the given threshold value.
            """
            def fpr(FP, TN):
                """
                Function to calculate the False Positive Rate (FPR) of the given classifier.
                """
                return FP / (FP + TN)
            
            def tpr(TP, FN):
                """
                Function to calculate the True Positive Rate (TPR) of the given classifier.
                """
                return TP / (TP + FN)
            
            tp, fp, fn, tn = 0, 0, 0, 0

            for x, colour in zip(x_vals, colours):
                if colour == 'GREEN':
                    if x > threshold_value:
                        # actual colour is green and prediction is also green
                        tp += 1
                    else:
                        # actual colour is green but prediction is red
                        fn += 1
                elif colour == 'RED':
                    if x > threshold_value:
                        # actual colour is red but prediction is green
                        fp += 1
                    else:
                        # actual colour is red and prediction is also red
                        tn += 1

            return Dot(ax2.c2p(fpr(fp, tn), tpr(tp, fn)), color=YELLOW)
            # return fpr(fp, tn), tpr(tp, fn)
        
        ax2 = Axes(
            x_range=[0, 1.1, 0.5], 
            y_range=[0, 1.1, 0.5],
            x_length=10,
            y_length=10,
            axis_config={"include_numbers": True})
        
        # ax2_labels = ax2.get_axis_labels(Tex("FPR"), Tex("TPR"))
        x_label_ax2 = ax2.get_x_axis_label(Tex("FPR")).next_to(ax2.get_x_axis(), RIGHT, buff=0.5)
        y_label_ax2 = ax2.get_y_axis_label(Tex("TPR")).next_to(ax2.get_y_axis(), UP, buff=0.5)

        roc_curve = always_redraw(lambda : performance_metrics(k.get_value(), x_vals, colours))
        roc_curve_trace = TracedPath(roc_curve.get_center, stroke_width=2, stroke_color=YELLOW)
        random_guesser = DashedLine(ax2.c2p(0, 0), ax2.c2p(1, 1), color=TEAL_B)
        random_guesser_label = Tex("Random Guesser", color=TEAL_B).move_to([6, 3, 0]).scale(0.5)

        # scaling down and moving plots
        dot_plot = VGroup(ax, x_label, *dots, threshold_line).scale(0.5).move_to([-3.5, 0, 0])
        data_plot = VGroup(ax2, x_label_ax2, y_label_ax2, roc_curve, random_guesser).scale(0.5).move_to([3.5, 0, 0])

        self.add(ax, x_label, *dots, ax2, x_label_ax2, y_label_ax2, roc_curve_trace, random_guesser, random_guesser_label)
        self.wait()

        self.play(Create(threshold_line))
        self.play(Create(roc_curve))
        self.wait()
        self.play(k.animate.set_value(1), rate_func=linear, run_time=7)
        self.wait()
        

        self.wait(2)


class roc_multiple_classifiers(Scene):
    """
    Similar to the previous scene, but with two classifiers.
    (Yes I'm aware that there was probably a better way to code this rather than copy-pasting the code lol)
    """
    def construct(self):
        n = 50
        np.random.seed(8)

        def generate_coloured_points(n, k):
            """
            Function to generate n points in the interval (-1,1), with most 
            points on the left being red and most points on the right being green.

            k is the slope parameter
            """
            x = np.random.uniform(0, 1, n)

            # probability of being green based on x value
            threshold = 1/(1 + np.exp(-k * (x-0.5)))

            # generate color labels based on probability
            # Gaussian noise added to mix up colours
            colours = np.where(x+np.random.normal(scale=0.2, size=n) > threshold, 'GREEN', 'RED')

            return x, colours
        
        # plot for first classifier
        ax = Axes(
            x_range=[0,1,0.25], 
            y_range=[0,3],
            x_axis_config={'include_numbers': True},
            y_axis_config={'include_ticks': False, 'include_numbers': False})
        
        x_label = ax.get_x_axis_label(Tex("Model 1 Threshold")).shift(DOWN*1.5)
        
        

        spread_y_vals = np.linspace(0, 3, n)

        x_vals, colours = generate_coloured_points(n, 2)
        dots = [
            Dot(ax.c2p(x, y), color=color)
            for x, color, y in zip(x_vals, colours, spread_y_vals)
        ]

        k = ValueTracker(0)

        threshold_line = always_redraw(
            lambda : Line(ax.c2p(k.get_value(), 0), ax.c2p(k.get_value(), 3), color=YELLOW)
        )

        # plot for second classifier
        ax_new = Axes(
            x_range=[0,1,0.25], 
            y_range=[0,3],
            x_axis_config={'include_numbers': True},
            y_axis_config={'include_ticks': False, 'include_numbers': False})
        
        x_label_new = ax_new.get_x_axis_label(Tex("Model 2 Threshold")).shift(DOWN*1.5)

        spread_y_vals_new = np.linspace(0, 3, n)

        x_vals_new, colours_new = generate_coloured_points(n, 0.5)
        dots_new = [
            Dot(ax_new.c2p(x, y), color=colour)
            for x, colour, y in zip(x_vals_new, colours_new, spread_y_vals_new)
        ]

        k_new = ValueTracker(0)

        threshold_line_new = always_redraw(
            lambda : Line(ax_new.c2p(k_new.get_value(), 0), ax_new.c2p(k_new.get_value(), 3), color=ORANGE)
        )

        def performance_metrics(threshold_value, x_vals, colours, point_colour=YELLOW):
            """
            Calculate the current performance metrics with respect to the given threshold value.
            """
            def fpr(FP, TN):
                """
                Function to calculate the False Positive Rate (FPR) of the given classifier.
                """
                return FP / (FP + TN)
            
            def tpr(TP, FN):
                """
                Function to calculate the True Positive Rate (TPR) of the given classifier.
                """
                return TP / (TP + FN)
            
            tp, fp, fn, tn = 0, 0, 0, 0

            for x, colour in zip(x_vals, colours):
                if colour == 'GREEN':
                    if x > threshold_value:
                        # actual colour is green and prediction is also green
                        tp += 1
                    else:
                        # actual colour is green but prediction is red
                        fn += 1
                elif colour == 'RED':
                    if x > threshold_value:
                        # actual colour is red but prediction is green
                        fp += 1
                    else:
                        # actual colour is red and prediction is also red
                        tn += 1

            return Dot(ax2.c2p(fpr(fp, tn), tpr(tp, fn)), color=point_colour)
        
        # plot for ROC curves
        ax2 = Axes(
            x_range=[0, 1.1, 0.5], 
            y_range=[0, 1.1, 0.5],
            x_length=10,
            y_length=10,
            axis_config={"include_numbers": True})
        
        # ax2_labels = ax2.get_axis_labels(Tex("FPR"), Tex("TPR"))
        x_label_ax2 = ax2.get_x_axis_label(Tex("FPR")).next_to(ax2.get_x_axis(), RIGHT, buff=0.5)
        y_label_ax2 = ax2.get_y_axis_label(Tex("TPR")).next_to(ax2.get_y_axis(), UP, buff=0.5)

        random_guesser = DashedLine(ax2.c2p(0, 0), ax2.c2p(1, 1), color=TEAL_B)
        random_guesser_label = Tex("Random Guesser", color=TEAL_B).move_to([6, 3, 0]).scale(0.5)

        # first ROC curve
        roc_curve = always_redraw(lambda : performance_metrics(k.get_value(), x_vals, colours))
        roc_curve_trace = TracedPath(roc_curve.get_center, stroke_width=2, stroke_color=YELLOW)

        # second ROC curve
        roc_curve_new = always_redraw(lambda : performance_metrics(k_new.get_value(), x_vals_new, colours_new, ORANGE))
        roc_curve_trace_new = TracedPath(roc_curve_new.get_center, stroke_width=2, stroke_color=ORANGE)

        # scaling down and moving plots
        classifier1_plot = VGroup(ax, x_label, *dots, threshold_line).scale(0.4).move_to([-3.5, 1.75, 0])
        classifier2_plot = VGroup(ax_new, x_label_new, *dots_new, threshold_line_new).scale(0.4).move_to([-3.5, -1.75, 0])
        data_plot = VGroup(ax2, x_label_ax2, y_label_ax2, roc_curve, roc_curve_new, random_guesser).scale(0.5).move_to([3.5, 0, 0])

        self.add(ax, x_label, *dots, 
                 ax_new, x_label_new, *dots_new,
                 ax2, x_label_ax2, y_label_ax2, random_guesser, random_guesser_label,
                 roc_curve_trace, roc_curve_trace_new)
        self.wait()

        # plotting first ROC curve
        self.play(Create(threshold_line))
        self.play(Create(roc_curve))
        self.play(Flash(roc_curve))
        self.wait()
        self.play(k.animate.set_value(1), rate_func=linear, run_time=7)
        self.wait(2)

        # plotting second ROC curve
        self.play(Create(threshold_line_new))
        self.play(Create(roc_curve_new))
        self.play(Flash(roc_curve_new, color=ORANGE))
        self.wait()
        self.play(k_new.animate.set_value(1), rate_func=linear, run_time=7)

        self.wait(2)