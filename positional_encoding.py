import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from manim import *

def pos_enc_even(pos, d_model, x):
    return np.sin(pos / (10000**(2*x / d_model)))

def pos_enc_odd(pos, d_model, x):
    return np.cos(pos / (10000**(2*x / d_model)))

class pos_enc_even_plot(Scene):
    """
    Animation for the even dimensions of the positional encoding function, which uses sine.
    """
    def construct(self):
        self.camera.background_color = '#161718'
        ax = Axes(
            x_range=[0, 500, 100],
            x_length = 15,
            y_range=[-1, 1],
            y_axis_config = {'numbers_to_include': [-1, 0, 1]},
            tips = False
        ).add_coordinates().scale(0.8).move_to([0, -1, 0])

        pos = ValueTracker(1)
        d_model = 512

        formula = MathTex(r"PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)", color='#ffd459').scale(0.8).move_to([3,3,0])
        pos_text = always_redraw(
            lambda: MathTex(r"pos = " + f"{pos.get_value():.2f}", color='#ffd459').move_to([-3,3,0])
        )

        pos_enc_even_curve = always_redraw(
                lambda: ax.plot(
                lambda x: pos_enc_even(pos.get_value(), d_model, x),
                color='#ffd459'
            )
        )

        imprints = []

        def leave_imprint():
            imprint = ax.plot(
                lambda x: pos_enc_even(pos.get_value(), d_model, x),
                color='#ffd459',
                stroke_opacity=0.5
            )
            self.add(imprint)
            imprints.append(imprint)
            
            # Fade all previous imprints
            for old_imprint in imprints[:-1]:
                new_opacity = old_imprint.get_stroke_opacity() * 0.4
                old_imprint.set_stroke(opacity=new_opacity)

        self.add(ax, pos_enc_even_curve, formula, pos_text)
        self.wait(2)

        # Leave imprints at regular intervals
        leave_imprint()
        self.play(
            pos.animate.set_value(pos.get_value() + 79/2),
            rate_func=rate_functions.ease_in_quad,
            run_time=5
        )
        leave_imprint()
        self.play(
            pos.animate.set_value(pos.get_value() + 79/2),
            rate_func=rate_functions.ease_out_quad,
            run_time=5
        )
        self.wait(2)


class pos_enc_odd_plot(Scene):
    """
    Animation for the odd dimensions of the positional encoding function, which uses cosine.
    """
    def construct(self):
        self.camera.background_color = '#161718'
        ax = Axes(
            x_range=[0, 500, 100],
            x_length = 15,
            y_range=[-1, 1],
            y_axis_config = {'numbers_to_include': [-1, 0, 1]},
            tips = False
        ).add_coordinates().scale(0.8).move_to([0, -1, 0])

        pos = ValueTracker(1)
        d_model = 512

        formula = MathTex(r"PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)", color='#5ce1e6').scale(0.8).move_to([3,3,0])
        pos_text = always_redraw(
            lambda: MathTex(r"pos = " + f"{pos.get_value():.2f}", color='#5ce1e6').move_to([-3,3,0])
        )

        pos_enc_odd_curve = always_redraw(
                lambda: ax.plot(
                lambda x: pos_enc_odd(pos.get_value(), d_model, x),
                color='#5ce1e6'
            )
        )

        imprints = []

        def leave_imprint():
            imprint = ax.plot(
                lambda x: pos_enc_odd(pos.get_value(), d_model, x),
                color='#5ce1e6',
                stroke_opacity=0.5
            )
            self.add(imprint)
            imprints.append(imprint)
            
            # Fade all previous imprints
            for old_imprint in imprints[:-1]:
                new_opacity = old_imprint.get_stroke_opacity() * 0.4
                old_imprint.set_stroke(opacity=new_opacity)

        self.add(ax, pos_enc_odd_curve, formula, pos_text)
        self.wait(2)

        # Leave imprints at regular intervals
        leave_imprint()
        self.play(
            pos.animate.set_value(pos.get_value() + 79/2),
            rate_func=rate_functions.ease_in_quad,
            run_time=5
        )
        leave_imprint()
        self.play(
            pos.animate.set_value(pos.get_value() + 79/2),
            rate_func=rate_functions.ease_out_quad,
            run_time=5
        )
        self.wait(2)

if __name__ == "__main__":
    def pos_enc_matrix(sentence_length, d_model):
        """
        Generates a positional encoding matrix for a given sentence length and model dimension.
        """
        pos_enc_matrix = np.zeros((sentence_length, d_model))

        for pos in range(sentence_length):
            for i in range(d_model // 2): # Half-loop since we fill two dimensions for each value of i
                if i % 2 == 0:
                    pos_enc_matrix[pos, 2*i] = pos_enc_even(pos, d_model, i)
                else:
                    pos_enc_matrix[pos, 2*i + 1] = pos_enc_odd(pos, d_model, i)

        return pos_enc_matrix
    
    # Plot the positional encoding matrix
    sentence_length = 30 # Change this if you want to explore longer/shorter sentences
    d_model = 512
    pos_matrix = pos_enc_matrix(sentence_length, d_model)

    plt.figure(figsize=(10, 6), facecolor='#161718')
    ax = plt.gca()
    ax.set_facecolor('#161718')

    im = plt.imshow(pos_matrix, cmap='magma', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    ax.xaxis.tick_top()

    # Set all spines, ticks, and labels to white
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    plt.savefig("positional_encoding_matrix_512_dimensions.png", facecolor='#161718')