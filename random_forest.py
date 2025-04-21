import numpy as np
from manim import *

class random_forest(Scene):
    """
    Visualization of multiple decision tree diagrams side by side to represent a random forest.
    """
    def construct(self):
        circle = Circle(radius=0.6, color=BLUE).move_to([0,3,0])
        text = Text("Data", font_size=30).move_to(circle.get_center())

        data = VGroup(circle, text)

        training_data1 = Dot(point=data.get_center(), radius=0.08, color=BLUE)
        training_data2 = Dot(point=data.get_center(), radius=0.08, color=BLUE)
        training_data3 = Dot(point=data.get_center(), radius=0.08, color=BLUE)

        tree1 = self.create_decision_tree_diagram(
            leaf1_pos=[-1.5, 0, 0],
            leaf1_color='RED_C',
            leaf2_pos=[1.5, 0, 0],
            leaf2_color='YELLOW_C',
            leaf3_pos=[0, -2, 0],
            leaf3_color='GREEN_C',
            leaf4_pos=[3, -2, 0],
            leaf4_color='RED_C'
        ).scale(0.5).move_to([-5, 0, 0])

        tree2 = self.create_decision_tree_diagram(
            leaf2_pos=[-1.5, 0, 0],
            leaf2_color='YELLOW_C',
            leaf1_pos=[1.5, 0, 0],
            leaf1_color='GREEN_C',
            leaf3_pos=[-3, -2, 0],
            leaf3_color='RED_C',
            leaf4_pos=[0, -2, 0],
            leaf4_color='GREEN_C'
        ).scale(0.5).move_to([-1, 0, 0])

        ellipsis = VGroup([Dot(point=[0.25*i + 1.75,0,0], radius=0.05) for i in range(3)])
        mini_ellipsis = VGroup([Dot(point=[0.25*i + 0.25,-2,0], radius=0.03) for i in range(3)])

        tree3 = self.create_decision_tree_diagram(
            leaf1_pos=[-1.5, 0, 0],
            leaf1_color='GREEN_C',
            leaf2_pos=[1.5, 0, 0],
            leaf2_color='YELLOW_C',
            leaf3_pos=[0, -2, 0],
            leaf3_color='RED_C',
            leaf4_pos=[3, -2, 0],
            leaf4_color='GREEN_C'
        ).scale(0.5).move_to([5, 0, 0])

        tree_text1 = Text("Tree 1", font_size=20).next_to(tree1[0], UP)
        tree_text2 = Text("Tree 2", font_size=20).next_to(tree2[0], UP)
        tree_text3 = Text("Tree n", font_size=20).next_to(tree3[0], UP)

        trees = VGroup(tree1, tree2, tree3)

        # separate leaf and arrow animations
        leaf_animations = []
        arrow_animations = []
        for tree in trees:
            for element in tree:
                if isinstance(element, Rectangle):
                    leaf_animations.append(Create(element))
                elif isinstance(element, Arrow):
                    arrow_animations.append(GrowArrow(element))

        # animations
        self.play(Create(data))
        self.wait()

        self.play(*leaf_animations)
        self.play(*arrow_animations, Create(ellipsis))
        self.play(FadeIn(tree_text1), FadeIn(tree_text2), FadeIn(tree_text3))
        self.wait(2)

        self.play(
            training_data1.animate.move_to(tree1[0].get_center()),
            training_data2.animate.move_to(tree2[0].get_center()),
            training_data3.animate.move_to(tree3[0].get_center())
            )
        self.wait()

        self.play(
            training_data1.animate.move_to(tree1[3].get_center()),
            training_data2.animate.move_to(tree2[1].get_center()),
            training_data3.animate.move_to(tree3[3].get_center())
            )
        self.wait()

        self.play(
            training_data1.animate.move_to(tree1[7].get_center()),
            training_data3.animate.move_to(tree3[7].get_center())
            )
        self.wait()
        
        # colour the nodes according to the leaf they end up in
        self.play(Flash(training_data1, color=RED), training_data1.animate.set_color('RED_C'))
        self.play(Flash(training_data2, color=GREEN), training_data2.animate.set_color('GREEN_C'))
        self.play(Flash(training_data3, color=GREEN), training_data3.animate.set_color('GREEN_C'))
        self.wait()

        # bring the nodes together at the bottom of the screen
        self.play(
            training_data1.animate.move_to([-1.5, -2, 0]),
            training_data2.animate.move_to([-0.4, -2, 0]),
            training_data3.animate.move_to([1.5, -2, 0]),
            FadeIn(mini_ellipsis)
        )
        self.wait()

        labels = VGroup(training_data1, training_data2, training_data3, mini_ellipsis)
        brace = Brace(labels)
        brace_text = Tex("Prediction").next_to(brace, DOWN)

        self.play(Create(brace), Write(brace_text))
        self.wait()

        self.play(ApplyWave(labels))
        self.play(brace_text.animate.set_color('GREEN_C'))
        
        self.wait(2)

    def create_decision_tree_diagram(self, leaf1_pos, leaf1_color, leaf2_pos, leaf2_color, leaf3_pos, leaf3_color, leaf4_pos, leaf4_color):
        """
        Create decision tree diagrams without labels. The leaf colours are to be passed as parameters.
        """
        root = Rectangle(width=2, height=1, color='YELLOW_C').move_to([0, 2, 0])
        leaf1 = Rectangle(width=2, height=1, color=leaf1_color).move_to(leaf1_pos)
        arrow1 = Arrow(start=root.get_bottom(), end=leaf1.get_top())
        leaf2 = Rectangle(width=2, height=1, color=leaf2_color).move_to(leaf2_pos)
        arrow2 = Arrow(start=root.get_bottom(), end=leaf2.get_top())
        leaf3 = Rectangle(width=2, height=1, color=leaf3_color).move_to(leaf3_pos)
        arrow3 = Arrow(start=leaf2.get_bottom(), end=leaf3.get_top())
        leaf4 = Rectangle(width=2, height=1, color=leaf4_color).move_to(leaf4_pos)
        arrow4 = Arrow(start=leaf2.get_bottom(), end=leaf4.get_top())

        return VGroup(root, leaf1, arrow1, leaf2, arrow2, leaf3, arrow3, leaf4, arrow4)