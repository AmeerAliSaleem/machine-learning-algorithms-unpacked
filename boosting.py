import numpy as np
from manim import *

class stump_boost(Scene):
    """
    Visualization of gradient boosting with decision tree stumps.
    """
    def construct(self):
        circle = Circle(radius=0.6, color=BLUE).move_to([0,3,0])
        text = Text("Data", font_size=30).move_to(circle.get_center())

        data = VGroup(circle, text)

        dot1 = Dot(point=data.get_center(), radius=0.08, color=GREEN_C)
        dot2 = Dot(point=data.get_center(), radius=0.08, color=RED_C)
        dot3 = Dot(point=data.get_center(), radius=0.08, color=GREEN_C)
        dot4 = Dot(point=data.get_center(), radius=0.08, color=RED_C)

        dot1_2 = dot1.copy()
        dot2_2 = dot2.copy()
        dot3_2 = dot3.copy()
        dot4_2 = dot4.copy()

        dot1_3 = dot1.copy()
        dot2_3 = dot2.copy()
        dot3_3 = dot3.copy()
        dot4_3 = dot4.copy()

        training_data1 = Dot(point=data.get_center(), radius=0.08, color=BLUE)
        # training_data2 = Dot(point=data.get_center(), radius=0.08, color=BLUE)
        # training_data3 = Dot(point=data.get_center(), radius=0.08, color=BLUE)

        tree1 = self.create_decision_tree_diagram(
            leaf1_colour='RED_C',
            leaf2_colour='GREEN_C'
        ).scale(0.5).move_to([-5.25, 0, 0])

        tree2 = self.create_decision_tree_diagram(
            leaf1_colour='RED_C',
            leaf2_colour='GREEN_C',
            leaf3_colour='RED_C',
            leaf4_colour='GREEN_C'
        ).scale(0.5).move_to([-1, 0, 0])

        tree3 = self.create_decision_tree_diagram(
            leaf1_colour='RED_C',
            leaf2_colour='GREEN_C',
            leaf3_colour='RED_C',
            leaf4_colour='GREEN_C',
            leaf5_colour='RED_C',
            leaf6_colour='GREEN_C'
        ).scale(0.5).move_to([4.25, 0, 0])

        tree1_text = MathTex(f"F_1").next_to(tree1[0], UP)
        tree2_text = MathTex(f"F_2").next_to(tree2[0], UP)
        tree3_text = MathTex(f"F_m").next_to(tree3[0], UP)

        ellipsis = VGroup([Dot(point=[0.2*i + 1.5,0,0], radius=0.04) for i in range(3)])
        mini_ellipsis = VGroup([Dot(point=[0.25*i + 0.25,-2,0], radius=0.03) for i in range(3)])

        tree1_leaves = [Create(leaf) for leaf in tree1 if isinstance(leaf, Rectangle)]
        tree1_arrows = [Create(arrow) for arrow in tree1 if isinstance(arrow, Arrow)]
        tree2_leaves = [Create(leaf) for leaf in tree2 if isinstance(leaf, Rectangle)]
        tree2_arrows = [Create(arrow) for arrow in tree2 if isinstance(arrow, Arrow)]
        tree3_leaves = [Create(leaf) for leaf in tree3 if isinstance(leaf, Rectangle)]  
        tree3_arrows = [Create(arrow) for arrow in tree3 if isinstance(arrow, Arrow)]

        # animations
        self.play(Create(data))
        self.wait()

        # tree 1
        self.play(*tree1_leaves)
        self.play(*tree1_arrows, Write(tree1_text))
        self.wait()

        self.play(
            dot1.animate.move_to(tree1[0].get_center() + [-0.375,0,0]),
            dot2.animate.move_to(tree1[0].get_center() + [-0.125,0,0]),
            dot3.animate.move_to(tree1[0].get_center() + [0.125,0,0]),
            dot4.animate.move_to(tree1[0].get_center() + [0.375,0,0])
            )
        self.wait()
        
        self.play(
            dot1.animate.move_to(tree1[1].get_center() + [-0.25,0,0]),
            dot2.animate.move_to(tree1[1].get_center()),
            dot3.animate.move_to(tree1[1].get_center() + [0.25,0,0]),
            dot4.animate.move_to(tree1[3].get_center())
        )
        self.wait()

        self.play(
            Indicate(dot1, color=YELLOW),
            Indicate(dot3, color=YELLOW),
            Indicate(dot4, color=YELLOW)
        )
        self.wait()

        self.play(
            dot1.animate.move_to([-5.5,-2.5,0]),
            dot3.animate.move_to([-5.25,-2.5,0]),
            dot4.animate.move_to([-5,-2.5,0]),
        )
        self.wait()

        # tree 2
        self.play(*tree2_leaves)
        self.play(*tree2_arrows, Write(tree2_text))
        self.wait()

        self.play(
            dot1_2.animate.move_to(tree2[0].get_center() + [-0.375,0,0]),
            dot2_2.animate.move_to(tree2[0].get_center() + [-0.125,0,0]),
            dot3_2.animate.move_to(tree2[0].get_center() + [0.125,0,0]),
            dot4_2.animate.move_to(tree2[0].get_center() + [0.375,0,0])
            )
        self.wait()
        
        self.play(
            dot1_2.animate.move_to(tree2[1].get_center() + [-0.25,0,0]),
            dot2_2.animate.move_to(tree2[1].get_center()),
            dot3_2.animate.move_to(tree2[1].get_center() + [0.25,0,0]),
            dot4_2.animate.move_to(tree2[3].get_center())
        )
        self.wait()

        self.play(
            dot1_2.animate.move_to(tree2[5].get_center()),
            dot3_2.animate.move_to(tree2[7].get_center())
        )
        self.wait()
        
        self.play(
            Indicate(dot1_2, color=YELLOW),
            Indicate(dot4_2, color=YELLOW)
        )
        self.wait()

        self.play(
            dot1_2.animate.move_to([-1.125,-2.5,0]),
            dot4_2.animate.move_to([-0.875,-2.5,0]),
        )
        self.wait()

        self.play(Create(ellipsis))

        # tree m
        self.play(*tree3_leaves)
        self.play(*tree3_arrows, Write(tree3_text))
        self.wait()

        self.play(
            dot1_3.animate.move_to(tree3[0].get_center() + [-0.375,0,0]),
            dot2_3.animate.move_to(tree3[0].get_center() + [-0.125,0,0]),
            dot3_3.animate.move_to(tree3[0].get_center() + [0.125,0,0]),
            dot4_3.animate.move_to(tree3[0].get_center() + [0.375,0,0])
            )
        self.wait()
        
        self.play(
            dot1_3.animate.move_to(tree3[1].get_center() + [-0.25,0,0]),
            dot2_3.animate.move_to(tree3[1].get_center()),
            dot3_3.animate.move_to(tree3[1].get_center() + [0.25,0,0]),
            dot4_3.animate.move_to(tree3[3].get_center())
        )
        self.wait()

        self.play(
            dot1_3.animate.move_to(tree3[5].get_center()),
            dot3_3.animate.move_to(tree3[7].get_center()),
            dot4_3.animate.move_to(tree3[9].get_center())
        )
        self.wait()

        self.play(dot1_3.animate.move_to(tree3[15].get_center()))


        self.wait(2)


    def create_decision_tree_diagram(self, leaf1_colour, leaf2_colour,
                                     leaf3_colour=None, leaf4_colour=None,
                                     leaf5_colour=None, leaf6_colour=None):
        """
        Create a decision tree diagram without text labels
        """
        root = Rectangle(width=2, height=1, color='YELLOW_C').move_to([0, 2, 0])
        leaf1 = Rectangle(width=2, height=1, color=leaf1_colour).move_to([-2.5, 0, 0])
        arrow1 = Arrow(start=root.get_bottom(), end=leaf1.get_top())
        leaf2 = Rectangle(width=2, height=1, color=leaf2_colour).move_to([2.5, 0, 0])
        arrow2 = Arrow(start=root.get_bottom(), end=leaf2.get_top())

        if leaf3_colour is None:
            # tree 1
            return VGroup(root, leaf1, arrow1, leaf2, arrow2)
        else:
            # tree 2
            leaf3 = Rectangle(width=2, height=1, color=leaf3_colour).move_to([-3.7, -2, 0])
            arrow3 = Arrow(start=leaf1.get_bottom(), end=leaf3.get_top())
            leaf4 = Rectangle(width=2, height=1, color=leaf4_colour).move_to([-1.2, -2, 0])
            arrow4 = Arrow(start=leaf1.get_bottom(), end=leaf4.get_top())
            if leaf5_colour is None:
                return VGroup(root, leaf1, arrow1, leaf2, arrow2, leaf3, arrow3, leaf4, arrow4)
            else:
                # tree m
                leaf5 = Rectangle(width=2, height=1, color=leaf5_colour).move_to([1.3, -2, 0])
                arrow5 = Arrow(start=leaf2.get_bottom(), end=leaf5.get_top())
                leaf6 = Rectangle(width=2, height=1, color=leaf6_colour).move_to([3.7, -2, 0])
                arrow6 = Arrow(start=leaf2.get_bottom(), end=leaf6.get_top())
                leaf7 = Rectangle(width=2, height=1, color='RED_C').move_to([-4.9, -4, 0])
                arrow7 = Arrow(start=leaf3.get_bottom(), end=leaf7.get_top())
                leaf8 = Rectangle(width=2, height=1, color='GREEN_C').move_to([-2.5, -4, 0])
                arrow8 = Arrow(start=leaf3.get_bottom(), end=leaf8.get_top())
                return VGroup(root, leaf1, arrow1, leaf2, arrow2, leaf3, arrow3, leaf4, arrow4,
                              leaf5, arrow5, leaf6, arrow6, leaf7, arrow7, leaf8, arrow8)