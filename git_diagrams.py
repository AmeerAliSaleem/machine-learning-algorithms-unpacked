from manim import *

class working_dirs(Scene):
    """
    A diagram depicting file locations and the git commands used to move files between them
    """

    def construct(self):
        working_directory = Ellipse(3, 2, color=PURPLE_E).set_fill(color=PURPLE_E, opacity=1).move_to([-5.25, 2.5, 0])
        stage = Ellipse(3, 2, color=BLUE_E, fill_color=BLUE_E).set_fill(color=BLUE_E, opacity=1).move_to([-1.75, 2.5, 0])
        local_repo = Ellipse(3, 2, color=GREEN_E, fill_color=GREEN_E).set_fill(color=GREEN_E, opacity=1).move_to([1.75, 2.5, 0])
        remote_repo = Ellipse(3, 2, color=GOLD_E, fill_color=GOLD_E).set_fill(color=GOLD_E, opacity=1).move_to([5.25, 2.5, 0])

        working_directory_label = Text("Working directory", font_size=24, font='Rockwell').move_to(working_directory.get_center())
        stage_label = Text("Staging area", font_size=24, font='Rockwell').move_to(stage.get_center())
        local_repo_directory_label = Text("Local repo", font_size=24, font='Rockwell').move_to(local_repo.get_center())
        remote_repo_directory_label = Text("Remote repo", font_size=24, font='Rockwell').move_to(remote_repo.get_center())

        git_add = Text("git add", font_size=24, font='Consolas').move_to([-3.5, 0.5, 0])
        git_commit = Text("git commit", font_size=24, font='Consolas').move_to([0, -0.5, 0])
        git_push = Text("git push", font_size=24, font='Consolas').move_to([3.5, -1.5, 0])
        git_pull = Text("git pull", font_size=24, font='Consolas').move_to([3.5, -3, 0])

        git_add_arrow = Arrow(color=PURPLE).next_to(git_add, UP*0.5)
        git_commit_arrow = Arrow(color=BLUE).next_to(git_commit, UP*0.5)
        git_push_arrow = Arrow(color=GREEN).next_to(git_push, UP*0.5)
        git_pull_arrow = Arrow(start=RIGHT, end=LEFT, color=GOLD).next_to(git_pull, UP*0.5)

        dash1 = DashedLine([-5.25,6,0], [-5.25,-6,0], color=PURPLE_E)
        dash2 = DashedLine([-1.75,6,0], [-1.75,-6,0], color=BLUE_E)
        dash3 = DashedLine([1.75,6,0], [1.75,-6,0], color=GREEN_E)
        dash4 = DashedLine([5.25,6,0], [5.25,-6,0], color=GOLD_E)

        self.add(dash1, dash2, dash3, dash4)
        self.add(working_directory, stage, local_repo, remote_repo)
        self.add(working_directory_label, stage_label, local_repo_directory_label, remote_repo_directory_label)
        self.add(git_add, git_commit, git_push, git_pull)
        self.add(git_add_arrow, git_commit_arrow, git_push_arrow, git_pull_arrow)