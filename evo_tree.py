from copy import deepcopy
from dataclasses import dataclass
import random
from typing import Callable


@dataclass
class TreeNode:
    parameter_key: int
    parameter_split: float
    left_child: "TreeNode | None" = None
    right_child: "TreeNode | None" = None
    represented_class: int | None = None
    last_score: float = 0.0

    def get_class(self, case: list[float]) -> int:
        if (
            self.left_child is None
            and self.right_child is None
            and self.represented_class is None
        ):
            return -1

        if (
            self.left_child is None
            and self.right_child is None
            and self.represented_class is not None
        ):
            return self.represented_class

        assert self.left_child is not None
        assert self.right_child is not None

        if case[self.parameter_key] <= self.parameter_split:
            return self.left_child.get_class(case)
        else:
            return self.right_child.get_class(case)

    def all_nodes(self) -> list["TreeNode"]:
        ret: list["TreeNode"] = [self]
        if self.left_child is not None:
            assert self.right_child is not None
            ret.extend(self.left_child.all_nodes())
            ret.extend(self.right_child.all_nodes())
        return ret

    def mutate(
        self,
        class_rng: Callable[[], int],
        parameter_rng: Callable[[], int],
        delta_split_rng: Callable[[], float],
        split_rng: Callable[[], float],
        add_child_prob: float,
    ):
        mutation_idx = random.randint(0, 4)

        if mutation_idx == 1:
            self.represented_class = class_rng()
        elif mutation_idx == 2:
            self.parameter_key = parameter_rng()
        elif mutation_idx == 3:
            self.parameter_split += delta_split_rng()

        if random.random() < add_child_prob:
            self.left_child = random_node(class_rng, parameter_rng)
            self.right_child = random_node(class_rng, parameter_rng)

    def random_descendant(self) -> "TreeNode":
        return random.choice(self.all_nodes())

    def children(self) -> list["TreeNode"]:
        if self.left_child is not None:
            assert self.right_child is not None
            return [self.left_child, self.right_child]
        return []

    def find_parent(self, target: "TreeNode") -> "TreeNode | None":
        for child in self.children():
            if child is target:
                return self
            parent = self.find_parent(child)
            if parent:
                return parent
        return None


type Tree = TreeNode


def random_node(
    class_rng: Callable[[], int], parameter_rng: Callable[[], int]
) -> TreeNode:
    return TreeNode(
        parameter_key=parameter_rng(),
        represented_class=class_rng(),
        parameter_split=random.random(),
    )


def swap_nodes(node_1: TreeNode, node_2: TreeNode):
    node_1.parameter_key, node_2.parameter_key = (
        node_2.parameter_key,
        node_1.parameter_key,
    )
    node_1.left_child, node_2.left_child = node_2.left_child, node_1.left_child
    node_1.right_child, node_2.right_child = node_2.right_child, node_1.right_child
    node_1.represented_class, node_2.represented_class = (
        node_2.represented_class,
        node_1.represented_class,
    )
    node_1.parameter_split, node_2.parameter_split = (
        node_2.parameter_split,
        node_1.parameter_split,
    )


def cross_breed(parent_1: Tree, parent_2: Tree) -> tuple[Tree, Tree]:
    child_1 = deepcopy(parent_1)
    child_2 = deepcopy(parent_2)

    node_1 = child_1.random_descendant()
    node_2 = child_2.random_descendant()

    swap_nodes(node_1, node_2)
    return child_1, child_2
