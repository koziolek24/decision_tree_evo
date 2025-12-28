from copy import deepcopy
from dataclasses import dataclass
import random
from typing import Callable, Self


@dataclass
class TreeNode:
    parameter_key: int
    children: list[tuple[float, Self]]
    represented_class: int

    def get_class(self, case: list[float]) -> int:
        if len(self.children) == 0:
            return self.represented_class

        for i in range(1, len(self.children)):
            if self.children[i][0] > case[self.parameter_key]:
                return self.children[i - 1][1].get_class(case)
        return self.children[0][1].get_class(case)

    def all_nodes(self) -> list[Self]:
        ret = [self]
        for _, child in self.children:
            ret.extend(child.all_nodes())
        return ret

    def mutate(
        self,
        class_rng: Callable[[], int],
        parameter_rng: Callable[[], int],
        delta_split_rng: Callable[[], float],
    ):
        self.represented_class = class_rng()
        self.parameter_key = parameter_rng()

        child = random.randint(0, len(self.children) - 1)
        self.children[child] = (
            self.children[child][0] + delta_split_rng(),
            self.children[child][1],
        )
        self.children.sort(key=lambda x: x[0])

    def insert_random_split(self, rng: Callable[[], float], child: Self):
        split = rng()
        self.children.append((split, child))
        self.children.sort(key=lambda x: x[0])

    def random_descendant(self) -> Self:
        return random.choice(self.all_nodes())

    def find_parent(self, target: Self) -> Self | None:
        for _, child in self.children:
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
        parameter_key=parameter_rng(), represented_class=class_rng(), children=[]
    )


def swap_nodes(node_1: TreeNode, node_2: TreeNode):
    node_1.parameter_key, node_2.parameter_key = (
        node_2.parameter_key,
        node_1.parameter_key,
    )
    node_1.children, node_2.children = node_2.children, node_1.children
    node_1.represented_class, node_2.represented_class = (
        node_2.represented_class,
        node_1.represented_class,
    )


def cross_breed(parent_1: Tree, parent_2: Tree) -> tuple[Tree, Tree]:
    child_1 = deepcopy(parent_1)
    child_2 = deepcopy(parent_2)

    node_1 = child_1.random_descendant()
    node_2 = child_2.random_descendant()

    swap_nodes(node_1, node_2)
    return child_1, child_2
