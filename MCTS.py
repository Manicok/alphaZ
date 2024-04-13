#!/usr/bin/env python
# coding: utf-8

# In[7]:
from game import Player

# In[8]:


import numpy as np
from copy import deepcopy
from typing import Dict, List, Tuple, TypeVar, Callable, Sequence, Union

import logging

logging.basicConfig(filename="MCTS_log.txt",
                    format="%(asctime)s - %(funcName)s - %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger()

Node = TypeVar("Node", bound='TreeNode')
Board = TypeVar("Board", bound="Board")

action = int
prob = float
action_prob = Tuple[action, prob]
PolicyValueFunction = Callable[[...], Tuple[List[action_prob], float]]


# In[10]:


class TreeNode:
    """蒙特卡洛树节点"""
    def __init__(self, parent: Union[None, Node], prob: float, Cpuct=4) -> None:
        self.parent: TreeNode = parent
        self.children: Dict[int, TreeNode] = {}  # 当前节点在行为a后达到新的节点s'
        self.p = prob
        self.u = 0
        self.n = 0
        self.q = 0  # 明确： Q(s, a)中的s为父节点下的状态
        self.Cpuct = Cpuct

    def _get_value(self) -> float:
        """更新U，返回Q+U"""
        self.u = self.Cpuct * self.p * np.sqrt(self.parent.n) / (1 + self.n)
        return self.q + self.u

    def _update(self, value: float) -> None:
        """更新N，Q"""
        self.n += 1
        self.q += (value - self.q) / self.n

    def update_subtree(self, value) -> None:
        """更新该分支上的所有节点的Q"""
        if self.parent:
            self.parent.update_subtree(-value)
        self._update(value)

    def select(self) -> (int, Node):
        """返回有最大价值的(action, Child)"""
        return max(self.children.items(),
                   key=lambda x: x[1]._get_value())

    def expand(self, action_prob_pairs: List[action_prob]) -> None:
        """扩张子节点"""
        for action, prob in action_prob_pairs:
            if action not in self.children.keys():
                self.children[action] = TreeNode(self, prob, self.Cpuct)

    def no_child(self) -> bool:
        """判断节点是否需扩张"""
        return self.children == {}

    def is_root(self) -> bool:
        """判断是否为根节点"""
        return self.parent is None


# In[11]:


class MCTS:
    """蒙特卡洛树搜索"""
    def __init__(self,
                 PolicyValueF: PolicyValueFunction,
                 rounds: int,
                 Cpunt: float = 4) -> None:
        self.root = TreeNode(None, 1.0)
        self.policy = PolicyValueF
        self.Cpuct = Cpunt
        self.rounds = rounds

    def _playout(self, board_copy: Board) -> None:
        """从根节点搜索到叶节点，得到叶节点的价值，反向更新该分支上的所有节点"""
        node = self.root
        while True:
            if node.no_child():
                break
            action, node = node.select()  # 获得最佳aciton，更新节点
            if not board_copy.action(action):
                raise Exception("Invalid Action")
        #  通过神经网路得到action-prob和节点价值
        result = board_copy.end(show=False)  # 只有end()后才会更新current_space等数据
        action_prob_pairs, value = self.policy(board_copy)
        if not result:
            node.expand(action_prob_pairs)
        else:
            value = 1.0 if result == board_copy.current_player else -1
        #  明确：value指的是当前局面s'对当前执棋者player'的价值
        #       而反向更新是Q(s, a)，即在局面s下，动作a对player的价值
        node.update_subtree(-value)

    def get_action_probs(self, board: Board, temp=1e-3) -> Tuple[List[action], Sequence[prob]]:
        """根据树搜素返回actions和probs"""
        if len(board.states) <= 30:  # 前30步设置temp为1
            temp = 1.0
        for _ in range(self.rounds):
            board_copy = deepcopy(board)
            self._playout(board_copy)
        actions = [action for action in self.root.children.keys()]
        scoped_visits = np.array([node.n for node in self.root.children.values()])
        scoped_visits = scoped_visits / scoped_visits.sum()
        temp_modified = np.power(scoped_visits, 1 / temp) + 1e-10
        probs = temp_modified / np.sum(temp_modified)
        return actions, probs

    def action_forward(self, action: int) -> None:
        """动作对应的子节点成为新的根节点；
        保留这个节点下面的子树、丢弃其余部分
        """
        self.root = self.root.children[action]
        self.root.parent = None

    def clear(self) -> None:
        """删除所有树的信息"""
        self.root = TreeNode(None, 1.0)


# In[14]:


class MCTSPlayer(Player):
    def __init__(self,
                 PolicyValueF: PolicyValueFunction,
                 color: int = 0, name="MCTSPlayer",
                 rounds=1600, Cpunt=4) -> None:
        super().__init__(name=name, color=color)
        self.MCTS = MCTS(PolicyValueF, rounds=rounds, Cpunt=Cpunt)

    def get_action(self, board: Board, temp=1e-3, self_play=False) -> Union[action, action_prob]:
        """返回MCTS选择的action, 自我对弈时还返回所有动作的概率"""
        accessible_actions = board.current_space  # AlphaZero 允许自填真眼
        if not accessible_actions:
            raise Exception("No action accessible")
        all_probs = np.zeros(board.grids + 1)
        actions, probs = self.MCTS.get_action_probs(board, temp)
        all_probs[actions] = probs
        if self_play:
            Dir_noise = np.random.dirichlet(np.ones(len(probs)))
            p = .75 * probs + .25 * Dir_noise
            action = np.random.choice(actions, p=p)
            self.MCTS.action_forward(action)
            logger.info("action %s", action)
            return action, all_probs
        else:
            action = actions[np.argmax(probs)]
            self.reset()
            return action

    def reset(self):
        """重置树"""
        self.MCTS.clear()




