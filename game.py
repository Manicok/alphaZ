#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
from typing import List, Tuple, Dict, Union
import matplotlib.pyplot as plt 
from copy import deepcopy
import logging

logging.basicConfig(filename="game_log.txt",
                    format="%(asctime)s - %(funcName)s - %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger()

class Board:
    """棋盘类"""
    def __init__(self, side_length: int = 19) -> None:
        self.length = side_length
        self.grids = self.length**2
        self.players = [-1, 1]
        self.current_player = -1
        self.fig = None  # 画布
        self.eaten = False  # 吃子后要更新画布
        self.states: List[Tuple[int, int]] = []  # 记录（action， player）
        #  分别定义两个玩家的局面落子状态，0 为空闲， 1为己方已落子，-1为对方已落子，-2为暂时不可落子
        self.action_space = {-1: np.zeros(self.grids),
                             1: np.zeros(self.grids)}
        #  分别记录双方相连的棋子坐标及其对应的"气"的来源坐标，及{棋子},{气的来源}对组成的列表
        self.souce_of_qi: Dict[int, List[List[set, set]]] = {-1: [],
                                                             1: []}
        #  记录每个格子四个方向上的敌方落子数，辅助判断行为空间
        self.be_taken = self._get_be_taken()
        #  当前执棋者的行为空间
        self.current_space = self._get_action_space() | {self.grids, }  # 包含自填真眼，可通过 -= self.true_eyes去除真眼.-1为pass
        #  记录历史局面
        self.history = [np.zeros((self.length, self.length)) for _ in range(16)]


    def _get_be_taken(self) -> Dict[int, np.array]:
        """初始化格子四个方向上的敌方落子数，边界、角落起始站分别为1、2，其余为0"""
        corners = [self._location_change((x, y)) for x in (0, self.length - 1) for y in (0, self.length - 1)]
        edges = [self._location_change((x, y)) for x in range(1, self.length - 1) for y in (0, self.length - 1)]
        edges.extend(self._location_change((x, y)) for x in (0, self.length - 1) for y in range(1, self.length - 1))
        self.revise = np.zeros(self.grids)  # 用于_get_action_space中的修正
        self.revise[corners] = -2
        self.revise[edges] = -1
        taken = np.zeros(self.grids)
        taken[corners] += 2
        taken[edges] += 1
        return {-1: taken,
                1: np.copy(taken)}

    def _copy(self):
        """复制一份数据"""
        self.action_space_copy = deepcopy(self.action_space)
        self.souce_of_qi_copy = deepcopy(self.souce_of_qi)
        self.be_taken_copy = deepcopy(self.be_taken)

    def _back(self, player_change=False):
        """用备份数据悔棋和在下无效棋后还原数据"""
        if player_change:
            self.current_player *= -1
        self.states.pop()
        self.action_space = self.action_space_copy
        self.souce_of_qi = self.souce_of_qi_copy
        self.be_taken = self.be_taken_copy

    def _location_change(self, loc: Union[Tuple[int, int], int]) -> Union[int, Tuple[int, int]]:
        """两种坐标的转换"""
        try:
            x, y = loc
            return self.length * y + x
        except (TypeError, ValueError):
            y, x = divmod(loc, self.length)
            return (x, y)

    def _qi_loc(self, loc: int) -> List[int]:
        """返回一个坐标上下左右相邻的、棋盘内的坐标"""
        x, y = self._location_change(loc)
        neighbor_2d = [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                       if dx*dy == 0 and dx != dy
                       if x + dx in range(self.length) if y + dy in range(self.length)]
        
        return [self._location_change(n) for n in neighbor_2d]  # (左，右，下，上)

    def _get_friends(self, loc) -> List[int]:
        """返回在当前玩家下loc四周的有己方棋子的坐标"""
        return [l for l in self._qi_loc(loc)
                if self.action_space[self.current_player][l] == 1]
        
    def _get_opposites(self, loc) -> List[int]:
        """返回在当前玩家下loc四周的有敌方棋子的坐标"""
        return [l for l in self._qi_loc(loc)
                if self.action_space[self.current_player * -1][l] == 1]

    def _valid_action(self, action: int) -> bool:
        """已知该位置为空闲的情况下，返回该action是否有效"""
        assert self.action_space[self.current_player][action] == 0
        if self.action(action):
            self._back(player_change=True)
            return True
        return False
    
    def _get_action_space(self) -> set[int]:
        """输出当前执棋者真正的行为空间"""
        void_space = set(np.where(self.action_space[self.current_player] == 0)[0])  # 所有空闲格
        self.counter_eyes = set(np.where(self.be_taken[self.current_player] == 4)[0]).intersection(void_space)  # 所有对方的目
        self.eyes = set(np.where(self.be_taken[self.current_player * -1] == 4)[0]).intersection(void_space)  # 所有己方的目
        possible_space = set(np.where(self.be_taken[-1] + self.be_taken[1] + self.revise == 4)[0])
        possible_space = possible_space.intersection(void_space) - self.counter_eyes
        space = void_space - self.counter_eyes - possible_space
        for s in possible_space:
            if self._valid_action(s):
                space.add(s)
        return space

    def _get_true_eyes(self) -> set[int]:
        """判断self.eyes中真眼"""
        true_eyes = set()
        for eye in self.eyes:
            eye_x, eye_y = self._location_change(eye)
            diags = [self._location_change((eye_x + dx, eye_y + dy))
                     for dx in (1, -1) for dy in (1, -1)
                     if eye_x + dx in range(self.length) if eye_y + dy in range(self.length)]
            takenup = len(np.where(self.action_space[self.current_player][diags] == 1)[0])
            voids = np.where(self.action_space[self.current_player][diags] == 0)[0]
            self.current_player *= -1
            for void in voids:
                if self.action_space[self.current_player][void] != 0:
                    continue
                if not self._valid_action(void):
                    takenup += 1
            self.current_player *= -1
            if len(diags) in (1, 2):  # 要求眼角全占
                if takenup == len(diags):
                    true_eyes.add(eye)
            else:  # 要求眼角占三个
                if takenup == 3:
                    true_eyes.add(eye)
        return true_eyes
    
    def _eat(self, loc: int, opposites: List[int]) -> List[int]:
        """返回loc处落子后被吃了的棋子，修改[棋，气]对"""
        eaten = []
        pairs = [pair for pair in self.souce_of_qi[self.current_player*-1]
                 if any([oppo in pair[0] for oppo in opposites])]
        for pair in pairs:
            pair[1].remove(loc)
            if len(pair[1]) == 0:
                eaten.extend(list(pair[0]))
                self.souce_of_qi[self.current_player*-1].remove(pair)
        return eaten

    def _link(self, loc: int, friends: List[int]) -> bool:
        """合并相连的[棋，气]对，然后判断会不会因落子导致的最后的气的来源消失，返回是否为有效落子"""
        friends.append(loc)
        pairs = [pair for pair in self.souce_of_qi[self.current_player]
                 if any([l in pair[0] for l in friends])]
        grids = set()
        souces = set()
        for pair in pairs:
            self.souce_of_qi[self.current_player].remove(pair)
            grids.update(pair[0])
            souces.update(pair[1])
        souces.remove(loc)  # 相邻的棋子失去这个气的来源
        self.souce_of_qi[self.current_player].append([grids, souces])
        return len(souces) != 0

    def _pass(self) -> None:
        #  备份数据
        self._copy()
        #  下一回合对方的暂时不可下棋变为可下棋
        idx = self.action_space[self.current_player*-1] == -2
        self.action_space[self.current_player*-1][idx] = 0
        #  记录（action， player）
        self.states.append((self.grids, self.current_player))
        #  改变当前执棋者
        self.current_player *= -1
    
    def action(self, loc: Union[int, Tuple] = None) -> bool:
        """执行传入的下棋动作，默认None或self.length为pass
        改变：行为空间，气的计数，当前下棋者，返回当前落子是否有效"""
        if loc is None or loc == self.grids:
            self._pass()
            return True
        if isinstance(loc, tuple):
            loc = self._location_change(loc)
        #  初步判断是否可落子
        if self.action_space[self.current_player][loc] != 0:
            return False
        #  备份数据
        self._copy()
        #  下一回合对方的暂时不可下棋变为可下棋
        idx = self.action_space[self.current_player*-1] == -2
        self.action_space[self.current_player*-1][idx] = 0
        #  记录（action， player）
        self.states.append((loc, self.current_player))
        #  修改落子状态等
        self.action_space[self.current_player][loc] = 1
        self.action_space[self.current_player * -1][loc] = -1
        self.be_taken[self.current_player * -1][self._qi_loc(loc)] += 1
        #  获取周围信息
        neighbours = self._qi_loc(loc)
        friends = self._get_friends(loc)
        opposites = self._get_opposites(loc)
        void = set(neighbours) - set(friends) - set(opposites)
        #  无论有没有气的来源，先构建pair
        self.souce_of_qi[self.current_player].append([{loc, }, void])

        #  先吃子, 即改两边的落子状态，然后改己方[棋，气]对中可能增加的气的来源
        if opposites:
            eaten = self._eat(loc, opposites)
            self.eaten = False
            if eaten:
                self.eaten = True
                if len(eaten) == 1:  # 吃一只与吃多子分别考虑
                    #  吃一子可能触发暂时不可落子
                    if len(self._get_opposites(loc)) == 4 and len(self._get_friends(eaten[0])) == 4:
                        self.action_space[self.current_player*-1][eaten[0]] = -2
                    else:  # 非暂时不可落则为一定不可落
                        self.action_space[self.current_player*-1][eaten[0]] = -1
                    self.action_space[self.current_player][eaten[0]] = 0
                else:  # 吃多子时，一定是可落子的
                    self.action_space[self.current_player][eaten] = 0
                    self.action_space[self.current_player * -1][eaten] = 0
                for l in eaten:
                    # 吃的子可能成为新的气的来源
                    souce_add_loc = self._get_friends(l)
                    pairs = [pair for pair in self.souce_of_qi[self.current_player]
                             if any([friend in pair[0] for friend in souce_add_loc])]
                    for pair in pairs:
                        pair[1].add(l)
                    self.be_taken[self.current_player][self._qi_loc(l)] -= 1

        #  判断是否为有效落子
        if friends:  # 落子导致棋子相连，共享气
            vaild = self._link(loc, friends)
            if not vaild:
                self._back(player_change=False)
                return False
        else:  # 判断该处落子后最终有没有气的来源
            if not self.souce_of_qi[self.current_player][-1][1]:
                self._back(player_change=False)
                return False
        #  改变当前执棋者
        self.current_player *= -1
        return True

    def _get_souce(self, player: int) -> set:
        """得到一个玩家的所有气的来源的集合"""
        all_souces = [pair[-1] for pair in self.souce_of_qi[player]]
        collection = set()
        for s in all_souces:
            collection.update(s)
        return collection

    def end(self, show=True):
        """
        判断是否结束
        1.只在无棋可下时才进行数子判断
        2.自填真眼不算有棋可下
        3.数字超过棋盘一半时判赢
        返回： 0 -> 未结束/还有棋可下，-1/1 -> 赢家
        """
        self.current_space = self._get_action_space() | {self.grids, }
        
        #  连续两次pass即判断结束
        if len(self.states) >= 2 and (self.states[-1][0] == self.grids and self.states[-2][0] == self.grids):
            # Tromp-Tylor记分法
            black_cross = self._get_souce(-1)
            white_cross = self._get_souce(1)
            #  仅与一方相邻的空白交叉点
            black_only_cross = black_cross - white_cross
            white_only_cross = white_cross - black_cross
            #  落子的数量
            black_color = np.sum(self.action_space[-1] == 1)
            white_color = np.sum(self.action_space[1] == 1)
            #  记分
            black = black_color + len(black_only_cross)
            white = white_color + len(white_only_cross)
            if show:
                print(f"黑棋：{black}子\n白棋：{white}子")
            if black > white:
                return -1
            else:
                return 1
        else:
            return 0
        
    def regret(self) -> None:
        """悔棋"""
        self._back(True)

    def clear(self) -> None:
        """清空该棋盘"""
        self.current_player = -1
        self.states.clear()
        self.action_space = {-1: np.zeros(self.grids),
                             1: np.zeros(self.grids)}
        self.souce_of_qi: Dict[int, List[List[set, set]]] = {-1: [],
                                                             1: []}
        self.be_taken = self._get_be_taken()
        self.current_space = self._get_action_space() | {-1, }
        self.history = [np.zeros((self.length, self.length)) for _ in range(16)]
        
    def _get_board(self, player: int) -> np.arange:
        """获得player的棋盘局面，返回数据为2维数组"""
        locs = [self._location_change(loc)
                for loc in np.where(self.action_space[player] == 1)[0]]
        state = np.zeros((self.length, self.length))
        if locs:
            xs, ys = list(zip(*locs))
            state[xs, ys] = 1
        return state
    
    def get_data(self) -> np.array:
        """存储神经网路所需的17xself.lengthxself.lenth的状态数据"""
        player_value = 0 if self.current_player == -1 else 1
        player = np.full((1, self.length, self.length), fill_value=player_value)
        reserve = [[self.history[y], self.history[x]]
                   for x in range(0, 14, 2)
                   for y in range(1, 14, 2)
                   if x < y and y - x == 1]
        self.history.clear()
        for pair in reserve:
            self.history.extend(pair)
        new_history = [self._get_board(self.current_player),
                       self._get_board(self.current_player * -1)]
        new_history.extend(self.history)
        self.history = new_history
        return np.append(np.array(self.history), player, axis=0)
        
    def picture(self) -> None:
        size = (int(self.length / 3), ) * 2
        s = int(self.length / 6 * 100)
        if not self.fig or self.eaten:
            self.fig, self.ax = plt.subplots(figsize=size)
            self.ax.grid(linestyle='--', linewidth=1, alpha=.2)
            self.ax.set_xticks(range(self.length));
            self.ax.set_xlim(0, self.length - 1)
            self.ax.set_ylim(0, self.length - 1)
            self.ax.xaxis.set_ticks_position('top')
            self.ax.set_yticks(range(self.length));
            self.ax.invert_yaxis()
            plt.close(self.fig)
        else:
            player_0_idx = np.where(self.action_space[-1] == 1)[0]
            player_0_2d_loc = [self._location_change(l) for l in player_0_idx]
            player_0_x_y = list(zip(*player_0_2d_loc))
            
            player_1_idx = np.where(self.action_space[1] == 1)[0]
            player_1_2d_loc = [self._location_change(l) for l in player_1_idx]
            player_1_x_y = list(zip(*player_1_2d_loc))
            
            if player_0_x_y:
                self.ax.scatter(player_0_x_y[0], player_0_x_y[1], c='k', s=s);
            if player_1_x_y:
                self.ax.scatter(player_1_x_y[0], player_1_x_y[1], c='r', s=s);
                return self.fig
        # 未吃子只多画最后一步即可
        loc, player = self.states[-1]
        c = 'k' if player == -1 else 'r'
        x, y = self._location_change(loc)
        self.ax.scatter(x, y, c=c, s=s)
        return self.fig
        


# In[145]:


class Player:
    """玩家基类"""
    def __init__(self, name: str, color: int) -> None:
        self.name = name
        self.color = color

    def get_action(self, board: Board) -> int:
        ...

    def __str__(self):
        colors = {-1: "black",
                  1: "white"}
        return self.name + " holding " + colors.get(self.color, "")


# In[157]:


class Game:
    def __init__(self, board_length=19):
        self.board = Board(board_length)

    def self_play(self, player, show=False):
        states, search_probs, players = [], [], []
        while True:
            action, probs = player.get_action(self.board, self_play=True)
            states.append(self.board.get_data())
            search_probs.append(probs)
            players.append(self.board.current_player)
            self.board.action(action)
            if show:
                self.board.picture()
            result = self.board.end(show=show)
            if result:
                logger.info("end a game: %s", self.board.states)
                winner = np.where(np.array(players) == result, 1, -1)
                player.reset()
                self.board.clear()
                return list(zip(states, search_probs, winner))


    def play(self, p1, p2, show=True):
        p1.color = -1
        p2.color = 1
        players = {-1: p1, 1: p2}
        while True:
            print(self.board.current_player)
            print(self.board.current_space)
            player = players[self.board.current_player]
            action = player.get_action(self.board)
            print(action)
            self.board.action(action)
            if show:
                self.board.picture()
            result = self.board.end()
            if result:
                p1.reset()
                p2.reset()
                self.board.clear()
                return result