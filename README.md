主要的内容
# game.py
* cls Board 棋盘类
  * self.current_space 当前执棋者可执行动作空间集合（可自填真眼）
  * def get_data  返回训练用棋局数据
* cls Player 玩家基类
* cls Game 棋局类
  * def play 双方对弈
  * def self_play 自我对弈
# MCTS.py
* cls TreeNode  蒙特卡洛树节点
* cls MCTS 蒙特卡洛树搜索
  * def _playout 执行一次树搜索
  * def get_action_probs 执行self.rounds次树搜索，返回*动作-概率对*
* cls MCTSPlayer 蒙特卡洛玩家
  * def get_action 返回动作
# network.py
* cls ResBlock 残差块
* cls PolicyValueNet  策略价值神经网路
  * def PolicyValueFunction 返回动作概率与局面价值（对当前执棋者）
# train.ipynb
* cls Train 训练类
