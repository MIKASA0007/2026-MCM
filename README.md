# 2026 MCM Problem C Solution: Data With The Stars 💃📊

> **Team Number:** [你的队伍编号]
> **Topic:** Modeling Fan Voting & Elimination Mechanics in "Dancing with the Stars"

## 📖 项目背景 (Project Overview)

本项目是 **2026年美国大学生数学建模竞赛 (MCM)** C题 "Data With The Stars" 的解决方案代码仓库。

本题的核心挑战在于：现实中的电视节目《与星共舞》(DWTS) **并不公开具体的粉丝投票数**，只公布评委打分和最终淘汰结果。我们需要建立数学模型来：
1.  **反向推演 (Inverse Modeling):** 基于淘汰结果和评委分，重构历史上缺失的粉丝投票数据。
2.  **机制对比 (Mechanism Analysis):** 比较“排名制 (Ranking)”与“百分比制 (Percentage)”两种计分方式的公平性与差异。
3.  **敏感性分析 (Sensitivity Analysis):** 探索导致选手被淘汰的“危险区域 (Kill Zones)”。

## 📂 文件结构 (File Structure)

| 文件名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `1.py` | 核心模型 | **不确定性推断模型**。用于处理推演出的粉丝投票数据，计算投票份额的置信区间 (95% CI)、RCIW (区间宽度) 及不确定性水平。 |
| `2-1-1.py` | 仿真模拟 | **计分机制对比引擎**。实现了 Ranking 和 Percentage 两种赛制的逻辑，计算选手的“双重排名”，并标记不同赛制下的淘汰结果差异。 |
| `2-1-2.py` | 可视化 | **生存格局可视化**。绘制 "Kill Zones" 图表，识别“悲剧区 (High Judge, Low Fan)”和“奇迹区 (High Fan, Low Judge)”。 |
| `final_metrics_fan_votes_v2.csv` | 数据集 | 经过预处理和初步反演的清洗数据，包含赛季、周数、评委分及估算的粉丝票数。 |
| `2026_MCM_Problem_C.pdf` | 题目 | 原始题目描述文件。 |

## 💡 建模思路 (Modeling Methodology)

### 1. 粉丝投票的反向重构 (The Invisible Hand)
由于粉丝投票数不可见，我们采用了一种**贝叶斯逆向推断**的思路（体现在数据预处理与 `1.py` 中）：
- 假设每一周的粉丝投票服从某种分布。
- 利用 `actual_elimination`（实际淘汰结果）作为约束条件。
- **核心逻辑：** 如果某位选手评委分很高却被淘汰，说明其粉丝票数极低。模型通过 `Entropy`（熵）和 `Constraint Tightness` 来量化这种可能性的分布。
- **代码实现 (`1.py`):** 计算 `est_vote_share_mean` (估计得票率均值) 及其置信区间，量化模型对每个推断结果的“不确定性 (Uncertainty Level)”。

### 2. 双重赛制模拟 (Ranking vs. Percentage)
代码 `2-1-1.py` 精确复现了题目要求的两种历史计分规则：
- **Ranking System (排名制):** $Score = Rank_{Judge} + Rank_{Fan}$。总排名越小越好。
- **Percentage System (百分比制):** $Score = \%_{Judge} + \%_{Fan}$。总百分比越高越好。
- **冲突检测：** 我们比较了同一组数据在两种规则下的不同命运，找出了那些“生于排名，死于百分比”或反之的边缘案例。

### 3. 生存空间可视化 (The "Kill Zones")
代码 `2-1-2.py` 通过可视化的方式揭示了比赛的残酷性。我们定义了两个关键区域：
- **🟢 Miracle Zone (奇迹区):** 评委分很低（Rank高），但靠着极高的粉丝人气存活下来。
- **🔴 Tragic Zone (悲剧区):** 评委分很高，但因为粉丝投票不足而惨遭淘汰。
- 通过 `seaborn` 绘制散点图，横轴为评委不满度，纵轴为粉丝不满度，直观展示了不同赛制下的生存边界。

## 🚀 快速开始 (Getting Started)

### 环境依赖
请确保安装以下 Python 库：
```bash
pip install pandas numpy scipy matplotlib seaborn tqdm
