# 2026 MCM Problem C: Analysis of Voting Mechanics in "Dancing with the Stars"

> **基于动态贝叶斯逆向推断的DWTS投票机制公平性与偏向性分析** > **Dynamic Bayesian Inverse Inference for Fairness & Bias Analysis in DWTS Voting Systems**

## 📖 项目背景 (Background)

本项目旨在解决 **2026 MCM (美赛) C题**，通过数学建模分析《与星共舞》(Dancing with the Stars) 历史数据，探讨评分机制对比赛结果的影响。

由于比赛官方仅公布评委分数（Judge Scores）和最终淘汰结果（Elimination），而**粉丝投票数据（Fan Votes）是未知的黑箱**。本项目的核心难点在于**如何在数据缺失的情况下，定量评估不同计分规则（排名制 vs 百分比制）的公平性及偏向性**。

## 🧠 核心建模思路 (Modeling Methodology)

本项目采用**“动态贝叶斯逆向推断” (Dynamic Bayesian Inverse Inference)** 框架，通过蒙特卡洛模拟重构缺失的粉丝投票数据，进而对比不同赛制的影响。

### 1. 模型架构图

**🛠️ 模型架构流程 (Model Pipeline)**

1. **📥 数据摄入与处理 (Input & ETL)**
* 输入：比赛原始分数与淘汰结果。
* 处理：解析非结构化文本，生成时间序列长表 (Long-format Data)。


2. **🔄 贝叶斯逆向推断 (Inference Engine)**
* 🧠 **记忆模型**：基于上一周的后验分布构建本周先验 (Dynamic Prior)。
* 🎲 **随机模拟**：执行蒙特卡洛采样 () 生成潜在粉丝票数。
* 🛡️ **逻辑过滤**：剔除那些“会导致错误淘汰结果”的不合理样本。
* 📉 **后验估计**：聚合有效样本，输出粉丝得票率的统计分布。


3. **⚖️ 双盲平行模拟 (Dual Simulation)**
* 🅰️ **排名制 (Rank)**：
* 🅱️ **百分比制 (Percent)**：


4. **📊 结果分析 (Bias Analysis)**
* 计算 Spearman 相关系数。
* 识别并统计“分歧案例” (Divergence Cases)。


---


### 2. 关键算法实现

#### A. 缺失数据重构：蒙特卡洛逆向工程

由于无法直接获取粉丝票数，我们将其视为**隐变量**。

* **先验分布 (Dynamic Prior)**：假设选手的粉丝基础具有时间连续性。 周的先验均值来源于  周的后验均值（引入衰减因子  防止过度拟合）。
* **似然函数 (Likelihood as Logic Check)**：如果在某次模拟中，生成的粉丝票数导致了与历史事实（即实际被淘汰的选手）不符的结果，则该样本被视为“不可能事件”并剔除。
* **自适应松弛 (Adaptive Relaxation)**：对于极端数据（如“爆冷”），标准逻辑过滤器可能导致无解。模型引入了自适应松弛机制，允许在极高不确定性下放宽约束，确保算法的鲁棒性。

#### B. 赛制对比：双盲测试

利用重构出的完整数据集（评委分 + 估算粉丝票），我们在每一季数据上分别运行两套平行规则：

1. **排名制 (Ranking System)**：。
2. **百分比制 (Percentage System)**：。

#### C. 偏向性定义 (Bias Metrics)

我们通过以下指标定义“偏向性”：

* **Spearman 相关系数**：计算最终排名与粉丝票排名的相关性。系数越高，说明赛制越尊重粉丝意愿。
* **分歧点分析 (Divergence Analysis)**：专门捕捉“评委低分、粉丝高票”的偏科选手，观察他们在两种赛制下的存活率。

## 📂 代码结构说明 (Code Structure)

完整代码位于 `advanced_dwts_solver.py`，主要包含 `AdvancedDWTSSolver` 类，模块划分如下：

| 模块 | 方法名 | 功能描述 |
| --- | --- | --- |
| **ETL** | `preprocess_raw_data` | 处理原始宽表数据，解析淘汰周，生成时间序列长表。 |
| **Inference** | `run_inference` | **核心算法**。执行蒙特卡洛模拟，推算每位选手的粉丝得票率。 |
| **Simulation** | `calculate_both_methods` | 应用两种数学规则，生成平行宇宙下的排名结果。 |
| **Analysis** | `analyze_bias_and_contrast` | 计算相关性矩阵，输出“粉丝宠儿”保护能力的对比报告。 |

## 🚀 快速开始 (Quick Start)

### 1. 环境依赖

```bash
pip install pandas numpy scipy tqdm

```

### 2. 数据准备

请确保目录下包含比赛原始数据文件 `2026_MCM_Problem_C_Data.csv`。

### 3. 运行模型

```bash
python advanced_dwts_solver.py

```

### 4. 输出结果

程序运行结束后，将在当前目录生成以下文件：

* `dwts_comparison_inference.csv`: 包含推算的粉丝票数（隐变量显性化结果）。
* `dwts_comparison_detailed_rankings.csv`: 包含双赛制下的详细排名对比。
* `dwts_comparison_fan_bias_cases.csv`: **关键文件**，列出了所有赛制产生分歧的案例（即证明某种赛制更偏向粉丝的证据）。

## 📊 结论预览 (Findings)

基于模型运行结果，我们在控制台中输出如下结论：

1. **相关性差异**：**排名制 (Rank System)** 的最终结果与粉丝排名的 Spearman 相关系数通常高于百分比制。
2. **机制偏向**：当选手出现严重偏科（评委分极低但粉丝票极高）时，**排名制**提供了更强的“生存保护”。
* *原理*：在排名制中，粉丝投票第一名（Rank 1）的权重极高，可以强力拉升评委打分的劣势；而在百分比制中，极低的评委分（如占总分3%）会严重拖累总成绩，即便粉丝票很高也难以挽回。



---

## 📝 License

此项目代码为 2026 MCM 参赛辅助代码，遵循 MIT License。
