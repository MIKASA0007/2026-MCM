import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr
from tqdm import tqdm
import warnings
import os

# 全局配置
warnings.filterwarnings('ignore')
np.random.seed(2026)
INPUT_FILE = '2026_MCM_Problem_C_Data.csv'
OUTPUT_PREFIX = 'dwts_comparison_'

class AdvancedDWTSSolver:
    def __init__(self, raw_data_path):
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"未找到文件: {raw_data_path}")
        self.raw_df = pd.read_csv(raw_data_path)
        self.long_df = None
        self.inference_df = None
        self.final_df = None
        self.total_vote_pool = 14000000

    # ==========================================================
    # 模块 1: 数据预处理 (ETL) - 保持不变
    # ==========================================================
    def preprocess_raw_data(self):
        print(f"[1/4] 读取并清洗原始数据...")
        long_data = []
        for _, row in self.raw_df.iterrows():
            season = row['season']
            name = row['celebrity_name']
            res_text = str(row['results'])
            
            elim_week = 99
            if 'Eliminated Week' in res_text:
                try:
                    elim_week = int(res_text.split('Week')[-1].strip())
                except: pass
            
            for w in range(1, 16):
                cols = [c for c in self.raw_df.columns if f'week{w}_' in c and 'score' in c]
                if not cols: break
                week_scores = [row[c] for c in cols if pd.notnull(row[c])]
                if not week_scores: continue
                
                judge_total = sum(week_scores)
                if judge_total == 0 and w > 1: continue
                if w > elim_week: continue
                
                long_data.append({
                    'season': season, 'week': w, 'celebrity_name': name,
                    'judge_score': judge_total, 'actual_elimination': 1 if w == elim_week else 0
                })
        
        self.long_df = pd.DataFrame(long_data)
        self.long_df.sort_values(['season', 'week'], inplace=True)
        return self.long_df

    # ==========================================================
    # 模块 2: 逆向推断 (Inference) - 保持不变
    # ==========================================================
    def _get_rank_score(self, judge_vals, fan_vals):
        j_ranks = rankdata(-np.array(judge_vals), method='min')
        f_ranks = rankdata(-np.array(fan_vals), method='min')
        return j_ranks + f_ranks

    def run_inference(self, n_simulations=3000):
        print(f"[2/4] 运行粉丝票数逆向推断 (N={n_simulations})...")
        if self.long_df is None: self.preprocess_raw_data()
        results = []
        
        for season, seas_group in tqdm(self.long_df.groupby('season'), desc="Seasons"):
            prev_week_priors = {}
            for week in sorted(seas_group['week'].unique()):
                group = seas_group[seas_group['week'] == week]
                contestants = group['celebrity_name'].tolist()
                judge_scores = group['judge_score'].values
                eliminated = group['actual_elimination'].values
                n = len(contestants)
                if n < 2: continue
                
                # 动态先验构建
                prior_means = np.zeros(n)
                prior_std = np.ones(n) * 0.5
                for i, name in enumerate(contestants):
                    if name in prev_week_priors:
                        prior_means[i] = prev_week_priors[name] * 0.9
                        prior_std[i] = 0.3

                # 采样与筛选
                raw_log = np.random.normal(prior_means, prior_std, size=(n_simulations, n))
                raw_samples = np.exp(raw_log)
                valid_samples = []
                
                # 使用宽松的排名规则作为过滤器，确保推断的票数在逻辑上可行
                elim_idx = np.where(eliminated == 1)[0]
                for sample in raw_samples:
                    if len(elim_idx) == 0: 
                        valid_samples.append(sample)
                        continue
                    
                    # 检查：被淘汰者是否处于危险区（排名倒数）
                    # 这是一个通用逻辑，用于过滤掉明显不可能的票数分布
                    t = self._get_rank_score(judge_scores, sample)
                    sorted_idx = np.argsort(-t) # 降序，头部为最差
                    threshold = len(elim_idx) + 2 # 宽松阈值
                    worst_k = sorted_idx[:threshold]
                    
                    if all(idx in worst_k for idx in elim_idx):
                        valid_samples.append(sample)

                # 参数估计
                if not valid_samples:
                    est_share = np.exp(prior_means) / np.exp(prior_means).sum()
                else:
                    valid_arr = np.array(valid_samples)
                    shares = valid_arr / valid_arr.sum(axis=1, keepdims=True)
                    est_share = shares.mean(axis=0)
                    
                    # 更新先验记忆
                    log_shares = np.log(shares + 1e-9).mean(axis=0)
                    log_shares -= log_shares.mean()
                    for i, name in enumerate(contestants):
                        prev_week_priors[name] = log_shares[i]

                for i, name in enumerate(contestants):
                    results.append({
                        'season': season, 'week': week, 'celebrity_name': name,
                        'judge_score': judge_scores[i], 
                        'est_vote_share': est_share[i],
                        'est_votes_count': int(est_share[i] * self.total_vote_pool)
                    })
        self.inference_df = pd.DataFrame(results)
        return self.inference_df

    # ==========================================================
    # 模块 3: 双系统规则计算 (Calculation)
    # ==========================================================
    def calculate_both_methods(self):
        print("[3/4] 同时应用【排名制】与【百分比制】计算排名...")
        if self.inference_df is None: self.run_inference()
        
        results_list = []
        df = self.inference_df.copy()
        
        for (season, week), group in df.groupby(['season', 'week']):
            g = group.copy()
            n = len(g)
            
            # --- 方法 A: 排名制 (Rank System) ---
            # 逻辑：Judge Rank + Fan Rank = Total Rank. 数值越小越好。
            g['Judge Rank'] = g['judge_score'].rank(ascending=False, method='min')
            g['Fan Rank'] = g['est_vote_share'].rank(ascending=False, method='min')
            g['Rank Sum'] = g['Judge Rank'] + g['Fan Rank']
            
            # 排名生成：Rank Sum 小者在前；若平分，Fan Rank 小者(票多)在前
            g_rank = g.sort_values(by=['Rank Sum', 'Fan Rank'], ascending=[True, True])
            # 创建排名映射表 {index: rank_place}
            rank_map = dict(zip(g_rank.index, range(1, n+1)))
            g['Place (Rank System)'] = g.index.map(rank_map)
            
            # --- 方法 B: 百分比制 (Percentage System) ---
            # 逻辑：Judge % + Fan % = Total %. 数值越大越好。
            j_sum = g['judge_score'].sum()
            g['Judge Pct'] = g['judge_score'] / j_sum if j_sum > 0 else 0
            g['Fan Pct'] = g['est_vote_share']
            g['Total Pct'] = g['Judge Pct'] + g['Fan Pct']
            
            # 排名生成：Total Pct 大者在前；若平分，Fan Pct 大者在前
            g_pct = g.sort_values(by=['Total Pct', 'Fan Pct'], ascending=[False, False])
            pct_map = dict(zip(g_pct.index, range(1, n+1)))
            g['Place (Percent System)'] = g.index.map(pct_map)
            
            results_list.append(g)
            
        self.final_df = pd.concat(results_list)
        # 保存双系统对比表
        self.final_df.to_csv(f"{OUTPUT_PREFIX}detailed_rankings.csv", index=False)
        return self.final_df

    # ==========================================================
    # 模块 4: 差异对比与偏向性分析 (Modified Analysis)
    # ==========================================================
    def analyze_bias_and_contrast(self):
        print(f"[4/4] 生成差异对比与粉丝偏向性分析报告...")
        df = self.final_df.copy()
        
        # 1. 全局相关性对比 (Global Correlation Comparison)
        # 比较两种最终排名方式与“粉丝票数排名”的 Spearman 相关系数。
        # 系数越高(越接近1)，说明最终结果与粉丝意愿越一致。
        
        stats = []
        for (s, w), g in df.groupby(['season', 'week']):
            if len(g) < 3: continue
            
            # 两种系统最终排名 vs 粉丝票排名的相关性
            # 注意：排名数值都是“越小越好”，所以预期是正相关
            r_rank_fan = spearmanr(g['Place (Rank System)'], g['Fan Rank']).correlation
            r_pct_fan = spearmanr(g['Place (Percent System)'], g['Fan Rank']).correlation
            
            stats.append({
                'season': s, 'week': w, 
                'corr_rank_fan': r_rank_fan, 
                'corr_pct_fan': r_pct_fan
            })
            
        stats_df = pd.DataFrame(stats)
        
        # 2. 差异案例分析 (Divergence Analysis)
        # 专门寻找：评委分低(后50%) 但 粉丝票高(前50%) 的“偏科”选手
        # 统计这类选手在哪个系统中获得了更好的名次
        
        divergence_cases = []
        bias_counts = {'Rank System': 0, 'Percent System': 0}
        
        print("\n正在扫描历史数据中的系统分歧点...")
        for idx, row in df.iterrows():
            g = df[(df['season'] == row['season']) & (df['week'] == row['week'])]
            n = len(g)
            if n < 4: continue
            
            rank_place = row['Place (Rank System)']
            pct_place = row['Place (Percent System)']
            
            # 如果两个系统给出的排名不同
            if rank_place != pct_place:
                # 判断是否为“粉丝宠儿” (Fan Favorite: Judge Bad, Fan Good)
                judge_bad = row['Judge Rank'] > (n / 2)
                fan_good = row['Fan Rank'] <= (n / 2)
                
                if judge_bad and fan_good:
                    favored = ""
                    if rank_place < pct_place: # 数值小代表排名高
                        bias_counts['Rank System'] += 1
                        favored = "Rank System"
                    elif pct_place < rank_place:
                        bias_counts['Percent System'] += 1
                        favored = "Percent System"
                    
                    if favored:
                        divergence_cases.append({
                            'Season': row['season'],
                            'Week': row['week'],
                            'Celebrity': row['celebrity_name'],
                            'Fan Rank': row['Fan Rank'],
                            'Judge Rank': row['Judge Rank'],
                            'Rank System Place': rank_place,
                            'Percent System Place': pct_place,
                            'Better Outcome In': favored
                        })

        div_df = pd.DataFrame(divergence_cases)
        if not div_df.empty:
            div_df.to_csv(f"{OUTPUT_PREFIX}fan_bias_cases.csv", index=False)

        # 3. 输出最终结论
        print("\n" + "="*60)
        print("【对比分析结论：哪种机制更偏向粉丝？】")
        print("="*60)
        
        # 结论 A: 基于相关性
        mean_r_rank = stats_df['corr_rank_fan'].mean()
        mean_r_pct = stats_df['corr_pct_fan'].mean()
        
        print(f"1. 统计相关性 (Spearman Correlation):")
        print(f"   (数值越高，表示最终结果与粉丝排名的重合度越高)")
        print(f"   - 排名制 (Rank System) 平均相关性: {mean_r_rank:.4f}")
        print(f"   - 百分比制 (Percent System) 平均相关性: {mean_r_pct:.4f}")
        
        winner_corr = "排名制 (Rank System)" if mean_r_rank > mean_r_pct else "百分比制 (Percent System)"
        print(f"   >> 结论：【{winner_corr}】与粉丝投票结果的一致性更强。")

        # 结论 B: 基于案例
        print(f"\n2. '粉丝宠儿'保护能力 (针对 评委低分/粉丝高票 选手):")
        print(f"   在 {len(div_df)} 个分歧案例中，各系统给予这类选手更好排名的次数：")
        print(f"   - 排名制 (Rank System):   {bias_counts['Rank System']} 次")
        print(f"   - 百分比制 (Percent System): {bias_counts['Percent System']} 次")
        
        winner_bias = "排名制 (Rank System)" if bias_counts['Rank System'] > bias_counts['Percent System'] else "百分比制 (Percent System)"
        print(f"   >> 结论：当评委打压而粉丝支持时，【{winner_bias}】更容易让选手存活。")
        
        print("="*60)
        print(f"详细分歧案例已保存至: {OUTPUT_PREFIX}fan_bias_cases.csv")

# ==========================================================
# 主程序入口
# ==========================================================
if __name__ == "__main__":
    solver = AdvancedDWTSSolver(INPUT_FILE)
    solver.preprocess_raw_data()
    solver.run_inference(n_simulations=3000)
    solver.calculate_both_methods()
    solver.analyze_bias_and_contrast()