"""
生成IEEE TSG论文级统计对比表
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import pandas as pd


class IEEETableGenerator:
    """生成IEEE期刊标准的对比表"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.summary_file = self.results_dir / "experiment_summary.json"
        
        # 加载汇总数据
        with open(self.summary_file, 'r') as f:
            self.data = json.load(f)
    
    def generate_table1_data_quality(self) -> str:
        """
        表1: Data Quality Comparison
        对比STRICT/LIGHT/RAW三种数据模式
        """
        print("\n生成表1: Data Quality Comparison...")
        
        # 从实验4提取数据
        exp4_data = self.data.get('exp4', {})
        
        modes = ['strict', 'light', 'raw']
        
        # 构建表格数据
        table_data = []
        for mode in modes:
            if mode in exp4_data:
                d = exp4_data[mode]
                row = {
                    'Data Mode': mode.upper(),
                    'Mean Reward': f"{d['reward_mean']:.2f}",
                    'Std Reward': f"{d['reward_std']:.2f}",
                    'CVaR↑': f"{d['cvar_mean']:.4f}",
                    'Max Loss↓': f"{d['max_loss_mean']:.4f}",
                }
                table_data.append(row)
        
        # 生成Markdown表格
        df = pd.DataFrame(table_data)
        markdown_table = df.to_markdown(index=False)
        
        # 生成LaTeX表格
        latex_table = self._generate_latex_table(
            df,
            caption="Data Quality Comparison: Performance metrics across different preprocessing modes",
            label="tab:data_quality"
        )
        
        # 保存
        output_file = self.results_dir / "TABLE1_data_quality.md"
        with open(output_file, 'w') as f:
            f.write("# Table 1: Data Quality Comparison\n\n")
            f.write("## Markdown Format\n\n")
            f.write(markdown_table)
            f.write("\n\n## LaTeX Format\n\n")
            f.write("```latex\n")
            f.write(latex_table)
            f.write("\n```\n")
        
        print(f"✅ 表1已保存: {output_file}")
        return markdown_table
    
    def generate_table2_distribution_shift(self) -> str:
        """
        表2: Distribution Shift Performance
        对比Alice和Yulara的泛化性能
        """
        print("\n生成表2: Distribution Shift Performance...")
        
        # 从实验3提取数据
        exp3_data = self.data.get('exp3', {})
        
        locations = ['alice', 'yulara']
        
        # 构建表格数据
        table_data = []
        for loc in locations:
            if loc in exp3_data:
                d = exp3_data[loc]
                row = {
                    'Test Location': loc.capitalize(),
                    'Mean Reward': f"{d['reward_mean']:.2f}",
                    'Std Reward': f"{d['reward_std']:.2f}",
                    'CVaR↑': f"{d['cvar_mean']:.4f}",
                    'Max Loss↓': f"{d['max_loss_mean']:.4f}",
                }
                table_data.append(row)
        
        # 计算性能差异
        if len(table_data) == 2:
            alice_reward = exp3_data['alice']['reward_mean']
            yulara_reward = exp3_data['yulara']['reward_mean']
            performance_gap = abs(alice_reward - yulara_reward)
            performance_gap_pct = (performance_gap / abs(alice_reward)) * 100
            
            table_data.append({
                'Test Location': 'Performance Gap',
                'Mean Reward': f"{performance_gap:.2f} ({performance_gap_pct:.2f}%)",
                'Std Reward': '-',
                'CVaR↑': '-',
                'Max Loss↓': '-',
            })
        
        # 生成Markdown表格
        df = pd.DataFrame(table_data)
        markdown_table = df.to_markdown(index=False)
        
        # 生成LaTeX表格
        latex_table = self._generate_latex_table(
            df,
            caption="Distribution Shift: Generalization performance from Alice Springs to Yulara",
            label="tab:distribution_shift"
        )
        
        # 保存
        output_file = self.results_dir / "TABLE2_distribution_shift.md"
        with open(output_file, 'w') as f:
            f.write("# Table 2: Distribution Shift Performance\n\n")
            f.write("## Markdown Format\n\n")
            f.write(markdown_table)
            f.write("\n\n## LaTeX Format\n\n")
            f.write("```latex\n")
            f.write(latex_table)
            f.write("\n```\n")
        
        print(f"✅ 表2已保存: {output_file}")
        return markdown_table
    
    def generate_table3_ablation_study(self) -> str:
        """
        表3: Ablation Study
        对比不同算法和配置
        """
        print("\n生成表3: Ablation Study (Baseline Comparison)...")
        
        # 从实验5提取STRICT模式的基线对比
        exp5_strict = self.data.get('exp5_strict', {})
        
        methods = ['DDPG', 'PPO', 'DR3L_rho0', 'DR3L_full']
        method_names = {
            'DDPG': 'DDPG (Baseline)',
            'PPO': 'PPO (Baseline)',
            'DR3L_rho0': 'DR3L (ρ=0, No Robustness)',
            'DR3L_full': 'DR3L (Full, ρ=0.05)'
        }
        
        # 构建表格数据
        table_data = []
        for method in methods:
            if method in exp5_strict:
                d = exp5_strict[method]
                row = {
                    'Method': method_names.get(method, method),
                    'Mean Reward↑': f"{d['reward_mean']:.2f}",
                    'Std Reward': f"{d['reward_std']:.2f}",
                    'CVaR↑': f"{d['cvar_mean']:.4f}",
                    'Max Loss↓': f"{d['max_loss_mean']:.4f}",
                }
                table_data.append(row)
        
        # 生成Markdown表格
        df = pd.DataFrame(table_data)
        markdown_table = df.to_markdown(index=False)
        
        # 生成LaTeX表格
        latex_table = self._generate_latex_table(
            df,
            caption="Ablation Study: Comparison of different RL algorithms and DR3L configurations",
            label="tab:ablation_study"
        )
        
        # 保存
        output_file = self.results_dir / "TABLE3_ablation_study.md"
        with open(output_file, 'w') as f:
            f.write("# Table 3: Ablation Study (Baseline Comparison)\n\n")
            f.write("## Markdown Format\n\n")
            f.write(markdown_table)
            f.write("\n\n## LaTeX Format\n\n")
            f.write("```latex\n")
            f.write(latex_table)
            f.write("\n```\n")
        
        print(f"✅ 表3已保存: {output_file}")
        return markdown_table
    
    def generate_table4_lambda_tradeoff(self) -> str:
        """
        表4: Lambda Trade-off Analysis (STRICT模式)
        """
        print("\n生成表4: Lambda Trade-off Analysis...")
        
        # 从实验1提取STRICT模式数据
        exp1_strict = self.data.get('exp1_strict', {})
        
        lambdas = [0.0, 0.1, 0.5, 1.0, 2.0]
        
        # 构建表格数据
        table_data = []
        for lam in lambdas:
            key = f"lambda_{lam}"
            if key in exp1_strict:
                d = exp1_strict[key]
                row = {
                    'λ': f"{lam}",
                    'Mean Reward': f"{d['reward_mean']:.2f}",
                    'Std Reward': f"{d['reward_std']:.2f}",
                    'CVaR↑': f"{d['cvar_mean']:.4f}",
                    'Max Loss↓': f"{d['max_loss_mean']:.4f}",
                }
                table_data.append(row)
        
        # 生成Markdown表格
        df = pd.DataFrame(table_data)
        markdown_table = df.to_markdown(index=False)
        
        # 生成LaTeX表格
        latex_table = self._generate_latex_table(
            df,
            caption="Lambda Trade-off: Impact of risk weight parameter λ on performance",
            label="tab:lambda_tradeoff"
        )
        
        # 保存
        output_file = self.results_dir / "TABLE4_lambda_tradeoff.md"
        with open(output_file, 'w') as f:
            f.write("# Table 4: Lambda Trade-off Analysis\n\n")
            f.write("## Markdown Format\n\n")
            f.write(markdown_table)
            f.write("\n\n## LaTeX Format\n\n")
            f.write("```latex\n")
            f.write(latex_table)
            f.write("\n```\n")
        
        print(f"✅ 表4已保存: {output_file}")
        return markdown_table
    
    def generate_table5_robust_comparison(self) -> str:
        """
        表5: Robustness Parameter Comparison (STRICT模式)
        """
        print("\n生成表5: Robustness Parameter Comparison...")
        
        # 从实验2提取STRICT模式数据
        exp2_strict = self.data.get('exp2_strict', {})
        
        rhos = [0.0, 0.01, 0.05]
        
        # 构建表格数据
        table_data = []
        for rho in rhos:
            key = f"rho_{rho}"
            if key in exp2_strict:
                d = exp2_strict[key]
                row = {
                    'ρ': f"{rho}",
                    'Mean Reward': f"{d['reward_mean']:.2f}",
                    'Std Reward': f"{d['reward_std']:.2f}",
                    'CVaR↑': f"{d['cvar_mean']:.4f}",
                    'Max Loss↓': f"{d['max_loss_mean']:.4f}",
                }
                table_data.append(row)
        
        # 生成Markdown表格
        df = pd.DataFrame(table_data)
        markdown_table = df.to_markdown(index=False)
        
        # 生成LaTeX表格
        latex_table = self._generate_latex_table(
            df,
            caption="Robustness Comparison: Impact of CVaR parameter ρ on risk sensitivity",
            label="tab:robust_comparison"
        )
        
        # 保存
        output_file = self.results_dir / "TABLE5_robust_comparison.md"
        with open(output_file, 'w') as f:
            f.write("# Table 5: Robustness Parameter Comparison\n\n")
            f.write("## Markdown Format\n\n")
            f.write(markdown_table)
            f.write("\n\n## LaTeX Format\n\n")
            f.write("```latex\n")
            f.write(latex_table)
            f.write("\n```\n")
        
        print(f"✅ 表5已保存: {output_file}")
        return markdown_table
    
    def _generate_latex_table(self, df: pd.DataFrame, caption: str, label: str) -> str:
        """生成LaTeX表格"""
        n_cols = len(df.columns)
        col_format = 'l' + 'c' * (n_cols - 1)
        
        latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{col_format}}}
\\hline
"""
        
        # 表头
        headers = ' & '.join(df.columns) + ' \\\\'
        latex += headers + '\n\\hline\n'
        
        # 数据行
        for _, row in df.iterrows():
            row_str = ' & '.join([str(val) for val in row.values]) + ' \\\\'
            latex += row_str + '\n'
        
        latex += """\\hline
\\end{tabular}
\\end{table}"""
        
        return latex
    
    def generate_all_tables(self):
        """生成所有表格"""
        print("="*80)
        print("生成IEEE TSG论文级统计对比表")
        print("="*80)
        
        self.generate_table1_data_quality()
        self.generate_table2_distribution_shift()
        self.generate_table3_ablation_study()
        self.generate_table4_lambda_tradeoff()
        self.generate_table5_robust_comparison()
        
        print("\n" + "="*80)
        print("✅ 所有表格生成完成！")
        print(f"保存位置: {self.results_dir}")
        print("="*80)


def main():
    """主函数"""
    generator = IEEETableGenerator()
    generator.generate_all_tables()


if __name__ == "__main__":
    main()
