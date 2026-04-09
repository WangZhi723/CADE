"""
生成IEEE TSG论文级统计对比表（简化版，无外部依赖）
"""

import json
from pathlib import Path


class IEEETableGenerator:
    """生成IEEE期刊标准的对比表"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.summary_file = self.results_dir / "experiment_summary.json"
        
        # 加载汇总数据
        with open(self.summary_file, 'r') as f:
            self.data = json.load(f)
    
    def generate_markdown_table(self, headers, rows):
        """生成Markdown表格"""
        # 表头
        table = "| " + " | ".join(headers) + " |\n"
        table += "|" + "|".join(["-" * (len(h) + 2) for h in headers]) + "|\n"
        
        # 数据行
        for row in rows:
            table += "| " + " | ".join([str(cell) for cell in row]) + " |\n"
        
        return table
    
    def generate_latex_table(self, headers, rows, caption, label):
        """生成LaTeX表格"""
        n_cols = len(headers)
        col_format = 'l' + 'c' * (n_cols - 1)
        
        latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{col_format}}}
\\hline
"""
        
        # 表头
        latex += ' & '.join(headers) + ' \\\\\n\\hline\n'
        
        # 数据行
        for row in rows:
            latex += ' & '.join([str(cell) for cell in row]) + ' \\\\\n'
        
        latex += """\\hline
\\end{tabular}
\\end{table}"""
        
        return latex
    
    def generate_table1_data_quality(self):
        """表1: Data Quality Comparison"""
        print("\n生成表1: Data Quality Comparison...")
        
        exp4_data = self.data.get('exp4', {})
        modes = ['strict', 'light', 'raw']
        
        headers = ['Data Mode', 'Mean Reward', 'Std Reward', 'CVaR↑', 'Max Loss↓']
        rows = []
        
        for mode in modes:
            if mode in exp4_data:
                d = exp4_data[mode]
                row = [
                    mode.upper(),
                    f"{d['reward_mean']:.2f}",
                    f"{d['reward_std']:.2f}",
                    f"{d['cvar_mean']:.4f}",
                    f"{d['max_loss_mean']:.4f}"
                ]
                rows.append(row)
        
        markdown_table = self.generate_markdown_table(headers, rows)
        latex_table = self.generate_latex_table(
            headers, rows,
            "Data Quality Comparison: Performance metrics across different preprocessing modes",
            "tab:data_quality"
        )
        
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
    
    def generate_table2_distribution_shift(self):
        """表2: Distribution Shift Performance"""
        print("\n生成表2: Distribution Shift Performance...")
        
        exp3_data = self.data.get('exp3', {})
        locations = ['alice', 'yulara']
        
        headers = ['Test Location', 'Mean Reward', 'Std Reward', 'CVaR↑', 'Max Loss↓']
        rows = []
        
        for loc in locations:
            if loc in exp3_data:
                d = exp3_data[loc]
                row = [
                    loc.capitalize(),
                    f"{d['reward_mean']:.2f}",
                    f"{d['reward_std']:.2f}",
                    f"{d['cvar_mean']:.4f}",
                    f"{d['max_loss_mean']:.4f}"
                ]
                rows.append(row)
        
        # 计算性能差异
        if len(rows) == 2:
            alice_reward = exp3_data['alice']['reward_mean']
            yulara_reward = exp3_data['yulara']['reward_mean']
            performance_gap = abs(alice_reward - yulara_reward)
            performance_gap_pct = (performance_gap / abs(alice_reward)) * 100
            
            rows.append([
                'Performance Gap',
                f"{performance_gap:.2f} ({performance_gap_pct:.2f}%)",
                '-', '-', '-'
            ])
        
        markdown_table = self.generate_markdown_table(headers, rows)
        latex_table = self.generate_latex_table(
            headers, rows,
            "Distribution Shift: Generalization performance from Alice Springs to Yulara",
            "tab:distribution_shift"
        )
        
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
    
    def generate_table3_ablation_study(self):
        """表3: Ablation Study"""
        print("\n生成表3: Ablation Study (Baseline Comparison)...")
        
        exp5_strict = self.data.get('exp5_strict', {})
        
        methods = ['DDPG', 'PPO', 'DR3L_rho0', 'DR3L_full']
        method_names = {
            'DDPG': 'DDPG (Baseline)',
            'PPO': 'PPO (Baseline)',
            'DR3L_rho0': 'DR3L (ρ=0, No Robustness)',
            'DR3L_full': 'DR3L (Full, ρ=0.05)'
        }
        
        headers = ['Method', 'Mean Reward↑', 'Std Reward', 'CVaR↑', 'Max Loss↓']
        rows = []
        
        for method in methods:
            if method in exp5_strict:
                d = exp5_strict[method]
                row = [
                    method_names.get(method, method),
                    f"{d['reward_mean']:.2f}",
                    f"{d['reward_std']:.2f}",
                    f"{d['cvar_mean']:.4f}",
                    f"{d['max_loss_mean']:.4f}"
                ]
                rows.append(row)
        
        markdown_table = self.generate_markdown_table(headers, rows)
        latex_table = self.generate_latex_table(
            headers, rows,
            "Ablation Study: Comparison of different RL algorithms and DR3L configurations",
            "tab:ablation_study"
        )
        
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
    
    def generate_table4_lambda_tradeoff(self):
        """表4: Lambda Trade-off Analysis"""
        print("\n生成表4: Lambda Trade-off Analysis...")
        
        exp1_strict = self.data.get('exp1_strict', {})
        lambdas = [0.0, 0.1, 0.5, 1.0, 2.0]
        
        headers = ['λ', 'Mean Reward', 'Std Reward', 'CVaR↑', 'Max Loss↓']
        rows = []
        
        for lam in lambdas:
            key = f"lambda_{lam}"
            if key in exp1_strict:
                d = exp1_strict[key]
                row = [
                    f"{lam}",
                    f"{d['reward_mean']:.2f}",
                    f"{d['reward_std']:.2f}",
                    f"{d['cvar_mean']:.4f}",
                    f"{d['max_loss_mean']:.4f}"
                ]
                rows.append(row)
        
        markdown_table = self.generate_markdown_table(headers, rows)
        latex_table = self.generate_latex_table(
            headers, rows,
            "Lambda Trade-off: Impact of risk weight parameter λ on performance",
            "tab:lambda_tradeoff"
        )
        
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
    
    def generate_table5_robust_comparison(self):
        """表5: Robustness Parameter Comparison"""
        print("\n生成表5: Robustness Parameter Comparison...")
        
        exp2_strict = self.data.get('exp2_strict', {})
        rhos = [0.0, 0.01, 0.05]
        
        headers = ['ρ', 'Mean Reward', 'Std Reward', 'CVaR↑', 'Max Loss↓']
        rows = []
        
        for rho in rhos:
            key = f"rho_{rho}"
            if key in exp2_strict:
                d = exp2_strict[key]
                row = [
                    f"{rho}",
                    f"{d['reward_mean']:.2f}",
                    f"{d['reward_std']:.2f}",
                    f"{d['cvar_mean']:.4f}",
                    f"{d['max_loss_mean']:.4f}"
                ]
                rows.append(row)
        
        markdown_table = self.generate_markdown_table(headers, rows)
        latex_table = self.generate_latex_table(
            headers, rows,
            "Robustness Comparison: Impact of CVaR parameter ρ on risk sensitivity",
            "tab:robust_comparison"
        )
        
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
