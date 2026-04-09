"""
完整实验自动化运行脚本
适用于IEEE TSG论文投稿
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json


class ExperimentOrchestrator:
    """实验编排器 - 管理所有实验的执行"""
    
    def __init__(self, results_dir: str = "results", use_wandb: bool = False):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self.log_file = self.results_dir / f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    def log(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def run_command(self, command: list, description: str) -> bool:
        """运行命令并记录结果"""
        self.log(f"开始: {description}")
        self.log(f"命令: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            
            self.log(f"✅ 成功: {description}")
            if result.stdout:
                self.log(f"输出: {result.stdout[:500]}")  # 限制输出长度
            
            return True
        
        except subprocess.CalledProcessError as e:
            self.log(f"❌ 失败: {description}")
            self.log(f"错误: {e.stderr[:500]}")
            return False
        
        except Exception as e:
            self.log(f"❌ 异常: {description}")
            self.log(f"错误: {str(e)}")
            return False
    
    def check_prerequisites(self) -> bool:
        """检查前置条件"""
        self.log("="*80)
        self.log("检查前置条件")
        self.log("="*80)
        
        # 检查数据文件
        required_files = [
            "processed_data/strict/alice/train_rl.pkl.gz",
            "processed_data/strict/alice/test_rl.pkl.gz",
            "processed_data/light/alice/train_rl.pkl.gz",
            "processed_data/light/alice/test_rl.pkl.gz",
            "processed_data/raw/alice/train_rl.pkl.gz",
            "processed_data/raw/alice/test_rl.pkl.gz",
        ]
        
        all_exist = True
        for file_path in required_files:
            if not Path(file_path).exists():
                self.log(f"❌ 缺失文件: {file_path}")
                all_exist = False
            else:
                self.log(f"✅ 找到文件: {file_path}")
        
        if not all_exist:
            self.log("\n⚠️  部分数据文件缺失。请先运行数据预处理：")
            self.log("   python preprocess_dkasc_data_v2.py strict")
            self.log("   python preprocess_dkasc_data_v2.py light")
            self.log("   python preprocess_dkasc_data_v2.py raw")
            return False
        
        self.log("\n✅ 所有前置条件满足")
        return True
    
    def run_experiments(self, modes: list = None, experiments: list = None):
        """运行实验"""
        if modes is None:
            modes = ['strict', 'light', 'raw']
        
        if experiments is None:
            experiments = ['lambda', 'robust', 'baselines']
        
        self.log("\n" + "="*80)
        self.log("开始运行实验")
        self.log("="*80)
        self.log(f"数据模式: {modes}")
        self.log(f"实验类型: {experiments}")
        
        results = {
            'success': [],
            'failed': [],
            'skipped': []
        }
        
        for mode in modes:
            for exp in experiments:
                exp_name = f"{exp}_{mode}"
                
                # 构建命令
                command = [
                    sys.executable,  # Python解释器
                    "run_experiments.py",
                    "--experiment", exp,
                    "--data_mode", mode,
                    "--seed", "42"
                ]
                
                if self.use_wandb:
                    command.append("--use_wandb")
                
                # 运行实验
                success = self.run_command(
                    command,
                    f"实验 {exp_name}"
                )
                
                if success:
                    results['success'].append(exp_name)
                else:
                    results['failed'].append(exp_name)
        
        # 运行实验3（分布偏移）和实验4（数据质量）
        if 'shift' in experiments or 'all' in experiments:
            success = self.run_command(
                [sys.executable, "run_experiments.py", "--experiment", "shift"],
                "实验3: 分布偏移"
            )
            if success:
                results['success'].append('shift')
            else:
                results['failed'].append('shift')
        
        if 'quality' in experiments or 'all' in experiments:
            success = self.run_command(
                [sys.executable, "run_experiments.py", "--experiment", "quality"],
                "实验4: 数据质量"
            )
            if success:
                results['success'].append('quality')
            else:
                results['failed'].append('quality')
        
        return results
    
    def generate_outputs(self):
        """生成所有输出（表格、图表、分析）"""
        self.log("\n" + "="*80)
        self.log("生成论文输出")
        self.log("="*80)
        
        outputs = []
        
        # 生成表格
        success = self.run_command(
            [sys.executable, "generate_ieee_tables.py"],
            "生成IEEE表格"
        )
        if success:
            outputs.append("tables")
        
        # 生成图表
        success = self.run_command(
            [sys.executable, "generate_ieee_figures.py"],
            "生成IEEE图表"
        )
        if success:
            outputs.append("figures")
        
        # 生成分析文本
        success = self.run_command(
            [sys.executable, "generate_ieee_analysis.py"],
            "生成IEEE分析文本"
        )
        if success:
            outputs.append("analysis")
        
        # 生成汇总
        success = self.run_command(
            [sys.executable, "generate_summary_comparison.py"],
            "生成实验汇总"
        )
        if success:
            outputs.append("summary")
        
        return outputs
    
    def create_final_report(self, exp_results: dict, output_results: list):
        """创建最终报告"""
        self.log("\n" + "="*80)
        self.log("创建最终报告")
        self.log("="*80)
        
        report = f"""# 实验完成报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 实验执行结果

### 成功的实验 ({len(exp_results['success'])})
"""
        
        for exp in exp_results['success']:
            report += f"- ✅ {exp}\n"
        
        if exp_results['failed']:
            report += f"\n### 失败的实验 ({len(exp_results['failed'])})\n"
            for exp in exp_results['failed']:
                report += f"- ❌ {exp}\n"
        
        if exp_results['skipped']:
            report += f"\n### 跳过的实验 ({len(exp_results['skipped'])})\n"
            for exp in exp_results['skipped']:
                report += f"- ⏭️  {exp}\n"
        
        report += f"""

## 输出文件生成

"""
        
        if 'tables' in output_results:
            report += """### 表格
- ✅ TABLE1_data_quality.md
- ✅ TABLE2_distribution_shift.md
- ✅ TABLE3_ablation_study.md
- ✅ TABLE4_lambda_tradeoff.md
- ✅ TABLE5_robust_comparison.md

"""
        
        if 'figures' in output_results:
            report += """### 图表
- ✅ FIGURE1_convergence_curves.pdf
- ✅ FIGURE2_data_quality.pdf
- ✅ FIGURE3_distribution_shift.pdf
- ✅ FIGURE4_reward_tail_distribution.pdf

"""
        
        if 'analysis' in output_results:
            report += """### 分析文本
- ✅ IEEE_ANALYSIS_REPORT.md

"""
        
        if 'summary' in output_results:
            report += """### 汇总报告
- ✅ EXPERIMENT_SUMMARY_REPORT.md
- ✅ experiment_summary.json

"""
        
        report += f"""

## 文件位置

所有结果保存在 `{self.results_dir}` 目录下。

## 下一步

1. 查看分析报告: `cat {self.results_dir}/IEEE_ANALYSIS_REPORT.md`
2. 查看图表: 打开 `{self.results_dir}/FIGURE*.pdf`
3. 查看表格: 打开 `{self.results_dir}/TABLE*.md`

## 论文写作建议

### 实验部分

使用生成的表格和图表：
- 表1-3：放在Results部分
- 图1-4：放在Results和Discussion部分

### 分析部分

参考 `IEEE_ANALYSIS_REPORT.md` 中的分析文本，包括：
- 收敛性分析
- 鲁棒性分析
- 分布偏移分析
- 风险敏感性分析
- 数据质量影响分析

### 投稿清单

- [ ] 检查所有图表是否清晰（300 DPI）
- [ ] 确认表格格式符合IEEE标准
- [ ] 验证所有数值的准确性
- [ ] 添加适当的引用和说明
- [ ] 准备补充材料（如需要）

---

*此报告由自动化脚本生成*
"""
        
        # 保存报告
        report_file = self.results_dir / "FINAL_REPORT.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        self.log(f"✅ 最终报告已保存: {report_file}")
        
        return report_file
    
    def run_complete_pipeline(self, modes: list = None, experiments: list = None):
        """运行完整流程"""
        self.log("="*80)
        self.log("IEEE TSG 完整实验流程")
        self.log("="*80)
        self.log(f"日志文件: {self.log_file}")
        
        # 1. 检查前置条件
        if not self.check_prerequisites():
            self.log("\n❌ 前置条件检查失败，终止执行")
            return False
        
        # 2. 运行实验
        exp_results = self.run_experiments(modes, experiments)
        
        # 3. 生成输出
        output_results = self.generate_outputs()
        
        # 4. 创建最终报告
        report_file = self.create_final_report(exp_results, output_results)
        
        # 5. 总结
        self.log("\n" + "="*80)
        self.log("实验流程完成")
        self.log("="*80)
        self.log(f"成功: {len(exp_results['success'])} 个实验")
        self.log(f"失败: {len(exp_results['failed'])} 个实验")
        self.log(f"输出: {len(output_results)} 类文件")
        self.log(f"\n最终报告: {report_file}")
        self.log(f"日志文件: {self.log_file}")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='IEEE TSG完整实验自动化运行脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 运行所有实验
  python run_complete_experiments.py --all

  # 只运行STRICT模式
  python run_complete_experiments.py --modes strict

  # 只运行lambda和robust实验
  python run_complete_experiments.py --experiments lambda robust

  # 启用WandB日志
  python run_complete_experiments.py --all --use_wandb

  # 只生成输出（不运行实验）
  python run_complete_experiments.py --generate_only
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='运行所有实验和生成所有输出')
    parser.add_argument('--modes', nargs='+', 
                       choices=['strict', 'light', 'raw'],
                       help='指定数据模式')
    parser.add_argument('--experiments', nargs='+',
                       choices=['lambda', 'robust', 'baselines', 'shift', 'quality'],
                       help='指定实验类型')
    parser.add_argument('--generate_only', action='store_true',
                       help='只生成输出文件，不运行实验')
    parser.add_argument('--use_wandb', action='store_true',
                       help='启用WandB日志记录')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='结果保存目录')
    
    args = parser.parse_args()
    
    # 创建编排器
    orchestrator = ExperimentOrchestrator(
        results_dir=args.results_dir,
        use_wandb=args.use_wandb
    )
    
    if args.generate_only:
        # 只生成输出
        orchestrator.log("只生成输出模式")
        output_results = orchestrator.generate_outputs()
        orchestrator.create_final_report({'success': [], 'failed': [], 'skipped': []}, output_results)
    
    elif args.all:
        # 运行完整流程
        orchestrator.run_complete_pipeline()
    
    else:
        # 自定义运行
        modes = args.modes if args.modes else ['strict', 'light', 'raw']
        experiments = args.experiments if args.experiments else ['lambda', 'robust', 'baselines']
        
        orchestrator.run_complete_pipeline(modes, experiments)


if __name__ == "__main__":
    main()
