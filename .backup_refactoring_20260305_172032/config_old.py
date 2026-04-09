"""
Configuration File for DR3L PV-BESS System
光伏储能系统配置文件
"""

class Config:
    """全局配置类"""
    
    # ========== 基础配置 ==========
    SEED = 42
    
    # ========== 光伏系统参数 ==========
    PV_CAPACITY = 2.0  # MW，光伏容量
    
    # ========== 储能系统参数 (BESS) ==========
    BESS_CAPACITY = 4.0  # MWh，储能容量
    BESS_POWER_RATED = 1.0  # MW，额定功率
    BESS_SOC_MIN = 0.1  # 最小 SoC
    BESS_SOC_MAX = 0.9  # 最大 SoC
    BESS_SOC_INIT = 0.5  # 初始 SoC
    BESS_ETA_CHARGE = 0.95  # 充电效率
    BESS_ETA_DISCHARGE = 0.95  # 放电效率
    BESS_RAMP_RATE = 0.25  # 爬坡率限制（每步最大变化率）
    
    # ========== 负荷参数 ==========
    LOAD_BASE = 0.8  # MW，基础负荷
    LOAD_PEAK = 1.5  # MW，峰值负荷
    
    # ========== 电价参数 ==========
    PRICE_BUY = 100.0  # $/MWh，购电价格
    PRICE_SELL = 80.0  # $/MWh，售电价格
    PRICE_PEAK_MULTIPLIER = 1.5  # 峰时电价倍数
    
    # ========== CVaR 风险参数 ==========
    CVAR_ALPHA = 0.05  # CVaR 置信水平（关注最差5%的情况）
    CVAR_WINDOW = 100  # CVaR 计算的滑动窗口大小
    LAMBDA_CVAR = 1.0  # CVaR 惩罚权重
    LAMBDA_EXTREME = 0.5  # 极端偏差惩罚权重
    
    # ========== 违约惩罚参数（新增）==========
    LAMBDA_VIOLATION = 1.0  # 违约惩罚权重
    LAMBDA_RAMP_VIOLATION = 0.5  # ramp违约惩罚权重
    VIOLATION_THRESHOLD = 0.01  # 违约率阈值（1%）
    RAMP_VIOLATION_THRESHOLD = 0.05  # ramp违约率阈值（5%）
    
    # ========== 惩罚系数 ==========
    PENALTY_ENERGY_GAP = 200.0  # 能量缺口惩罚系数
    PENALTY_SOC_VIOLATION = 500.0  # SoC 越界惩罚系数
    PENALTY_RAMP_VIOLATION = 300.0  # 功率爬坡惩罚系数
    EXTREME_THRESHOLD = 0.3  # 极端偏差阈值（30%）
    
    # ========== 训练参数 ==========
    MAX_EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 288  # 24小时 * 12步/小时（5分钟间隔）
    BATCH_SIZE = 64
    LEARNING_RATE_ACTOR = 1e-4
    LEARNING_RATE_CRITIC = 1e-3
    GAMMA = 0.99  # 折扣因子
    TAU = 0.005  # 软更新系数
    
    # ========== 路径配置 ==========
    LOG_DIR = "./logs"
    MODEL_DIR = "./models"
    FIGURE_DIR = "./figures"
    TENSORBOARD_LOG = "./runs"
    DATA_DIR = "./processed_data"
    DATA_PATH = "./Dataset/DKASC"  # DKASC数据集路径
    
    # ========== 数据集配置 ==========
    MASTER_METER_FILE = "master_meter.csv"  # 主表文件名
    WINDOW_SIZE = 24  # 滑动窗口大小（时间步数）
    TRAIN_RATIO = 0.7  # 训练集比例
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    VAL_RATIO = 0.15  # 验证集比例
    TEST_SPLIT = 0.15
