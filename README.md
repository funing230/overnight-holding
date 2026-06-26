# 一夜持股法 (Overnight Holding) — A股隔夜持股量化决策系统

基于 Hermes Agent 的 A 股"一夜持股法"实盘决策系统。每日 14:30-14:57 自动运行，从全市场筛选隔夜持股候选池。

## 流水线

```
14:30  快照采集 → 全A实时行情（腾讯 qt.gtimg.cn）
14:31  确定性召回 → 17因素评分 Top50
14:32  数据补全 → 雪球+Twitter+微博热搜 + 风险扫描
14:33  Heavy审查 → deepseek-v4-pro deepthink（Top50→15）
14:49  Light审查 → qwen3.7-max chat（Top15复核）
14:54  最终价格 → 刷新实时价格兜底
14:55  FinalFusion → live×0.60 + heavy×0.25 + light×0.15
14:57  输出Top5 → CSV + Markdown报告
```

## 数据源

| 层级 | 来源 | 用途 |
|---|---|---|
| 行情 | Tushare + 腾讯 | 日线/实时价量/PE/PB |
| 基本面 | 东财 datacenter + AkShare | 利润表 |
| 情绪 | 雪球/Twitter/微博热搜 | 散户+国际情绪 |
| 风险 | 东财 datacenter | 业绩预亏/ST扫描 |
| 北向 | AkShare hsgt | 外资流向 |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
export YUANLAN_API_KEY="your_key"
export XUEQIU_COOKIE="xq_a_token=..."
# Twitter 代理等见 .env 配置

# 运行
PYTHONPATH=. python3.12 scripts/run_overnight.py --date 20260626
```

## 项目结构

```
overnight_holding/
├── config/          # 模型配置（deepseek-v4-pro, qwen3.7-max）
├── dataflows/       # 数据采集 + 评分引擎
├── llm/             # 多模型调度池
├── backtest/        # 回测引擎
├── scripts/         # 入口脚本
├── tests/           # 测试
└── data/            # 运行时输出（快照+结果）
```

## 模型

| 环节 | 模型 | 厂商 | 模式 |
|---|---|---|---|
| Heavy Review | deepseek-v4-pro | DeepSeek (via yuanlan) | deepthink |
| Light Review | qwen3.7-max | 阿里 (via yuanlan) | chat |
