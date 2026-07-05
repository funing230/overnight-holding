好的，我们先做人类可读分析，然后输出 JSON。

---

## 一、市场与行业结构

- 候选池共 15 只，主板 12 只、创业板 3 只，行业极度分散（14 个行业），仅“电气设备”出现 2 只（阳光电源、德业股份），无过度集中风险。
- 池内股票均为全市场筛选后的一夜持股候选，均通过初步风控（live_pass_risk_filter=true，无 reject reasons）。
- openclaw_context 未加载，我们仅依据盘中量价、历史隔夜数据及 overnight_live_score 进行判断。

## 二、个股评价与风险提示

以下按原始 overnight_live_score 从高到低梳理：

| 股票 | score | 日内涨幅 | 高位回落 | 日区间位置 | 历史隔夜胜率(5日) | 主要风险 |
|------|-------|----------|----------|------------|-------------------|----------|
| 东方盛虹 | 0.7190 | +1.33% | -0.35% | 94% | 0.2 | 化纤，尾盘极强势但历史隔夜样本表现一般 |
| 铜陵有色 | 0.7058 | -0.16% | -0.93% | 78% | 0.2 | 日内收跌，历史隔夜均值-4.3%，不建议一夜持有 |
| 东方财富 | 0.7050 | +1.39% | -0.92% | 77% | 0.2 | 创业板头部券商，流动性极佳，但隔夜正向率仅 0.2 |
| 华友钴业 | 0.7019 | +1.22% | -0.66% | 86% | 0.4 | 小金属，历史隔夜正向率稍好，风险可控 |
| 阳光电源 | 0.7009 | +2.17% | -0.77% | 84% | 0.2 | 创业板权重，日内涨幅较好，但历史隔夜为负 |
| 东鹏饮料 | 0.6984 | +1.18% | -0.65% | 82% | 0.2 | 消费蓝筹，流动性良好，稳健 |
| 海尔智家 | 0.6951 | -0.05% | -0.78% | 72% | 0.2 | 白电龙头，当日几乎平盘，防守属性强，隔夜弹性有限 |
| 奔图科技 | 0.6933 | +2.19% | -1.33% | 66% | 0.2 | IT设备小票，尾盘从偏低位置拉升，但历史隔夜偏弱 |
| 盐湖股份 | 0.6794 | +1.89% | -1.79% | 63% | 0.2 | 农药化肥，从日高位大幅回落，尾盘跟随度一般 |
| 汇川技术 | 0.6753 | +1.69% | -1.41% | 67% | 0.4 | 电器仪表，历史隔夜曾录得正收益，位置偏低有修复预期 |
| 德业股份 | 0.6538 | +1.83% | -0.88% | 82% | 0.2 | 电气设备，score偏低，与阳光电源同行业，注意行业微集中 |
| 古井贡酒 | 0.6506 | +1.22% | -0.25% | 93% | 0.2 | 白酒小票，成交额约 2.7 亿，流动性稍弱，次日开盘滑点风险 |
| 合盛硅业 | 0.6273 | +6.18% | -1.45% | 79% | 0.6 | 化工原料，当日大涨但 model 评分低，历史隔夜胜率 0.6 |
| 川投能源 | 0.6247 | +0.14% | -0.07% | 97% | 0.6 | 水电防御股，流动性较低，预期隔夜收益空间小 |
| 恒立液压 | 0.6112 | +0.36% | -2.20% | 58% | 0.6 | 工程机械，尾盘弱势，当日从高位大幅回落 |

## 三、风险与组合构造考虑

- **不宣否决（veto）**：未发现需要直接剔除的异常波动、流动性枯竭或基本面事件，所有股票 agent_veto=false。
- **需谨慎下调的标的**：铜陵有色（日内收跌+历史隔夜极差）、恒立液压（尾盘结构弱）、古井贡酒（流动性偏差）。
- **积极上调的标的**：合盛硅业（日内强势+历史隔夜胜率高）、东方财富/东方盛虹/华友钴业（强势收盘或流动性护体）。
- **行业分散已自然实现**，若最终 top5 选择电气设备两只（阳光电源+德业股份）可接受，但需注意权重。

## 四、14:55 前排序调整建议（融合后参考）

1. 东方盛虹 – 强势、高 score，可作重点候选。
2. 东方财富 – 高流动性、隔夜预期平稳。
3. 华友钴业 – 胜率稍高，风险可控。
4. 东鹏饮料 – 防御+稳健。
5. 合盛硅业 – 日内最强，值得博取惯性。

实际最终 top5 应以 14:55 的实时行情再次确认。

---

AGENT_REVIEW_JSON_START
{
  "trade_date": "2026-06-30",
  "target_top_n": 5,
  "reviews": [
    {
      "ts_code": "300274.SZ",
      "agent_score": 0.69,
      "agent_risk_level": "medium",
      "agent_veto": false,
      "agent_adjustment": -0.02,
      "agent_reason": "日内涨2.17%但历史隔夜负收益，下调排序"
    },
    {
      "ts_code": "000301.SZ",
      "agent_score": 0.73,
      "agent_risk_level": "medium",
      "agent_veto": false,
      "agent_adjustment": 0.02,
      "agent_reason": "尾盘极度强势，日区间高位，上调"
    },
    {
      "ts_code": "300059.SZ",
      "agent_score": 0.72,
      "agent_risk_level": "low",
      "agent_veto": false,
      "agent_adjustment": 0.03,
      "agent_reason": "流动性极好，大盘蓝筹，稳健上调"
    },
    {
      "ts_code": "605499.SH",
      "agent_score": 0.71,
      "agent_risk_level": "low",
      "agent_veto": false,
      "agent_adjustment": 0.01,
      "agent_reason": "消费蓝筹，走势稳健，小幅上调"
    },
    {
      "ts_code": "000630.SZ",
      "agent_score": 0.63,
      "agent_risk_level": "medium",
      "agent_veto": false,
      "agent_adjustment": -0.08,
      "agent_reason": "日内微跌，历史隔夜均值-4.3%，显著下调"
    },
    {
      "ts_code": "603799.SH",
      "agent_score": 0.72,
      "agent_risk_level": "medium",
      "agent_veto": false,
      "agent_adjustment": 0.02,
      "agent_reason": "隔夜正向率0.4，日区间高位，上调"
    },
    {
      "ts_code": "600690.SH",
      "agent_score": 0.69,
      "agent_risk_level": "low",
      "agent_veto": false,
      "agent_adjustment": 0.0,
      "agent_reason": "平盘防守型，隔夜弹性有限，维持中性"
    },
    {
      "ts_code": "002180.SZ",
      "agent_score": 0.68,
      "agent_risk_level": "medium",
      "agent_veto": false,
      "agent_adjustment": -0.01,
      "agent_reason": "低位拉升但历史隔夜偏弱，微下调"
    },
    {
      "ts_code": "300124.SZ",
      "agent_score": 0.69,
      "agent_risk_level": "medium",
      "agent_veto": false,
      "agent_adjustment": 0.01,
      "agent_reason": "历史隔夜有正收益，偏低位置，小幅上调"
    },
    {
      "ts_code": "605117.SH",
      "agent_score": 0.65,
      "agent_risk_level": "medium",
      "agent_veto": false,
      "agent_adjustment": -0.02,
      "agent_reason": "同行业已有更优标的，score偏低，下调"
    },
    {
      "ts_code": "000792.SZ",
      "agent_score": 0.67,
      "agent_risk_level": "medium",
      "agent_veto": false,
      "agent_adjustment": -0.02,
      "agent_reason": "高位回落较大，尾盘动能减弱，下调"
    },
    {
      "ts_code": "600674.SH",
      "agent_score": 0.63,
      "agent_risk_level": "low",
      "agent_veto": false,
      "agent_adjustment": 0.01,
      "agent_reason": "高胜率但波动小，流动性一般，微上调"
    },
    {
      "ts_code": "603260.SH",
      "agent_score": 0.69,
      "agent_risk_level": "medium",
      "agent_veto": false,
      "agent_adjustment": 0.08,
      "agent_reason": "日内大涨6.18%，历史隔夜胜率高，显著上调"
    },
    {
      "ts_code": "000596.SZ",
      "agent_score": 0.62,
      "agent_risk_level": "medium",
      "agent_veto": false,
      "agent_adjustment": -0.05,
      "agent_reason": "成交额偏小，次日开盘滑点风险，下调"
    },
    {
      "ts_code": "601100.SH",
      "agent_score": 0.59,
      "agent_risk_level": "medium",
      "agent_veto": false,
      "agent_adjustment": -0.08,
      "agent_reason": "尾盘弱势，日高位回落2.2%，显著下调"
    }
  ]
}
AGENT_REVIEW_JSON_END
