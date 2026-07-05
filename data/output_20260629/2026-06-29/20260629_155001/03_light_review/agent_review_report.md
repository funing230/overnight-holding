好的，这是针对候选池的 pre-close 审查分析，包括人类可读的分析部分和 JSON 输出。

---

**人类可读分析**

**1. 市场与风格概览**
当前候选池共 15 只，主板占绝对主导（13 只），另有科创板与创业板各 1 只。整体结构偏向大盘蓝筹与周期性行业。从 `live_return_vs_pre_close`（相对昨日收盘涨幅）看，日内有普涨特征，幅度在 +0.5% 到 +8.1% 之间，多数集中在 +1.5% ~ +3.3%。需要注意的是科创板寒武纪的单票波动极大，且日内最高与现价背离严重，可能对整个组合的稳定性构成干扰。

**2. 行业集中度**
行业分布较散，但存在两个小集中区：火力发电（华电国际、华能国际）和铜（江西铜业、铜陵有色）。其余行业各一只。若最终 Top5 同时纳入两只同行业个股，会降低组合分散度，但单一行业两只尚在可接受范围内。水电（国投电力）与煤炭（兖矿能源）亦有较强表现，形成新旧能源共同占优的格局。

**3. 个股审查要点**
- **华电国际**：尾盘距日内最高仅 -0.22%，几乎收在全天最高价，`live_range_pos` 高达 0.95，结构极强且波动可控，是典型的一夜持股优选标的。
- **兖矿能源**：`overnight_live_score` 排名第一，日内从高点仅微幅回落，`from_day_high` -0.38%，强势煤炭股，尾部动能充足。
- **国投电力**：水电防御属性强，尾盘同样紧贴日高，波动小，适合稳健型尾盘介入。
- **华能国际**：与华电同属火电，日内位置稍弱（`live_range_pos` 0.83），但整体健康，可作为火电板块的次优选择。
- **江西铜业**：涨幅近 5%，回撤控制在 1% 以内，铜价强势驱动，流动性充裕，具备弹性。
- **铜陵有色**：铜板块二弟，涨幅适中（2.4%），回撤更小（-0.62%），安全边际较高。
- **德业股份**：当日涨幅最大（+8.06%），但过度拉伸可能透支次日隔夜空间，追涨风险偏高，需谨慎对待。
- **汇川技术、美的集团**：均为制造龙头，涨幅 3% 左右，结构中性，可作为分散配置，但弹性偏温和。
- **寒武纪**：科创板超高价标的，日内冲高后大幅回落（`from_day_high` -3.89%），`live_range_pos` 仅 0.51，显示午后抛压沉重，尾盘弱势结构明显。高波动叠加盘口脆弱性，不适合作为尾盘买入、次日开盘卖出的策略标的。
- **建设银行、京沪高铁**：涨幅极低，波动率小，隔夜收益预期有限，不适合弹性优先的策略。
- **申万宏源**：券商股今日表现平平，同样缺乏超额收益潜力。

**4. 风险提示**
- **极端波动风险**：寒武纪（科创板）日内振幅超 8%，且收盘大幅脱离高点，尾盘承接力不足，存在次日大幅低开或继续调整的风险。
- **过度拉伸风险**：德业股份单日涨幅已创候选池最高，小时级别乖离率较大，不宜列为最优先。
- **流动性风险**：全场个股成交额均较为充裕，未出现流动性困境。
- **行业政策/事件风险**：未发现需要特别警示的消息面信号，全部候选股均通过实时风险过滤器。

**5. 组合构造建议**
建议 14:55 最终出票时，优先采纳以下排序逻辑：
1. 强力首推 **华电国际**、**兖矿能源**、**国投电力**，三者兼具高评分与极低回撤优势。
2. 次选 **华能国际** 或 **江西铜业**，作为周期板块补充；若希望分散火电，可选铜陵有色替代华能国际。
3. 中等评级的美的集团、汇川技术、盐湖股份可作为第五位的候补，但预期弹性略逊。
4. 对 **寒武纪** 启动 VETO，不建议进入 Top3/Top5；德业股份虽不 VETO，但应大幅下调其排名。
5. 银行、铁路、券商类标的整体下调，避免占据宝贵名额。

---

AGENT_REVIEW_JSON_START
{
  "trade_date": "2026-06-29",
  "target_top_n": 5,
  "reviews": [
    {
      "ts_code": "688256.SH",
      "agent_score": 0.25,
      "agent_risk_level": "high",
      "agent_veto": true,
      "agent_adjustment": -0.2,
      "agent_reason": "科创板高波动，日内从高点大幅回落，尾盘结构弱势，不适合一夜持股"
    },
    {
      "ts_code": "600188.SH",
      "agent_score": 0.9,
      "agent_risk_level": "low",
      "agent_veto": false,
      "agent_adjustment": 0.15,
      "agent_reason": "评分最高，尾盘紧贴日高，回撤极小，煤炭强势"
    },
    {
      "ts_code": "600886.SH",
      "agent_score": 0.88,
      "agent_risk_level": "low",
      "agent_veto": false,
      "agent_adjustment": 0.13,
      "agent_reason": "水电防御性强，尾盘结构健康，回撤小"
    },
    {
      "ts_code": "600027.SH",
      "agent_score": 0.92,
      "agent_risk_level": "low",
      "agent_veto": false,
      "agent_adjustment": 0.18,
      "agent_reason": "火电日内几乎收于最高，极低回撤，结构最优"
    },
    {
      "ts_code": "600011.SH",
      "agent_score": 0.82,
      "agent_risk_level": "low",
      "agent_veto": false,
      "agent_adjustment": 0.1,
      "agent_reason": "火电次选，尾盘位置较高但略弱于华电国际"
    },
    {
      "ts_code": "600362.SH",
      "agent_score": 0.78,
      "agent_risk_level": "low",
      "agent_veto": false,
      "agent_adjustment": 0.08,
      "agent_reason": "铜板块强势，涨幅较高且回撤可控"
    },
    {
      "ts_code": "603799.SH",
      "agent_score": 0.55,
      "agent_risk_level": "medium",
      "agent_veto": false,
      "agent_adjustment": -0.05,
      "agent_reason": "小金属股，日内回撤略大，动能相对偏弱"
    },
    {
      "ts_code": "605117.SH",
      "agent_score": 0.6,
      "agent_risk_level": "medium",
      "agent_veto": false,
      "agent_adjustment": -0.1,
      "agent_reason": "涨幅过高存在透支风险，需谨慎追入"
    },
    {
      "ts_code": "000630.SZ",
      "agent_score": 0.68,
      "agent_risk_level": "low",
      "agent_veto": false,
      "agent_adjustment": 0.03,
      "agent_reason": "铜板块，涨幅适中，回撤小，流动性良好"
    },
    {
      "ts_code": "000333.SZ",
      "agent_score": 0.65,
      "agent_risk_level": "medium",
      "agent_veto": false,
      "agent_adjustment": 0.0,
      "agent_reason": "家电龙头，涨幅稳健但弹性一般"
    },
    {
      "ts_code": "000792.SZ",
      "agent_score": 0.6,
      "agent_risk_level": "medium",
      "agent_veto": false,
      "agent_adjustment": -0.02,
      "agent_reason": "农药化肥，日内回撤较大，上冲动能稍弱"
    },
    {
      "ts_code": "601939.SH",
      "agent_score": 0.3,
      "agent_risk_level": "low",
      "agent_veto": false,
      "agent_adjustment": -0.15,
      "agent_reason": "银行股波动极低，隔夜收益预期有限"
    },
    {
      "ts_code": "300124.SZ",
      "agent_score": 0.62,
      "agent_risk_level": "medium",
      "agent_veto": false,
      "agent_adjustment": 0.0,
      "agent_reason": "创业板工控龙头，涨幅适中，但波动大于主板"
    },
    {
      "ts_code": "601816.SH",
      "agent_score": 0.3,
      "agent_risk_level": "low",
      "agent_veto": false,
      "agent_adjustment": -0.15,
      "agent_reason": "铁路低波动标的，弹性不足，不适合短炒"
    },
    {
      "ts_code": "000166.SZ",
      "agent_score": 0.35,
      "agent_risk_level": "low",
      "agent_veto": false,
      "agent_adjustment": -0.1,
      "agent_reason": "券商涨幅较小，日内弹性一般"
    }
  ]
}
AGENT_REVIEW_JSON_END
