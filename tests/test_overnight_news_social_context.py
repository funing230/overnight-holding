from dataflows.overnight_news_social_context import (
    _dedupe_articles,
    _parse_markdown_news,
    summarize_news_social_context,
    NewsContextResult,
    _match_social_hot_to_candidates,
    _parse_tophub_rows,
    _match_theme_hot_to_candidates,
)
import pandas as pd


def test_parse_markdown_news_basic():
    text = """## 000001.SZ News\n\n### Title A (source: Yahoo Finance)\nSummary line 1\nLink: https://example.com/a\n\n### Title B (source: EastMoney)\nPublished: 2026-05-14 10:00:00\nLink: https://example.com/b\n"""
    items = _parse_markdown_news(text, ticker="000001.SZ")
    assert len(items) == 2
    assert items[0]["title"] == "Title A"
    assert items[0]["ticker"] == "000001.SZ"
    assert items[0]["link"] == "https://example.com/a"
    assert items[1]["source"] == "EastMoney"


def test_dedupe_articles_by_title_and_ticker():
    items = [
        {"title": "Same Title", "ticker": "000001.SZ"},
        {"title": "Same Title", "ticker": "000001.SZ"},
        {"title": "Same Title", "ticker": "000002.SZ"},
    ]
    deduped = _dedupe_articles(items)
    assert len(deduped) == 2


def test_summarize_news_social_context_counts():
    ctx = NewsContextResult(
        news_top10=[{"ticker": "000001.SZ"}, {"ticker": "000002.SZ"}],
        social_top10=[{"ticker": "000001.SZ"}],
        global_news_top10=[{"ticker": "GLOBAL"}],
        vendor="akshare",
        notes=["ok"],
    )
    summary = summarize_news_social_context(ctx)
    assert summary["vendor"] == "akshare"
    assert summary["news_top10_count"] == 2
    assert summary["social_top10_count"] == 1
    assert summary["news_ticker_counts"]["000001.SZ"] == 1


def test_match_social_hot_to_candidates_builds_soft_bonus_features():
    candidate_pool = pd.DataFrame(
        {
            "ts_code": ["000333.SZ", "600941.SH", "601728.SH"],
            "name": ["美的集团", "中国移动", "中国电信"],
        }
    )
    hot_items = [
        {"source_type": "weibo", "source_rank": 3, "title": "美的集团发布新动作", "summary": "美的再成焦点", "published_at": "2026-05-14 12:00:00"},
        {"source_type": "zhihu", "source_rank": 9, "title": "如何看中国移动估值", "summary": "中国移动讨论升温", "published_at": "2026-05-14 13:00:00"},
        {"source_type": "bilibili", "source_rank": 5, "title": "美的家电到底强在哪", "summary": "Midea", "published_at": "2026-05-14 14:00:00"},
    ]
    rows, summary = _match_social_hot_to_candidates(candidate_pool, hot_items, "2026-05-14")
    by_code = {x["ts_code"]: x for x in rows}
    assert by_code["000333.SZ"]["hot_mention_count"] == 2
    assert by_code["000333.SZ"]["hot_source_count"] == 2
    assert by_code["000333.SZ"]["hot_best_rank"] == 3
    assert by_code["000333.SZ"]["social_bonus_score"] > 0.0
    assert by_code["600941.SH"]["hot_mention_count"] == 1
    assert summary["matched_candidate_count"] == 2


def test_parse_tophub_rows_weibo_table():
    html = """
    <table>
      <tbody>
        <tr>
          <td align="center">1.</td>
          <td><a href="https://s.weibo.com/weibo?q=test1" target="_blank">中美领导人致辞</a></td>
          <td class="ws">197万</td>
          <td align="right"><a href="https://s.weibo.com/weibo?q=test1" title="查看详细"></a></td>
        </tr>
        <tr>
          <td align="center">2.</td>
          <td><a href="https://s.weibo.com/weibo?q=test2" target="_blank">iPhone17Pro降价2000</a></td>
          <td class="ws">163万</td>
          <td align="right"><a href="https://s.weibo.com/weibo?q=test2" title="查看详细"></a></td>
        </tr>
      </tbody>
    </table>
    """
    items = _parse_tophub_rows(html, source="weibo", limit=10)
    assert len(items) == 2
    assert items[0]["source_type"] == "weibo"
    assert items[0]["source_rank"] == 1
    assert items[0]["title"] == "中美领导人致辞"
    assert items[0]["summary"] == "197万"
    assert items[0]["url"] == "https://s.weibo.com/weibo?q=test1"
    assert items[1]["source_rank"] == 2


def test_parse_tophub_rows_zhihu_table_with_cover_column():
    html = """
    <table>
      <tbody>
        <tr>
          <td align="center">1.</td>
          <td class="al" align="center"><img src="https://example.com/cover.png" width="100" height="64" /></td>
          <td class="al">
            <div><a href="https://www.zhihu.com/question/1" target="_blank">中美元首举行会谈，有哪些信息值得关注？</a></div>
          </td>
          <td>1206 万热度</td>
          <td align="right"><a href="https://www.zhihu.com/question/1" title="查看详细"></a></td>
        </tr>
      </tbody>
    </table>
    """
    items = _parse_tophub_rows(html, source="zhihu", limit=10)
    assert len(items) == 1
    assert items[0]["source_type"] == "zhihu"
    assert items[0]["source_rank"] == 1
    assert items[0]["title"] == "中美元首举行会谈，有哪些信息值得关注？"
    assert items[0]["summary"] == "1206 万热度"
    assert items[0]["url"] == "https://www.zhihu.com/question/1"


def test_match_theme_hot_to_candidates_builds_theme_bonus_features():
    candidate_pool = pd.DataFrame(
        {
            "ts_code": ["688256.SH", "600050.SH", "002594.SZ", "601888.SH"],
            "name": ["寒武纪", "中国联通", "比亚迪", "中国中免"],
            "industry": ["半导体", "电信运营", "汽车整车", "旅游服务"],
        }
    )
    news_items = [
        {"title": "信通院启动人工智能终端智能化分级测试工作 加速推进新国标落地实施", "summary": "AI 终端与国标推进"},
        {"title": "预告：国新办18日举行新闻发布会 介绍加力优化离境退税措施扩大入境消费有关情况", "summary": "离境退税与入境消费"},
    ]
    social_items = [
        {"title": "中美元首举行会谈，有哪些信息值得关注？", "summary": "1904 万热度"},
        {"title": "国家反诈中心APP 检测AI图", "summary": "37万"},
    ]
    rows, summary = _match_theme_hot_to_candidates(candidate_pool, news_items, social_items)
    by_code = {x["ts_code"]: x for x in rows}
    assert by_code["688256.SH"]["theme_match_count"] > 0
    assert by_code["600050.SH"]["theme_match_count"] > 0
    assert by_code["002594.SZ"]["theme_match_count"] > 0
    assert by_code["601888.SH"]["theme_match_count"] > 0
    assert by_code["688256.SH"]["theme_bonus_score"] > 0.0
    assert summary["matched_candidate_count"] >= 3
