#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI文本检测范式演示工具 —— FastAPI 服务入口
==============================================
配套论文：《AI文本检测的同源困境与治理重构——基于高等教育学术评价异化的实证反思》

本服务是论文第五章（三）"归谬式同构实验"的配套演示程序。
其设计目的并非提供准确的AI文本判定，而是通过复现主流检测工具
的统计特征范式，揭示该范式固有的"同源困境"。

参见论文第三至五章。
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from analyzer import TextAnalyzer
import uvicorn

# ============================================================
# FastAPI 应用初始化
# ============================================================
app = FastAPI(
    title="AI文本检测范式演示工具",
    description="基于17维统计特征的文本分析引擎，配套学术论文《AI文本检测的同源困境与治理重构》",
    version="4.0.0"
)

# CORS 中间件：允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API 端点：文本分析
# ============================================================
@app.post("/api/analyze")
async def analyze(request: Request):
    """
    分析文本的AI生成概率。

    请求体 (JSON):
        text: str         - 待分析文本
        preset: str       - 预设评分体系 (balanced / long_sentence_heavy / academic_norm)
        weights: dict     - 自定义权重 (仅当 preset 为空时使用)

    返回 (JSON):
        ai_probability: float      - AI生成概率评分 (0-100)
        human_probability: float   - 人类创作概率 (0-100)
        risk_level: str            - 风险等级 (高风险 / 中风险 / 低风险)
        confidence: float          - 置信度 (0.3-0.9)
        linguistic_regularity: float - 语言规整度 (0-1)
        risk_factors: list         - 风险因子列表
        dimension_scores: dict     - 各维度得分 (0-100)
        report_text: str           - 详细分析报告
        preset_name: str           - 使用的评分体系名称
        feature_count: int         - 特征维度 (20)
        stats: dict                - 原始统计值
    """
    try:
        # 解析请求体
        data = await request.json()
        text = data.get("text", "")

        # 文本长度校验：过短文本无法进行稳定的特征提取
        if not text or len(text.strip()) < 50:
            return JSONResponse(
                status_code=400,
                content={"error": "文本过短，至少需要50个字符以进行有意义的特征提取"}
            )

        # 获取预设名称或自定义权重
        preset = data.get("preset", None)
        custom_weights = data.get("weights", None)

        # 初始化分析器并执行17维特征提取与评分
        analyzer = TextAnalyzer(text)
        result = analyzer.get_ai_score(preset=preset, custom_weights=custom_weights)

        return result

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# 挂载静态文件（前端演示界面）
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
