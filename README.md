# AI文本检测范式演示工具

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## ⚠️ 重要声明

**本工具是学术论文的配套实证程序，并非商业AI检测产品。**

本工具是论文《AI文本检测的同源困境与治理重构——基于高等教育学术评价异化的实证反思》（liuming-kust，2026）第五章“归谬式同构实验”及附录A的配套源代码。

**设计目的**：通过复现主流检测工具公开的统计特征，揭示该范式固有的“同源困境”——规范的人类文本与AI生成文本在统计特征上无法可靠区分。

**禁止用途**：任何将本工具评分作为学术诚信判断依据的行为，均违背本研究的基本立场。

## 功能概述

- 提取文本的22维统计特征（句法复杂度、词汇多样性、篇章结构、AI特征、补充统计）
- 支持多套预设评分体系 + 自定义权重调节
- 提供可视化分析报告（雷达图、句子分布图等）
- 支持文件上传（.txt/.docx/.pdf）

## 快速开始

### 环境要求

- Python 3.8+

### 安装与运行

```bash
git clone https://github.com/liuming-kust/ai-detect.git
cd ai-detect
pip install -r requirements.txt
python server.py
# 浏览器打开 http://localhost:8000
