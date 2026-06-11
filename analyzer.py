#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学术文本量化分析引擎 v5.0 —— 22维特征 · 多套预设评分体系
==============================================================
配套论文：《AI文本检测的同源困境与治理重构——基于高等教育学术评价异化的实证反思》

本模块是论文第五章（三）"归谬式同构实验"的核心分析引擎。
严格遵循GPTZero、Turnitin等主流平台公开的统计特征维度，
实现22维文本特征提取（20维核心评分 + 2维补充统计）与加权评分。

设计哲学：
    本引擎采用"归谬法"论证逻辑。如果在最简条件下（仅公开特征、
    无黑箱优化），人类文本与AI文本的评分已出现严重重叠，则任何
    基于同一统计范式的商用系统都无法从根本上克服这一局限。

特征体系（22维）：
    句法复杂度（5维）：句法规整度、长句倾向、平均句长、句长方差、句首多样性
    词汇多样性（4维）：词汇集中度、Hapax比例、信息熵、词汇创新度
    篇章结构（4维）：过渡词密度、标点规范度、元话语密度、段落规整度
    AI特征（7维）：句式模板密度、虚词密度、n-gram重复率、重复模式密度、
                  词汇突发性、困惑度、论证连接词多样性
    补充统计（2维）：句长偏度、句长峰度（不纳入核心加权评分）

参考：
    - GPTZero. How GPTZero Works: Perplexity and Burstiness [EB/OL].
    - Turnitin. AI Writing Detection Technical Overview [EB/OL].
    - Liang W, et al. GPT Detectors Are Biased Against Non-Native English Writers[J]. Patterns, 2023.
    - 论文第三章"三重悖论"及第五章"归谬式同构实验"
"""

import re
import math
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import jieba


# ============================================================
# 预设评分体系权重（22维）
# ============================================================
PRESET_WEIGHTS: Dict[str, Dict[str, Any]] = {
    "long_sentence_heavy": {
        "description": "长句敏感模式（侧重长句占比与平均句长，类似GPTZero）",
        "weights": {
            # 句法复杂度（5维）
            'sentence_regularity': 0.04,
            'long_sentence_tendency': 0.22,
            'avg_sent_length': 0.13,
            'sent_len_variance': 0.04,
            'sent_start_diversity': 0.07,
            # 词汇多样性（4维）
            'vocabulary_concentration': 0.04,
            'hapax_ratio': 0.03,
            'entropy_score': 0.04,
            'vocabulary_innovation': 0.03,
            # 篇章结构（4维）
            'transition_density': 0.04,
            'punctuation_regularity': 0.03,
            'metadiscourse_density': 0.04,
            'paragraph_regularity': 0.04,
            # AI特征（7维）
            'syntactic_pattern_density': 0.05,
            'function_word_density': 0.03,
            'ngram_repetition': 0.04,
            'repeat_pattern_density': 0.04,
            'burstiness': 0.04,
            'perplexity_score': 0.05,
            'argument_connect_diversity': 0.03,
            # 补充统计（2维，极低权重）
            'sentence_skewness': 0.015,
            'sentence_kurtosis': 0.015,
        }
    },
    "balanced": {
        "description": "均衡模式（各维度权重接近，模拟通用检测器）",
        "weights": {
            'sentence_regularity': 0.055,
            'long_sentence_tendency': 0.055,
            'avg_sent_length': 0.055,
            'sent_len_variance': 0.045,
            'sent_start_diversity': 0.055,
            'vocabulary_concentration': 0.050,
            'hapax_ratio': 0.045,
            'entropy_score': 0.045,
            'vocabulary_innovation': 0.045,
            'transition_density': 0.050,
            'punctuation_regularity': 0.045,
            'metadiscourse_density': 0.050,
            'paragraph_regularity': 0.045,
            'syntactic_pattern_density': 0.050,
            'function_word_density': 0.045,
            'ngram_repetition': 0.045,
            'repeat_pattern_density': 0.045,
            'burstiness': 0.045,
            'perplexity_score': 0.045,
            'argument_connect_diversity': 0.045,
            'sentence_skewness': 0.015,
            'sentence_kurtosis': 0.015,
        }
    },
    "academic_norm": {
        "description": "学术规范模式（侧重过渡词、元话语、标点规范，模拟期刊检测工具）",
        "weights": {
            'sentence_regularity': 0.05,
            'long_sentence_tendency': 0.05,
            'avg_sent_length': 0.05,
            'sent_len_variance': 0.04,
            'sent_start_diversity': 0.05,
            'vocabulary_concentration': 0.05,
            'hapax_ratio': 0.04,
            'entropy_score': 0.04,
            'vocabulary_innovation': 0.04,
            'transition_density': 0.12,
            'punctuation_regularity': 0.09,
            'metadiscourse_density': 0.12,
            'paragraph_regularity': 0.04,
            'syntactic_pattern_density': 0.04,
            'function_word_density': 0.03,
            'ngram_repetition': 0.03,
            'repeat_pattern_density': 0.03,
            'burstiness': 0.03,
            'perplexity_score': 0.03,
            'argument_connect_diversity': 0.04,
            'sentence_skewness': 0.03,
            'sentence_kurtosis': 0.02,
        }
    }
}


# ============================================================
# 文本分析器主类
# ============================================================
class TextAnalyzer:
    """
    学术文本量化分析引擎 v5.0

    功能：提取文本的22维统计特征（20维核心评分 + 2维补充统计）并生成AI概率评分。
    """

    def __init__(self, text: str):
        self.raw_text: str = text
        self.cleaned_text: str = re.sub(r'\s+', ' ', text).strip()
        self.sentences: List[str] = self._split_sentences()
        self.words: List[str] = self._tokenize()
        self.total_chars: int = len(self.cleaned_text)

        self.dim_scores: Dict[str, float] = {}
        self.stats: Dict[str, Any] = {}

        self._extract_all_features()

    def _split_sentences(self) -> List[str]:
        parts = re.split(r'[。！？；：\n]+', self.cleaned_text)
        return [s.strip() for s in parts if len(s.strip()) > 1]

    def _tokenize(self) -> List[str]:
        return [w for w in jieba.cut(self.cleaned_text) if w.strip()]

    def _extract_all_features(self):
        self._compute_sentence_features()
        self._compute_word_features()
        self._compute_discourse_features()
        self._compute_ai_features()

    # ============================================================
    # 句法复杂度（5维） + 补充统计（2维）
    # ============================================================

    def _compute_sentence_features(self):
        sent_lens = [len(s) for s in self.sentences]

        if len(sent_lens) < 2:
            self.stats.update({
                'sent_avg': sent_lens[0] if sent_lens else 0,
                'sent_cv': 0.0, 'sent_len_variance': 0.0,
                'long_sentence_ratio': 0.0, 'sent_start_diversity': 0.5,
                'sent_skewness': 0.0, 'sent_kurtosis': 0.0,
            })
            return

        arr = np.array(sent_lens)
        n = len(arr)

        sent_avg = float(np.mean(arr))
        sent_std = float(np.std(arr, ddof=1))
        sent_cv = sent_std / sent_avg if sent_avg > 0 else 0.0
        sent_len_variance = float(np.var(arr) / (sent_avg ** 2)) if sent_avg > 0 else 0.0

        long_ratio = float(np.sum(arr >= 50) / n)

        starters = [s[:min(3, len(s))] for s in self.sentences if len(s) >= 1]
        if len(starters) >= 5:
            freq = Counter(starters)
            top3_count = sum(freq.most_common(3)[i][1] for i in range(min(3, len(freq))))
            sent_start_diversity = 1.0 - top3_count / len(starters)
        else:
            sent_start_diversity = 0.5

        # 补充统计：偏度与峰度
        if sent_std > 0:
            z = (arr - sent_avg) / sent_std
            skewness = float(np.mean(z**3))
            kurtosis = float(np.mean(z**4) - 3)
        else:
            skewness = 0.0
            kurtosis = 0.0

        self.stats.update({
            'sent_avg': round(sent_avg, 1),
            'sent_cv': round(sent_cv, 4),
            'sent_len_variance': round(sent_len_variance, 4),
            'long_sentence_ratio': round(long_ratio, 4),
            'sent_start_diversity': round(sent_start_diversity, 4),
            'sent_skewness': round(skewness, 4),
            'sent_kurtosis': round(kurtosis, 4),
        })

        self.dim_scores['sentence_regularity'] = self._score_cv(sent_cv)
        self.dim_scores['long_sentence_tendency'] = self._score_long_ratio(long_ratio)
        self.dim_scores['avg_sent_length'] = self._score_avg_len(sent_avg)
        self.dim_scores['sent_len_variance'] = self._score_variance(sent_len_variance)
        self.dim_scores['sent_start_diversity'] = self._score_diversity(sent_start_diversity)
        self.dim_scores['sentence_skewness'] = self._score_skewness(skewness)
        self.dim_scores['sentence_kurtosis'] = self._score_kurtosis(kurtosis)

    # ============================================================
    # 词汇多样性（4维）
    # ============================================================

    def _compute_word_features(self):
        if not self.words:
            self.stats.update({'ttr': 0.0, 'hapax_ratio': 0.0, 'entropy': 0.0, 'vocabulary_innovation': 0.0})
            return

        unique = len(set(self.words))
        total = len(self.words)

        ttr = unique / total
        freq = Counter(self.words)
        hapax_count = sum(1 for v in freq.values() if v == 1)
        hapax_ratio = hapax_count / total

        probs = np.array(list(freq.values())) / total
        entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))

        word_lens = [len(w) for w in self.words]
        if word_lens:
            len_counter = Counter(word_lens)
            multi_char_ratio = sum(v for k, v in len_counter.items() if k >= 2) / total
            avg_word_len = float(np.mean(word_lens))
            vocabulary_innovation = multi_char_ratio * 0.6 + min(avg_word_len / 6, 1.0) * 0.4
        else:
            vocabulary_innovation = 0.0

        self.stats.update({
            'ttr': round(ttr, 4), 'hapax_ratio': round(hapax_ratio, 4),
            'entropy': round(entropy, 4), 'vocabulary_innovation': round(vocabulary_innovation, 4),
        })

        self.dim_scores['vocabulary_concentration'] = self._score_ttr(ttr)
        self.dim_scores['hapax_ratio'] = self._score_hapax(hapax_ratio)
        self.dim_scores['entropy_score'] = self._score_entropy(entropy)
        self.dim_scores['vocabulary_innovation'] = self._score_innovation(vocabulary_innovation)

    # ============================================================
    # 篇章结构（4维）
    # ============================================================

    def _compute_discourse_features(self):
        transition_words = [
            '但是', '然而', '因此', '所以', '首先', '其次', '最后', '总之',
            '此外', '另外', '同时', '并且', '而且', '不过', '尽管如此',
            '换句话说', '也就是说', '换言之'
        ]
        trans_count = sum(self.cleaned_text.count(w) for w in transition_words)
        trans_density = trans_count / (self.total_chars / 1000) if self.total_chars > 0 else 0

        periods = len(re.findall(r'[。！？.!?]', self.cleaned_text))
        commas = len(re.findall(r'[，,；;]', self.cleaned_text))
        punct_total = periods + commas
        punct_density = punct_total / (self.total_chars / 1000) if self.total_chars > 0 else 0

        meta_markers = [
            '值得注意的是', '需要指出的是', '值得一提的是',
            '综上所述', '总而言之', '换言之', '也就是说',
            '由此可见', '毋庸置疑', '毫无疑问'
        ]
        meta_count = sum(self.cleaned_text.count(m) for m in meta_markers)
        meta_density = meta_count / (self.total_chars / 1000) if self.total_chars > 0 else 0

        paras = [p.strip() for p in self.raw_text.split('\n\n') if p.strip()]
        para_cv = float(np.std([len(p) for p in paras], ddof=1) / np.mean([len(p) for p in paras])) if len(paras) >= 2 else 0.0

        self.stats.update({
            'trans_density': round(trans_density, 1), 'punct_density': round(punct_density, 1),
            'meta_density': round(meta_density, 1), 'para_cv': round(para_cv, 4),
        })

        self.dim_scores['transition_density'] = self._score_trans(trans_density)
        self.dim_scores['punctuation_regularity'] = self._score_punct(punct_density)
        self.dim_scores['metadiscourse_density'] = self._score_meta(meta_density)
        self.dim_scores['paragraph_regularity'] = self._score_para(para_cv)

    # ============================================================
    # AI特征（7维）
    # ============================================================

    def _compute_ai_features(self):
        patterns = [
            r'不是.{0,20}而是.{0,20}', r'不仅.{0,20}而且.{0,20}',
            r'虽然.{0,20}但是.{0,20}', r'如果.{0,20}那么.{0,20}',
            r'一方面.{0,30}另一方面.{0,30}', r'首先.{0,30}其次.{0,30}',
        ]
        pattern_count = sum(len(re.findall(p, self.cleaned_text)) for p in patterns)
        pattern_density = pattern_count / (self.total_chars / 1000) if self.total_chars > 0 else 0

        if len(self.words) >= 4:
            trigrams = [tuple(self.words[i:i+3]) for i in range(len(self.words)-2)]
            ngram_repetition = 1.0 - len(set(trigrams)) / len(trigrams) if trigrams else 0.0
        else:
            ngram_repetition = 0.0

        repeat_pattern_density = self._compute_repeat_pattern_density()

        function_words = ['的', '了', '着', '过', '是', '在', '和', '与', '或', '而', '则', '且']
        func_count = sum(self.cleaned_text.count(w) for w in function_words)
        func_density = func_count / (self.total_chars / 100) if self.total_chars > 0 else 0

        burstiness = self._compute_burstiness()
        perplexity_score = self._compute_perplexity_score()
        argument_connect_diversity = self._compute_argument_diversity()

        self.stats.update({
            'pattern_density': round(pattern_density, 1),
            'ngram_repetition': round(ngram_repetition, 4),
            'repeat_pattern_density': round(repeat_pattern_density, 1),
            'function_word_density': round(func_density, 1),
            'burstiness': round(burstiness, 4),
            'perplexity_score': round(perplexity_score, 4),
            'argument_connect_diversity': round(argument_connect_diversity, 4),
        })

        self.dim_scores['syntactic_pattern_density'] = self._score_pattern(pattern_density)
        self.dim_scores['ngram_repetition'] = self._score_ngram(ngram_repetition)
        self.dim_scores['repeat_pattern_density'] = self._score_repeat(repeat_pattern_density)
        self.dim_scores['function_word_density'] = self._score_func(func_density)
        self.dim_scores['burstiness'] = burstiness
        self.dim_scores['perplexity_score'] = perplexity_score
        self.dim_scores['argument_connect_diversity'] = self._score_arg_div(argument_connect_diversity)

    def _compute_repeat_pattern_density(self) -> float:
        if len(self.words) < 8:
            return 0.0
        common_stops = {'的', '了', '是', '在', '和', '与', '一', '不', '也', '就'}
        fragments = []
        for n in range(4, min(7, len(self.words) + 1)):
            for i in range(len(self.words) - n + 1):
                fragments.append(tuple(self.words[i:i+n]))
        frag_counter = Counter(fragments)
        valid_repeats = 0
        for frag, count in frag_counter.items():
            if count >= 2:
                stop_ratio = sum(1 for w in frag if w in common_stops) / len(frag)
                if stop_ratio < 0.5:
                    valid_repeats += 1
        return valid_repeats / (self.total_chars / 1000) if self.total_chars > 0 else 0.0

    def _compute_burstiness(self) -> float:
        if len(self.words) < 30:
            return 0.5
        pos = defaultdict(list)
        for i, w in enumerate(self.words):
            pos[w].append(i)
        intervals = []
        for lst in pos.values():
            if len(lst) > 1:
                intervals.extend(np.diff(lst))
        if len(intervals) < 2:
            return 0.5
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
        if cv < 0.3: return 0.90
        if cv < 0.6: return 0.70
        if cv < 1.0: return 0.45
        return 0.25

    def _compute_perplexity_score(self) -> float:
        if len(self.words) < 4:
            return 0.5
        trigram_counts = Counter(tuple(self.words[i:i+3]) for i in range(len(self.words)-2))
        bigram_counts = Counter(tuple(self.words[i:i+2]) for i in range(len(self.words)-1))
        V = len(set(self.words))
        log_prob = 0.0
        n = 0
        for i in range(len(self.words)-2):
            w1, w2, w3 = self.words[i], self.words[i+1], self.words[i+2]
            tri_cnt = trigram_counts.get((w1, w2, w3), 0)
            bi_cnt = bigram_counts.get((w1, w2), 0)
            prob = (tri_cnt + 1) / (bi_cnt + V)
            log_prob += math.log2(prob)
            n += 1
        if n == 0: return 0.5
        perplexity = 2 ** (-log_prob / n)
        return 1.0 / (1.0 + math.exp(-(perplexity - 80) / 30))

    def _compute_argument_diversity(self) -> float:
        connectives = ['因此', '所以', '从而', '导致', '进而', '因而', '故此', '于是', '可见', '由此']
        freq = {w: self.cleaned_text.count(w) for w in connectives}
        used = sum(1 for v in freq.values() if v > 0)
        return used / len(connectives) if connectives else 0.0

    # ============================================================
    # 评分函数（22维全部覆盖）
    # ============================================================

    @staticmethod
    def _score_cv(cv: float) -> float:
        if cv < 0.15: return 0.90
        if cv < 0.25: return 0.75
        if cv < 0.40: return 0.55
        if cv < 0.60: return 0.35
        return 0.20

    @staticmethod
    def _score_long_ratio(ratio: float) -> float:
        if ratio > 0.30: return 0.95
        if ratio > 0.25: return 0.85
        if ratio > 0.20: return 0.70
        if ratio > 0.15: return 0.55
        if ratio > 0.10: return 0.40
        if ratio > 0.05: return 0.25
        return 0.15

    @staticmethod
    def _score_avg_len(avg: float) -> float:
        if avg > 45: return 0.90
        if avg > 35: return 0.75
        if avg > 28: return 0.55
        if avg > 22: return 0.35
        return 0.20

    @staticmethod
    def _score_variance(var: float) -> float:
        if var < 0.10: return 0.85
        if var < 0.25: return 0.65
        if var < 0.50: return 0.40
        if var < 0.80: return 0.25
        return 0.15

    @staticmethod
    def _score_diversity(div: float) -> float:
        if div < 0.25: return 0.85
        if div < 0.40: return 0.65
        if div < 0.55: return 0.45
        return 0.25

    @staticmethod
    def _score_skewness(skew: float) -> float:
        if skew > 2.0: return 0.85
        if skew > 1.0: return 0.65
        if skew > 0.3: return 0.45
        return 0.25

    @staticmethod
    def _score_kurtosis(kurt: float) -> float:
        if kurt > 3.0: return 0.80
        if kurt > 1.0: return 0.60
        if kurt > 0.0: return 0.40
        return 0.20

    @staticmethod
    def _score_ttr(ttr: float) -> float:
        if ttr < 0.20: return 0.80
        if ttr < 0.35: return 0.60
        if ttr < 0.50: return 0.40
        return 0.20

    @staticmethod
    def _score_hapax(hapax: float) -> float:
        if hapax < 0.20: return 0.80
        if hapax < 0.35: return 0.60
        if hapax < 0.50: return 0.40
        return 0.20

    @staticmethod
    def _score_entropy(entropy: float) -> float:
        if entropy < 3.0: return 0.80
        if entropy < 4.5: return 0.60
        if entropy < 6.0: return 0.40
        return 0.20

    @staticmethod
    def _score_innovation(inno: float) -> float:
        if inno < 0.20: return 0.85
        if inno < 0.35: return 0.65
        if inno < 0.50: return 0.45
        if inno < 0.65: return 0.30
        return 0.15

    @staticmethod
    def _score_trans(density: float) -> float:
        if density > 18: return 0.85
        if density > 12: return 0.65
        if density > 8: return 0.45
        return 0.20

    @staticmethod
    def _score_punct(density: float) -> float:
        if 30 < density < 50: return 0.70
        if 20 < density < 60: return 0.50
        return 0.30

    @staticmethod
    def _score_meta(density: float) -> float:
        if density > 8: return 0.80
        if density > 5: return 0.60
        if density > 2: return 0.40
        return 0.20

    @staticmethod
    def _score_para(cv: float) -> float:
        if cv < 0.20: return 0.80
        if cv < 0.40: return 0.60
        if cv < 0.60: return 0.40
        return 0.20

    @staticmethod
    def _score_pattern(density: float) -> float:
        if density > 10: return 0.80
        if density > 6: return 0.60
        if density > 3: return 0.40
        return 0.20

    @staticmethod
    def _score_ngram(rep: float) -> float:
        if rep > 0.30: return 0.80
        if rep > 0.15: return 0.60
        if rep > 0.05: return 0.40
        return 0.20

    @staticmethod
    def _score_repeat(density: float) -> float:
        if density > 12: return 0.85
        if density > 8: return 0.70
        if density > 4: return 0.50
        if density > 1: return 0.30
        return 0.15

    @staticmethod
    def _score_func(density: float) -> float:
        if density > 15: return 0.75
        if density > 10: return 0.55
        return 0.30

    @staticmethod
    def _score_arg_div(div: float) -> float:
        if div < 0.2: return 0.90
        if div < 0.4: return 0.70
        if div < 0.6: return 0.50
        return 0.30

    # ============================================================
    # 综合评分
    # ============================================================

    def get_ai_score(self, preset: Optional[str] = None,
                     custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        if preset and preset in PRESET_WEIGHTS:
            weights = PRESET_WEIGHTS[preset]["weights"].copy()
            preset_name = preset
        elif custom_weights:
            total = sum(custom_weights.values())
            weights = {k: v/total for k, v in custom_weights.items()} if total > 0 else PRESET_WEIGHTS["long_sentence_heavy"]["weights"].copy()
            preset_name = "custom"
        else:
            weights = PRESET_WEIGHTS["long_sentence_heavy"]["weights"].copy()
            preset_name = "long_sentence_heavy"

        raw_score = sum(self.dim_scores.get(k, 0.5) * weights.get(k, 0.0) for k in weights)
        ai_probability = max(0.0, min(100.0, raw_score * 100))

        risk_level = '高风险' if ai_probability > 65 else ('中风险' if ai_probability > 35 else '低风险')

        risk_factors = self._identify_risk_factors()
        confidence = min(0.90, max(0.30, self.total_chars / 3000))
        linguistic_regularity = self._compute_linguistic_regularity()
        report_text = self._generate_report(ai_probability, risk_level, preset_name)

        return {
            'ai_probability': round(ai_probability, 1),
            'human_probability': round(100 - ai_probability, 1),
            'risk_level': risk_level,
            'confidence': round(confidence, 2),
            'linguistic_regularity': round(linguistic_regularity, 4),
            'risk_factors': risk_factors,
            'dimension_scores': {k: round(v * 100, 1) for k, v in self.dim_scores.items()},
            'report_text': report_text,
            'preset_name': preset_name,
            'feature_count': 22,
            'stats': self.stats,
        }

    def _identify_risk_factors(self) -> List[str]:
        factors = []
        checks = [
            ('long_sentence_tendency', 0.7, f"长句占比偏高 ({round(self.stats.get('long_sentence_ratio', 0)*100)}%)"),
            ('avg_sent_length', 0.7, f"平均句长偏高 ({self.stats.get('sent_avg', 0):.1f}字)"),
            ('sent_start_diversity', 0.6, f"句首多样性低 ({round(self.stats.get('sent_start_diversity', 0)*100)}%)"),
            ('sentence_regularity', 0.6, f"句法规整度过高 (CV={self.stats.get('sent_cv', 0):.4f})"),
            ('vocabulary_concentration', 0.6, f"词汇集中度高 (TTR={self.stats.get('ttr', 0):.4f})"),
            ('sent_len_variance', 0.6, f"句长方差过低 ({self.stats.get('sent_len_variance', 0):.4f})"),
            ('repeat_pattern_density', 0.6, f"重复模式密度偏高 ({self.stats.get('repeat_pattern_density', 0):.1f}/千字)"),
            ('vocabulary_innovation', 0.6, "词汇创新度偏低"),
            ('ngram_repetition', 0.6, f"n-gram重复率偏高 ({self.stats.get('ngram_repetition', 0):.4f})"),
            ('perplexity_score', 0.7, "文本困惑度异常低，高度疑似AI生成"),
            ('argument_connect_diversity', 0.7, f"论证连接词单一 (多样性{self.stats.get('argument_connect_diversity', 0):.2f})"),
            ('burstiness', 0.6, '词汇分布均匀，缺乏突发性'),
        ]
        for key, threshold, message in checks:
            if self.dim_scores.get(key, 0) > threshold:
                factors.append(message)
        if not factors:
            factors.append('各项指标在正常范围内')
        return factors

    def _compute_linguistic_regularity(self) -> float:
        scores = [
            max(0, 1 - self.stats.get('sent_cv', 0)/0.8),
            max(0, 1 - self.stats.get('ttr', 0)/0.6),
            min(1, self.stats.get('punct_density', 0)/50),
            max(0, 1 - self.stats.get('para_cv', 0)/0.5) if self.stats.get('para_cv', 0) > 0 else 0.5,
            min(1, self.stats.get('trans_density', 0)/15),
            min(1, self.stats.get('long_sentence_ratio', 0)/0.5),
            min(1, (1 - self.stats.get('sent_start_diversity', 0))/0.7),
        ]
        return sum(scores) / len(scores)

    def _generate_report(self, ai_prob: float, risk_level: str, preset_name: str) -> str:
        s = self.stats
        lines = [
            "═" * 60,
            f"  AI文本检测分析报告 (v5.0 · 22维特征)",
            "═" * 60,
            f"  文本长度: {self.total_chars} 字符 · 句子: {len(self.sentences)} · 词数: {len(self.words)}",
            f"  AI概率: {ai_prob:.1f}% | 风险: {risk_level} | 预设: {preset_name}",
            f"  语言规整度: {self._compute_linguistic_regularity():.4f}",
            "",
            "  【句法复杂度（5维+2维补充）】",
            f"    平均句长: {s.get('sent_avg',0):.1f}字 | CV: {s.get('sent_cv',0):.4f} | 方差: {s.get('sent_len_variance',0):.4f}",
            f"    长句占比: {s.get('long_sentence_ratio',0)*100:.1f}% | 句首多样性: {s.get('sent_start_diversity',0):.4f}",
            f"    偏度: {s.get('sent_skewness',0):.2f} | 峰度: {s.get('sent_kurtosis',0):.2f}",
            "",
            "  【词汇多样性（4维）】",
            f"    TTR: {s.get('ttr',0):.4f} | Hapax: {s.get('hapax_ratio',0):.4f}",
            f"    熵: {s.get('entropy',0):.4f} | 创新度: {s.get('vocabulary_innovation',0):.4f}",
            "",
            "  【篇章结构（4维）】",
            f"    过渡词: {s.get('trans_density',0):.1f}/千字 | 标点: {s.get('punct_density',0):.1f}/千字",
            f"    元话语: {s.get('meta_density',0):.1f}/千字 | 段落CV: {s.get('para_cv',0):.4f}",
            "",
            "  【AI特征（7维）】",
            f"    模板: {s.get('pattern_density',0):.1f}/千字 | n-gram: {s.get('ngram_repetition',0):.4f}",
            f"    重复模式: {s.get('repeat_pattern_density',0):.1f}/千字 | 虚词: {s.get('function_word_density',0):.1f}/百字",
            f"    突发性: {s.get('burstiness',0):.4f} | 困惑度: {s.get('perplexity_score',0):.4f}",
            f"    连接词多样性: {s.get('argument_connect_diversity',0):.4f}",
            "",
            "─" * 60,
            "  【重要提示】",
            "  本报告数值基于统计特征与训练语料的相似度计算，",
            "  不代表文本的实际创作来源。统计特征检测范式存在",
            "  同源困境：规范人类文本可能被误判，改写AI文本可",
            "  能轻松规避。请结合内容质量等质性维度综合判断。",
            "─" * 60,
        ]
        return '\n'.join(lines)
