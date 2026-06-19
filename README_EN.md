
# AI Text Detection Paradigm Demo

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## ⚠️ Statement of Purpose

**This tool is an academic companion program, not a commercial AI detection product.**

It is the companion source code for the paper *The Transgression of Intelligent Digital Governance: The Dilemma and Return of AI Text Detection* (Liu et al., 2026), specifically for the cross-genre text "fluency trap" analysis in Section 4.3.

**This tool does NOT claim to "detect" AI-generated text. It demonstrates why detection, under the statistical paradigm, cannot work.**

## What This Tool Actually Does

This tool implements **22 publicly documented statistical features**—syntactic, lexical, and structural—that mainstream detectors (GPTZero, Turnitin, etc.) rely on. It uses fixed, transparent weights with **no data-trained classifier layer**.

Its purpose is to reveal a structural condition:

> **Under any fixed, transparent statistical framework, normatively regular human texts and machine-generated texts occupy overlapping feature spaces.**

This is not a claim that "this text is AI-generated." It is a demonstration that the statistical features used for detection **do not distinguish** between human and machine authorship once the text reaches a certain level of fluency and regularity.

## Why Multiple Scoring Systems?

This tool provides multiple preset scoring systems (e.g., `long_sentence_heavy`, `balanced`, `lexical_focused`). **Different systems produce different scores for the same text.**

This is not a bug. It is the point.

Commercial detectors do the same thing—they choose different features, different weights, different proprietary calibrations. The same text (Lao She's *Forest of Trees*) receives AI scores ranging from **0% to 99.9%** across different commercial platforms. This is not because the text changes. It is because the **detector's design changes**.

> **The AI score is not a property of the text. It is a property of the detector's design—its feature selection, weighting scheme, and proprietary calibration.**

## The Logic: A Reductio ad Absurdum Experiment

This tool is built on a **reductio ad absurdum** argument:

1. If even a minimal, transparent feature set—using only publicly documented statistical dimensions with fixed weights—already produces severe overlap between normative human texts and machine outputs (~55–60% for both)…
2. …then adding more opaque, data-trained layers on top cannot resolve the structural problem. It can only obscure it.

**The tool exposes what commercial black boxes hide.**

Commercial systems appear "accurate" not because they have solved the overlap, but because their proprietary classifier layers have learned to reinterpret and suppress the raw statistical proximity. Their sophistication is not evidence that the paradigm works; it is evidence that the overlap can be hidden behind an opaque decision boundary.

## How to Interpret Your Results

If you run this tool on a well-written human text and receive a "high risk" score:

- **This does NOT mean the text is AI-generated.**
- It means that, under the statistical features that all detectors rely on, your text falls into the same feature space as machine-generated text.

This phenomenon is what the paper terms the **"fluency trap"**: the more standard and fluent a human text, the more likely it is to be flagged as AI-generated.

If you switch between scoring systems and observe significant score variation, that itself is a finding: **the score is a function of the detector's design, not an intrinsic property of the text.**

## What This Tool Is Not

| Not this | But this |
|----------|----------|
| A commercial AI detector | An academic demonstration tool |
| A product for academic integrity decisions | A research instrument for exposing paradigm limits |
| A claim to be "more accurate" | A claim to be **more transparent** |

## Prohibited Use

Any use of this tool's scores as a basis for academic integrity decisions violates the fundamental stance of this research. The tool was built to show that such scores **should not be used** as the basis for high-stakes judgments.

## Features

- 22-dimensional statistical feature extraction (syntactic, lexical, structural)
- Multiple preset scoring systems + custom weight adjustment
- Visual analysis reports (radar charts, sentence distribution, etc.)
- File upload support (.txt/.docx/.pdf)
- Fully transparent and reproducible

## Quick Start

### Requirements

- Python 3.8+

### Installation and Run

```bash
git clone https://github.com/liuming-kust/ai-detect.git
cd ai-detect
pip install -r requirements.txt
python server.py
# Open http://localhost:8000 in your browser
```

## Related Paper

This tool accompanies:

- Liu M,The Transgression of Intelligent Digital Governance: The Dilemma and Return of AI Text Detection. (Under review)

The paper provides the full theoretical framework—the three-paradox causal chain, the category transgression diagnosis, and the governance implications—of which this tool is the empirical demonstration.

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



**The score is not in the text. It is in the detector's design.**
```
