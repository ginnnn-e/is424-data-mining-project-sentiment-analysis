# Financial News & Social Media Sentiment Analysis Pipeline (2020–2024)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FinBERT%20%7C%20Llama%203-yellow.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-NLP%20Pipeline-ee4c2c.svg)
![Data Scale](https://img.shields.io/badge/Corpus%20Size-2.03M%2B%20Records-success.svg)

> **Project Context:** This repository showcases my individual workstream for our IS424 Data Mining group project: [**Stock Price Direction Prediction using Machine Learning & Sentiment Analysis (`HengZhengKai/is424-data-mining-project`)**](https://github.com/HengZhengKai/is424-data-mining-project/tree/main).
>
> While the full team project combines **multi-source financial sentiment** with **technical analysis indicators** across 9 ML/DL models to predict 4–10 day stock price movements, **this repository and README focus specifically on the News & Social Media Sentiment Analysis pipeline that I owned and built.**

---

## My Role & Contributions

In our group project, **I led the end-to-end News & Social Media Sentiment Analysis workstream** (Stages 1–3 of the main pipeline), which supplied the daily sentiment features consumed by the team's downstream ML/LSTM trading models:

* **Owned Multi-Source Financial News & Social Media Collection:** Extracted historical stock news from the **FNSPID** corpus (8,500+ tickers) and built custom web scrapers for **MarketWatch** and **Google News** to backfill missing 2020–2024 date windows across `AAPL`, `META`, `NVDA`, `SPY`, `TSLA`, `LLY`, `AMZN`, `MSFT`, and `ORCL`.
* **Built the Data Cleaning & Standardization Pipeline:** Unified **2,033,186 raw records** from **8,057 news publishers** and **3 social media platforms** (Reddit, StockTwits, X/Twitter) into a single canonical schema with ISO-8601 timestamps, URL domain extraction, 512-token truncation, and source-aware deduplication.
* **Executed NLP Sentiment Modeling & LLM Benchmarking:** Selected **`FinBERT-Tone` (`yiyanghkust/finbert-tone`)** as the **final production sentiment model** after benchmarking against **VADER**, **`ProsusAI/finbert`**, and **Llama 3 / 3.1 / 3.2 (1B–8B)**—running batch inference across 5 years of data and engineering the **0.70 News / 0.30 Social Media** daily `Weighted Sentiment Score` (`9,135` stock-date records) fed directly into the final ML/LSTM training datasets.
* **Conducted Text Mining & Sentiment EDA:** Performed bigram topic extraction, word-cloud generation, cross-ticker co-occurrence analysis, and 2020–2024 macroeconomic sentiment regime analysis.

---

## Executive Summary & Key Achievements

* **Large-Scale Multi-Source Data Engineering:** Built an end-to-end data collection, scraping, and standardization pipeline aggregating **2,033,186 combined records** across **8,057 unique news publishers** and **3 major social media platforms** (Reddit, StockTwits, X/Twitter) from **Jan 1, 2020 to Dec 31, 2024**.
* **Custom Gap-Filling Web Scrapers:** Engineered automated date-windowed scrapers for **MarketWatch** and **Google Search/News** (`BeautifulSoup`, `requests`, `newspaper3k`) to bridge coverage gaps in public financial datasets (such as the Dec 16, 2023 cutoff in the **FNSPID** Nasdaq dataset and Meta's `FB` $\rightarrow$ `META` ticker transition).
* **Domain-Specific Transformer & LLM Benchmarking:** Evaluated lexicon-based (**VADER**), finance-tuned BERT (**`ProsusAI/finbert`**, **`yiyanghkust/finbert-tone`**), and generative LLM (**`Llama-3.1-8B-Instruct`**, **`Llama-3.2-3B-Instruct`**, **`Llama-3.2-1B`**) approaches—selecting **`finbert-tone`** (pretrained on 4.9B tokens of corporate filings, earnings calls, and analyst reports) for optimal accuracy-to-throughput across 2M+ records.
* **Source-Weighted Daily Sentiment Index:** Designed a noise-resistant daily aggregation scheme combining continuous polarity scores ($P(\text{Positive}) - P(\text{Negative})$) with source reliability weights (**0.70 Financial News / 0.30 Social Media**), producing **9,135 daily stock-sentiment observations** that helped drive downstream out-of-sample test ROC AUCs of **0.57–0.79** across stocks.

---

## End-to-End Sentiment Pipeline Architecture

```mermaid
flowchart LR
    subgraph S1["1. Multi-Source Data Collection"]
        N1["FNSPID Dataset<br/>(Nasdaq 8.5k+ Tickers)"]
        N2["Custom Web Scrapers<br/>(MarketWatch & Google News)"]
        N3["Public Benchmarks<br/>(Kaggle Financial News)"]
        SM1["Reddit API (PRAW)<br/>(r/wallstreetbets, r/stocks)"]
        SM2["StockTwits & X/Twitter<br/>(Cashtag Streams)"]
    end

    subgraph S2["2. Standardization & Cleaning"]
        C1["Unified 6-Column Schema<br/>(ISO-8601 Datetime, URL Parser)"]
        C2["Source-Aware Deduplication<br/>& 512-Token Length Filtering"]
    end

    subgraph S3["3. NLP Sentiment Inference"]
        M1["VADER Baseline"]
        M2["ProsusAI/finbert"]
        M3["yiyanghkust/finbert-tone<br/>(Selected Production Model)"]
        M4["Llama 3.1 / 3.2 (1B–8B)<br/>(LLM Benchmarking)"]
    end

    subgraph S4["4. Feature Engineering & EDA"]
        F1["Continuous Polarity<br/>(Pos Prob - Neg Prob)"]
        F2["Weighted Daily Aggregation<br/>(0.7 News + 0.3 Social)"]
        F3["Downstream ML/LSTM<br/>& Trading Backtester"]
    end

    S1 --> S2 --> S3 --> S4
```

---

## 1. Multi-Source Data Collection & Web Scraping

### Tickers Covered
* **Primary Modeling Portfolio (2020–2024):** `AAPL` (Apple), `META` (Meta Platforms), `NVDA` (NVIDIA), `SPY` (S&P 500 ETF), `TSLA` (Tesla)
* **Extended Data Collection & Exploratory Scope:** `LLY` (Eli Lilly), `AMZN` (Amazon), `MSFT` (Microsoft), `ORCL` (Oracle)

### Bridging Coverage Gaps Across Sources
No single public dataset provided complete, uninterrupted daily coverage from 2020 through 2024. To construct a continuous 5-year corpus without temporal blind spots, I combined four distinct data streams:

| Data Stream | Role in Pipeline | Key Implementation Details & Repository Links |
| :--- | :--- | :--- |
| **FNSPID (Nasdaq External Data)** | Historical news backbone (up to Dec 16, 2023) | Processed the multi-gigabyte raw Nasdaq corpus in `1,000,000`-row chunks (`finspid.ipynb`) across 8,563 tickers to extract target stock histories (`AAPL`: 9,338 rows; `NVDA`: 11,862 rows; `SPY`: 7,225 rows; `LLY`: 3,545 rows; `FB`: 433 rows). |
| **Custom MarketWatch Scraper** | Backfill post-Dec 2023 cutoff & historical gaps | Built [`news/scraping/scrape_marketwatch.ipynb`](https://github.com/HengZhengKai/is424-data-mining-project/blob/main/news/scraping/scrape_marketwatch.ipynb) to iterate day-by-day (`daterange`) across `marketwatch.com/search`, handling ticker rebrands (`FB` $\rightarrow$ `META` in June 2022) and 2024 out-of-sample test data. |
| **Custom Google News Scraper** | High-recall headline & publisher aggregation | Built [`news/scraping/scrape_google.ipynb`](https://github.com/HengZhengKai/is424-data-mining-project/blob/main/news/scraping/scrape_google.ipynb) (`scrape.ipynb`) using date-bounded query parameters (`tbs=cdr:1,cd_min:...,cd_max:...`) and `User-Agent` rotation, collecting ~95.5k supplementary rows across `SPY` (~38k), `TSLA` (~18.5k), `META` (~14k), `NVDA` (~13k), and `AAPL` (~12k). |
| **Social Media Streams (Reddit, StockTwits, X)** | Retail investor sentiment & momentum signals | Extracted Reddit submissions/comments via `PRAW` ([`social_media/reddit/reddit.py`](https://github.com/HengZhengKai/is424-data-mining-project/blob/main/social_media/reddit/reddit.py)) and cleaned multi-year **StockTwits** ([`social_media/stocktwits/`](https://github.com/HengZhengKai/is424-data-mining-project/tree/main/social_media/stocktwits)) and **X/Twitter** ([`social_media/x/notebooks/`](https://github.com/HengZhengKai/is424-data-mining-project/tree/main/social_media/x/notebooks)) cashtag datasets. |

---

## 2. Data Standardization, Cleaning & Deduplication

Raw data from FNSPID, MarketWatch, Google Search, Kaggle, Reddit, StockTwits, and X arrived with incompatible schemas, mixed timezones, HTML artifacts, and aggregator duplicates.

In [`news/datasets/raw/cleaning_scripts/`](https://github.com/HengZhengKai/is424-data-mining-project/tree/main/news/datasets/raw/cleaning_scripts), [`clean_news.ipynb`](https://github.com/HengZhengKai/is424-data-mining-project/blob/main/news/datasets/raw_post_midterm/cleaning_scripts/clean_news.ipynb), [`clean_all.ipynb`](https://github.com/HengZhengKai/is424-data-mining-project/blob/main/news_social_media_data/cleaning_scripts/clean_all.ipynb), and [`combine_all.ipynb`](https://github.com/HengZhengKai/is424-data-mining-project/blob/main/news_social_media_data/cleaning_scripts/combine_all.ipynb), I standardized all records into a unified 6-column schema:

```text
['Datetime', 'Stock', 'Content', 'Type', 'Source', 'Sentiment']
```

### Key Data Quality Engineering Steps
1. **ISO-8601 Temporal Normalization:** Converted heterogeneous date formats (UTC strings, relative dates, localized timestamps) into strict `%Y-%m-%dT%H:%M:%S` format and filtered to the `2020-01-01` – `2024-12-31` study window.
2. **Automated Publisher Extraction:** Parsed root domain names from raw article URLs via `urllib.parse.urlparse` to track publisher provenance across **8,057 unique news domains**.
3. **Transformer Token-Limit Alignment:** Removed corrupted rows (e.g., newline-only author bylines) and constrained text length to fit within BERT's **512-token (~2,000-character)** context window.
4. **Source-Aware Deduplication:**
   * For **`Type == 'News'`**, deduplicated identical headlines (`subset=['Content']`) caused by syndication across Yahoo Finance, Nasdaq, MarketWatch, and Google News.
   * For **`Type == 'Social Media'`**, preserved recurring retail phrases across *different* dates so genuine surges in retail chatter were not artificially flattened.

---

## 3. Final Sentiment Model Used vs. Alternatives Explored

### Final Production Sentiment Model: **FinBERT-Tone (`yiyanghkust/finbert-tone`)**

The **final sentiment model used** to generate the `Weighted Sentiment Score` feature in the downstream model training datasets ([`training/data/{STOCK}_data_model_training.csv`](https://github.com/HengZhengKai/is424-data-mining-project/tree/main/training/data)) is **`FinBERT-Tone` (`yiyanghkust/finbert-tone`)**, implemented in [`financial-news-sentiment-analysis-finbert tone.ipynb`](https://github.com/HengZhengKai/is424-data-mining-project/blob/main/financial-news-sentiment-analysis-finbert%20tone.ipynb) and aggregated in [`combine_sentiment_scores_per_date.ipynb`](https://github.com/HengZhengKai/is424-data-mining-project/blob/main/combine_sentiment_scores_per_date.ipynb).

#### Why `yiyanghkust/finbert-tone` Was Chosen
* **Domain-Specific Pretraining at Scale:** Built on BERT and pretrained on **4.9 billion tokens** of financial communications (**2.5B tokens** of 10-K & 10-Q corporate filings, **1.3B tokens** of earnings call transcripts, and **1.1B tokens** of analyst reports).
* **Financial Context Accuracy:** Accurately interprets domain-specific phrasing (e.g., *"beats consensus"*, *"guidance raised"*, *"short squeeze"*, *"margin compression"*) that general-purpose lexicons misclassify.
* **High Throughput for 2.03M+ Rows:** Fast enough for full-corpus batch GPU inference across 5 years of data (`2020–2024`), unlike generative LLMs (`Llama-3.1-8B` / `Llama-3.2-3B`).

#### How the Final Sentiment Score Is Calculated (Step-by-Step)
1. **Tokenization & Inference (`yiyanghkust/finbert-tone`):** Each cleaned headline or social media post (`Content`, truncated to $\le 512$ tokens) is passed through `BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")`.
2. **Softmax Probability Extraction:** Model logits are converted via `softmax(dim=-1)` into three class probabilities for every text $i$:
   * $P(\text{Positive}_i)$ (`scores[0]`)
   * $P(\text{Neutral}_i)$ (`scores[1]`)
   * $P(\text{Negative}_i)$ (`scores[2]`)
3. **Continuous Row-Level Polarity Score:** Instead of taking a hard `argmax` label (which discards confidence), a continuous polarity score is computed for each record:
   $$\text{Sentiment\_Score}_i = P(\text{Positive}_i) - P(\text{Negative}_i) \in [-1, 1]$$
4. **Source-Weighted Daily Aggregation (`0.70 News / 0.30 Social Media`):** In [`combine_sentiment_scores_per_date.ipynb`](https://github.com/HengZhengKai/is424-data-mining-project/blob/main/combine_sentiment_scores_per_date.ipynb), records on each date $t$ for stock $s$ are assigned reliability weights ($w_{\text{News}} = 0.70$, $w_{\text{Social Media}} = 0.30$) so high-volume retail chatter does not drown out institutional journalism, and aggregated via a weighted average:
   $$\text{Weighted Sentiment Score}_{s,t} = \frac{\sum_{i \in D_{s,t}} w_{\text{Type}(i)} \cdot \text{Sentiment\_Score}_i}{\sum_{i \in D_{s,t}} w_{\text{Type}(i)}}$$
5. **Downstream Feature Integration:** The resulting daily series (`news_sentiment_finbert_tone_weighted_{stock}.csv`, **9,135 total stock-date rows**) is merged by `Date` directly into [`training/data/{STOCK}_data_model_training.csv`](https://github.com/HengZhengKai/is424-data-mining-project/tree/main/training/data) as the **`Weighted Sentiment Score`** feature alongside `Close` price and the 10 technical indicator combinations (`Combi 1` – `Combi 10`).

---

### Summary of All Sentiment Models Benchmarked

Before locking in **`yiyanghkust/finbert-tone`** as the final model, I evaluated three other NLP families across the pipeline:

| Model / Approach | Status | Notebook Reference | Why It Was Selected or Rejected |
| :--- | :--- | :--- | :--- |
| **`FinBERT-Tone` (`yiyanghkust/finbert-tone`)** | **FINAL MODEL USED** | [`financial-news-sentiment-analysis-finbert tone.ipynb`](https://github.com/HengZhengKai/is424-data-mining-project/blob/main/financial-news-sentiment-analysis-finbert%20tone.ipynb) | **Selected for final production:** Best balance of financial domain calibration (pretrained on 4.9B financial tokens) and inference speed across all 2.03M+ rows. |
| **`FinBERT` (`ProsusAI/finbert`)** | Explored / Benchmarked | [`financial-news-sentiment-analysis-finbert.ipynb`](https://github.com/HengZhengKai/is424-data-mining-project/blob/main/financial-news-sentiment-analysis-finbert.ipynb) | Tested on sample/first-month slices; superseded by `yiyanghkust/finbert-tone`, which was trained on a much larger financial corpus and used for full 2020–2024 batch scoring. |
| **Llama 3 / 3.1 / 3.2 (`1B`, `3B`, `8B` Instruct)** | Explored / Rejected | [`news_social_media_data/llama/`](https://github.com/HengZhengKai/is424-data-mining-project/tree/main/news_social_media_data/llama) | **Rejected for full-corpus batching:** `Meta-Llama-3-8B`, `Llama-3.1-8B-Instruct`, and `Llama-3.2-3B-Instruct` were prohibitively slow / exhausted GPU memory on 2M+ rows, while `Llama-3.2-1B` produced inconsistent outputs. |
| **VADER (`vaderSentiment`)** | Initial Baseline Only | `Colab_Wk3 Sentiment Analysis using VADER.ipynb` | **Rejected:** Rule/lexicon-based approach lacked financial context (e.g., misinterpreting financial terms and earnings jargon). |

---

## 4. Exploratory Data Analysis (EDA) & Corpus Insights

In [`news_social_media_data/EDA_sentiment_analysis.ipynb`](https://github.com/HengZhengKai/is424-data-mining-project/blob/main/news_social_media_data/EDA_sentiment_analysis.ipynb) and [`news_social_media_data/EDA2_sentiment_analysis.ipynb`](https://github.com/HengZhengKai/is424-data-mining-project/blob/main/news_social_media_data/EDA2_sentiment_analysis.ipynb), I conducted text mining and distributional analysis across the **2,033,186** records and **9,135** aggregated daily stock-date observations:

### Corpus Source & Publisher Distribution (out of 8,057 Sources)

| Rank | Publisher Domain | Article Count |
| :---: | :--- | :---: |
| 1 | **Yahoo Finance** (`Yahoo`) | 23,780 |
| 2 | **Nasdaq** (`Nasdaq`) | 18,932 |
| 3 | **Barron's** (`Barrons`) | 8,742 |
| 4 | **MarketWatch** (`Marketwatch`) | 7,355 |
| 5 | **Bloomberg** (`Bloomberg.com`) | 4,522 |

| Source Breakdown by Stock (`News` vs. `Social Media`) | Social Media Platform Breakdown |
| :---: | :---: |
| ![Sentiment Data Source Distribution by Stock](assets/source_distribution_by_stock.png) | ![Social Media Distribution](assets/social_media_distribution.png) |

![Top 5 News Sources](assets/top_5_news_sources.png)

### Cross-Ticker Co-Occurrence & Bigram Themes
* **Top Co-Occurring Stock Pairs:** Extracted cross-mentions across all 2.03M texts, revealing strong macro-tech coupling between `('AAPL', 'SPY')` (**111,111** co-mentions), `('AAPL', 'TSLA')` (**96,652**), `('SPY', 'TSLA')` (**43,051**), `('AAPL', 'NVDA')` (**30,376**), and `('NVDA', 'TSLA')` (**22,893**).

![Top 5 Co-Occurring Stock Pairs](assets/top_co_occurring_pairs.png)

* **Dominant Bigram Topics:** After HTML entity decoding, URL stripping, and custom financial stopword removal, bigram frequency and word-cloud analysis surfaced distinct narratives per ticker—such as *"price target"*, *"data center"*, and *"amd intc"* for **`NVDA`**, versus *"interest rates"*, *"dow jones"*, and *"price action"* for **`SPY`**.

| AAPL Bigram Word Cloud | NVDA Bigram Word Cloud |
| :---: | :---: |
| ![AAPL Word Cloud](assets/wordcloud_aapl.png) | ![NVDA Word Cloud](assets/wordcloud_nvda.png) |
| **TSLA Bigram Word Cloud** | **META Bigram Word Cloud** |
| ![TSLA Word Cloud](assets/wordcloud_tsla.png) | ![META Word Cloud](assets/wordcloud_meta.png) |

<p align="center">
  <img src="assets/wordcloud_spy.png" alt="SPY Word Cloud" width="65%" />
</p>

### Multi-Year Sentiment Regime Shifts (2020–2024)
Analyzing the daily `weighted_sentiment` distributions (KDE plots and yearly trajectories) across the 5 core tickers captured clear macroeconomic regime changes:
* **5-Year Mean Weighted Sentiment:** `META` (`0.725`), `SPY` (`0.712`), `TSLA` (`0.601`), `NVDA` (`0.565`), `AAPL` (`0.565`).
* **2020–2021 Bull Market Optimism vs. 2022–2024 Compression:** Average sentiment peaked during the 2020–2021 liquidity expansion (e.g., `AAPL` at **`0.707`** and `TSLA` at **`0.723`** in 2021) before compressing sharply during the 2022–2024 inflation and rate-hike cycle (`AAPL` at **`0.455`** and `TSLA` at **`0.453`** in 2024).

![Sentiment Score KDE per Stock](assets/sentiment_kde_per_stock.png)

| Yearly Sentiment Trajectory (2020–2024) | 5-Year Average Sentiment per Stock |
| :---: | :---: |
| ![Yearly Sentiment Trends](assets/yearly_sentiment_trends.png) | ![Average Sentiment Score per Stock](assets/avg_sentiment_per_stock.png) |

---

## 5. How the Sentiment Scores Were Used in Downstream Model Training

Once my pipeline produced the daily **`Weighted Sentiment Score`** (`news_sentiment_finbert_tone_weighted_{stock}.csv`), it was handed off to the group's modeling pipeline ([`training/Final_{STOCK}_model_training.ipynb`](https://github.com/HengZhengKai/is424-data-mining-project/tree/main/training)) as a core predictive feature:

1. **Feature Matrix Integration (`1,267` Trading Days per Stock):**
   * Merged by `Date` into [`training/data/{STOCK}_data_model_training.csv`](https://github.com/HengZhengKai/is424-data-mining-project/tree/main/training/data) alongside the daily `Close` price and **10 technical indicator combinations** (`Combi 1` through `Combi 10`, encoded as `Buy = 1`, `Hold = 0`, `Sell = -1`, plus each indicator's optimal lookahead window `Combi X Days`).
   * For every prediction horizon $h \in \{4, \dots, 10\}$ trading days, the model input vector was constructed directly around the sentiment score:
     ```python
     features = ["Close", "Weighted Sentiment Score"] + rel_signals
     ```
     where `rel_signals` are the technical indicator combinations whose modal optimal horizon equals $h$ days.

2. **Horizon-Specific Return Threshold Optimization:**
   * For each horizon $h$, an automated grid search (`find_best_threshold_for_horizon`) swept return thresholds $\text{th} \in [0.0025, 0.05)$ in increments of `0.0025` to define the binary classification target:
     $$Y_{h\text{d}} = \mathbb{I}\left(\frac{\text{Close}_{t+h} - \text{Close}_t}{\text{Close}_t} > \text{th}\right)$$
   * Evaluated using `XGBoost`, `LightGBM`, and `RandomForest` with `Weighted Sentiment Score` included in the feature set to find the highest-signal return threshold per horizon.

3. **Out-of-Sample Model Training (2020–2023 Train / 2024 Test) & Backtesting:**
   * Using a strict temporal split (**2020–2023** for training, **2024** for out-of-sample testing), the combined feature vector (`Close` + `Weighted Sentiment Score` + technical signals) was used to train and compare **9 ML/DL architectures**:
     * **Tree & Ensemble Models:** Random Forest, XGBoost, LightGBM, Gradient Boosting, Decision Tree
     * **Linear, Kernel & Probabilistic Models:** Logistic Regression, SVM (RBF Kernel), Gaussian Naive Bayes
     * **Deep Learning:** 30-day lookback sequence **LSTM** (`PyTorch`)
   * Across the 5 tickers and prediction horizons, the best horizon models achieved **out-of-sample Test ROC AUCs between 0.57 and 0.79**, driving non-overlapping Long/Short backtests with **10 bps (0.10%) round-trip transaction fees**.

---

## Repository Directory Map (My Workstream vs. Full Project)

Below is how the files in this local workspace and the [main group repository (`HengZhengKai/is424-data-mining-project`)](https://github.com/HengZhengKai/is424-data-mining-project/tree/main) map to each stage of the sentiment pipeline:

```text
├── news/
│   ├── scraping/
│   │   ├── scrape_marketwatch.ipynb       # Automated daily MarketWatch headline scraper
│   │   └── scrape_google.ipynb            # Date-windowed Google Search/News scraper
│   └── datasets/
│       ├── raw/cleaning_scripts/          # Per-ticker cleaners (clean_aapl, clean_meta, clean_nvda, clean_spy, clean_lly)
│       └── raw_post_midterm/              # Round 2 expanded scraping & clean_news.ipynb (~95.5k Google Search rows)
│
├── social_media/
│   ├── reddit/                            # PRAW scraper (reddit.py) & per-ticker cleaning notebooks
│   ├── stocktwits/                        # Multi-year StockTwits cleaners (AAPL, META, NVDA)
│   └── x/notebooks/                       # Twitter/X financial & stock tweet cleaning pipelines
│
├── news_social_media_data/
│   ├── cleaning_scripts/
│   │   ├── clean_all.ipynb                # Standardizes News, StockTwits, Reddit, and X into 6-column schema
│   │   └── combine_all.ipynb              # Merges all sources per ticker & runs source-aware deduplication
│   ├── final_combined/                    # Partitioned master datasets (2.03M+ total rows across tickers)
│   ├── llama/                             # Llama 3 / 3.1 / 3.2 (1B, 3B, 8B) local GPU inference benchmarks
│   ├── EDA_sentiment_analysis.ipynb       # Source distribution, bigram extraction, co-occurrence & word clouds
│   └── EDA2_sentiment_analysis.ipynb      # Weighted sentiment KDE distributions & 2020–2024 yearly trends
│
├── financial-news-sentiment-analysis-finbert.ipynb       # ProsusAI/finbert inference pipeline
├── financial-news-sentiment-analysis-finbert tone.ipynb  # yiyanghkust/finbert-tone batch inference by ticker & year
└── combine_sentiment_scores_per_date.ipynb               # 0.7 News / 0.3 Social Media daily weighted sentiment aggregation
```

*(For the downstream technical indicator engineering, 9-model ML/LSTM training, and trading strategy backtesting that consume these sentiment features, see [`time_series/`](https://github.com/HengZhengKai/is424-data-mining-project/tree/main/time_series) and [`training/`](https://github.com/HengZhengKai/is424-data-mining-project/tree/main/training) in the [main repository](https://github.com/HengZhengKai/is424-data-mining-project/tree/main).)*

---

## Tech Stack & Libraries Used

* **NLP & Deep Learning:** `transformers` (Hugging Face), `torch` (PyTorch), `vaderSentiment`, `nltk`, `scikit-learn` (`CountVectorizer`)
* **Web Scraping & APIs:** `requests`, `beautifulsoup4`, `newspaper3k`, `lxml`, `praw` (Reddit API)
* **Data Engineering & Analytics:** `pandas`, `numpy`, `python-dateutil`, `urllib.parse`
* **Visualization:** `matplotlib`, `seaborn`, `wordcloud`

