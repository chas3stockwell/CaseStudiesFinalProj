# Personal ML Projects

Two independent machine learning projects exploring NLP and predictive analytics.

---

## Project 1: Answer Geography Questions using Machine Learning

A semantic parsing system that translates natural language geography questions into executable logical forms.

**Example:**
- Input: `"what states border texas?"`
- Output: `_answer(NV, (_state(V0), _next_to(V0, NV), _const(V0, _stateid(texas))))`

### Models

- **NearestNeighborSemanticParser** — Baseline using Jaccard similarity over word overlap
- **Seq2SeqSemanticParser** — Neural encoder-decoder LSTM that learns to map questions to logical forms

### Technologies

- **Python 3**
- **PyTorch** — LSTM encoder-decoder architecture, embeddings, Adam optimizer
- **NumPy** — Array manipulation and tensor padding
- **Java** — GeoQuery knowledge base evaluator backend
- **Dataset** — GeoQuery benchmark (~880 labeled question/logical-form pairs)

### Usage

```bash
python main.py                        # Train seq2seq model (default)
python main.py --do_nearest_neighbor  # Run nearest neighbor baseline
python main.py --no_java_eval         # Skip Java evaluation
```

### Evaluation Metrics

- **Exact Match** — String equality between predicted and gold logical form
- **Token Accuracy** — Position-by-position token overlap
- **Denotation Match** — Correctness of query execution against the knowledge base

---

## Project 2: Case Studies Final Project

Predictive analytics on college graduation data using classical machine learning techniques.

### Analyses

- **College GPA Prediction** (`college_gpa_predict.py`) — Linear regression predicting college GPA from student demographics and test scores
- **Student Clustering** (`hs_cluster_analysis.py`) — K-means clustering segmenting students by parental income and high school GPA

### Technologies

- **Python 3**
- **scikit-learn** — `LinearRegression`, `KMeans`, `train_test_split`
- **Pandas** — Data loading and manipulation
- **NumPy** — Feature engineering and normalization
- **Matplotlib / Seaborn** — Visualization and cluster plots
- **SciPy** — Statistical functions
- **Dataset** — `graduation_rate.csv` (~200 student records with ACT/SAT scores, GPA, parental income/education)
