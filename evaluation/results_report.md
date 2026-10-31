# PharmGuard Clinical RAG: Ablation & Evaluation Benchmark

## 1. Executive Summary
This evaluation benchmarks **Baseline Naive Dense RAG** (single-vector ChromaDB search with `bge-base-en-v1.5`) against **PharmGuard Production RAG** (ChromaDB + BM25 Lexical + Reciprocal Rank Fusion + BGE Cross-Encoder Reranking) across 15 high-risk clinical pharmacology scenarios.

### Key Findings:
- **Context Precision:** Improved from **0.820** to **0.933** (+0.113).
- **Context Recall:** Improved from **0.739** to **0.811** (+0.072).
- **MRR (Mean Reciprocal Rank):** Improved from **0.811** to **0.933** (+0.122).
- **Hit Rate @ 1:** Improved from **73.3%** to **86.7%** (+13.3%).
- **Hit Rate @ 4:** Achieved **100.0%** full coverage across all 15 scenarios.

---

## 2. Aggregate Benchmark Comparison Table
| Evaluation Metric | Baseline Naive Dense | PharmGuard (Hybrid + Reranker) | Absolute Improvement | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Context Precision (Rank-Weighted)** | 0.8204 | **0.9333** | +0.1129 | +13.8% |
| **Context Recall (Entity/Fact Coverage)** | 0.7389 | **0.8111** | +0.0722 | +9.8% |
| **Mean Reciprocal Rank (MRR)** | 0.8111 | **0.9333** | +0.1222 | +15.1% |
| **Hit Rate @ 1** | 0.7333 | **0.8667** | +0.1334 | +18.2% |
| **Hit Rate @ 4** | 0.9333 | **1.0000** | +0.0667 | +7.1% |

---

## 3. Clinical Pharmacology Analysis
### Why Naive Dense Retrieval Fails in Critical Clinical Settings:
1. **Lexical Anchor Blindness**: Dense embeddings frequently map semantically similar medical terms to close vector neighbors without distinguishing critical active moieties (e.g. confusing *Atorvastatin* with *Rosuvastatin*, or *Phenelzine* with other general antidepressants).
2. **Specific Acronyms & Enzyme Codes**: Pharmacogenomic terms like `CYP3A4`, `P-gp`, `5-HT1A`, and numerical cutoffs like `eGFR < 30` often suffer from semantic dilution in dense embeddings. BM25 guarantees that rare clinical alphanumeric tokens are prioritized.
3. **Cross-Encoder Context Fusion**: The bi-encoder ChromaDB embedding only scores query-to-sentence similarity independently. The BGE Cross-Encoder performs all-to-all cross-attention between the entire clinical question and the expanded 7-sentence window, ensuring that dosing conditions, contraindication caveats, and organ thresholds are correctly prioritized in top-1 position.

---

## 4. Per-Scenario Detailed Breakdown
| ID | Clinical Query | Baseline Precision | PharmGuard Precision | Baseline Recall | PharmGuard Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `CLIN-01` | What is the interaction risk between Paxlovid and Atorv... | 0.64 | **1.00** | 1.00 | **1.00** |
| `CLIN-02` | Can a patient taking Phenelzine be prescribed Sumatript... | 1.00 | **1.00** | 1.00 | **1.00** |
| `CLIN-03` | What is the eGFR cutoff for Metformin and what fatal co... | 1.00 | **1.00** | 1.00 | **0.50** |
| `CLIN-04` | What are the clinical consequences of co-administering ... | 1.00 | **1.00** | 0.50 | **0.50** |
| `CLIN-05` | Is Clarithromycin safe to prescribe with Colchicine in ... | 1.00 | **1.00** | 0.67 | **0.67** |
| `CLIN-06` | How should Apixaban dosing be adjusted based on renal f... | 0.00 | **0.58** | 0.50 | **1.00** |
| `CLIN-07` | Why is the combination of Paxlovid and Amiodarone contr... | 1.00 | **1.00** | 0.67 | **1.00** |
| `CLIN-08` | What are the contraindications for Spironolactone in ch... | 1.00 | **1.00** | 0.50 | **1.00** |
| `CLIN-09` | What is the required washout period when switching from... | 1.00 | **1.00** | 1.00 | **1.00** |
| `CLIN-10` | Can Tramadol and Venlafaxine be safely combined for pai... | 1.00 | **1.00** | 1.00 | **1.00** |
| `CLIN-11` | What dosage adjustments or precautions are mandated for... | 0.33 | **1.00** | 0.50 | **0.50** |
| `CLIN-12` | How does Ciprofloxacin affect Tacrolimus blood levels a... | 1.00 | **1.00** | 1.00 | **1.00** |
| `CLIN-13` | What is the 'Triple Whammy' in renal pharmacotherapy an... | 1.00 | **1.00** | 0.25 | **0.50** |
| `CLIN-14` | What are the renal thresholds for discontinuing Empagli... | 1.00 | **0.92** | 1.00 | **1.00** |
| `CLIN-15` | What severe adverse reaction occurs with Allopurinol ac... | 0.33 | **0.50** | 0.50 | **0.50** |
