# LegalSyntheticData

Automated pipeline for generating high-quality Vietnamese legal synthetic QA data, aligned with the **VLegalBench** benchmark (5 Bloom Taxonomy levels, 22 tasks). Data is sourced from a Neo4j Knowledge Graph of Vietnamese legal documents, with 85% priority on Ministry of Finance (BTC) documents.

---

## Pipeline Flow

```
Neo4j KG
   │
   ▼
[Seed Sampling]  ──── BTC-priority (85%)
   │
   ▼
[KG Graph Expansion]  ──── AMENDS / MENTIONS / BASED_ON / REPLACES
   │               (up to MAX_LEGAL_UNITS articles per block)
   ▼
[Task Selection]  ──── weighted random from 22 Bloom-level tasks
   │
   ▼
Agent 1: ContextMaster  ──── generates a legal scenario (100–200 words)
   │
   ▼  (retry loop up to MAX_RETRIES)
Agent 2: QACrafter  ──── generates question + reference answer
   │
   ├──► Agent 3A: Solver (temp=0.3)  ─┐
   └──► Agent 3B: Solver (temp=0.6)  ─┤── parallel
                                       ▼
                              Agent 4: Checker
                              4 criteria weighted score:
                                • consensus    20%
                                • factuality   40%  (must ≥ 0.6)
                                • classification 25% (must ≥ 0.6)
                                • clarity      15%
                                score ≥ 0.7 → PASS
                                         │
                        ┌────────────────┴──────────────┐
                       PASS                             FAIL
                        │                                │
                  Save to JSONL              feedback → QACrafter retry
```

---

## Output Schema

Each accepted sample is written to `output/level_X/task_X_Y.jsonl`:

```json
{
  "qid": "BTC_00001",
  "question": "Chủ tịch Hội Liên hiệp Phụ nữ xã là cán bộ hay công chức?",
  "relevant_laws": ["33/2023/NĐ-CP - Điều 5"],
  "answer": "Vấn đề đặt ra là việc xác định Chủ tịch Hội Liên hiệp Phụ nữ xã...",
  "level": "1.1",
  "instruction": "Trả lời câu hỏi pháp lý dưới đây dựa trên quy định pháp luật Việt Nam...",
  "level_name": "Nhận diện điều khoản pháp luật (Recognition & Recall)",
  "source_doc_ids": ["33_2023_ND-CP"],
  "context": "...",
  "ministry_focus": "BTC",
  "generation_metadata": {
    "attempts": 1,
    "checker_score": 0.85,
    "checker_details": {"consensus": 0.9, "factuality": 0.85, "classification": 0.8, "clarity": 0.9}
  }
}
```

### Output folder structure

```
output/
├── level_1/
│   ├── task_1_1.jsonl   # Nhận diện điều khoản pháp luật
│   ├── task_1_2.jsonl
│   └── task_1_3.jsonl
├── level_2/
│   ├── task_2_1.jsonl
│   └── ...
├── level_3/
│   └── ...
├── level_4/
│   └── ...
├── level_5/
│   └── ...
└── pipeline.log
```

---

## Bloom Taxonomy Tasks

| ID  | Level | Name (VI) |
|-----|-------|-----------|
| 1.1 | 1 | Nhận diện điều khoản pháp luật |
| 1.2 | 1 | Truy xuất thông tin pháp lý cụ thể |
| 1.3 | 1 | Nhận dạng văn bản và điều khoản liên quan |
| 2.1 | 2 | Tóm tắt nội dung điều luật |
| 2.2 | 2 | Phân loại loại văn bản pháp lý |
| 2.3 | 2 | Trích xuất mối quan hệ pháp lý |
| 2.4 | 2 | Giải thích thuật ngữ pháp lý |
| 3.1 | 3 | Dự đoán điều khoản áp dụng (Reasoning & Inference) |
| 3.2 | 3 | Suy luận hậu quả pháp lý |
| 3.3 | 3 | So sánh và đối chiếu điều khoản |
| 3.4 | 3 | Phát hiện mâu thuẫn hoặc xung đột pháp lý |
| 4.1 | 4 | Phân tích tình huống pháp lý phức tạp |
| 4.2 | 4 | Lập luận pháp lý theo IRAC |
| 4.3 | 4 | Tư vấn pháp lý dựa trên văn bản |
| 4.4 | 4 | Soạn thảo văn bản pháp lý đơn giản |
| 4.5 | 4 | Đánh giá tính hợp pháp của hành vi |
| 5.1 | 5 | Phát hiện thiên kiến trong văn bản pháp luật |
| 5.2 | 5 | Đánh giá tác động xã hội của điều luật |
| 5.3 | 5 | Lập luận đạo đức pháp lý |
| 5.4 | 5 | Phân tích công bằng và bình đẳng trong pháp luật |

Run `python run.py --list-tasks` to see all tasks with weights.

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/tranthai189765/LegalSyntheticData.git
cd LegalSyntheticData

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your API key and Neo4j credentials
```

### `.env` file

```env
FPT_API_KEY=sk-your-api-key
FPT_API_URL=https://mkp-api.fptcloud.com/v1/chat/completions
FPT_MODEL=gpt-oss-120b

NEO4J_URL=http://localhost:7474
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password

BTC_PRIORITY_RATIO=0.85
MAX_LEGAL_UNITS=5
KG_EXPANSION_HOPS=2
MAX_RETRIES=3
LLM_CONCURRENCY=3
CHECKER_PASS_THRESHOLD=0.7
```

---

## Usage

```bash
# Generate 100 samples across all 22 tasks (default)
python run.py --n 100

# Generate 500 samples into a custom output directory
python run.py --n 500 --output-dir output/run_v2

# Generate only task 3.3 (Compare & Contrast clauses)
python run.py --n 50 --task 3.3

# Increase concurrency for faster generation
python run.py --n 200 --concurrency 5

# List all available task IDs
python run.py --list-tasks
```

---

## Demo (single run)

```
$ python run.py --n 5 --task 1.1

12:00:01 | INFO | Starting pipeline: n=5  output_dir=output  task=1.1
12:00:01 | INFO | Config: model=gpt-oss-120b  btc_ratio=0.85  concurrency=3  max_retries=3
Generating samples:  20%|██        | 1/5 [00:08<00:32]
✓ BTC_00001  level=1.1  ministry=BTC  score=0.876  → level_1/task_1_1.jsonl
✓ BTC_00002  level=1.1  ministry=BTC  score=0.912  → level_1/task_1_1.jsonl
✓ GEN_00003  level=1.1  ministry=OTHER  score=0.841  → level_1/task_1_1.jsonl
✓ BTC_00004  level=1.1  ministry=BTC  score=0.893  → level_1/task_1_1.jsonl
✓ BTC_00005  level=1.1  ministry=BTC  score=0.855  → level_1/task_1_1.jsonl

============================================================
PIPELINE COMPLETE
Attempted=7  Accepted=5  Rejected=2  Errors=0  AcceptRate=71.4%  Elapsed=45s
Output directory: output/
  Per-task files: level_X/task_X_Y.jsonl
============================================================
```

---

## Architecture

| File | Role |
|------|------|
| `config.py` | Load `.env`, expose all settings |
| `neo4j_client.py` | Neo4j HTTP client, BTC-priority sampling, KG expansion |
| `llm_client.py` | Async OpenAI-compatible client (FPT Cloud), JSON extraction |
| `tasks/definitions.py` | 22 TaskDefinition objects with prompts & weights |
| `agents/context_master.py` | Agent 1: generate legal scenario from LegalBlock |
| `agents/qa_crafter.py` | Agent 2: generate question + reference answer as JSON |
| `agents/solver.py` | Agent 3 (×2): independent IRAC answers at different temperatures |
| `agents/checker.py` | Agent 4: 4-criterion quality gate with feedback |
| `pipeline.py` | Orchestration: concurrency, retry loop, per-task file writing |
| `run.py` | CLI entry point |

### Answer format by task type

- **paragraph** tasks (IRAC): flowing Vietnamese legal paragraph — no section headers.
  Uses hidden IRAC structure:
  > "Vấn đề đặt ra là... Theo quy định tại [Điều X Văn bản Y]... Đối chiếu với... Từ đó có thể kết luận rằng..."
- **list** tasks: numbered list with legal citations per item
- **json_triplets** tasks: `entity 1 | relation | entity 2` (one per line)
- **classification** tasks: short label answer
- **yes_no** tasks: starts with "Có." or "Không.", then 2–3 sentence justification

---

## Requirements

- Python 3.10+
- Neo4j 5.x running locally (or accessible via HTTP)
- FPT Cloud API key with access to `gpt-oss-120b`

```
openai>=1.0.0
aiohttp>=3.9.0
python-dotenv>=1.0.0
tenacity>=8.0.0
tqdm>=4.65.0
loguru>=0.7.0
```
