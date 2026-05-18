"""
Main Pipeline – Synthetic Legal QA Data Generator

Flow per sample:
  1. Sample a seed LegalUnit (BTC-priority).
  2. Build a LegalBlock via KG graph expansion.
  3. Select a task type (weighted random from 22 tasks).
  4. Agent 1 (ContextMaster): generate legal scenario.
  5. For up to MAX_RETRIES attempts:
       a. Agent 2 (QACrafter): generate question + answer hint
          (passes Checker feedback on retry > 0).
       b. Agents 3A & 3B (Solver): generate independent answers IN PARALLEL.
       c. Agent 4 (Checker): evaluate on 4 criteria.
       d. If pass → save sample, break.
       e. If fail → continue loop with feedback.
  6. If all retries exhausted → discard, log failure.

Output format (JSONL, per-task file under output/level_X/task_X_Y.jsonl):
{
  "qid": "BTC_00001",
  "question": "...",
  "relevant_laws": ["30/2023/TT-BTC - Điều 5"],
  "answer": "...",
  "level": "3.1",
  "instruction": "...",
  "level_name": "Article / Clause Prediction",
  "source_doc_ids": ["30_2023_TT-BTC"],
  "context": "...",
  "ministry_focus": "BTC",
  "generation_metadata": {
    "attempts": 1,
    "checker_score": 0.92,
    "checker_details": {...}
  }
}
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from loguru import logger
from tqdm import tqdm

import config
from agents.checker import CheckResult, Checker
from agents.context_master import ContextMaster
from agents.qa_crafter import QACrafter
from agents.solver import Solver
from llm_client import get_llm
from neo4j_client import LegalBlock, LegalUnit, Neo4jClient
from tasks.definitions import TaskDefinition, sample_task


# ── Output schema ─────────────────────────────────────────────────────────────

@dataclass
class SyntheticSample:
    qid: str
    question: str
    relevant_laws: List[str]
    answer: str
    level: str
    instruction: str
    # Extra fields (kept for research / filtering)
    level_name: str
    source_doc_ids: List[str]
    context: str
    ministry_focus: str
    generation_metadata: dict = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# ── Stats ─────────────────────────────────────────────────────────────────────

@dataclass
class PipelineStats:
    attempted: int = 0
    accepted: int  = 0
    rejected: int  = 0
    errors: int    = 0
    start_time: float = field(default_factory=time.time)

    def acceptance_rate(self) -> float:
        if self.attempted == 0:
            return 0.0
        return self.accepted / self.attempted

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def summary(self) -> str:
        return (
            f"Attempted={self.attempted}  Accepted={self.accepted}  "
            f"Rejected={self.rejected}  Errors={self.errors}  "
            f"AcceptRate={self.acceptance_rate():.1%}  "
            f"Elapsed={self.elapsed():.0f}s"
        )


# ── Per-task file management ──────────────────────────────────────────────────

def _task_output_path(output_dir: str, task: TaskDefinition) -> str:
    """Return the per-task JSONL path, e.g. output/level_3/task_3_1.jsonl"""
    level_folder = f"level_{task.level}"
    task_file    = f"task_{task.id.replace('.', '_')}.jsonl"
    return os.path.join(output_dir, level_folder, task_file)


class _FileRegistry:
    """Open and cache one file handle per task output path."""

    def __init__(self):
        self._handles: Dict[str, object] = {}

    def write(self, path: str, line: str):
        if path not in self._handles:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._handles[path] = open(path, "a", encoding="utf-8")
        fh = self._handles[path]
        fh.write(line + "\n")
        fh.flush()

    def close_all(self):
        for fh in self._handles.values():
            try:
                fh.close()
            except Exception:
                pass
        self._handles.clear()


# ── Core pipeline logic ───────────────────────────────────────────────────────

class Pipeline:
    def __init__(self):
        llm = get_llm()
        self.kg        = Neo4jClient()
        self.ctx_agent = ContextMaster(llm)
        self.crafter   = QACrafter(llm)
        self.solver_a  = Solver(llm, name="A", temperature=0.3)
        self.solver_b  = Solver(llm, name="B", temperature=0.6)
        self.checker   = Checker(llm)
        self._qid_counter = 0

    def _next_qid(self, is_btc: bool) -> str:
        self._qid_counter += 1
        prefix = "BTC" if is_btc else "GEN"
        return f"{prefix}_{self._qid_counter:05d}"

    async def _generate_one(
        self,
        task_override: str | None = None,
    ) -> Optional[SyntheticSample]:
        # 1. Seed
        seed: Optional[LegalUnit] = await self.kg.sample_seed_unit()
        if seed is None:
            logger.warning("No seed unit found, skipping.")
            return None

        # 2. KG expansion → LegalBlock
        block: LegalBlock = await self.kg.build_legal_block(seed)
        if not block.units:
            logger.warning("Empty legal block, skipping.")
            return None

        # 3. Task selection
        task: TaskDefinition = (
            sample_task() if task_override is None
            else __import__("tasks.definitions", fromlist=["get_task"]).get_task(task_override)
        )

        # 4. Context
        context: str = await self.ctx_agent.generate(block, task)

        # 5. QA + Solve + Check loop
        feedback: Optional[str] = None
        result: Optional[CheckResult] = None

        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                # 5a. Generate QA
                qa = await self.crafter.generate(block, context, task, feedback)

                # 5b. Two solvers in parallel
                answer_a, answer_b = await asyncio.gather(
                    self.solver_a.solve(qa.question, block, task),
                    self.solver_b.solve(qa.question, block, task),
                )

                # 5c. Quality check
                result = await self.checker.check(
                    question=qa.question,
                    answer_hint=qa.answer_hint,
                    answer_a=answer_a,
                    answer_b=answer_b,
                    block=block,
                    task=task,
                )

                if result.passed:
                    qid = self._next_qid(block.is_btc_focused)
                    return SyntheticSample(
                        qid=qid,
                        question=qa.question,
                        relevant_laws=block.law_references,
                        answer=result.best_answer,
                        level=task.id,
                        instruction=qa.instruction,
                        level_name=f"{task.name_vi} ({task.level_name})",
                        source_doc_ids=block.doc_ids,
                        context=context,
                        ministry_focus="BTC" if block.is_btc_focused else "OTHER",
                        generation_metadata={
                            "attempts": attempt,
                            "checker_score": round(result.score, 4),
                            "checker_details": result.details,
                        },
                    )

                # 5d. Retry with feedback
                feedback = result.feedback
                logger.debug(
                    f"  Attempt {attempt}/{config.MAX_RETRIES} FAIL "
                    f"score={result.score:.2f} feedback={feedback[:80]!r}"
                )

            except Exception as exc:
                logger.error(f"  Attempt {attempt} exception: {exc}")
                feedback = f"Lỗi sinh dữ liệu: {exc}. Hãy tạo câu hỏi đơn giản và rõ ràng hơn."

        logger.warning(f"All {config.MAX_RETRIES} retries exhausted for task {task.id}.")
        return None

    # ── public interface ──────────────────────────────────────────────────────

    async def run(
        self,
        n_samples: int,
        output_dir: str,
        task_filter: Optional[str] = None,
    ) -> PipelineStats:
        os.makedirs(output_dir, exist_ok=True)
        stats = PipelineStats()
        registry = _FileRegistry()

        semaphore = asyncio.Semaphore(config.LLM_CONCURRENCY)

        async def bounded_generate() -> Optional[SyntheticSample]:
            async with semaphore:
                return await self._generate_one(task_override=task_filter)

        try:
            pbar = tqdm(total=n_samples, desc="Generating samples", unit="sample")

            while stats.accepted < n_samples:
                batch_size = min(
                    config.LLM_CONCURRENCY * 2,
                    n_samples - stats.accepted + 2,
                )
                tasks_coros = [bounded_generate() for _ in range(batch_size)]
                results = await asyncio.gather(*tasks_coros, return_exceptions=True)

                for res in results:
                    stats.attempted += 1
                    if isinstance(res, Exception):
                        stats.errors += 1
                        logger.error(f"Pipeline exception: {res}")
                    elif res is None:
                        stats.rejected += 1
                    else:
                        stats.accepted += 1
                        # Write to per-task file
                        task_def = __import__(
                            "tasks.definitions", fromlist=["get_task"]
                        ).get_task(res.level)
                        path = _task_output_path(output_dir, task_def)
                        registry.write(path, res.to_jsonl())
                        pbar.update(1)
                        logger.info(
                            f"✓ {res.qid}  level={res.level}  "
                            f"ministry={res.ministry_focus}  "
                            f"score={res.generation_metadata['checker_score']}  "
                            f"→ {os.path.relpath(path, output_dir)}"
                        )
                    if stats.accepted >= n_samples:
                        break

            pbar.close()
        finally:
            registry.close_all()

        logger.info(stats.summary())
        return stats
