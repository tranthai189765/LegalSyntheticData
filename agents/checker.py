"""
Agent 4 – Checker (LLM-as-a-Judge)

Evaluates a generated QA sample on four criteria:
  1. Consensus  – Do Solver A and B substantially agree on the answer?
  2. Factuality – Is the answer grounded in the provided legal text (no hallucinated laws)?
  3. Classification – Does the QA pair match the stated task type / Bloom level?
  4. Clarity    – Is the question and answer clear, natural Vietnamese legal writing?

Returns a CheckResult with:
  • passed    – bool (all four criteria must pass)
  • score     – float [0, 1] overall confidence
  • best_answer – the better answer (A or B) if passed; empty string otherwise
  • feedback  – detailed feedback string for QACrafter on retry
"""

from __future__ import annotations
from dataclasses import dataclass

from loguru import logger

from llm_client import LLMClient
from neo4j_client import LegalBlock
from tasks.definitions import TaskDefinition
from config import CHECKER_PASS_THRESHOLD


@dataclass
class CheckResult:
    passed: bool
    score: float               # 0–1 aggregate confidence
    best_answer: str           # the selected final answer
    feedback: str              # guidance for QACrafter on retry
    details: dict              # per-criterion scores


_SYSTEM = """\
Bạn là chuyên gia kiểm duyệt chất lượng dữ liệu huấn luyện mô hình ngôn ngữ pháp lý.
Nhiệm vụ: đánh giá một cặp câu hỏi – câu trả lời theo 4 tiêu chí.

QUY TẮC ĐẦU RA BẮT BUỘC:
- Phần <output> PHẢI là một JSON object hợp lệ và HOÀN CHỈNH.
- KHÔNG đặt text hoặc suy nghĩ vào phần content output.
- CHỈ trả về JSON object, bắt đầu bằng { và kết thúc bằng }.
- Không dùng markdown, không dùng ```json.
"""


def _build_user(
    question: str,
    answer_a: str,
    answer_b: str,
    answer_hint: str,
    block: LegalBlock,
    task: TaskDefinition,
) -> str:
    return f"""\
=== THÔNG TIN ĐÁNH GIÁ ===

Loại nhiệm vụ: {task.id} – {task.name_vi} (Level {task.level}: {task.level_name})
Định dạng câu trả lời kỳ vọng: {task.answer_format}

--- Điều luật nguồn ---
{block.combined_text}

--- Câu hỏi ---
{question}

--- Câu trả lời của Solver A ---
{answer_a}

--- Câu trả lời của Solver B ---
{answer_b}

--- Câu trả lời tham khảo (QACrafter) ---
{answer_hint}

=== YÊU CẦU ĐÁNH GIÁ ===

Đánh giá theo 4 tiêu chí sau (mỗi tiêu chí cho điểm từ 0.0 đến 1.0):

1. **consensus** (0–1): Mức độ đồng thuận giữa Solver A và B.
   - 1.0 = hoàn toàn đồng thuận về nội dung cốt lõi
   - 0.5 = đồng thuận một phần
   - 0.0 = mâu thuẫn hoàn toàn

2. **factuality** (0–1): Câu trả lời có dựa CHÍNH XÁC vào điều luật cung cấp không?
   - Phạt nặng (-0.5) nếu trích dẫn điều luật sai/bịa
   - Phạt nhẹ (-0.2) nếu bỏ sót điều luật quan trọng

3. **classification** (0–1): Câu hỏi và câu trả lời có đúng với loại nhiệm vụ {task.id} không?
   - 1.0 = hoàn toàn khớp với mô tả loại nhiệm vụ
   - 0.0 = sai loại hoàn toàn

4. **clarity** (0–1): Ngôn ngữ có rõ ràng, tự nhiên, phù hợp văn phong pháp lý tiếng Việt không?

Ngoài ra:
- **best_solver**: "A" hoặc "B" (solver nào trả lời tốt hơn, hoặc "A" nếu ngang nhau)
- **feedback**: Nếu có tiêu chí dưới 0.7, viết phản hồi cụ thể cho QACrafter để sửa
  (nêu rõ vấn đề và cách cải thiện). Nếu tất cả ≥ 0.7, để trống "".

Trả về JSON (KHÔNG markdown):
{{
  "consensus": <float>,
  "factuality": <float>,
  "classification": <float>,
  "clarity": <float>,
  "best_solver": "<A hoặc B>",
  "feedback": "<chuỗi phản hồi hoặc rỗng>"
}}
"""


class Checker:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    async def check(
        self,
        question: str,
        answer_hint: str,
        answer_a: str,
        answer_b: str,
        block: LegalBlock,
        task: TaskDefinition,
    ) -> CheckResult:
        user_prompt = _build_user(question, answer_a, answer_b, answer_hint, block, task)
        try:
            result = await self._llm.chat_json(
                system=_SYSTEM,
                user=user_prompt,
                temperature=0.1,
                max_tokens=2048,
            )
        except Exception as exc:
            logger.error(f"Checker error: {exc}")
            return CheckResult(
                passed=False, score=0.0, best_answer="",
                feedback=f"Lỗi hệ thống khi kiểm tra: {exc}",
                details={},
            )

        consensus      = float(result.get("consensus",      0.0))
        factuality     = float(result.get("factuality",     0.0))
        classification = float(result.get("classification", 0.0))
        clarity        = float(result.get("clarity",        0.0))
        best_solver    = result.get("best_solver", "A")
        feedback       = result.get("feedback", "")

        # Aggregate score (factuality weighted more heavily)
        score = (
            consensus      * 0.20 +
            factuality     * 0.40 +
            classification * 0.25 +
            clarity        * 0.15
        )

        # Must pass threshold AND factuality floor (anti-hallucination)
        passed = (
            score >= CHECKER_PASS_THRESHOLD
            and factuality >= 0.6
            and classification >= 0.6
        )

        best_answer = answer_a if best_solver == "A" else answer_b

        if not passed and not feedback:
            # Auto-generate generic feedback
            issues = []
            if consensus < 0.7:
                issues.append("Hai câu trả lời mâu thuẫn nhau – câu hỏi có thể quá mơ hồ.")
            if factuality < 0.7:
                issues.append("Câu trả lời chứa thông tin không có trong điều luật – hãy đặt câu hỏi bám sát hơn vào văn bản.")
            if classification < 0.7:
                issues.append(f"Câu hỏi/câu trả lời không khớp với loại nhiệm vụ {task.id} ({task.name_vi}).")
            if clarity < 0.7:
                issues.append("Ngôn ngữ chưa tự nhiên, cần viết lại rõ ràng hơn.")
            feedback = " | ".join(issues)

        return CheckResult(
            passed=passed,
            score=score,
            best_answer=best_answer,
            feedback=feedback,
            details={
                "consensus": consensus,
                "factuality": factuality,
                "classification": classification,
                "clarity": clarity,
            },
        )
