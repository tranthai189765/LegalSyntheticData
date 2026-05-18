"""
Agent 4 – Checker (LLM-as-a-Judge)

Evaluates a generated QA sample on four criteria:
  1. Consensus      – Do Solver A and B substantially agree?
  2. Factuality     – Are all citations in Q+A grounded in the provided block?
                      Uses a 3-step structured verification:
                        Step 1 – enumerate every citation in Q + A
                        Step 2 – verify each citation against the block
                        Step 3 – compute score; Python hard-overrides to 0.0
                                  if any hallucinated citation is confirmed
  3. Classification – Does the QA pair match the stated task type?
  4. Clarity        – Is the language clear, natural Vietnamese legal writing?

Returns a CheckResult with:
  • passed       – bool
  • score        – float [0, 1] weighted aggregate
  • best_answer  – the better solver answer
  • feedback     – detailed guidance for QACrafter on retry
  • details      – per-criterion scores + hallucinated citations list
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
    score: float
    best_answer: str
    feedback: str
    details: dict


_SYSTEM = """\
Bạn là chuyên gia kiểm duyệt chất lượng dữ liệu huấn luyện mô hình ngôn ngữ pháp lý.
Nhiệm vụ: đánh giá một cặp câu hỏi – câu trả lời theo quy trình 3 bước.

QUY TẮC ĐẦU RA BẮT BUỘC:
- CHỈ trả về JSON object, bắt đầu bằng { và kết thúc bằng }.
- Không dùng markdown, không dùng ```json, không thêm text ngoài JSON.
"""


def _build_user(
    question: str,
    answer_a: str,
    answer_b: str,
    answer_hint: str,
    block: LegalBlock,
    task: TaskDefinition,
) -> str:
    allowed_refs = "\n".join(f"  - {r}" for r in block.law_references) or "  (không có)"
    return f"""\
=== NGỮ LIỆU ĐẦU VÀO ===

Loại nhiệm vụ: {task.id} – {task.name_vi} (Level {task.level}: {task.level_name})
Định dạng câu trả lời kỳ vọng: {task.answer_format}

--- Điều luật nguồn (TOÀN BỘ nội dung hợp lệ để trích dẫn) ---
{block.combined_text}

--- Danh sách tham chiếu hợp lệ ---
{allowed_refs}

--- Câu hỏi ---
{question}

--- Câu trả lời Solver A ---
{answer_a}

--- Câu trả lời Solver B ---
{answer_b}

--- Câu trả lời tham khảo (QACrafter) ---
{answer_hint}

=== QUY TRÌNH ĐÁNH GIÁ 3 BƯỚC ===

**BƯỚC 1 – LIỆT KÊ TRÍCH DẪN**
Tìm TẤT CẢ các trích dẫn pháp lý xuất hiện trong CÂU HỎI VÀ CẢ HAI câu trả lời Solver.
Một trích dẫn là: số hiệu văn bản (ví dụ "03/1998/TT-BTC") và/hoặc số điều/khoản (ví dụ "Điều 5", "Khoản 2 Điều 3").
Bao gồm cả cụm như "Nghị định số X/CP", "Thông tư số Y/TT-BTC", "Quyết định Z/QĐ-BTC".

**BƯỚC 2 – ĐỐI CHIẾU VỚI BLOCK**
Với mỗi trích dẫn tìm được:
(a) Số hiệu văn bản có xuất hiện trong "Điều luật nguồn" không?
(b) Số điều/khoản đó có tồn tại trong "Điều luật nguồn" không?
(c) Nội dung được diễn giải có khớp với những gì "Điều luật nguồn" nêu không?
→ Nếu (a) hoặc (b) sai → trích dẫn bịa (hallucination).
→ Nếu (c) sai → trích dẫn sai nội dung.

**BƯỚC 3 – ĐÁNH GIÁ 4 TIÊU CHÍ**

3a. **consensus** (0.0–1.0): Solver A và B có đồng thuận về nội dung cốt lõi không?
  - 1.0 = đồng thuận hoàn toàn
  - 0.5 = đồng thuận một phần
  - 0.0 = mâu thuẫn

3b. **factuality** (0.0–1.0): Dựa trên kết quả Bước 2:
  - Bắt đầu từ 1.0
  - Trừ 0.5 cho mỗi trích dẫn bịa (văn bản hoặc điều khoản không tồn tại trong block)
  - Trừ 0.2 cho mỗi trích dẫn sai nội dung (văn bản tồn tại nhưng bị diễn giải sai)
  - Trừ 0.1 nếu bỏ sót điều luật quan trọng có trong block
  - Tối thiểu 0.0

3c. **classification** (0.0–1.0): QA có khớp đúng loại nhiệm vụ {task.id} không?
  - 1.0 = khớp hoàn toàn, câu hỏi đúng dạng, câu trả lời đúng format
  - 0.0 = sai loại nhiệm vụ hoàn toàn
  - Với nhiệm vụ Level {task.level} (≥4): trừ 0.3 nếu câu trả lời chỉ là
    "trích dẫn luật → restate facts → kết luận dứt khoát" mà không có:
    (a) phân tích ranh giới phân loại hoặc (b) lập luận có điều kiện ("Nếu X thì Y")
    hoặc (c) xem xét ngoại lệ. Đây là dấu hiệu reasoning chưa đủ chiều sâu cho level 4+.

3d. **clarity** (0.0–1.0): Ngôn ngữ tiếng Việt pháp lý có tự nhiên, rõ ràng, chính xác không?

3e. **best_solver**: "A" hoặc "B" (câu trả lời nào tốt hơn).

3f. **hallucinated_citations**: Danh sách các trích dẫn bịa đặt tìm được ở Bước 2
    (chuỗi mô tả ngắn mỗi mục, ví dụ "Điều 401 TT 18/2025/TT-BTC không tồn tại trong block").
    Để [] nếu không có hallucination.

3g. **feedback**: Nếu có tiêu chí nào dưới 0.7, mô tả cụ thể vấn đề và cách sửa cho QACrafter.
    Nếu factuality thấp do hallucination: liệt kê tên văn bản/điều khoản bị bịa.
    Để "" nếu tất cả tiêu chí ≥ 0.7.

Trả về JSON (KHÔNG markdown, KHÔNG text trước/sau):
{{
  "hallucinated_citations": ["<mô tả vi phạm 1>", ...],
  "consensus": <float 0.0-1.0>,
  "factuality": <float 0.0-1.0>,
  "classification": <float 0.0-1.0>,
  "clarity": <float 0.0-1.0>,
  "best_solver": "<A hoặc B>",
  "feedback": "<chuỗi hoặc rỗng>"
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

        hallucinated_citations: list = result.get("hallucinated_citations", [])
        consensus      = float(result.get("consensus",      0.0))
        factuality     = float(result.get("factuality",     0.0))
        classification = float(result.get("classification", 0.0))
        clarity        = float(result.get("clarity",        0.0))
        best_solver    = result.get("best_solver", "A")
        feedback       = result.get("feedback", "")

        # Hard override: if checker confirmed hallucinated citations, factuality = 0
        if hallucinated_citations:
            factuality = 0.0
            hal_str = "; ".join(hallucinated_citations)
            hallucination_note = f"HALLUCINATION CONFIRMED: {hal_str}"
            feedback = (hallucination_note + " | " + feedback).rstrip(" |")
            logger.debug(f"  Checker confirmed hallucination: {hal_str[:120]}")

        # Aggregate score (factuality weighted 40%)
        score = (
            consensus      * 0.20 +
            factuality     * 0.40 +
            classification * 0.25 +
            clarity        * 0.15
        )

        # Pass requires score threshold + factuality floor + classification floor
        passed = (
            score >= CHECKER_PASS_THRESHOLD
            and factuality >= 0.6
            and classification >= 0.6
        )

        best_answer = answer_a if best_solver == "A" else answer_b

        if not passed and not feedback:
            issues = []
            if consensus < 0.7:
                issues.append("Hai câu trả lời mâu thuẫn – câu hỏi có thể quá mơ hồ.")
            if factuality < 0.7:
                issues.append("Câu trả lời chứa thông tin không có trong điều luật – hãy bám sát văn bản hơn.")
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
                "hallucinated_citations": hallucinated_citations,
            },
        )
