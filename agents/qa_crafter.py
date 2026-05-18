"""
Agent 2 – QA Crafter

Generates a (question, answer_hint) pair for a specific task type.
Receives:
  • block   – the LegalBlock (raw legal articles)
  • context – scenario string from ContextMaster
  • task    – TaskDefinition specifying the Bloom level and task type
  • feedback – optional string from Checker (used on retry iterations)

Returns a QAPair dataclass.
"""

from __future__ import annotations
from dataclasses import dataclass

from loguru import logger

from llm_client import LLMClient
from neo4j_client import LegalBlock
from tasks.definitions import TaskDefinition


@dataclass
class QAPair:
    question: str
    answer_hint: str      # preliminary reference answer from Crafter
    instruction: str      # taken directly from task.instruction


_SYSTEM = """\
Bạn là chuyên gia xây dựng bộ dữ liệu đánh giá mô hình ngôn ngữ cho lĩnh vực pháp luật Việt Nam.
Nhiệm vụ của bạn là tạo ra một cặp câu hỏi – câu trả lời CHẤT LƯỢNG CAO theo đúng loại nhiệm vụ được yêu cầu.

Nguyên tắc bắt buộc:
1. Câu hỏi và câu trả lời phải DỰA HOÀN TOÀN vào nội dung điều luật được cung cấp.
2. KHÔNG được bịa đặt số điều/số nghị định/mức phạt không có trong văn bản.
3. Câu hỏi phải rõ ràng, tự nhiên, phù hợp văn phong pháp lý tiếng Việt.
4. Câu trả lời phải chính xác, có thể kiểm chứng từ văn bản pháp luật.
5. CHỐNG HALLUCINATION: Chỉ được nhắc đến số hiệu văn bản pháp luật có trong "Danh sách văn bản được phép trích dẫn" bên dưới.

QUY TẮC ĐẦU RA BẮT BUỘC:
- Phần output PHẢI là một JSON object hợp lệ và HOÀN CHỈNH.
- KHÔNG đặt text hoặc suy nghĩ vào phần output.
- CHỈ trả về JSON object, bắt đầu bằng { và kết thúc bằng }.
- Không dùng markdown, không dùng ```json.
"""


def _build_user(
    block: LegalBlock,
    context: str,
    task: TaskDefinition,
    feedback: str | None,
) -> str:
    feedback_section = ""
    if feedback:
        feedback_section = f"""
⚠️ PHẢN HỒI TỪ VÒNG TRƯỚC (cần sửa lại):
{feedback}

"""

    # Build explicit allowed-law list to prevent hallucination
    allowed_laws = "\n".join(f"  - {ref}" for ref in block.law_references) or "  (không có)"
    doc_numbers = ", ".join(
        sorted({u.official_number for u in block.units if u.official_number})
    )

    return f"""\
{feedback_section}\
Các điều luật được cung cấp:

{block.combined_text}

---
Danh sách văn bản được phép trích dẫn (KHÔNG được nhắc đến bất kỳ văn bản nào khác):
{allowed_laws}
Số hiệu văn bản: {doc_numbers}

---
Tình huống pháp lý:
{context}

---
Loại nhiệm vụ cần tạo: **{task.id} – {task.name_vi}** (Level {task.level}: {task.level_name})

Hướng dẫn chi tiết về loại nhiệm vụ này:
{task.qa_prompt}

---
Hãy tạo 1 cặp câu hỏi – câu trả lời theo đúng loại nhiệm vụ trên.
NHẮC LẠI: Chỉ được trích dẫn số hiệu văn bản từ danh sách trên.
Trả về JSON hợp lệ (không có markdown) theo định dạng:
{{
  "question": "<câu hỏi hoàn chỉnh>",
  "answer": "<câu trả lời tham khảo>"
}}
"""


class QACrafter:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    async def generate(
        self,
        block: LegalBlock,
        context: str,
        task: TaskDefinition,
        feedback: str | None = None,
    ) -> QAPair:
        user_prompt = _build_user(block, context, task, feedback)
        try:
            result = await self._llm.chat_json(
                system=_SYSTEM,
                user=user_prompt,
                temperature=0.75,
                max_tokens=3000,
            )
            question   = result.get("question", "").strip()
            answer     = result.get("answer", "").strip()
            if not question or not answer:
                raise ValueError("QACrafter: empty question or answer in response")
            return QAPair(
                question=question,
                answer_hint=answer,
                instruction=task.instruction,
            )
        except Exception as exc:
            logger.error(f"QACrafter error: {exc}")
            raise
