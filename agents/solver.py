"""
Agent 3 – Solver (two independent instances A and B)

Each Solver receives:
  • question  – from QACrafter
  • block     – LegalBlock with raw article text
  • task      – TaskDefinition (determines reasoning style)

And produces a standalone answer string.

Both Solvers use the same prompt template but run with slightly different
temperature (A=0.3 deterministic, B=0.6 more generative) so their outputs
can differ and be compared by the Checker.

For answer_format == "paragraph" tasks, the answer follows the IRAC
flowing-paragraph convention (no section headers, natural transitions):
  "Vấn đề đặt ra là... Theo quy định tại [Điều X văn bản Y]...
   Đối chiếu với... Từ đó có thể kết luận rằng..."
"""

from loguru import logger

from llm_client import LLMClient
from neo4j_client import LegalBlock
from tasks.definitions import TaskDefinition


# ── System prompts ────────────────────────────────────────────────────────────

_SYSTEM_IRAC = """\
Bạn là luật sư cao cấp với chuyên môn sâu về pháp luật Việt Nam, đặc biệt là lĩnh vực tài chính – ngân sách – thuế.

Khi trả lời câu hỏi pháp lý theo phong cách IRAC chuyển đổi (đoạn văn liền mạch):
1. CHỈ dựa vào nội dung điều luật được cung cấp.
2. KHÔNG bịa đặt điều luật, số hiệu văn bản, mức phạt.
3. Viết thành một đoạn văn LIỀN MẠCH, KHÔNG dùng tiêu đề (không viết "Issue:", "Rule:", v.v.).
4. Cấu trúc ngầm định theo IRAC:
   - Mở đầu: "Vấn đề đặt ra là..." – xác định vấn đề pháp lý.
   - Thân: "Theo quy định tại [Điều X Văn bản Y]..." – trích dẫn điều luật cụ thể.
   - Phân tích: "Đối chiếu với tình huống..." / "Căn cứ vào..." – áp dụng luật vào sự việc.
   - Kết: "Từ đó có thể kết luận rằng..." – kết luận trực tiếp.
5. Văn phong pháp lý chuẩn, trang trọng.
6. Phải trích dẫn đầy đủ tên văn bản (số hiệu, năm), điều, khoản cụ thể.
"""

_SYSTEM_GENERAL = """\
Bạn là luật sư cao cấp với chuyên môn sâu về pháp luật Việt Nam, đặc biệt là lĩnh vực tài chính – ngân sách – thuế.
Khi trả lời câu hỏi pháp lý, bạn:
1. CHỈ dựa vào nội dung điều luật được cung cấp và kiến thức pháp luật thực chứng.
2. KHÔNG bịa đặt điều luật, số hiệu, mức phạt.
3. Trả lời theo đúng định dạng phù hợp với loại câu hỏi.
4. Dùng văn phong pháp lý rõ ràng, chính xác.
"""

# Format hints per answer_format type
_FORMAT_HINTS: dict[str, str] = {
    "paragraph": (
        "Trả lời bằng đoạn văn pháp lý liên mạch theo cấu trúc IRAC ẩn "
        "(không dùng tiêu đề). Mở đầu bằng 'Vấn đề đặt ra là...', "
        "trích dẫn điều luật bằng 'Theo quy định tại [Điều X Văn bản Y]...', "
        "kết thúc bằng 'Từ đó có thể kết luận rằng...'."
    ),
    "list": (
        "Trả lời bằng danh sách có thứ tự, mỗi mục rõ ràng và có căn cứ pháp lý."
    ),
    "json_triplets": (
        "Trả lời bằng danh sách các triplet theo định dạng: "
        "thực thể 1 | quan hệ | thực thể 2 (mỗi triplet trên một dòng)."
    ),
    "classification": (
        "Trả lời ngắn gọn bằng nhãn phân loại. Không giải thích dài dòng."
    ),
    "yes_no": (
        "Bắt đầu câu trả lời bằng 'Có.' hoặc 'Không.' / 'Đúng.' hoặc 'Sai.', "
        "sau đó giải thích ngắn gọn (2–3 câu) với căn cứ pháp lý cụ thể "
        "(tên văn bản, điều khoản)."
    ),
}


def _build_user(question: str, block: LegalBlock, task: TaskDefinition) -> str:
    fmt_hint = _FORMAT_HINTS.get(task.answer_format, "")
    return f"""\
Các điều luật liên quan:

{block.combined_text}

---
Câu hỏi pháp lý:
{question}

---
Yêu cầu trả lời: {fmt_hint}
"""


class Solver:
    def __init__(self, llm: LLMClient, name: str = "A", temperature: float = 0.4):
        self._llm = llm
        self.name = name
        self._temperature = temperature

    async def solve(
        self,
        question: str,
        block: LegalBlock,
        task: TaskDefinition,
    ) -> str:
        user_prompt = _build_user(question, block, task)
        # Use IRAC-specific system prompt for paragraph-style answers
        system = _SYSTEM_IRAC if task.answer_format == "paragraph" else _SYSTEM_GENERAL
        try:
            answer = await self._llm.chat(
                system=system,
                user=user_prompt,
                temperature=self._temperature,
                max_tokens=1200,
            )
            return answer.strip()
        except Exception as exc:
            logger.error(f"Solver {self.name} error: {exc}")
            raise
