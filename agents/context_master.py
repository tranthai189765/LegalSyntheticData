"""
Agent 1 – Context Master

Given a LegalBlock (cluster of related legal articles), generates a
realistic legal scenario that:
  • Is grounded in the actual articles (no hallucinated laws).
  • Matches the perspective of the task level (citizen, enterprise, court, etc.).
  • Is task-aware so the scenario naturally leads to the required question type.

Returns a plain-text scenario string that QACrafter receives as context.
"""

import random
from loguru import logger

from llm_client import LLMClient
from neo4j_client import LegalBlock
from tasks.definitions import TaskDefinition


PERSPECTIVES = [
    "một công dân thông thường",
    "một doanh nghiệp nhỏ và vừa",
    "một cán bộ thuế",
    "một kế toán viên doanh nghiệp",
    "một luật sư tư vấn",
    "một nhà đầu tư nước ngoài",
    "một cơ quan nhà nước cấp huyện",
    "một công ty kiểm toán",
    "một ngân hàng thương mại",
    "một người nộp thuế",
]

_SYSTEM = """\
Bạn là chuyên gia pháp lý tài chính Việt Nam với 15 năm kinh nghiệm tư vấn.
Nhiệm vụ của bạn là xây dựng một TÌNH HUỐNG PHÁP LÝ THỰC TẾ dựa trên các điều luật được cung cấp.

Yêu cầu bắt buộc:
1. Tình huống phải bám CHÍNH XÁC vào nội dung các điều luật được cung cấp – không được bịa thêm điều luật.
2. Tình huống phải thực tế, xảy ra trong đời sống doanh nghiệp hoặc quản lý tài chính.
3. Đặt nhân vật cụ thể (tên, vai trò, bối cảnh).
4. Tình huống phải phù hợp với loại câu hỏi sẽ được đặt ra.
5. Độ dài: 100–200 từ.
6. Chỉ trả về mô tả tình huống, không thêm phân tích hay kết luận.
"""


def _build_user(block: LegalBlock, task: TaskDefinition, perspective: str) -> str:
    articles_text = block.combined_text

    # Level-specific scenario guidance
    level_hint = {
        1: "Tình huống nên đơn giản, trực tiếp đề cập đến các điều luật hoặc khái niệm trong văn bản.",
        2: "Tình huống nên có nhiều bên tham gia, mô tả rõ quan hệ pháp lý giữa các chủ thể.",
        3: "Tình huống nên phức tạp, đòi hỏi suy luận từ nhiều điều luật để đưa ra kết luận.",
        4: (
            "Tình huống phải tạo ra ĐIỂM MƠ HỒ PHÁP LÝ rõ ràng: hành vi hoặc khoản chi "
            "có thể được phân loại theo ≥2 cách khác nhau dẫn đến hệ quả pháp lý khác nhau. "
            "Không đưa ra gợi ý hay kết luận trong tình huống – chỉ mô tả sự kiện thuần túy. "
            "Tình huống phải có ít nhất 2 chủ thể với vai trò và nghĩa vụ khác nhau."
        ),
        5: "Tình huống nên đặt ra vấn đề đạo đức, công bằng hoặc xung đột lợi ích.",
    }.get(task.level, "")

    return f"""\
Dưới đây là các điều luật liên quan:

{articles_text}

---

Góc nhìn: Tình huống xảy ra với {perspective}.
Loại câu hỏi sẽ được đặt: {task.name_vi} (Level {task.id}).
Hướng dẫn thêm: {level_hint}

Hãy tạo một tình huống pháp lý thực tế (100–200 từ) dựa trên các điều luật trên.
Chỉ trả về mô tả tình huống, không thêm bình luận hay kết luận.
"""


class ContextMaster:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    async def generate(self, block: LegalBlock, task: TaskDefinition) -> str:
        perspective = random.choice(PERSPECTIVES)
        user_prompt = _build_user(block, task, perspective)
        try:
            context = await self._llm.chat(
                system=_SYSTEM,
                user=user_prompt,
                temperature=0.75,
                max_tokens=1500,
            )
            return context.strip()
        except Exception as exc:
            logger.error(f"ContextMaster error: {exc}")
            # Fallback: return raw combined text
            return block.combined_text[:600]
