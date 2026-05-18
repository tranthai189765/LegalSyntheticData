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

For answer_format == "summary", a plain structured summary is produced
(no IRAC structure).
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
2. KHÔNG bịa đặt điều luật, số hiệu văn bản, mức phạt, thời hạn.
3. Viết thành một đoạn văn LIỀN MẠCH, KHÔNG dùng tiêu đề (không viết "Issue:", "Rule:", v.v.).
4. Cấu trúc ngầm định theo IRAC với YÊU CẦU CHIỀU SÂU — đây là phần quan trọng nhất:

   I – VẤN ĐỀ: "Vấn đề đặt ra là..."
   - Xác định vấn đề pháp lý cốt lõi. KHÔNG giả định kết quả trước khi phân tích.
   - Nếu có yếu tố thời hạn, phải xác lập deadline theo luật TRƯỚC KHI kết luận vi phạm.
   - Nếu có nhiều vấn đề, chỉ đặt tiêu điểm vào vấn đề TRỌNG YẾU nhất.

   R – QUY ĐỊNH: "Theo quy định tại [Điều X Văn bản Y]..."
   - Trích dẫn đủ điều khoản. Chỉ dùng văn bản trong danh sách được phép.
   - Nếu có ≥2 điều khoản có thể áp dụng, nêu cả hai và giải thích lý do chọn điều nào.

   A – ÁP DỤNG (phần QUAN TRỌNG NHẤT, phải đủ 4 yếu tố sau):
   (a) PHÂN LOẠI PHÁP LÝ: hành vi/khoản chi/giao dịch trong tình huống thuộc danh mục nào theo luật?
       Nếu có ranh giới mơ hồ (ví dụ: "chi đào tạo" là operational hay administrative?),
       phải phân tích rõ cơ sở để phân loại, không bỏ qua.
   (b) PHÂN TÍCH TỪNG CHỦ THỂ: ai có nghĩa vụ gì, cơ sở pháp lý riêng cho từng người.
   (c) LẬP LUẬN CÓ ĐIỀU KIỆN: dùng cấu trúc "Nếu [điều kiện X] thì [hệ quả pháp lý Y];
       ngược lại / tuy nhiên nếu [điều kiện Z] thì [hệ quả pháp lý W]."
       Không kết luận dứt khoát khi thực tế còn thiếu thông tin để verify điều kiện.
   (d) XEM XÉT NGOẠI LỆ / MIỄN TRÁCH: có cơ sở miễn/giảm trách không?
       Nếu không có → nêu rõ "không có cơ sở miễn trách theo quy định được cung cấp."

   C – KẾT LUẬN: "Từ đó có thể kết luận rằng..."
   - Kết luận CÓ ĐIỀU KIỆN khi chưa xác lập đủ facts (ví dụ: "nếu phương án đã được
     phê duyệt đúng thẩm quyền thì... ; nếu chưa thì...").
   - Nêu cụ thể: chủ thể → nghĩa vụ cụ thể → cơ quan thẩm quyền (nếu luật quy định).
   - KHÔNG kết luận "tuân thủ" hay "vi phạm" khi còn điều kiện chưa được xác minh.
   - KHÔNG dùng kết luận chung chung như "cần cải thiện quy trình", "cần rà soát lại".

5. Văn phong pháp lý chuẩn, trang trọng.
6. TUYỆT ĐỐI không nhắc đến văn bản pháp luật nào không có trong điều luật được cung cấp.
"""

_SYSTEM_SUMMARY = """\
Bạn là chuyên gia pháp lý với chuyên môn sâu về pháp luật Việt Nam, đặc biệt là lĩnh vực tài chính – ngân sách – thuế.

Khi tóm tắt văn bản pháp lý:
1. CHỈ dựa vào nội dung điều luật được cung cấp.
2. KHÔNG bịa đặt, KHÔNG thêm thông tin ngoài văn bản.
3. Trình bày súc tích, có cấu trúc: mục đích → phạm vi áp dụng → nội dung chính → hiệu lực.
4. Không dùng IRAC, không dùng câu mở đầu "Vấn đề đặt ra là...".
5. Văn phong trung lập, khách quan.
6. TUYỆT ĐỐI không nhắc đến văn bản pháp luật nào không có trong điều luật được cung cấp.
"""

_SYSTEM_GENERAL = """\
Bạn là luật sư cao cấp với chuyên môn sâu về pháp luật Việt Nam, đặc biệt là lĩnh vực tài chính – ngân sách – thuế.
Khi trả lời câu hỏi pháp lý, bạn:
1. CHỈ dựa vào nội dung điều luật được cung cấp và kiến thức pháp luật thực chứng.
2. KHÔNG bịa đặt điều luật, số hiệu, mức phạt.
3. Trả lời theo đúng định dạng phù hợp với loại câu hỏi.
4. Dùng văn phong pháp lý rõ ràng, chính xác.
5. TUYỆT ĐỐI không nhắc đến văn bản pháp luật nào không có trong điều luật được cung cấp.
"""

# Format hints per answer_format type
_FORMAT_HINTS: dict[str, str] = {
    "paragraph": (
        "Trả lời bằng đoạn văn pháp lý liên mạch theo cấu trúc IRAC ẩn "
        "(không dùng tiêu đề). Mở đầu bằng 'Vấn đề đặt ra là...', "
        "trích dẫn điều luật bằng 'Theo quy định tại [Điều X Văn bản Y]...', "
        "kết thúc bằng 'Từ đó có thể kết luận rằng...'."
    ),
    "summary": (
        "Trả lời bằng đoạn văn tóm tắt KHÔNG theo IRAC. "
        "Trình bày có cấu trúc: mục đích – phạm vi – nội dung chính – hiệu lực. "
        "Không bắt đầu bằng 'Vấn đề đặt ra là...'. Súc tích, không quá 200 từ."
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

LƯU Ý: Chỉ trích dẫn văn bản pháp luật có trong danh sách sau, không trích dẫn văn bản khác:
{chr(10).join(f"- {r}" for r in block.law_references)}
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
        if task.answer_format == "paragraph":
            system = _SYSTEM_IRAC
        elif task.answer_format == "summary":
            system = _SYSTEM_SUMMARY
        else:
            system = _SYSTEM_GENERAL
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
