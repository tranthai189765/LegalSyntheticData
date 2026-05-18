"""
22 Task definitions spanning Bloom's Taxonomy Levels 1–5 for Vietnamese legal QA.

Each TaskDefinition carries:
  • id          — e.g. "3.1"
  • level       — int 1-5
  • level_name  — level label in Vietnamese
  • name        — task name (English)
  • name_vi     — task name (Vietnamese)
  • weight      — sampling weight (higher = more likely to be chosen)
  • instruction — the "instruction" field included in every output sample
                  (tells a downstream model HOW to answer this task type)
  • qa_prompt   — injected into QACrafter system prompt; describes how to
                  construct a question AND a reference answer for this task
  • answer_format — short hint for Solver / Checker ("paragraph", "list",
                    "json_triplets", "classification", "yes_no")
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field


@dataclass
class TaskDefinition:
    id: str
    level: int
    level_name: str
    name: str
    name_vi: str
    weight: float
    instruction: str
    qa_prompt: str
    answer_format: str   # "paragraph" | "list" | "json_triplets" | "classification" | "yes_no"


# ── Level 1: Recognition & Recall ────────────────────────────────────────────

T1_1 = TaskDefinition(
    id="1.1", level=1,
    level_name="Recognition & Recall",
    name="Legal Entity Recognition",
    name_vi="Nhận diện thực thể pháp lý",
    weight=1.0,
    instruction=(
        "Đọc đoạn văn bản pháp luật sau và trích xuất TẤT CẢ các thực thể pháp lý "
        "xuất hiện trong đó. Phân loại mỗi thực thể theo một trong các nhóm: "
        "PERSON (cá nhân), ORGANIZATION (tổ chức), POLITICAL_BODY (cơ quan nhà nước), "
        "LAW (văn bản pháp luật), MONEY (số tiền), DATE (ngày tháng), "
        "LOCATION (địa danh), LEGAL_ARTICLE (điều khoản), SOCIAL_ROLE (vai trò xã hội). "
        "Trả về theo định dạng: LOẠI: thực thể; LOẠI: thực thể; ..."
    ),
    qa_prompt="""
Nhiệm vụ: Tạo 1 cặp câu hỏi – câu trả lời loại **1.1 – Nhận diện thực thể pháp lý**.

Câu hỏi phải:
- Trích dẫn một đoạn văn bản lấy từ các điều luật được cung cấp (50-200 từ).
- Yêu cầu trích xuất và phân loại TẤT CẢ thực thể pháp lý trong đoạn đó.
- Bắt đầu bằng "Hãy trích xuất tất cả các thực thể pháp lý trong đoạn văn bản sau: ..."

Câu trả lời phải:
- Liệt kê đầy đủ các thực thể theo định dạng LOẠI: thực thể; ...
- Bao gồm ít nhất 5 thực thể thuộc ít nhất 3 loại khác nhau.
""",
    answer_format="list",
)

T1_2 = TaskDefinition(
    id="1.2", level=1,
    level_name="Recognition & Recall",
    name="Legal Topic Classification",
    name_vi="Phân loại chủ đề pháp lý",
    weight=1.0,
    instruction=(
        "Đọc câu hỏi pháp lý dưới đây và xác định nó thuộc lĩnh vực pháp luật nào. "
        "Chỉ cần nêu tên lĩnh vực, không cần giải thích."
    ),
    qa_prompt="""
Nhiệm vụ: Tạo 1 cặp câu hỏi – câu trả lời loại **1.2 – Phân loại chủ đề pháp lý**.

Câu hỏi phải:
- Đặt ra một tình huống pháp lý thực tế cụ thể (2–4 câu) liên quan đến các điều luật cung cấp.
- Bắt đầu bằng "Đọc tình huống sau và cho biết nó thuộc lĩnh vực pháp luật nào: ..."
- KHÔNG đề cập thẳng tên lĩnh vực trong câu hỏi.

Câu trả lời phải:
- Nêu đúng tên lĩnh vực pháp lý (ví dụ: "Tài chính – ngân sách", "Thuế", "Kế toán", ...).
- Ngắn gọn, 1–2 dòng.
""",
    answer_format="classification",
)

T1_3 = TaskDefinition(
    id="1.3", level=1,
    level_name="Recognition & Recall",
    name="Legal Concept Recall",
    name_vi="Ghi nhớ khái niệm pháp lý",
    weight=1.0,
    instruction=(
        "Định nghĩa khái niệm pháp lý dưới đây dựa trên quy định của pháp luật Việt Nam. "
        "Trả lời chính xác theo nội dung văn bản quy phạm pháp luật, không suy diễn thêm."
    ),
    qa_prompt="""
Nhiệm vụ: Tạo 1 cặp câu hỏi – câu trả lời loại **1.3 – Ghi nhớ khái niệm pháp lý**.

Câu hỏi phải:
- Hỏi định nghĩa của MỘT khái niệm pháp lý cụ thể xuất hiện trong các điều luật cung cấp.
- Dạng: "[Khái niệm X] là gì?" hoặc "Theo [văn bản Y], [khái niệm X] được hiểu như thế nào?"
- Khái niệm phải là thuật ngữ pháp lý chuyên ngành (không phải từ thông thường).

Câu trả lời phải:
- Trích dẫn trực tiếp hoặc diễn giải sát với định nghĩa trong điều luật.
- Bao gồm tên văn bản nguồn.
""",
    answer_format="paragraph",
)

T1_4 = TaskDefinition(
    id="1.4", level=1,
    level_name="Recognition & Recall",
    name="Article Recall",
    name_vi="Truy xuất điều khoản pháp luật",
    weight=1.0,
    instruction=(
        "Truy xuất và trình bày nội dung của điều khoản pháp luật được hỏi. "
        "Nêu đúng số điều, khoản, văn bản gốc và tóm tắt nội dung chính."
    ),
    qa_prompt="""
Nhiệm vụ: Tạo 1 cặp câu hỏi – câu trả lời loại **1.4 – Truy xuất điều khoản pháp luật**.

Câu hỏi phải:
- Hỏi về nội dung của một điều/khoản/điểm cụ thể trong các điều luật cung cấp.
- Dạng: "Điều X [văn bản Y] quy định về vấn đề gì?" hoặc
  "Khoản X Điều Y [văn bản Z] nêu nội dung gì?"

Câu trả lời phải:
- Tóm tắt chính xác nội dung điều/khoản được hỏi.
- Không suy diễn hay thêm thông tin ngoài văn bản.
""",
    answer_format="paragraph",
)

T1_5 = TaskDefinition(
    id="1.5", level=1,
    level_name="Recognition & Recall",
    name="Legal Schema Recall",
    name_vi="Nhận diện quan hệ văn bản pháp luật",
    weight=0.9,
    instruction=(
        "Xác định mối quan hệ pháp lý giữa các văn bản (sửa đổi, bổ sung, thay thế, "
        "bãi bỏ, căn cứ, hướng dẫn). Trả lời ngắn gọn, chính xác tên văn bản liên quan."
    ),
    qa_prompt="""
Nhiệm vụ: Tạo 1 cặp câu hỏi – câu trả lời loại **1.5 – Nhận diện quan hệ văn bản pháp luật**.

Câu hỏi phải:
- Hỏi về mối quan hệ giữa 2 văn bản pháp luật (sửa đổi / thay thế / bãi bỏ / căn cứ).
- Dạng: "[Văn bản A] sửa đổi/thay thế văn bản nào?" hoặc
  "[Văn bản A] được ban hành căn cứ vào văn bản nào?"
- Chỉ hỏi về các văn bản CÓ THỰC trong các điều luật được cung cấp.

Câu trả lời phải:
- Nêu đúng tên/số hiệu văn bản liên quan.
- Có thể thêm một câu giải thích về bản chất mối quan hệ.
""",
    answer_format="paragraph",
)


# ── Level 2: Understanding & Structuring ─────────────────────────────────────

T2_1 = TaskDefinition(
    id="2.1", level=2,
    level_name="Understanding & Structuring",
    name="Relation Extraction",
    name_vi="Trích xuất quan hệ pháp lý",
    weight=1.0,
    instruction=(
        "Đọc tình huống pháp lý dưới đây và xác định các quan hệ pháp luật xuất hiện. "
        "Trình bày theo dạng: Chủ thể – loại quan hệ – đối tượng."
    ),
    qa_prompt="""
Nhiệm vụ: Tạo 1 cặp câu hỏi – câu trả lời loại **2.1 – Trích xuất quan hệ pháp lý**.

Câu hỏi phải:
- Mô tả một tình huống pháp lý thực tế có 2–3 bên tham gia (100–250 từ).
  Ví dụ: tranh chấp hợp đồng, vi phạm hành chính, giao dịch tài chính.
- Hỏi: "Tình huống trên có những quan hệ pháp luật nào?"

Câu trả lời phải:
- Liệt kê ≥2 quan hệ pháp lý theo dạng:
  "[Chủ thể A] và [Chủ thể B] – [loại quan hệ] – [nội dung]"
- Dựa sát vào quy định trong các điều luật cung cấp.
""",
    answer_format="list",
)

T2_2 = TaskDefinition(
    id="2.2", level=2,
    level_name="Understanding & Structuring",
    name="Legal Element Recognition",
    name_vi="Nhận diện thành phần quy phạm pháp luật",
    weight=1.0,
    instruction=(
        "Xác định các thành phần của quy phạm pháp luật trong đoạn văn dưới đây: "
        "(1) Giả định – điều kiện áp dụng; "
        "(2) Quy định – hành vi được phép, bắt buộc hoặc cấm; "
        "(3) Chế tài – hậu quả pháp lý khi vi phạm. "
        "Nêu rõ từng thành phần và trích dẫn câu văn tương ứng."
    ),
    qa_prompt="""
Nhiệm vụ: Tạo 1 cặp câu hỏi – câu trả lời loại **2.2 – Nhận diện thành phần quy phạm pháp luật**.

Câu hỏi phải:
- Trích dẫn nguyên văn một điều/khoản đầy đủ từ các điều luật cung cấp.
- Hỏi: "Xác định các thành phần của quy phạm pháp luật (giả định, quy định, chế tài)
  trong điều khoản sau: [nội dung điều khoản]"

Câu trả lời phải:
- Nêu rõ ba phần: Giả định, Quy định, Chế tài (có thể là "Không có chế tài").
- Trích câu cụ thể từ điều khoản cho mỗi thành phần.
""",
    answer_format="list",
)

T2_3 = TaskDefinition(
    id="2.3", level=2,
    level_name="Understanding & Structuring",
    name="Legal Graph Structuring",
    name_vi="Cấu trúc hóa đồ thị pháp luật",
    weight=0.8,
    instruction=(
        "Chuyển đổi nội dung pháp lý dưới đây thành danh sách các triplet "
        "(thực thể 1, mối quan hệ, thực thể 2) biểu diễn quan hệ giữa các văn bản "
        "hoặc điều khoản. Mỗi triplet trên một dòng, phân tách bằng ' | '."
    ),
    qa_prompt="""
Nhiệm vụ: Tạo 1 cặp câu hỏi – câu trả lời loại **2.3 – Cấu trúc hóa đồ thị pháp luật**.

Câu hỏi phải:
- Trích dẫn một đoạn văn bản pháp luật mô tả mối quan hệ giữa các điều khoản/văn bản
  (ví dụ: "bãi bỏ", "thay thế", "sửa đổi", "bổ sung", "căn cứ vào").
- Hỏi: "Trích xuất các triplet (thực thể, quan hệ, thực thể) từ đoạn văn bản sau: ..."

Câu trả lời phải:
- Danh sách các triplet, mỗi triplet dạng: "Điều X / Văn bản Y | bãi bỏ | Điều Z / Văn bản W"
- Ít nhất 2 triplet.
""",
    answer_format="json_triplets",
)

T2_4 = TaskDefinition(
    id="2.4", level=2,
    level_name="Understanding & Structuring",
    name="Judgment Verification",
    name_vi="Xác minh phán quyết tòa án",
    weight=1.0,
    instruction=(
        "Dựa trên nội dung bản án hoặc tình huống pháp lý được mô tả, "
        "xác định nhận định dưới đây là ĐÚNG hay SAI. "
        "Giải thích ngắn gọn căn cứ pháp lý cho đánh giá của bạn."
    ),
    qa_prompt="""
Nhiệm vụ: Tạo 1 cặp câu hỏi – câu trả lời loại **2.4 – Xác minh phán quyết tòa án**.

Câu hỏi phải:
- Mô tả một vụ việc/tình huống pháp lý (100–200 từ) dựa trên các điều luật cung cấp.
- Đưa ra một nhận định về phán quyết hoặc cách áp dụng luật (có thể đúng hoặc sai).
- Hỏi: "Từ nội dung được cho, nhận định sau đây là đúng hay sai? [nhận định]"

Câu trả lời phải:
- Bắt đầu bằng "Đúng." hoặc "Sai."
- Giải thích 2–3 câu với căn cứ pháp lý cụ thể từ điều luật.
- 30% câu hỏi nên có đáp án "Sai" để tạo tính đa dạng.
""",
    answer_format="yes_no",
)

T2_5 = TaskDefinition(
    id="2.5", level=2,
    level_name="Understanding & Structuring",
    name="User Intent Understanding",
    name_vi="Hiểu ý định người dùng pháp lý",
    weight=0.9,
    instruction=(
        "Đọc câu hỏi/yêu cầu pháp lý dưới đây và xác định đúng intent (ý định). "
        "Chỉ chọn từ danh sách: chitchat | comparative_analysis | document_relationship | "
        "document_retrieval | external_analysis | general | legal_query | stats_summary. "
        "Có thể chọn nhiều nhãn nếu câu hỏi có nhiều intent; phân tách bằng ' và '."
    ),
    qa_prompt="""
Nhiệm vụ: Tạo 1 cặp câu hỏi – câu trả lời loại **2.5 – Hiểu ý định người dùng pháp lý**.

Câu hỏi phải:
- Bao gồm câu hỏi thực tế của người dùng (có thể phức tạp, có nhiều intent).
- Bắt đầu bằng: "Đọc câu hỏi sau và xác định intent: [câu hỏi của người dùng]"
- Nhắc lại danh sách intent hợp lệ trong câu hỏi:
  "Chọn từ: chitchat | comparative_analysis | document_relationship | document_retrieval | external_analysis | general | legal_query | stats_summary"

Câu trả lời phải:
- Chỉ chọn nhãn từ danh sách 8 nhãn trên, KHÔNG dùng nhãn nào khác.
- Liệt kê 1–3 nhãn phù hợp, phân tách bằng " và ".
- Ví dụ hợp lệ: "legal_query" hoặc "legal_query và comparative_analysis"
- TUYỆT ĐỐI không tự đặt nhãn mới ngoài danh sách trên.
""",
    answer_format="classification",
)


# ── Level 3: Reasoning & Inference ───────────────────────────────────────────

T3_1 = TaskDefinition(
    id="3.1", level=3,
    level_name="Reasoning & Inference",
    name="Article / Clause Prediction",
    name_vi="Dự đoán điều khoản áp dụng",
    weight=1.2,
    instruction=(
        "Đọc câu hỏi pháp lý dưới đây và xác định các điều/khoản/điểm của văn bản pháp luật "
        "có liên quan hoặc trực tiếp hỗ trợ trả lời câu hỏi. "
        "Nêu rõ số điều, khoản, tên văn bản (số hiệu)."
    ),
    qa_prompt="""
Nhiệm vụ: Tạo 1 cặp câu hỏi – câu trả lời loại **3.1 – Dự đoán điều khoản áp dụng**.

Câu hỏi phải:
- Đặt ra một tình huống/câu hỏi pháp lý ngắn (1–3 câu) liên quan đến các điều luật cung cấp.
- KHÔNG nhắc đến số điều/khoản trong câu hỏi.
- Bắt đầu bằng: "Đọc câu hỏi sau và trả về các điều khoản pháp luật liên quan: ..."

Câu trả lời phải:
- Liệt kê điều khoản cụ thể theo dạng: "Khoản X Điều Y [số hiệu văn bản]"
- Ít nhất 1, tối đa 3 điều khoản.
""",
    answer_format="list",
)

T3_2 = TaskDefinition(
    id="3.2", level=3,
    level_name="Reasoning & Inference",
    name="Legal Court Decision Prediction",
    name_vi="Dự đoán phán quyết tòa án",
    weight=1.2,
    instruction=(
        "Dựa trên tình tiết vụ án được mô tả và quy định pháp luật hiện hành, "
        "dự đoán kết quả phán quyết có khả năng xảy ra nhất. "
        "Trình bày ngắn gọn: phán quyết dự đoán và lý do pháp lý chính."
    ),
    qa_prompt="""
Nhiệm vụ: Tạo 1 cặp câu hỏi – câu trả lời loại **3.2 – Dự đoán phán quyết tòa án**.

Câu hỏi phải:
- Mô tả tình tiết một vụ kiện dân sự / hành chính / kinh doanh – thương mại (150–250 từ).
- Bao gồm: các bên tranh chấp, sự việc, yêu cầu của mỗi bên.
- Bắt đầu bằng: "Từ nội dung vụ án sau, dự đoán phán quyết có khả năng nhất của tòa: ..."

Câu trả lời phải:
- Nêu kết quả dự đoán (tòa chấp nhận/bác yêu cầu của ai, mức bồi thường nếu có).
- Trích dẫn điều luật áp dụng.
""",
    answer_format="paragraph",
)

T3_3 = TaskDefinition(
    id="3.3", level=3,
    level_name="Reasoning & Inference",
    name="Multi-Article Reasoning",
    name_vi="Suy luận đa điều khoản",
    weight=1.3,
    instruction=(
        "Dựa trên các điều luật liên quan được cung cấp phía trên, hãy trả lời câu hỏi pháp lý "
        "dưới đây bằng cách suy luận logic, kết hợp nội dung của nhiều điều khoản. "
        "Trình bày theo tư duy IRAC: Vấn đề – Quy định – Áp dụng – Kết luận."
    ),
    qa_prompt="""
Nhiệm vụ: Tạo 1 cặp câu hỏi – câu trả lời loại **3.3 – Suy luận đa điều khoản**.

Câu hỏi phải:
- Đặt ra một tình huống pháp lý phức tạp (1–3 câu) ĐÒI HỎI kết hợp ít nhất 2 điều luật.
- Phần đầu câu hỏi phải trình bày nội dung các điều luật liên quan (trích từ điều luật cung cấp).
- Dạng: "Nội dung điều luật liên quan:\n- [Điều A văn bản X]: ...\n- [Điều B văn bản Y]: ...\n
  Hãy trả lời câu hỏi dựa trên các điều luật trên:\n[Câu hỏi tình huống]"

Câu trả lời phải:
- Trả lời theo IRAC, kết hợp cả 2+ điều luật.
- Độ dài 100–200 từ.
""",
    answer_format="paragraph",
)

T3_4 = TaskDefinition(
    id="3.4", level=3,
    level_name="Reasoning & Inference",
    name="Conflict / Overlap Detection",
    name_vi="Phát hiện mâu thuẫn / chồng chéo pháp luật",
    weight=1.1,
    instruction=(
        "Phân tích hai quy phạm pháp luật dưới đây và xác định chúng có MÂU THUẪN "
        "hoặc CHỒNG CHÉO với nhau không. Trả lời 'Có' hoặc 'Không' và giải thích ngắn gọn."
    ),
    qa_prompt="""
Nhiệm vụ: Tạo 1 cặp câu hỏi – câu trả lời loại **3.4 – Phát hiện mâu thuẫn / chồng chéo**.

Câu hỏi phải:
- Đưa ra 2 quy phạm pháp luật (có thể mâu thuẫn hoặc không) lấy từ các điều luật cung cấp.
- Trình bày rõ: "Quy phạm 1: [nội dung]" và "Quy phạm 2: [nội dung]".
- Hỏi: "Hai quy phạm trên có mâu thuẫn hoặc chồng chéo với nhau không?"
- Đảm bảo ít nhất 40% cặp KHÔNG mâu thuẫn để tạo đa dạng.

Câu trả lời phải:
- Bắt đầu bằng "Có." hoặc "Không."
- Giải thích 2–3 câu vì sao.
""",
    answer_format="yes_no",
)

T3_5 = TaskDefinition(
    id="3.5", level=3,
    level_name="Reasoning & Inference",
    name="Penalty / Remedy Estimation",
    name_vi="Ước lượng mức chế tài / biện pháp khắc phục",
    weight=1.2,
    instruction=(
        "Dựa trên tình huống vi phạm pháp luật được mô tả và các quy định hiện hành, "
        "xác định mức xử phạt hoặc biện pháp khắc phục áp dụng. "
        "Nêu cụ thể mức phạt (tiền, hình phạt), căn cứ pháp lý."
    ),
    qa_prompt="""
Nhiệm vụ: Tạo 1 cặp câu hỏi – câu trả lời loại **3.5 – Ước lượng mức chế tài**.

Câu hỏi phải:
- Mô tả một hành vi vi phạm pháp lý CỤ THỂ có trong phạm vi các điều luật cung cấp.
- Nêu đủ tình tiết: ai vi phạm, vi phạm điều gì, mức độ vi phạm.
- Hỏi: "Mức xử phạt cho hành vi trên theo quy định pháp luật là gì?"

Câu trả lời phải:
- Nêu mức phạt cụ thể (nếu là phạt tiền: nêu mức từ–đến).
- Trích dẫn điều khoản áp dụng.
- Nếu có thể, nêu biện pháp khắc phục hậu quả đi kèm.
""",
    answer_format="paragraph",
)


# ── Level 4: Interpretation & Generation ─────────────────────────────────────

T4_1 = TaskDefinition(
    id="4.1", level=4,
    level_name="Interpretation & Generation",
    name="Legal Document Summarization",
    name_vi="Tóm tắt văn bản pháp lý",
    weight=1.2,
    instruction=(
        "Tóm tắt văn bản pháp lý dưới đây, giữ lại các thông tin trọng yếu: "
        "mục đích, phạm vi áp dụng, các quy định chính, hiệu lực thi hành. "
        "Trình bày súc tích, không quá 200 từ."
    ),
    qa_prompt="""
Nhiệm vụ: Tạo 1 cặp câu hỏi – câu trả lời loại **4.1 – Tóm tắt văn bản pháp lý**.

Câu hỏi phải:
- Cung cấp toàn bộ hoặc một phần lớn nội dung một văn bản pháp luật (từ điều luật cung cấp).
- Hỏi: "Hãy tóm tắt nội dung trọng yếu của [tên văn bản] dưới đây: [nội dung]"

Câu trả lời phải:
- Tóm tắt 100–200 từ, có cấu trúc: mục đích → phạm vi → nội dung chính → hiệu lực.
- KHÔNG dùng cấu trúc IRAC, KHÔNG bắt đầu bằng "Vấn đề đặt ra là...".
- Không thêm thông tin ngoài văn bản.
""",
    answer_format="summary",
)

T4_2 = TaskDefinition(
    id="4.2", level=4,
    level_name="Interpretation & Generation",
    name="Judicial Reasoning Generation",
    name_vi="Sinh lập luận tư pháp",
    weight=1.3,
    instruction=(
        "Viết lập luận tư pháp theo cấu trúc IRAC cho vụ việc dưới đây:\n"
        "- Vấn đề (Issue): xác định vấn đề pháp lý cần giải quyết\n"
        "- Quy định (Rule): nêu điều luật áp dụng\n"
        "- Áp dụng (Application): phân tích sự việc theo quy định\n"
        "- Kết luận (Conclusion): đưa ra kết luận pháp lý"
    ),
    qa_prompt="""
Nhiệm vụ: Tạo 1 cặp câu hỏi – câu trả lời loại **4.2 – Sinh lập luận tư pháp**.

=== YÊU CẦU CÂU HỎI ===
• TẬP TRUNG VÀO MỘT vấn đề pháp lý cốt lõi duy nhất (không liệt kê 5–6 sub-problems).
• Tình huống phải tạo ra GENUINE AMBIGUITY – người đọc không thể kết luận ngay mà phải suy luận:
    - Ví dụ: hành vi X có thể thuộc danh mục A (được phép) hoặc danh mục B (bị hạn chế).
    - Ví dụ: điều kiện Y có thể đã thỏa mãn hoặc chưa tùy cách hiểu quy định.
    - Ví dụ: hai điều khoản có thể áp dụng với hệ quả khác nhau.
• TUYỆT ĐỐI KHÔNG:
    - Nêu sẵn ngưỡng / mức % / mức phạt trong câu hỏi (đó là nhiệm vụ của phần Rule).
    - Dùng từ "lo ngại", "có thể không phù hợp", "vi phạm" trong câu hỏi – để model tự suy luận.
    - Gợi ý sẵn hướng giải quyết hay kết luận.
• Tình huống có ít nhất 2 chủ thể, thể hiện rõ ai đang ra quyết định và ai chịu hệ quả.
• Hỏi: "Trình bày lập luận pháp lý theo IRAC cho tình huống sau: ..."

=== YÊU CẦU ANSWER_HINT ===
answer_hint phải là lập luận IRAC chiều sâu, bao gồm đủ 5 yếu tố:
1. **Issue**: xác định vấn đề pháp lý cốt lõi duy nhất; nêu rõ điểm mơ hồ / ranh giới cần xác định.
2. **Rule**: trích đủ điều khoản áp dụng; nếu ≥2 điều có thể áp dụng, nêu cả hai.
3. **Classification analysis**: hành vi/khoản chi/giao dịch thuộc danh mục nào theo luật?
   Phải analyze ranh giới phân loại, không bỏ qua ambiguity.
4. **Application với conditional reasoning**:
   "Nếu [điều kiện A xác lập được] thì [hệ quả pháp lý X]; tuy nhiên nếu [điều kiện B] thì [hệ quả Y]."
   Phân tích trách nhiệm từng chủ thể. Kiểm tra ngoại lệ/miễn trách.
5. **Conclusion có điều kiện**: không kết luận dứt khoát khi còn điều kiện chưa verify;
   nêu (a) chủ thể, (b) nghĩa vụ cụ thể, (c) bước thủ tục tiếp theo theo luật.
   Tổng 250–400 từ.
""",
    answer_format="paragraph",
)

T4_3 = TaskDefinition(
    id="4.3", level=4,
    level_name="Interpretation & Generation",
    name="Objective Legal Opinion Generation",
    name_vi="Sinh ý kiến tư vấn pháp lý khách quan",
    weight=1.2,
    instruction=(
        "Dựa trên tình huống và quy định pháp luật liên quan, đưa ra ý kiến tư vấn pháp lý "
        "KHÁCH QUAN, CÂN BẰNG. Trình bày các mặt lợi, bất lợi và khuyến nghị cụ thể. "
        "Văn phong chuyên nghiệp, tránh thiên lệch."
    ),
    qa_prompt="""
Nhiệm vụ: Tạo 1 cặp câu hỏi – câu trả lời loại **4.3 – Sinh ý kiến tư vấn pháp lý khách quan**.

Câu hỏi phải:
- Mô tả một tình huống có tính tranh luận pháp lý (chính sách, quy định mới, v.v.).
- Yêu cầu phân tích tác động hoặc đưa ra ý kiến cân nhắc.
- Dạng: "Phân tích [vấn đề X] theo quan điểm pháp lý và đưa ra ý kiến khách quan."

Câu trả lời phải:
- Trình bày cả hai mặt (ủng hộ và phản đối).
- Nêu căn cứ pháp lý cho mỗi quan điểm.
- Kết luận bằng khuyến nghị thực tế.
- Tổng 150–300 từ.
""",
    answer_format="paragraph",
)


# ── Level 5: Ethics, Fairness & Bias ─────────────────────────────────────────

T5_1 = TaskDefinition(
    id="5.1", level=5,
    level_name="Ethics, Fairness & Bias",
    name="Bias Detection",
    name_vi="Phát hiện thiên lệch trong lập luận pháp lý",
    weight=0.9,
    instruction=(
        "Phân tích câu trả lời pháp lý dưới đây và xác định liệu nó có chứa thiên lệch "
        "(về giới tính, địa vị, tôn giáo, sắc tộc, v.v.) không. "
        "Nếu có, chỉ rõ loại thiên lệch và câu văn cụ thể."
    ),
    qa_prompt="""
Nhiệm vụ: Tạo 1 cặp câu hỏi – câu trả lời loại **5.1 – Phát hiện thiên lệch**.

Câu hỏi phải:
- Đưa ra một tình huống hoặc câu trả lời pháp lý (có thể chứa hoặc không chứa thiên lệch).
- Hỏi: "Câu trả lời/lập luận sau có chứa thiên lệch không phù hợp với nguyên tắc pháp lý không?"
- Ít nhất 40% câu hỏi nên có thiên lệch thực sự.

Câu trả lời phải:
- Xác định rõ: "Có thiên lệch." hoặc "Không có thiên lệch."
- Nếu có: nêu loại thiên lệch và câu văn cụ thể, giải thích tại sao không phù hợp.
- Nếu không: giải thích ngắn gọn tại sao lập luận là khách quan.
""",
    answer_format="paragraph",
)

T5_2 = TaskDefinition(
    id="5.2", level=5,
    level_name="Ethics, Fairness & Bias",
    name="Privacy & Data Protection",
    name_vi="Bảo vệ quyền riêng tư và dữ liệu",
    weight=0.8,
    instruction=(
        "Dựa trên quy định pháp luật hiện hành về bảo vệ dữ liệu cá nhân và quyền riêng tư, "
        "phân tích tình huống dưới đây: hành vi đó có vi phạm quy định không? "
        "Nêu quy định áp dụng và hậu quả pháp lý nếu có."
    ),
    qa_prompt="""
Nhiệm vụ: Tạo 1 cặp câu hỏi – câu trả lời loại **5.2 – Bảo vệ quyền riêng tư và dữ liệu**.

Câu hỏi phải:
- Mô tả tình huống liên quan đến thu thập, xử lý hoặc công bố thông tin cá nhân.
- Hỏi liệu hành vi đó có vi phạm quy định bảo vệ dữ liệu/quyền riêng tư không.
- Liên quan đến các điều luật được cung cấp.

Câu trả lời phải:
- Xác định có vi phạm hay không.
- Trích dẫn điều luật áp dụng.
- Nêu hậu quả pháp lý (nếu vi phạm) hoặc lý do hành vi được phép.
""",
    answer_format="paragraph",
)

T5_3 = TaskDefinition(
    id="5.3", level=5,
    level_name="Ethics, Fairness & Bias",
    name="Ethical Consistency Assessment",
    name_vi="Đánh giá nhất quán đạo đức nghề nghiệp pháp lý",
    weight=0.8,
    instruction=(
        "Đánh giá hành vi/lập luận trong tình huống dưới đây có phù hợp với "
        "chuẩn mực đạo đức nghề nghiệp pháp lý không. "
        "Nêu rõ quy tắc đạo đức bị vi phạm hoặc được tuân thủ."
    ),
    qa_prompt="""
Nhiệm vụ: Tạo 1 cặp câu hỏi – câu trả lời loại **5.3 – Đánh giá nhất quán đạo đức nghề nghiệp**.

Câu hỏi phải:
- Mô tả hành vi của luật sư, thẩm phán, công chứng viên hoặc cán bộ pháp lý.
- Hỏi: "Hành vi của [chủ thể] có vi phạm chuẩn mực đạo đức nghề nghiệp không?"
- Tình huống phải liên quan đến các quy định trong điều luật cung cấp.

Câu trả lời phải:
- Kết luận: vi phạm hay không vi phạm.
- Trích dẫn quy tắc/điều khoản đạo đức liên quan.
- Phân tích 2–3 câu.
""",
    answer_format="paragraph",
)

T5_4 = TaskDefinition(
    id="5.4", level=5,
    level_name="Ethics, Fairness & Bias",
    name="Unfair Contract Detection",
    name_vi="Phát hiện điều khoản hợp đồng không công bằng",
    weight=0.9,
    instruction=(
        "Phân tích điều khoản hợp đồng dưới đây: điều khoản đó có công bằng "
        "cho cả hai bên không? Nếu không, chỉ rõ điểm bất công bằng và "
        "quy định pháp luật liên quan."
    ),
    qa_prompt="""
Nhiệm vụ: Tạo 1 cặp câu hỏi – câu trả lời loại **5.4 – Phát hiện điều khoản hợp đồng không công bằng**.

Câu hỏi phải:
- Trích dẫn hoặc mô tả một điều khoản hợp đồng (thuê tài sản, dịch vụ, lao động, ...).
- Hỏi: "Điều khoản hợp đồng sau có công bằng cho cả hai bên không?"
- Khoảng 40% điều khoản nên CÔNG BẰNG để tạo đa dạng.

Câu trả lời phải:
- Nhận xét: "Công bằng." hoặc "Không công bằng – có lợi cho [bên A]."
- Giải thích 2–4 câu, có dẫn chiếu quy định pháp luật liên quan.
""",
    answer_format="paragraph",
)


# ── Registry ─────────────────────────────────────────────────────────────────

TASKS: list[TaskDefinition] = [
    T1_1, T1_2, T1_3, T1_4, T1_5,
    T2_1, T2_2, T2_3, T2_4, T2_5,
    T3_1, T3_2, T3_3, T3_4, T3_5,
    T4_1, T4_2, T4_3,
    T5_1, T5_2, T5_3, T5_4,
]

_TASK_MAP: dict[str, TaskDefinition] = {t.id: t for t in TASKS}
_WEIGHTS = [t.weight for t in TASKS]


def get_task(task_id: str) -> TaskDefinition:
    if task_id not in _TASK_MAP:
        raise KeyError(f"Unknown task id: {task_id}")
    return _TASK_MAP[task_id]


def sample_task(exclude_ids: list[str] | None = None) -> TaskDefinition:
    """Weighted random task selection, optionally excluding some IDs."""
    pool = TASKS
    weights = _WEIGHTS
    if exclude_ids:
        filtered = [(t, w) for t, w in zip(TASKS, _WEIGHTS) if t.id not in exclude_ids]
        if filtered:
            pool, weights = zip(*filtered)
            pool, weights = list(pool), list(weights)
    return random.choices(pool, weights=weights, k=1)[0]
