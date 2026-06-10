import time
import statistics

from chatbot import retriever, llm
from evaluation_set import evaluation_set

retrieval_times = []
llm_times = []
total_times = []

SYSTEM_PROMPT = """
Bạn là trợ lý học vụ HUST.

Quy tắc:

- Trả lời trực tiếp.
- Trả lời như đang tư vấn trực tiếp cho sinh viên.
- Không tự bịa ra thông tin. Nếu không biết, hãy nói "Tôi không tìm thấy thông tin!"
"""

print("===== RESPONSE TIME EVALUATION =====\n")

for sample in evaluation_set:

    question = sample["question"]

    # =====================
    # Total timer
    # =====================
    total_start = time.time()

    # =====================
    # Retrieval
    # =====================
    retrieval_start = time.time()

    docs = retriever.invoke(question)

    retrieval_time = (
        time.time() - retrieval_start
    )

    # =====================
    # Build Context
    # =====================
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = f"""
{SYSTEM_PROMPT}

======================
CONTEXT
======================

{context}

======================
QUESTION
======================

{question}

======================
ANSWER
======================
"""

    # =====================
    # LLM
    # =====================
    llm_start = time.time()

    response = llm.invoke(prompt)

    llm_time = (
        time.time() - llm_start
    )

    # =====================
    # Total
    # =====================
    total_time = (
        time.time() - total_start
    )

    retrieval_times.append(
        retrieval_time
    )

    llm_times.append(
        llm_time
    )

    total_times.append(
        total_time
    )

    print(
        f"{sample['id']} | "
        f"Retrieval={retrieval_time:.3f}s | "
        f"LLM={llm_time:.3f}s | "
        f"Total={total_time:.3f}s"
    )

# =====================
# Summary
# =====================

avg_retrieval = statistics.mean(
    retrieval_times
)

avg_llm = statistics.mean(
    llm_times
)

avg_total = statistics.mean(
    total_times
)

min_total = min(total_times)
max_total = max(total_times)

print("\n==============================")
print("AVERAGE RESPONSE TIME")
print("==============================")

print(
    f"Average Retrieval : "
    f"{avg_retrieval:.3f} s"
)

print(
    f"Average LLM       : "
    f"{avg_llm:.3f} s"
)

print(
    f"Average Total     : "
    f"{avg_total:.3f} s"
)

print(
    f"Fastest Query     : "
    f"{min_total:.3f} s"
)

print(
    f"Slowest Query     : "
    f"{max_total:.3f} s"
)