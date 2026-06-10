from chatbot import retriever
from evaluation_set import evaluation_set


def hit_at_k(docs, source, page, k):

    for doc in docs[:k]:

        if (
            doc.metadata.get("source") == source
            and doc.metadata.get("page") == page
        ):
            return 1

    return 0


recall1 = 0
recall3 = 0
recall5 = 0

for sample in evaluation_set:

    docs = retriever.invoke(
        sample["question"]
    )

    recall1 += hit_at_k(
        docs,
        sample["source"],
        sample["page"],
        1
    )

    recall3 += hit_at_k(
        docs,
        sample["source"],
        sample["page"],
        3
    )

    recall5 += hit_at_k(
        docs,
        sample["source"],
        sample["page"],
        5
    )

n = len(evaluation_set)

print()
print("===== RETRIEVAL EVALUATION =====")
print(f"Questions : {n}")
print(f"Recall@1  : {recall1/n:.2%}")
print(f"Recall@3  : {recall3/n:.2%}")
print(f"Recall@5  : {recall5/n:.2%}")