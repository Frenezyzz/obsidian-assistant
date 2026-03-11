from collections import OrderedDict

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama


PERSIST_DIR = "data"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1:8b"   # you can change this later
TOP_K = 5
MIN_RELEVANCE_SCORE = 0.55


def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    return db


def load_llm():
    llm = ChatOllama(
        model=CHAT_MODEL,
        temperature=0
    )
    return llm

def looks_like_english(text):
    common_spanish_words = {
        "el", "la", "los", "las", "de", "que", "sobre", "para",
        "con", "sin", "por", "aceptar", "inmigrantes", "puntos"
    }

    words = set(text.lower().split())
    matches = words.intersection(common_spanish_words)

    return len(matches) < 2

def reformulate_query(llm, question, target_language):
    prompt = f"""
You are translating a search query.

Task:
Translate the user's query into {target_language}.

Strict rules:
- Return the query ONLY in {target_language}.
- Do NOT answer the question.
- Do NOT explain anything.
- Do NOT keep the original language.
- Do NOT return bilingual text.
- If the original query is already in {target_language}, return it unchanged.

User query:
{question}
""".strip()

    response = llm.invoke(prompt)
    return response.content.strip()

def filter_docs_by_folder(docs, folder_filter):
    if not folder_filter:
        return docs

    folder_filter = folder_filter.lower().strip()

    filtered = []
    for doc in docs:
        source = doc.metadata.get("source", "").lower()
        if folder_filter in source:
            filtered.append(doc)

    return filtered

def build_context(docs):
    parts = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown source")
        content = doc.page_content.strip()

        parts.append(
            f"[Chunk {i}]\n"
            f"Source: {source}\n"
            f"Content:\n{content}\n"
        )

    return "\n\n".join(parts)

def retrieve_documents(db, llm, question, folder_filter=None, source_language_hint=None):
    queries = [question]

    if source_language_hint:
        source_language_hint = source_language_hint.lower().strip()

        if source_language_hint == "en":
            alt_query = reformulate_query(llm, question, "English")
            if alt_query and alt_query.lower() != question.lower():
                if looks_like_english(alt_query):
                    queries.append(alt_query)
                else:
                    print("\n[Warning] English reformulation failed. Using original query only.")

        elif source_language_hint == "es":
            alt_query = reformulate_query(llm, question, "Spanish")
            if alt_query and alt_query.lower() != question.lower():
                queries.append(alt_query)

    all_results = []

    for q in queries:
        results = db.similarity_search_with_relevance_scores(q, k=TOP_K)
        all_results.extend(results)

    deduped = {}
    for doc, score in all_results:
        source = doc.metadata.get("source", "")
        chunk_id = doc.metadata.get("chunk_id", "")
        key = f"{source}::{chunk_id}"

        if key not in deduped or score > deduped[key][1]:
            deduped[key] = (doc, score)

    filtered_results = [
        (doc, score)
        for doc, score in deduped.values()
        if score >= MIN_RELEVANCE_SCORE
    ]

    if folder_filter:
        folder_filter = folder_filter.lower().strip()
        filtered_results = [
            (doc, score)
            for doc, score in filtered_results
            if folder_filter in doc.metadata.get("folder", "").lower()
            or folder_filter in doc.metadata.get("relative_path", "").lower()
        ]

    filtered_results.sort(key=lambda x: x[1], reverse=True)

    return filtered_results[:TOP_K], queries

def ask_question(db, llm, question, folder_filter=None, source_language_hint=None):
    results, queries_used = retrieve_documents(
        db=db,
        llm=llm,
        question=question,
        folder_filter=folder_filter,
        source_language_hint=source_language_hint
    )

    if not results:
        return "Not found in your vault.", [], [], queries_used

    docs = [doc for doc, score in results]
    context = build_context(docs)

    prompt = f"""
You are an assistant that answers questions using ONLY the context provided below.

Rules:
- Answer only with information found in the context.
- Do not invent or assume anything.
- If the answer is not clearly supported by the context, reply exactly:
Not found in your vault.
- Answer in the same language as the user's question.
- Be clear and concise.
- If possible, summarize the answer in a useful study-friendly way.

Context:
{context}

Question:
{question}
""".strip()

    response = llm.invoke(prompt)
    answer = response.content.strip()

    return answer, docs, results, queries_used

def format_source(doc):
    relative_path = doc.metadata.get("relative_path") or doc.metadata.get("source", "Unknown source")
    h1 = doc.metadata.get("h1")
    h2 = doc.metadata.get("h2")
    h3 = doc.metadata.get("h3")

    headings = [h for h in [h1, h2, h3] if h]

    if headings:
        return f"{relative_path} | {' > '.join(headings)}"

    return relative_path

def extract_sources(docs):
    unique_sources = list(
        OrderedDict.fromkeys(
            format_source(doc)
            for doc in docs
        )
    )
    return unique_sources


def main():
    print("Loading vector database...")
    db = load_vectorstore()

    print("Loading chat model...")
    llm = load_llm()

    print("\nYour Obsidian assistant is ready.")
    print("Write your question and press Enter.")
    print("Type 'exit' to close.\n")

    print("Optional folder filter examples:")
    print("- 0300 Psicologo")
    print("- 0600 Guitarra")
    print("- Leave empty to search the whole vault\n")

    while True:
        question = input("Ask a question: ").strip()

        if question.lower() in {"exit", "quit", "salir"}:
            print("Goodbye.")
            break

        if not question:
            continue

        folder_filter = input("Folder filter (optional): ").strip()
        source_language_hint = input("Source language hint (optional: en/es): ").strip()
        


        answer, docs, results, queries_used = ask_question(
            db,
            llm,
            question,
            folder_filter=folder_filter,
            source_language_hint=source_language_hint
        )
        sources = extract_sources(docs)

        print("\nAnswer:\n")
        print(answer)

        print("\nSources:")
        if sources:
            for src in sources:
                print(f"- {src}")
        else:
            print("- No sources found")

        print("\nQueries used for retrieval:")
        for q in queries_used:
            print(f"- {q}")

        print("\nTop matches:")
        for doc, score in results:
            src = format_source(doc)
            print(f"- {score:.3f} | {src}")

        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()