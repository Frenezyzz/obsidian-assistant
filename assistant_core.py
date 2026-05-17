from collections import OrderedDict

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

import json
from datetime import datetime
from pathlib import Path

PERSIST_DIR = "data"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1:8b"   # you can change this later
TOP_K = 30
MIN_RELEVANCE_SCORE = 0.55
HISTORY_FILE = "history.json"


def get_folder_structure():
    folders = {
        "0300 Psicologo": [],
        "0600 Guitarra": [],
        "0800 Universidad": [
            "Marketing",
            "Arquitectura",
        ]
    }

    return folders

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

def retrieve_documents(
        db,
        llm,
        question,
        folder_filter=None,
        source_language_hint=None,
        study_mode="normal"
    ):
    queries = [question]
        # SPECIAL MODE: folder_quiz
    if study_mode == "folder_quiz" and folder_filter:

        folder_filter = (
            folder_filter
            .lower()
            .replace("\\", "/")
            .strip()
        )

        raw = db.get(include=["documents", "metadatas"])

        results = []

        for text, metadata in zip(
            raw["documents"],
            raw["metadatas"]
        ):

            folder = (
                metadata.get("folder", "")
                .lower()
                .replace("\\", "/")
            )

            if folder_filter in folder:

                from langchain_core.documents import Document

                doc = Document(
                    page_content=text,
                    metadata=metadata
                )

                results.append((doc, 1.0))

        return results[:TOP_K], queries

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

def get_mode_instructions(study_mode):
    if study_mode == "summary":
        return """
- Give a concise study summary.
- Use short paragraphs or bullet points if helpful.
- Focus only on the most important ideas.
"""

    if study_mode == "quiz":
        return """
- Do not give a direct explanatory answer.
- Create 5 study questions based only on the provided context.
- Questions should help the user test understanding.
- Do not include answers unless explicitly asked.
"""
    if study_mode == "folder_quiz":
        return """
- Create a study quiz using ALL the provided context.
- Generate 10 varied study questions.
- Mix concepts from different notes if relevant.
- Include:
  - short answer questions
  - conceptual questions
  - true/false
  - multiple choice
- Do NOT include the answers.
- Make the quiz feel like a real exam review.
"""
    if study_mode == "flashcards":
        return """
    - Create 15 high-quality flashcards based ONLY on the provided context.

    Rules:
    - Each flashcard must test ONLY ONE idea.
    - Answers must be SHORT and PRECISE.
    - Avoid long explanations.
    - Avoid redundant flashcards.
    - Use varied flashcard styles:
    - definitions
    - true/false
    - concept identification
    - fill-in-the-blank
    - cause/effect
    - examples

    Formatting rules:
    - Format EXACTLY like this:

    Flashcard 1
    Q: ...
    A: ...

    Flashcard 2
    Q: ...
    A: ...

    - Leave one blank line between flashcards.
    - Do not add commentary before or after the flashcards.
    """

    return """
- Answer the user's question clearly and concisely.
- If possible, summarize the answer in a useful study-friendly way.
"""

def ask_question(db, llm, question, folder_filter=None, source_language_hint=None, study_mode="normal"):
    mode_instructions = get_mode_instructions(study_mode)
    if study_mode in {"flashcards", "folder_quiz"}:
        global MIN_RELEVANCE_SCORE
        old_threshold = MIN_RELEVANCE_SCORE
        MIN_RELEVANCE_SCORE = 0.15
    results, queries_used = retrieve_documents(
        db=db,
        llm=llm,
        question=question,
        folder_filter=folder_filter,
        source_language_hint=source_language_hint,
        study_mode=study_mode
    )
    if study_mode in {"flashcards", "folder_quiz"}:
        MIN_RELEVANCE_SCORE = old_threshold

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
{mode_instructions}

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

def save_history(question, study_mode, answer, sources):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "study_mode": study_mode,
        "answer": answer,
        "sources": sources
    }

    history_path = Path(HISTORY_FILE)

    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []

    history.append(entry)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

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
        print("Available study modes:")
        print("- normal")
        print("- summary")
        print("- quiz")
        print("- flashcards")
        print("- folder_quiz\n")

        study_mode = input(
            "Study mode (normal/summary/quiz/flashcards/folder_quiz): "
        ).strip().lower()

        if not study_mode:
            study_mode = "normal"

        if study_mode == "folder_quiz":

            folder_filter = input(
                "Folder for quiz generation: "
            ).strip()

            question = f"""
        Generate a complete study quiz using ALL the notes
        from the folder: {folder_filter}
        """.strip()

        else:
            question = input("Ask a question: ").strip()

            if question.lower() in {"exit", "quit", "salir"}:
                print("Goodbye.")
                break

            if not question:
                continue

            folder_filter = input("Folder filter (optional): ").strip()
        
        source_language_hint = input(
            "Source language hint (optional: en/es): "
        ).strip()

        answer, docs, results, queries_used = ask_question(
            db,
            llm,
            question,
            folder_filter=folder_filter,
            source_language_hint=source_language_hint,
            study_mode=study_mode
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

        save_history(
            question=question,
            study_mode=study_mode,
            answer=answer,
            sources=sources
        )
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()