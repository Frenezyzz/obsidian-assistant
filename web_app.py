from flask import Flask, render_template, request
import markdown
from assistant_core import (
    load_vectorstore,
    load_llm,
    ask_question,
    extract_sources,
    get_folder_structure
)

app = Flask(__name__)

print("Loading vector database...")
db = load_vectorstore()

print("Loading LLM...")
llm = load_llm()


@app.route("/", methods=["GET", "POST"])
def index():

    answer = None
    sources = []

    if request.method == "POST":

        question = request.form.get("question", "")
        folder_filter = request.form.get("folder_filter", "")
        source_language_hint = request.form.get("source_language_hint", "")
        study_mode = request.form.get("study_mode", "normal")
        if study_mode == "folder_quiz":
            question = f"""
        Generate a complete study quiz using ALL the notes
        from the folder: {folder_filter}
        """.strip()

        answer, docs, results, queries_used = ask_question(
            db=db,
            llm=llm,
            question=question,
            folder_filter=folder_filter,
            source_language_hint=source_language_hint,
            study_mode=study_mode
        )

        sources = extract_sources(docs)
    folder_structure = get_folder_structure()
    if answer:
        answer = markdown.markdown(
            answer,
            extensions=["fenced_code", "tables"]
        )
    return render_template(
        "index.html",
        answer=answer,
        sources=sources,
        folder_structure=folder_structure
    )


if __name__ == "__main__":
    app.run(debug=True)