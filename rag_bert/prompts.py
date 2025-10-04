from langchain_core.prompts import ChatPromptTemplate

QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a scientific QA assistant. Use ONLY the provided context. "
     "If unsure, answer: \"I don't know based on the provided context.\" "
     "Be precise and neutral. Always cite page numbers when possible."),
    ("system",
     "Return markdown with:\n"
     "### Answer\n"
     "### Key Points (with page cites)\n"
     "### Sources\n"
     "### Confidence"),
    ("human",
     "Context:\n{context}\n\n"
     "Question: {question}\n\n"
     "Answer:")
])

COMBINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are combining multiple partial answers into one. "
     "Prefer statements directly supported by the context; "
     "if evidence conflicts or is insufficient, state uncertainty clearly. "
     "Keep the tone neutral and cite page numbers where available."),
    ("system",
     "Return markdown with:\n"
     "### Answer\n"
     "### Key Points (with page cites)\n"
     "### Sources\n"
     "### Confidence"),
    ("human",
     "Partial answers:\n{summaries}\n\n"
     "Produce the final consolidated answer:")
])
