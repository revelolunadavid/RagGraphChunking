from typing import List
from .openrouter_client import call_openrouter


def ask_from_chunks(chunks: List[str], query: str, max_len: int = 200) -> str:
    """Ask the LLM using only the provided chunks. Returns a concise single-line answer.

    If the answer is not present in the chunks, returns the token "NO_ENCONTRADO".
    """

    # Build the context with numbered chunks (keep them as provided)
    context = "\n\n".join([f"Chunk {i+1}:\n{c}" for i, c in enumerate(chunks)])

    system = {
        "role": "system",
        "content": (
            "Eres un asistente que responde usando SOLAMENTE la información provista en los chunks. "
            "No inventes información ni uses conocimiento externo. Responde de forma concisa en una sola línea."
        ),
    }

    user = {
        "role": "user",
        "content": (
            f"Context:\n{context}\n\nQuestion: {query}\n\n"
            f"Instrucciones: Usa únicamente el contexto anterior y responde en una sola línea. "
            f"Si la respuesta no puede encontrarse en el contexto, responde exactamente: NO_ENCONTRADO. "
            f"Máx {max_len} caracteres."
        ),
    }

    messages = [system, user]

    resp = call_openrouter(messages)

    # follow OpenRouter response structure
    choice = resp.get("choices", [])[0]
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = message.get("content", "")

    # Normalize to one line and enforce max length
    answer = " ".join(content.splitlines()).strip()
    if not answer:
        return "NO_ENCONTRADO"
    if len(answer) > max_len:
        answer = answer[:max_len]
    return answer
