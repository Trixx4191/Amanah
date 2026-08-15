"""
Command-line interface for the local agent.

Usage:
    python -m agent.cli add <filepath>          Add a document to the knowledge base
    python -m agent.cli remove <doc_id>          Remove a document by its ID
    python -m agent.cli list                     List documents in the knowledge base
    python -m agent.cli chat                      Start an interactive chat session
"""

from __future__ import annotations

import argparse
import sys

from agent.knowledge_base import KnowledgeBase
from agent.llm import LLMConfig, LocalLLM
from agent.memory import ConversationMemory
from agent.reasoning import run_agent

DEFAULT_MODEL_PATH = "models/model.gguf"  # see docs/setup.md for how to obtain one


def cmd_add(args, kb: KnowledgeBase):
    print(f"Ingesting {args.filepath} ...")
    info = kb.add_document(args.filepath)
    print(f"Added '{info['source_filename']}' as doc_id={info['doc_id']} "
          f"({info['chunks_added']} chunks). Knowledge base now has {len(kb)} chunks total.")


def cmd_remove(args, kb: KnowledgeBase):
    removed = kb.remove_document(args.doc_id)
    print(f"Removed {removed} chunks for doc_id={args.doc_id}.")


def cmd_list(args, kb: KnowledgeBase):
    docs = kb.list_documents()
    if not docs:
        print("Knowledge base is empty. Use 'add <filepath>' to add a document.")
        return
    print(f"{'doc_id':14} {'chunks':8} source")
    for d in docs:
        print(f"{d['doc_id']:14} {d['chunk_count']:<8} {d['source_filename']}")


def cmd_chat(args, kb: KnowledgeBase):
    if len(kb) == 0:
        print("Warning: knowledge base is empty. Add documents first with 'add <filepath>'.\n")

    print(f"Loading model from {args.model} ... (this can take a bit on first load)")
    llm = LocalLLM(LLMConfig(model_path=args.model))
    memory = ConversationMemory()
    cite = args.cite

    print("Ready. Type your questions, 'cite on'/'cite off' to toggle citations, "
          "or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "cite on":
            cite = True
            print("(citations on)")
            continue
        if user_input.lower() == "cite off":
            cite = False
            print("(citations off)")
            continue

        memory.add_turn("user", user_input)
        response = run_agent(
            user_input, llm, kb.store,
            conversation_context=memory.as_context_string(),
            cite_sources=cite,
        )

        if args.show_reasoning:
            for i, step in enumerate(response.steps, 1):
                print(f"  [step {i}] Thought: {step.thought}")
                if step.action_query:
                    print(f"  [step {i}] Searched: \"{step.action_query}\"")

        if response.low_confidence:
            print("Agent (low confidence — docs may not fully cover this):")
        else:
            print("Agent:")
        print(response.answer)
        print()

        memory.add_turn("assistant", response.answer)
        memory.summarize_older_turns(llm)


def main():
    parser = argparse.ArgumentParser(description="Local document-learning agent")
    parser.add_argument("--index", default="data/knowledge_base/index",
                         help="Path to the knowledge base index")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Add a document to the knowledge base")
    p_add.add_argument("filepath")

    p_remove = sub.add_parser("remove", help="Remove a document by doc_id")
    p_remove.add_argument("doc_id")

    sub.add_parser("list", help="List documents in the knowledge base")

    p_chat = sub.add_parser("chat", help="Start an interactive chat session")
    p_chat.add_argument("--model", default=DEFAULT_MODEL_PATH,
                         help="Path to a local GGUF model file")
    p_chat.add_argument("--cite", action="store_true",
                         help="Show source citations in answers")
    p_chat.add_argument("--show-reasoning", action="store_true",
                         help="Print the agent's Thought/Search steps")

    args = parser.parse_args()
    kb = KnowledgeBase(index_path=args.index)

    commands = {"add": cmd_add, "remove": cmd_remove, "list": cmd_list, "chat": cmd_chat}
    commands[args.command](args, kb)


if __name__ == "__main__":
    main()
