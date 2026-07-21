"""
Regenerates the Mermaid architecture diagram directly from the compiled
LangGraph object, so the diagram in README.md can never silently drift from
the actual graph topology in agent/graph.py.

Run whenever nodes or edges change:

    python scripts/generate_diagram.py

Then paste the printed block into the ```mermaid fence in README.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.memory import MemorySaver

from agent.graph import build_graph

if __name__ == "__main__":
    graph = build_graph(checkpointer=MemorySaver())
    print(graph.get_graph().draw_mermaid())
