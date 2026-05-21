import json
from networkx.readwrite import json_graph
from graphify.export import to_svg
from pathlib import Path

data = json.loads(Path('graphify-out/graph.json').read_text(encoding='utf-8'))
G = json_graph.node_link_graph(data, edges='links')

communities = {}
for n, d in G.nodes(data=True):
    cid = d.get('community')
    if cid is not None:
        communities.setdefault(int(cid), []).append(n)

labels_path = Path('graphify-out/.graphify_labels.json')
if labels_path.exists():
    labels_raw = json.loads(labels_path.read_text(encoding='utf-8'))
    labels = {int(k): v for k, v in labels_raw.items()}
else:
    labels = {
        0: "Image Extraction",
        1: "Question Generation",
        2: "Caption Description LLM",
        3: "LLM Descriptions",
        4: "Indexing Pipeline",
        5: "Backend API",
        6: "Search Engine"
    }

to_svg(G, communities, 'graphify-out/graph.svg', community_labels=labels)
print('graph.svg written successfully to graphify-out/graph.svg')
