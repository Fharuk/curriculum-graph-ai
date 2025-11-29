import json
from typing import List, Dict, Set, Optional, Any

class CurriculumManager:
    """
    Manages the state of the Dynamic Curriculum Graph.
    Designed for full serialization to Firestore (P1).
    """
    def __init__(self):
        self.nodes: Dict[str, dict] = {}
        self.adjacency: Dict[str, List[str]] = {}
        self.reverse_adjacency: Dict[str, List[str]] = {}
        self.completed_nodes: Set[str] = set()
        self.topic: Optional[str] = None
        self.context: Optional[str] = None

    def serialize(self) -> Dict[str, Any]:
        """Converts the object state into a JSON-friendly dictionary for Firestore."""
        return {
            'nodes': self.nodes,
            'adjacency': self.adjacency,
            'reverse_adjacency': self.reverse_adjacency,
            'completed_nodes': list(self.completed_nodes),
            'topic': self.topic,
            'context': self.context
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]):
        """Creates a CurriculumManager instance from Firestore data."""
        manager = cls()
        manager.nodes = data.get('nodes', {})
        manager.adjacency = data.get('adjacency', {})
        manager.reverse_adjacency = data.get('reverse_adjacency', {})
        manager.completed_nodes = set(data.get('completed_nodes', []))
        manager.topic = data.get('topic')
        manager.context = data.get('context')
        manager._update_node_statuses()
        return manager

    def load_from_json(self, graph_data: dict, topic: str, context: str):
        """Initializes the graph from the Architect Agent's output."""
        self.nodes = {}
        self.adjacency = {}
        self.reverse_adjacency = {}
        self.completed_nodes = set()
        self.topic = topic
        self.context = context

        # Load Nodes
        for node in graph_data.get('nodes', []):
            self.nodes[node['id']] = {
                'label': node['label'],
                'status': 'LOCKED' 
            }
            self.adjacency[node['id']] = []
            self.reverse_adjacency[node['id']] = []

        # Load Edges
        for edge in graph_data.get('edges', []):
            src, tgt = edge['source'], edge['target']
            if src in self.nodes and tgt in self.nodes:
                self.adjacency[src].append(tgt)
                self.reverse_adjacency[tgt].append(src)

        self._update_node_statuses()

    def _update_node_statuses(self):
        """Recalculates status for all nodes."""
        for node_id in self.nodes:
            if node_id in self.completed_nodes:
                self.nodes[node_id]['status'] = 'COMPLETED'
                continue

            prereqs = self.reverse_adjacency.get(node_id, [])
            if not prereqs:
                self.nodes[node_id]['status'] = 'AVAILABLE'
            else:
                all_met = all(p in self.completed_nodes for p in prereqs)
                self.nodes[node_id]['status'] = 'AVAILABLE' if all_met else 'LOCKED'

    def mark_completed(self, node_id: str):
        if node_id in self.nodes:
            self.completed_nodes.add(node_id)
            self._update_node_statuses()

    def inject_remedial_node(self, failed_node_id: str, remedial_data: dict) -> bool:
        """Dynamically alters the graph structure by inserting a new dependency."""
        new_id = remedial_data.get('remedial_node_id')
        new_label = remedial_data.get('remedial_node_label')
        
        if not new_id or new_id in self.nodes:
            return False 

        self.nodes[new_id] = {
            'label': new_label,
            'status': 'AVAILABLE' 
        }
        self.adjacency[new_id] = []
        self.reverse_adjacency[new_id] = []

        # New Node -> Failed Node (Dependency)
        self.adjacency[new_id].append(failed_node_id)
        self.reverse_adjacency[failed_node_id].append(new_id)

        if failed_node_id in self.nodes:
            self.nodes[failed_node_id]['status'] = 'LOCKED'

        self._update_node_statuses()
        return True

    def get_dot_graph(self) -> str:
        """Returns Graphviz DOT string for visualization."""
        dot = "digraph G {\n"
        dot += "  rankdir=LR;\n"
        dot += "  node [shape=box, style=filled, fontname=\"Helvetica\"];\n"

        for n_id, data in self.nodes.items():
            color = "white"
            fontcolor = "black"
            if data['status'] == 'COMPLETED':
                color = "#d4edda"
                fontcolor = "#155724"
            elif data['status'] == 'AVAILABLE':
                color = "#cce5ff"
                fontcolor = "#004085"
            elif data['status'] == 'LOCKED':
                color = "#e2e3e5"
                fontcolor = "#383d41"
            
            label = data['label']
            dot += f'  "{n_id}" [label="{label}", fillcolor="{color}", fontcolor="{fontcolor}"];\n'

        for src, targets in self.adjacency.items():
            for tgt in targets:
                dot += f'  "{src}" -> "{tgt}";\n'

        dot += "}"
        return dot