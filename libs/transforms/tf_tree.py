
from munch import munchify
from scipy.spatial.transform import Rotation
import numpy as np
import networkx as nx
from .cartesian import combine_tfs, inverse_tf
from matplotlib import pyplot as plt


class Transform(object):

    def __init__(self, parent_frame_id, child_frame_id, timestamp, T, R):
        self.parent_frame_id = parent_frame_id
        self.child_frame_id = child_frame_id
        self.timestamp = timestamp
        self.T = T
        self.R = R

    def inverse(self):
        inverted_tf = inverse_tf(self.tf())
        return Transform(
            self.parent_frame_id,
            self.child_frame_id,
            self.timestamp,
            inverted_tf.T,
            inverted_tf.R)

    def tf(self):
        return munchify({'T': self.T, 'R': self.R})

    @staticmethod
    def from_msg(msg):
        #{'header': {'stamp': {'sec': 1660442782, 'nanosec': 129495556}, 'frame_id': 'base_link_op'},
        # 'child_frame_id': 'back_base_link',
        # 'transform': {
        #     'translation': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        #     'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}}
        #}
        if isinstance(msg, dict):
            stamp = msg['header']['stamp']
            timestamp = stamp['sec'] + stamp['nanosec'] * 1e-9
            parent_frame_id = msg['header']['frame_id']
            child_frame_id = msg['child_frame_id']
            transform = msg['transform']
            t = transform['translation']
            q = transform['rotation']
            T = np.array([t['x'], t['y'], t['z']])
            quat = [q['x'], q['y'], q['z'], q['w']]
        else:
            stamp = msg.header.stamp
            timestamp = stamp.sec + stamp.nanosec * 1e-9
            parent_frame_id = msg.header.frame_id
            child_frame_id = msg.child_frame_id
            transform = msg.transform
            t = transform.translation
            q = transform.rotation
            T = np.array([t.x, t.y, t.z])
            quat = [q.x, q.y, q.z, q.w]
        if np.linalg.norm(quat) < 1e-5 or np.any(np.isnan(quat)):
            # DIAG
            return None
        R = Rotation.from_quat(quat)
        return Transform(parent_frame_id, child_frame_id, timestamp, T, R)

class TfTree(object):

  def __init__(self):
      self.G = nx.DiGraph()

  def add(self, transform):
      self.G.add_edge(transform.parent_frame_id,
                      transform.child_frame_id,
                      transform=transform)
      self.G.add_edge(transform.child_frame_id,
                      transform.parent_frame_id,
                      transform=transform.inverse())

  def query(self, parent_frame, child_frame):
      try:
          path = next(nx.shortest_simple_paths(self.G, parent_frame, child_frame))
      except (nx.exception.NodeNotFound, nx.exception.NetworkXNoPath) as e:
          return None
      
      transforms = []
      max_ts = float('-inf')
      for i in range(len(path) - 1):
        parent = path[i]
        child = path[i + 1]
        data = self.G.get_edge_data(parent, child, default=None)['transform']
        transforms.append(data.tf())
        max_ts = max(max_ts, data.timestamp)

      transform = combine_tfs(*transforms)
      return Transform(parent_frame, child_frame, max_ts, transform.T, transform.R)

  def _filter(self, start_node, name_substring, depth):
      graph = self.G
      
      # BFS to get nodes within the specified depth
      filtered_nodes = {start_node}
      queue = [(start_node, 0)]
      while queue:
          node, current_depth = queue.pop(0)
          if current_depth < depth or depth is None:
              neighbors = list(graph.neighbors(node))
              for neighbor in neighbors:
                  if neighbor not in filtered_nodes:
                      queue.append((neighbor, current_depth + 1))
                      filtered_nodes.add(neighbor)
      
      # Filter nodes by name substring
      if name_substring is not None:
          filtered_nodes = {
              node for node in filtered_nodes
              if name_substring in node  # Check if the substring exists in the node name
          }
      
      # Create a subgraph with filtered nodes
      subgraph = graph.subgraph(filtered_nodes).copy()
      return subgraph

  def draw(self, start_node=None, name_substring=None, depth=None,  node_size=700, node_color='lightblue', font_size=10, font_color='black', arrowsize=20):
      if start_node is not None:
          G = self._filter(start_node, name_substring, depth)
      else:
          G = self.G
      pos = nx.spring_layout(G)  # positions for all nodes
      nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightgreen", linewidths=1.0, font_size=10)
      plt.show()
