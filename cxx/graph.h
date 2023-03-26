// vim: ts=4:sw=4:et
#ifndef __GENEASM_GRAPH_H
#define __GENEASM_GRAPH_H
#define BOOST_ALLOW_DEPRECATED_HEADERS (1)
#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <cstdint>
#include <mutex>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <doctest/doctest.h>

#include "logger.h"
#include "seqemb.h"

using namespace std;

class GraphNode;
class GraphEdge;
class GraphPath;

typedef boost::adjacency_list<boost::listS, boost::listS, boost::bidirectionalS, GraphNode *,
                              GraphEdge *>
    Graph;
typedef Graph::vertex_descriptor GraphNodeDescriptor;
typedef Graph::edge_descriptor GraphEdgeDescriptor;

typedef int GraphDescriptorID;
typedef int GraphPathID;

class GraphNode {
public:
  int id;
  GraphNode() { this->get_next_id_(); }
  bool operator<(const GraphNode &o) const { return id < o.id; }
  bool operator==(const GraphNode &o) const { return id == o.id; }
  bool operator!=(const GraphNode &o) const { return id != o.id; }

  GraphNode(const GraphNode &node) {
    spdlog::error("XXX GraphNode Copy Constructor Called.\n{}", boost::stacktrace::stacktrace());
  };

private:
  void get_next_id_() {
    static int next_id_ = 1;
    id = next_id_++;
  }
};

class GraphEdge {
public:
  int id;
  GraphEdge() { this->get_next_id_(); }
  bool operator<(const GraphEdge &o) const { return id < o.id; }
  bool operator==(const GraphEdge &o) const { return id == o.id; }
  bool operator!=(const GraphEdge &o) const { return id != o.id; }

  GraphEdge(const GraphEdge &node) {
    spdlog::error("XXX GraphEdge Copy Constructor Called.\n{}", boost::stacktrace::stacktrace());
  };

  vector<char> *GetData();
  char GetData(int32_t off);
  size_t GetDataLen();

  vector<GeneSeqSlice> seq_slices;

  bool IsEpsilon() const {
    if (this->seq_slices.size() == 0) {
      return true;
    }
    return false;
  }
  string ShortNameString();

private:
  void get_next_id_() {
    static int next_id_ = 1;
    id = next_id_++;
  }
};

class GraphDescriptor {
public:
  GraphDescriptorID id;
  GraphDescriptor() { this->get_next_id_(); };
  ~GraphDescriptor();

  GraphDescriptor(const GraphDescriptor &node) {
    spdlog::error("XXX GraphDescriptor Copy Constructor Called.\n{}",
                  boost::stacktrace::stacktrace());
  };

  mutex graph_mutex;
  Graph graph;

  GraphNode *get_node_ref(GraphNodeDescriptor node_desc);
  GraphEdge *get_edge_ref(GraphEdgeDescriptor edge_desc);

  GraphNodeDescriptor AddNode(GraphNode **node_out);
  void RemoveNode(GraphNodeDescriptor node_descr);

  bool AddEdge(GraphNodeDescriptor src_node, GraphNodeDescriptor dst_node,
               GraphEdgeDescriptor *edge_desc_out, GraphEdge **edge_out);
  bool AddEdge(GraphNodeDescriptor src_node, GraphNodeDescriptor dst_node, char *seq,
               GeneSeqType seq_type, GraphEdgeDescriptor *edge_desc_out, GraphEdge **edge_out);
  void RemoveEdge(GraphEdgeDescriptor edge_descr, GraphEdge *graph_edge);

  vector<GraphPath *> *GetPathsBetweenNodes(GraphNodeDescriptor src_node,
                                            GraphNodeDescriptor dst_node, bool already_locked);
  void CalculatePaths(bool already_locked);

  // Priori: There is always exactly one start and one end node in a graph.
  GraphNodeDescriptor start_node;
  GraphNodeDescriptor end_node;

  mutex subgraph_mutex;

  vector<GraphPath *> *paths = NULL;

private:
  void get_next_id_() {
    static int next_id_ = 1;
    id = next_id_++;
  }
};

class GraphPath {
public:
  GraphPathID id;

  GraphPath(GraphDescriptor *graph_desc_in, vector<GraphEdgeDescriptor> *path_in);
  ~GraphPath();

  void AsCharArray(char **str_out, int *strlen_out);

  GraphDescriptor *graph_desc;
  vector<GraphEdgeDescriptor> *path;

private:
  char *str = NULL;
  int strlen = 0;

  void get_next_id_() {
    static GraphPathID next_id_ = 1;
    id = next_id_++;
  }
};

class GraphCursor {
public:
  int id;
  GraphCursor(GraphDescriptor *in_graph_desc_p);
  ~GraphCursor();

  GraphCursor(const GraphCursor &node) {
    spdlog::error("XXX GraphCursor Copy Constructor Called.\n{}", boost::stacktrace::stacktrace());
  };

  bool operator==(GraphCursor &o);
  int MatchLengthDirectional(GraphCursor &o, bool direction_is_up);
  vector<char> *ConsumeNextNNtds(bool direction_is_up, int n_ntds);

  friend ostream &operator<<(ostream &os, const GraphCursor &gcur);
  int GraphvizRepr(ostream &os, size_t maxNodeDist);
  bool FindMatchingEdge(char ntd, bool direction_is_up, GraphNodeDescriptor root_node_desc,
                        GraphEdgeDescriptor *edge_desc_out, int32_t *edge_off_out);

  GraphNode *get_cur_node(void) const;
  GraphEdge *get_cur_edge(bool direction_is_up);
  GraphNodeDescriptor get_src_node(void);
  GraphNodeDescriptor get_dst_node(void);
  void UpdateEdge(bool direction_is_up);

  void set_node(GraphNodeDescriptor node);

private:
  GraphDescriptor *graph_desc_p;
  GraphNodeDescriptor node_desc;
  GraphEdgeDescriptor edge_desc;
  int32_t edge_offset;
  bool edge_desc_null;

  void get_next_id_() {
    static int next_id_ = 1;
    id = next_id_++;
  }
};

#endif