// vim: ts=4:sw=4:et
#include "graph.h"
#include "logger.h"

#include <boost/graph/graphviz.hpp>

string GraphEdge::ShortNameString() {
  char buf[32];

  if (this->IsEpsilon()) {
    snprintf(buf, sizeof(buf), "E%d &epsilon;", this->id);
  } else {
    snprintf(buf, sizeof(buf), "E%d[%zu]", this->id, this->GetDataLen());
  }

  return string(buf);
}

vector<char> *GraphEdge::GetData() {
  assert(seq_slices.size() > 0);

  return seq_slices[0].GetData();
}

char GraphEdge::GetData(int32_t off) {
  assert(seq_slices.size() > 0);

  return seq_slices[0].GetData(off);
}

size_t GraphEdge::GetDataLen() {
  assert(seq_slices.size() > 0);

  return seq_slices[0].len;
}

GraphPath::GraphPath(GraphDescriptor *graph_desc_in, vector<GraphEdgeDescriptor> *path_in) {
  get_next_id_();

  graph_desc = graph_desc_in;
  path = path_in;
}

void GraphPath::AsCharArray(char **str_out, int *strlen_out) {
  // use cached value if available
  if (this->str) {
    *str_out = this->str;
    *strlen_out = this->strlen;
    return;
  }

  vector<char> path_str;

  for (GraphEdgeDescriptor &edge_desc : *(this->path)) {
    GraphEdge *edge = this->graph_desc->graph[edge_desc];

    vector<char> *edge_data = edge->GetData();

    path_str.insert(path_str.end(), edge_data->begin(), edge_data->end());
  }

  this->strlen = path_str.size();
  this->str = new char[this->strlen + 1];
  this->str[this->strlen] = '\0';
  memcpy(this->str, path_str.data(), this->strlen);
  *str_out = this->str;
  *strlen_out = this->strlen;
}

GraphPath::~GraphPath() {
  delete this->path;

  if (this->str) {
    delete this->str;
  }
}

GraphNodeDescriptor GraphDescriptor::AddNode(GraphNode **node_out) {
  GraphNode *new_node = new GraphNode();
  GraphNodeDescriptor node_desc;

  *node_out = new_node;
  node_desc = boost::add_vertex(new_node, this->graph);
  return node_desc;
}

void GraphDescriptor::RemoveNode(GraphNodeDescriptor node_descr) {
  delete graph[node_descr];
  boost::remove_vertex(node_descr, this->graph);
}

bool GraphDescriptor::AddEdge(GraphNodeDescriptor src_node, GraphNodeDescriptor dst_node,
                              GraphEdgeDescriptor *edge_desc_out, GraphEdge **edge_out) {
  bool success;
  GraphEdge *new_edge = new GraphEdge();

  tie(*edge_desc_out, success) = boost::add_edge(src_node, dst_node, this->graph);
  graph[*edge_desc_out] = new_edge;
  *edge_out = graph[*edge_desc_out];

  return success;
}

bool GraphDescriptor::AddEdge(GraphNodeDescriptor src_node, GraphNodeDescriptor dst_node, char *seq,
                              GeneSeqType seq_type, GraphEdgeDescriptor *edge_desc_out,
                              GraphEdge **edge_out) {
  bool success;

  success = this->AddEdge(src_node, dst_node, edge_desc_out, edge_out);

  if (seq == kGENE_SEQ_EPSILON) {
    return success;
  }

  GeneSeq *seqp = new GeneSeq(seq, strlen(seq), seq_type);
  (*edge_out)->seq_slices.push_back(GeneSeqSlice(seqp));

  return success;
}

void GraphDescriptor::RemoveEdge(GraphEdgeDescriptor edge_desc, GraphEdge *graph_edge) {
  boost::remove_edge(edge_desc, this->graph);
  delete graph_edge;
}

vector<GraphPath *> *GraphDescriptor::GetPathsBetweenNodes(GraphNodeDescriptor src_node,
                                                           GraphNodeDescriptor dst_node,
                                                           bool already_locked) {
  vector<GraphPath *> *paths = new vector<GraphPath *>;

  queue<vector<GraphEdgeDescriptor> *> pips; // paths in progress
  Graph::out_edge_iterator oe, oend;

  if (!already_locked) {
    graph_mutex.lock();
  }

  for (tie(oe, oend) = out_edges(src_node, this->graph); oe != oend; ++oe) {
    vector<GraphEdgeDescriptor> *cur_path = new vector<GraphEdgeDescriptor>;
    cur_path->push_back((GraphEdgeDescriptor)*oe);
    pips.push(cur_path);
  }

  while (!pips.empty()) {
    vector<GraphEdgeDescriptor> *cur_path;
    GraphNodeDescriptor tail_edge_dest_node;

    cur_path = pips.front();

    tail_edge_dest_node = target(cur_path->back(), this->graph);

    if (tail_edge_dest_node == dst_node) {
      pips.pop();

      GraphEdge *gep = this->graph[cur_path->back()];
      if (gep->IsEpsilon()) {
        cur_path->pop_back();
      }

      if (cur_path->size() == 0) {
        free(cur_path);
        continue;
      }

      paths->push_back(new GraphPath(this, cur_path));
      continue;
    }

    bool firstOutgoing = true;
    for (tie(oe, oend) = out_edges(tail_edge_dest_node, this->graph); oe != oend; ++oe) {
      if (firstOutgoing) {

        // remove epsilons as we go for memory efficiency.
        GraphEdge *gep = this->graph[cur_path->back()];
        if (gep->IsEpsilon()) {
          cur_path->pop_back();
        }

        cur_path->push_back((GraphEdgeDescriptor)*oe);
        firstOutgoing = false;
        continue;
      }

      vector<GraphEdgeDescriptor> *alt_path = new vector<GraphEdgeDescriptor>;
      *alt_path = *cur_path;
      alt_path->pop_back();
      alt_path->push_back((GraphEdgeDescriptor)*oe);
      pips.push(alt_path);
    }

    if (firstOutgoing == true) {
      delete cur_path;
      pips.pop();
    }
  }

  if (!already_locked) {
    graph_mutex.unlock();
  }

  return paths;
}

void GraphDescriptor::CalculatePaths(bool already_locked) {
  if (!already_locked) {
    graph_mutex.lock();
  }

  this->paths = this->GetPathsBetweenNodes(this->start_node, this->end_node, true);

  if (!already_locked) {
    graph_mutex.unlock();
  }
}

GraphNode *GraphDescriptor::get_node_ref(GraphNodeDescriptor node_desc) {
  return this->graph[node_desc];
}

GraphEdge *GraphDescriptor::get_edge_ref(GraphEdgeDescriptor edge_desc) {
  return this->graph[edge_desc];
}

GraphDescriptor::~GraphDescriptor() {
  if (this->paths != NULL) {
    delete this->paths;
    this->paths = NULL;
  }
}

GraphCursor::GraphCursor(GraphDescriptor *in_graph_desc_p) {
  this->get_next_id_();

  this->graph_desc_p = in_graph_desc_p;
  this->node_desc = NULL;
  this->edge_desc_null = true; /* work around BGL lack of null_edge */
  this->edge_offset = 0;
}

bool GraphCursor::operator==(GraphCursor &o) {
  if ((this->graph_desc_p == o.graph_desc_p) && (this->node_desc == o.node_desc)) {

    if (this->edge_desc_null && o.edge_desc_null) {
      return true;
    }

    if ((this->edge_desc == o.edge_desc) && (this->edge_offset == o.edge_offset)) {
      return true;
    }
  }

  return false;
}

ostream &operator<<(ostream &os, const GraphCursor &gcur) {
  os << "GraphCursor{id=" << gcur.id;

  if (gcur.graph_desc_p != NULL) {
    os << ", graph_id=" << gcur.graph_desc_p->id;

    if (gcur.node_desc != NULL) {
      GraphNode *n = gcur.get_cur_node();
      os << ", node_id=" << n->id;
    }

    if (gcur.edge_desc_null == false) {
      GraphEdge *e = gcur.graph_desc_p->get_edge_ref(gcur.edge_desc);
      os << ", edge_id=" << e->id << ", edge_offset=" << gcur.edge_offset;
    }
  }
  os << "}";

  return os;
}

int GraphCursor::GraphvizRepr(ostream &os, size_t max_node_dist) {
  Graph *graph_p = &this->graph_desc_p->graph;
  GraphEdge *start_edge = NULL;
  set<int> max_dist_node_set;

  assert(max_node_dist > 0);

  os << "digraph G {\n";

  if (this->get_cur_node() == NULL) {
    os << "}\n";
    return 0; // null start_node condition
  }

  // output the start_node and start_edge first and highlight.
  const GraphNode *start_node = this->graph_desc_p->graph[this->node_desc];
  os << start_node->id << " [label=N" << start_node->id
     << ", style=filled, fillcolor=dodgerblue];\n";

  if (this->edge_desc_null == false) {
    start_edge = this->graph_desc_p->graph[this->edge_desc];
    GraphEdge *ge = (*graph_p)[this->edge_desc];
    GraphNodeDescriptor snode = source(this->edge_desc, *graph_p);
    GraphNodeDescriptor tnode = target(this->edge_desc, *graph_p);
    GraphNode *snp = (*graph_p)[snode];
    GraphNode *tnp = (*graph_p)[tnode];
    os << snp->id << " -> " << tnp->id << " [label=\"" << ge->ShortNameString()
       << " edgeoff=" << this->edge_offset << "\", penwidth=3, color=dodgerblue];\n";
  }

  unordered_set<GraphNodeDescriptor> pending_nodes_uset;
  unordered_set<GraphNodeDescriptor> next_pending_nodes_uset;

  // XXXADS TODO, reduce code duplication in this function.

  pending_nodes_uset.insert(this->node_desc);
  for (int cur_dist = 0; cur_dist <= max_node_dist; cur_dist++) {
    bool no_pending_nodes = true;
    set<int> pending_nodes_set;

    while (!pending_nodes_uset.empty()) {
      GraphNodeDescriptor tnode = *pending_nodes_uset.begin();
      pending_nodes_uset.erase(tnode);
      GraphNode *gn = (*graph_p)[tnode];

      if (start_node->id != gn->id) {
        os << gn->id << " [label=N" << gn->id << "];\n";
      }

      Graph::in_edge_iterator ie, iend;
      for (tie(ie, iend) = in_edges(tnode, *graph_p); ie != iend; ++ie) {
        GraphEdge *ge = (*graph_p)[*ie];
        GraphNodeDescriptor snode = source(*ie, *graph_p);
        GraphNode *snp = (*graph_p)[snode];
        GraphNode *tnp = (*graph_p)[tnode];

        if (!start_edge || (start_edge && (start_edge->id != ge->id))) {
          os << snp->id << " -> " << tnp->id << " [label=\"" << ge->ShortNameString() << "\"];\n";
        }

        if (cur_dist < max_node_dist) {
          next_pending_nodes_uset.insert(snode);
          no_pending_nodes = false;
        } else {
          auto it = max_dist_node_set.find(snp->id);
          if (it == max_dist_node_set.end()) {
            max_dist_node_set.insert(snp->id);
            os << snp->id << " [label=\"N" << snp->id << "...\"];\n";
          }
        }
      }
    }

    pending_nodes_uset.swap(next_pending_nodes_uset);

    if (no_pending_nodes == true) {
      break;
    }
  }

  pending_nodes_uset.insert(this->node_desc);
  for (int cur_dist = 0; cur_dist <= max_node_dist; cur_dist++) {
    bool no_pending_nodes = true;

    while (!pending_nodes_uset.empty()) {
      GraphNodeDescriptor snode = *pending_nodes_uset.begin();
      pending_nodes_uset.erase(snode);
      GraphNode *gn = (*graph_p)[snode];

      if (start_node->id != gn->id) {
        os << gn->id << " [label=N" << gn->id << "];\n";
      }

      Graph::out_edge_iterator oe, oend;
      for (tie(oe, oend) = out_edges(snode, *graph_p); oe != oend; ++oe) {
        GraphEdge *ge = (*graph_p)[*oe];
        GraphNodeDescriptor tnode = target(*oe, *graph_p);
        GraphNode *snp = (*graph_p)[snode];
        GraphNode *tnp = (*graph_p)[tnode];

        if (!start_edge || (start_edge && (start_edge->id != ge->id))) {
          os << snp->id << " -> " << tnp->id << " [label=\"" << ge->ShortNameString() << "\"];\n";
        }

        if (cur_dist < max_node_dist) {
          next_pending_nodes_uset.insert(tnode);
          no_pending_nodes = false;
        } else {
          auto it = max_dist_node_set.find(tnp->id);
          if (it == max_dist_node_set.end()) {
            max_dist_node_set.insert(tnp->id);
            os << tnp->id << " [label=\"N" << tnp->id << "...\"];\n";
          }
        }
      }
    }

    pending_nodes_uset.swap(next_pending_nodes_uset);

    if (no_pending_nodes == true) {
      break;
    }
  }

  os << "}\n";
  return 0;
}

bool GraphCursor::FindMatchingEdge(char ntd, bool direction_is_up,
                                   GraphNodeDescriptor root_node_desc,
                                   GraphEdgeDescriptor *edge_desc_out, int32_t *edge_off_out) {
  Graph *graph_p = &this->graph_desc_p->graph;
  unordered_set<GraphNodeDescriptor> pending_nodes_uset;
  stack<GraphEdgeDescriptor> pending_edges_stack;

  int32_t edge_off;

  pending_nodes_uset.insert(root_node_desc);

  while (!pending_nodes_uset.empty()) {
    GraphNodeDescriptor node_desc = *pending_nodes_uset.begin();
    pending_nodes_uset.erase(node_desc);

    if (direction_is_up) {
      Graph::in_edge_iterator ie, iend;
      for (tie(ie, iend) = in_edges(node_desc, *graph_p); ie != iend; ++ie) {
        pending_edges_stack.push((GraphEdgeDescriptor)*ie);
      }
    } else {
      Graph::out_edge_iterator oe, oend;
      for (tie(oe, oend) = out_edges(node_desc, *graph_p); oe != oend; ++oe) {
        pending_edges_stack.push((GraphEdgeDescriptor)*oe);
      }
    }

    while (!pending_edges_stack.empty()) {
      GraphEdgeDescriptor edge_desc = pending_edges_stack.top();
      GraphEdge *edge = (*graph_p)[edge_desc];
      pending_edges_stack.pop();

      if (edge->IsEpsilon()) {
        // if EPSILON, queue next node (directionally) for processing.

        if (direction_is_up) {
          pending_nodes_uset.insert((GraphNodeDescriptor)source(edge_desc, *graph_p));
        } else {
          pending_nodes_uset.insert((GraphNodeDescriptor)target(edge_desc, *graph_p));
        }

        continue;
      }

      if (direction_is_up) {
        edge_off = edge->GetDataLen() - 1;
      } else {
        edge_off = 0;
      }

      if (edge->GetData(edge_off) == ntd) {
        *edge_desc_out = edge_desc;
        *edge_off_out = edge_off;
        return true;
      }

    } // end of for each edge in the pending_edges_stack
  }   // end of for each node in the pending_nodes_uset

  return false;
}

int GraphCursor::MatchLengthDirectional(GraphCursor &o, bool direction_is_up) {
  int posdelta;
  int slim, olim;       // self and other limit
  int32_t *soff, *ooff; // self and other offfset

  int matchlen;

  GraphEdge *this_edge_p = get_cur_edge(direction_is_up);
  GraphEdge *o_edge_p = o.get_cur_edge(direction_is_up);

  assert(o_edge_p != NULL);
  assert(this_edge_p != NULL);

  if (direction_is_up == true) {
    posdelta = -1;
    slim = 0;
    olim = 0;
  } else {
    // direction is down
    posdelta = 1;
    slim = this_edge_p->GetDataLen();
    olim = o_edge_p->GetDataLen();
  }

  soff = &this->edge_offset;
  ooff = &o.edge_offset;

  matchlen = 0;

  while (true) {
    if (direction_is_up == true) {
      if ((*soff < 0) || (*ooff < 0))
        return matchlen;
    } else {
      // direction is down
      if ((*soff >= slim) || (*ooff >= olim))
        return matchlen;
    }

    if (this_edge_p->GetData(*soff) != o_edge_p->GetData(*ooff))
      return matchlen;

    matchlen += 1;
    *soff += posdelta;
    *ooff += posdelta;
  }

  spdlog::error("INTERNAL ERROR: MatchLengthDirectional.");
}

GraphNode *GraphCursor::get_cur_node(void) const {
  assert(this->graph_desc_p != NULL);

  if (this->node_desc == NULL)
    return NULL;

  return this->graph_desc_p->graph[this->node_desc];
}

GraphEdge *GraphCursor::get_cur_edge(bool direction_is_up) {
  assert(this->graph_desc_p != NULL);

  if (this->edge_desc_null == true) {
    this->UpdateEdge(direction_is_up);
    if (this->edge_desc_null == true) {
      return NULL;
    }
  }

  return this->graph_desc_p->graph[this->edge_desc];
}

GraphNodeDescriptor GraphCursor::get_src_node(void) {
  assert(this->graph_desc_p != NULL);

  if (this->edge_desc_null == true)
    return NULL;

  return source(this->edge_desc, this->graph_desc_p->graph);
}

GraphNodeDescriptor GraphCursor::get_dst_node(void) {
  assert(this->graph_desc_p != NULL);

  if (this->edge_desc_null == true)
    return NULL;

  return target(this->edge_desc, this->graph_desc_p->graph);
}

void GraphCursor::set_node(GraphNodeDescriptor node) {
  this->node_desc = node;

  // reset edge status
  this->edge_desc_null = true; /* work around BGL lack of null_edge */
  this->edge_offset = 0;
}

void GraphCursor::UpdateEdge(bool direction_is_up) {
  Graph *graph_p = &this->graph_desc_p->graph;

  this->edge_offset = 0;

  if (direction_is_up == true) {
    Graph::in_edge_iterator ie, iend;
    for (tie(ie, iend) = in_edges(this->node_desc, *graph_p); ie != iend; ++ie) {
      this->edge_desc = *ie;
      GraphEdge *ge = (*graph_p)[*ie];
      this->edge_offset = ge->GetDataLen() - 1;
      this->edge_desc_null = false;
      break;
    }
  } else {
    /* direction is down */
    Graph::out_edge_iterator oe, oend;

    for (tie(oe, oend) = out_edges(this->node_desc, *graph_p); oe != oend; ++oe) {
      this->edge_desc = *oe;
      this->edge_desc_null = false;
      break;
    }
  }
}

vector<char> *GraphCursor::ConsumeNextNNtds(bool direction_is_up, int n_ntds) {
  GraphEdge *cur_edge_p = get_cur_edge(direction_is_up);

  assert(cur_edge_p != NULL);

  if (direction_is_up == true) {
    assert(n_ntds <= (this->edge_offset + 1));
    this->edge_offset -= n_ntds;

    vector<char> *subseq =
        slice(*cur_edge_p->GetData(), this->edge_offset + 1, this->edge_offset + n_ntds);
    reverse(subseq->begin(), subseq->end());

    if (this->edge_offset == -1)
      this->set_node(get_src_node());

    return subseq;
  }

  // direction is down
  assert(n_ntds <= (cur_edge_p->GetDataLen() - this->edge_offset));
  this->edge_offset += n_ntds;

  vector<char> *subseq =
      slice(*cur_edge_p->GetData(), this->edge_offset - n_ntds, this->edge_offset - 1);

  if (this->edge_offset == (int32_t)cur_edge_p->GetDataLen())
    this->set_node(get_dst_node());

  return subseq;
}

GraphCursor::~GraphCursor() {}

TEST_CASE("GraphCursor ConsumeNextNNtds 001") {
  GraphDescriptor graph_desc;

  GraphNode *n1, *n2;
  GraphNodeDescriptor v1, v2;
  GraphEdge *e1;
  GraphEdgeDescriptor ed1;
  bool success;

  spdlog::info("GraphCursor ConsumeNextNNtds 001");

  v1 = graph_desc.AddNode(&n1);
  v2 = graph_desc.AddNode(&n2);

  success = graph_desc.AddEdge(v1, v2, SEQ("ACGT"), kGeneSeqTypeDNA, &ed1, &e1);
  // spdlog::info("    e1->seq: {}", *(e1->GetData()));

  GraphCursor gcur(&graph_desc);
  gcur.set_node(v1);

  vector<char> *cvec = gcur.ConsumeNextNNtds(/*direction_is_up*/ false, 4);

  string msg("");
  for (int i = 0; i < cvec->size(); i++) {
    msg.push_back((*cvec)[i]);
  }

  CHECK(msg == "ACGT");

  spdlog::info("    ntds: {}", msg);
}

TEST_CASE("GraphCursor ConsumeNextNNtds 002") {
  GraphDescriptor graph_desc;

  GraphNode *n1, *n2;
  GraphNodeDescriptor v1, v2;
  GraphEdge *e1;
  GraphEdgeDescriptor ed1;
  bool success;

  spdlog::info("GraphCursor ConsumeNextNNtds 002");

  v1 = graph_desc.AddNode(&n1);
  v2 = graph_desc.AddNode(&n2);

  success = graph_desc.AddEdge(v1, v2, SEQ("ACGT"), kGeneSeqTypeDNA, &ed1, &e1);
  // spdlog::info("    e1->seq: {}", *(e1->GetData()));

  GraphCursor gcur(&graph_desc);
  gcur.set_node(v1);

  spdlog::info("    graph cursor: {}", gcur);

  vector<char> *cvec = gcur.ConsumeNextNNtds(/*direction_is_up*/ false, 2);
  string msg("");
  for (int i = 0; i < cvec->size(); i++) {
    msg.push_back((*cvec)[i]);
  }
  CHECK(msg == "AC");

  spdlog::info("    ntds: {}", msg);
  spdlog::info("    graph cursor after first call to ConsumeNextNNtds: {}", gcur);

  vector<char> *cvec2 = gcur.ConsumeNextNNtds(/*direction_is_up*/ false, 2);
  for (int i = 0; i < cvec2->size(); i++) {
    msg.push_back((*cvec2)[i]);
  }
  CHECK(msg == "ACGT");
  spdlog::info("    ntds: {}", msg);
  spdlog::info("    graph cursor after second call to ConsumeNextNNtds: {}", gcur);
}

TEST_CASE("GraphCursor ConsumeNextNNtds 003") {
  GraphDescriptor graph_desc;

  GraphNode *n1, *n2;
  GraphNodeDescriptor v1, v2;
  GraphEdge *e1;
  GraphEdgeDescriptor ed1;
  bool success;

  spdlog::info("GraphCursor ConsumeNextNNtds 003");

  v1 = graph_desc.AddNode(&n1);
  v2 = graph_desc.AddNode(&n2);

  success = graph_desc.AddEdge(v1, v2, SEQ("ACGT"), kGeneSeqTypeDNA, &ed1, &e1);
  // spdlog::info("    e1->seq: {}", *(e1->GetData()));

  GraphCursor gcur(&graph_desc);
  gcur.set_node(v2);

  vector<char> *cvec = gcur.ConsumeNextNNtds(/*direction_is_up*/ true, 4);

  string msg("");
  for (int i = 0; i < cvec->size(); i++) {
    msg.push_back((*cvec)[i]);
  }
  CHECK(msg == "TGCA");

  spdlog::info("    ntds: {}", msg);
}

TEST_CASE("GraphCursor ConsumeNextNNtds 004") {
  GraphDescriptor graph_desc;

  GraphNode *n1, *n2;
  GraphNodeDescriptor v1, v2;
  GraphEdge *e1;
  GraphEdgeDescriptor ed1;
  bool success;

  spdlog::info("GraphCursor ConsumeNextNNtds 004");

  v1 = graph_desc.AddNode(&n1);
  v2 = graph_desc.AddNode(&n2);

  success = graph_desc.AddEdge(v1, v2, SEQ("ACGT"), kGeneSeqTypeDNA, &ed1, &e1);
  // spdlog::info("    e1->seq: {}", *(e1->GetData()));

  GraphCursor gcur(&graph_desc);
  gcur.set_node(v2);

  spdlog::info("    graph cursor: {}", gcur);

  vector<char> *cvec = gcur.ConsumeNextNNtds(/*direction_is_up*/ true, 2);
  string msg("");
  for (int i = 0; i < cvec->size(); i++) {
    msg.push_back((*cvec)[i]);
  }
  CHECK(msg == "TG");

  CHECK((gcur == gcur));

  spdlog::info("    ntds: {}", msg);
  spdlog::info("    graph cursor after first call to ConsumeNextNNtds: {}", gcur);

  vector<char> *cvec2 = gcur.ConsumeNextNNtds(/*direction_is_up*/ true, 2);
  for (int i = 0; i < cvec2->size(); i++) {
    msg.push_back((*cvec2)[i]);
  }
  CHECK(msg == "TGCA");
  spdlog::info("    ntds: {}", msg);
  spdlog::info("    graph cursor after second call to ConsumeNextNNtds: {}", gcur);
}

TEST_CASE("GraphCursor MatchLengthDirectional 001") {
  GraphDescriptor graph_desc1;

  GraphNode *g1n1, *g1n2;
  GraphNodeDescriptor g1v1, g1v2;
  GraphEdge *g1e1;
  GraphEdgeDescriptor g1ed1;

  GraphDescriptor graph_desc2;
  GraphNode *g2n1, *g2n2;
  GraphNodeDescriptor g2v1, g2v2;
  GraphEdge *g2e1;
  GraphEdgeDescriptor g2ed1;

  bool success;

  spdlog::info("GraphCursor MatchLengthDirectional 001");

  // Graph 1 init
  g1v1 = graph_desc1.AddNode(&g1n1);
  g1v2 = graph_desc1.AddNode(&g1n2);

  success = graph_desc1.AddEdge(g1v1, g1v2, SEQ("ACGT"), kGeneSeqTypeDNA, &g1ed1, &g1e1);

  // Graph 2 init
  g2v1 = graph_desc2.AddNode(&g2n1);
  g2v2 = graph_desc2.AddNode(&g2n2);

  success = graph_desc2.AddEdge(g2v1, g2v2, SEQ("ACGT"), kGeneSeqTypeDNA, &g2ed1, &g2e1);

  // Setup 2 graph cursors
  GraphCursor gcur1(&graph_desc1);
  gcur1.set_node(g1v1);
  GraphCursor gcur2(&graph_desc2);
  gcur2.set_node(g2v1);

  CHECK((gcur1 == gcur1));
  CHECK(!(gcur1 == gcur2));

  spdlog::info("    graph cursors: {}, {}", gcur1, gcur2);

  int match_len = gcur1.MatchLengthDirectional(gcur2, /*direction_is_up*/ false);
  CHECK(match_len == 4);

  spdlog::info("    graph cursors after MatchLengthDirectional: {}, {}", gcur1, gcur2);
  spdlog::info("    match_len: {}", match_len);
}

TEST_CASE("GraphCursor MatchLengthDirectional 002") {
  GraphDescriptor graph_desc1;

  GraphNode *g1n1, *g1n2;
  GraphNodeDescriptor g1v1, g1v2;
  GraphEdge *g1e1;
  GraphEdgeDescriptor g1ed1;

  GraphDescriptor graph_desc2;
  GraphNode *g2n1, *g2n2;
  GraphNodeDescriptor g2v1, g2v2;
  GraphEdge *g2e1;
  GraphEdgeDescriptor g2ed1;

  bool success;

  spdlog::info("GraphCursor MatchLengthDirectional 002");

  // Graph 1 init
  g1v1 = graph_desc1.AddNode(&g1n1);
  g1v2 = graph_desc1.AddNode(&g1n2);

  success = graph_desc1.AddEdge(g1v1, g1v2, SEQ("ACGT"), kGeneSeqTypeDNA, &g1ed1, &g1e1);

  // Graph 2 init
  g2v1 = graph_desc2.AddNode(&g2n1);
  g2v2 = graph_desc2.AddNode(&g2n2);

  success = graph_desc2.AddEdge(g2v1, g2v2, SEQ("ACGT"), kGeneSeqTypeDNA, &g2ed1, &g2e1);

  // Setup 2 graph cursors
  GraphCursor gcur1(&graph_desc1);
  gcur1.set_node(g1v2);
  GraphCursor gcur2(&graph_desc2);
  gcur2.set_node(g2v2);

  spdlog::info("    graph cursors: {}, {}", gcur1, gcur2);

  int match_len = gcur1.MatchLengthDirectional(gcur2, /*direction_is_up*/ true);
  CHECK(match_len == 4);

  spdlog::info("    graph cursors after MatchLengthDirectional: {}, {}", gcur1, gcur2);
  spdlog::info("    match_len: {}", match_len);
}

TEST_CASE("GraphCursor MatchLengthDirectional 003") {
  GraphDescriptor graph_desc1;

  GraphNode *g1n1, *g1n2;
  GraphNodeDescriptor g1v1, g1v2;
  GraphEdge *g1e1;
  GraphEdgeDescriptor g1ed1;

  GraphDescriptor graph_desc2;
  GraphNode *g2n1, *g2n2;
  GraphNodeDescriptor g2v1, g2v2;
  GraphEdge *g2e1;
  GraphEdgeDescriptor g2ed1;

  bool success;

  spdlog::info("GraphCursor MatchLengthDirectional 003");

  // Graph 1 init
  g1v1 = graph_desc1.AddNode(&g1n1);
  g1v2 = graph_desc1.AddNode(&g1n2);

  success = graph_desc1.AddEdge(g1v1, g1v2, SEQ("AGCT"), kGeneSeqTypeDNA, &g1ed1, &g1e1);

  // Graph 2 init
  g2v1 = graph_desc2.AddNode(&g2n1);
  g2v2 = graph_desc2.AddNode(&g2n2);

  success = graph_desc2.AddEdge(g2v1, g2v2, SEQ("ACGT"), kGeneSeqTypeDNA, &g2ed1, &g2e1);

  // Setup 2 graph cursors
  GraphCursor gcur1(&graph_desc1);
  gcur1.set_node(g1v2);
  GraphCursor gcur2(&graph_desc2);
  gcur2.set_node(g2v2);

  spdlog::info("    graph cursors: {}, {}", gcur1, gcur2);

  int match_len = gcur1.MatchLengthDirectional(gcur2, /*direction_is_up*/ true);
  CHECK(match_len == 1);

  spdlog::info("    graph cursors after MatchLengthDirectional: {}, {}", gcur1, gcur2);
  spdlog::info("    match_len: {}", match_len);
}

TEST_CASE("GraphCursor MatchLengthDirectional 004") {
  GraphDescriptor graph_desc1;

  GraphNode *g1n1, *g1n2;
  GraphNodeDescriptor g1v1, g1v2;
  GraphEdge *g1e1;
  GraphEdgeDescriptor g1ed1;

  GraphDescriptor graph_desc2;
  GraphNode *g2n1, *g2n2;
  GraphNodeDescriptor g2v1, g2v2;
  GraphEdge *g2e1;
  GraphEdgeDescriptor g2ed1;

  bool success;

  spdlog::info("GraphCursor MatchLengthDirectional 004");

  // Graph 1 init
  g1v1 = graph_desc1.AddNode(&g1n1);
  g1v2 = graph_desc1.AddNode(&g1n2);

  success = graph_desc1.AddEdge(g1v1, g1v2, SEQ("ACGC"), kGeneSeqTypeDNA, &g1ed1, &g1e1);

  // Graph 2 init
  g2v1 = graph_desc2.AddNode(&g2n1);
  g2v2 = graph_desc2.AddNode(&g2n2);

  success = graph_desc2.AddEdge(g2v1, g2v2, SEQ("ACGT"), kGeneSeqTypeDNA, &g2ed1, &g2e1);

  // Setup 2 graph cursors
  GraphCursor gcur1(&graph_desc1);
  gcur1.set_node(g1v1);
  GraphCursor gcur2(&graph_desc2);
  gcur2.set_node(g2v1);

  spdlog::info("    graph cursors: {}, {}", gcur1, gcur2);

  int match_len = gcur1.MatchLengthDirectional(gcur2, /*direction_is_up*/ false);
  CHECK(match_len == 3);

  spdlog::info("    graph cursors after MatchLengthDirectional: {}, {}", gcur1, gcur2);
  spdlog::info("    match_len: {}", match_len);
}

TEST_CASE("GraphCursor MatchLengthDirectional 005") {
  GraphDescriptor graph_desc1;

  GraphNode *g1n1, *g1n2;
  GraphNodeDescriptor g1v1, g1v2;
  GraphEdge *g1e1;
  GraphEdgeDescriptor g1ed1;

  GraphDescriptor graph_desc2;
  GraphNode *g2n1, *g2n2;
  GraphNodeDescriptor g2v1, g2v2;
  GraphEdge *g2e1;
  GraphEdgeDescriptor g2ed1;

  bool success;

  spdlog::info("GraphCursor MatchLengthDirectional 005");

  // Graph 1 init
  g1v1 = graph_desc1.AddNode(&g1n1);
  g1v2 = graph_desc1.AddNode(&g1n2);

  success = graph_desc1.AddEdge(g1v1, g1v2, SEQ("ACGT"), kGeneSeqTypeDNA, &g1ed1, &g1e1);

  // Graph 2 init
  g2v1 = graph_desc2.AddNode(&g2n1);
  g2v2 = graph_desc2.AddNode(&g2n2);

  success = graph_desc2.AddEdge(g2v1, g2v2, SEQ("CCGT"), kGeneSeqTypeDNA, &g2ed1, &g2e1);

  // Setup 2 graph cursors
  GraphCursor gcur1(&graph_desc1);
  gcur1.set_node(g1v1);
  GraphCursor gcur2(&graph_desc2);
  gcur2.set_node(g2v1);

  spdlog::info("    graph cursors: {}, {}", gcur1, gcur2);

  int match_len = gcur1.MatchLengthDirectional(gcur2, /*direction_is_up*/ false);
  CHECK(match_len == 0);

  spdlog::info("    graph cursors after MatchLengthDirectional: {}, {}", gcur1, gcur2);
  spdlog::info("    match_len: {}", match_len);
}

TEST_CASE("GraphCursor MatchLengthDirectional 006") {
  GraphDescriptor graph_desc1;

  GraphNode *g1n1, *g1n2;
  GraphNodeDescriptor g1v1, g1v2;
  GraphEdge *g1e1;
  GraphEdgeDescriptor g1ed1;

  GraphDescriptor graph_desc2;
  GraphNode *g2n1, *g2n2;
  GraphNodeDescriptor g2v1, g2v2;
  GraphEdge *g2e1;
  GraphEdgeDescriptor g2ed1;

  bool success;

  spdlog::info("GraphCursor MatchLengthDirectional 006");

  // Graph 1 init
  g1v1 = graph_desc1.AddNode(&g1n1);
  g1v2 = graph_desc1.AddNode(&g1n2);

  success = graph_desc1.AddEdge(g1v1, g1v2, SEQ("ACGA"), kGeneSeqTypeDNA, &g1ed1, &g1e1);

  // Graph 2 init
  g2v1 = graph_desc2.AddNode(&g2n1);
  g2v2 = graph_desc2.AddNode(&g2n2);

  success = graph_desc2.AddEdge(g2v1, g2v2, SEQ("ACGT"), kGeneSeqTypeDNA, &g2ed1, &g2e1);

  // Setup 2 graph cursors
  GraphCursor gcur1(&graph_desc1);
  gcur1.set_node(g1v2);
  GraphCursor gcur2(&graph_desc2);
  gcur2.set_node(g2v2);

  spdlog::info("    graph cursors: {}, {}", gcur1, gcur2);

  int match_len = gcur1.MatchLengthDirectional(gcur2, /*direction_is_up*/ true);
  CHECK(match_len == 0);

  spdlog::info("    graph cursors after MatchLengthDirectional: {}, {}", gcur1, gcur2);
  spdlog::info("    match_len: {}", match_len);
}

TEST_CASE("GraphCursor GraphvizRepr 001") {
  GraphDescriptor graph_desc;

  GraphNode *n1, *n2;
  GraphNodeDescriptor v1, v2;
  GraphEdge *e1;
  GraphEdgeDescriptor ed1;
  bool success;

  spdlog::info("GraphCursor GraphvizRepr 001");

  v1 = graph_desc.AddNode(&n1);
  v2 = graph_desc.AddNode(&n2);

  success = graph_desc.AddEdge(v1, v2, SEQ("ACGT"), kGeneSeqTypeDNA, &ed1, &e1);

  GraphCursor gcur(&graph_desc);
  gcur.set_node(v1);

  gcur.GraphvizRepr(cout, 2);

  vector<char> *cvec = gcur.ConsumeNextNNtds(/*direction_is_up*/ false, 2);

  gcur.GraphvizRepr(cout, 2);

  string msg("");
  for (int i = 0; i < cvec->size(); i++) {
    msg.push_back((*cvec)[i]);
  }

  spdlog::info("    ntds: {}", msg);
}

TEST_CASE("GraphCursor GraphvizRepr 002") {
  GraphDescriptor graph_desc;

  GraphNode *n1, *n2;
  GraphNodeDescriptor v1, v2;
  GraphEdge *e1;
  GraphEdgeDescriptor ed1;
  bool success;

  spdlog::info("GraphCursor GraphvizRepr 002");

  v1 = graph_desc.AddNode(&n1);
  v2 = graph_desc.AddNode(&n2);

  success = graph_desc.AddEdge(v1, v2, SEQ("ACGT"), kGeneSeqTypeDNA, &ed1, &e1);

  GraphCursor gcur(&graph_desc);
  gcur.set_node(v2);

  gcur.GraphvizRepr(cout, 2);

  vector<char> *cvec = gcur.ConsumeNextNNtds(/*direction_is_up*/ true, 2);

  gcur.GraphvizRepr(cout, 2);

  string msg("");
  for (int i = 0; i < cvec->size(); i++) {
    msg.push_back((*cvec)[i]);
  }

  spdlog::info("    ntds: {}", msg);
}

TEST_CASE("GraphCursor GraphvizRepr 003") {
  GraphDescriptor graph_desc;

  GraphNode *n1, *n2, *n3, *n4;
  GraphNodeDescriptor v1, v2, v3, v4;
  GraphEdge *e1, *e2, *e3;
  GraphEdgeDescriptor ed1, ed2, ed3;
  bool success;

  spdlog::info("GraphCursor GraphvizRepr 003");

  v1 = graph_desc.AddNode(&n1);
  v2 = graph_desc.AddNode(&n2);
  v3 = graph_desc.AddNode(&n3);
  v4 = graph_desc.AddNode(&n4);

  success = graph_desc.AddEdge(v1, v2, SEQ("ACGT"), kGeneSeqTypeDNA, &ed1, &e1);

  success = graph_desc.AddEdge(v2, v3, SEQ("ACGT"), kGeneSeqTypeDNA, &ed2, &e2);

  success = graph_desc.AddEdge(v3, v4, SEQ("ACGT"), kGeneSeqTypeDNA, &ed3, &e3);

  GraphCursor gcur(&graph_desc);
  gcur.set_node(v1);

  gcur.GraphvizRepr(cout, 2);

  vector<char> *cvec = gcur.ConsumeNextNNtds(/*direction_is_up*/ false, 2);

  gcur.GraphvizRepr(cout, 2);

  string msg("");
  for (int i = 0; i < cvec->size(); i++) {
    msg.push_back((*cvec)[i]);
  }

  spdlog::info("    ntds: {}", msg);
}

TEST_CASE("GraphCursor GraphvizRepr 004") {
  GraphDescriptor graph_desc;

  GraphNode *n1, *n2, *n3, *n4;
  GraphNodeDescriptor v1, v2, v3, v4;
  GraphEdge *e1, *e2, *e3;
  GraphEdgeDescriptor ed1, ed2, ed3;
  bool success;

  spdlog::info("GraphCursor GraphvizRepr 004");

  v1 = graph_desc.AddNode(&n1);
  v2 = graph_desc.AddNode(&n2);
  v3 = graph_desc.AddNode(&n3);
  v4 = graph_desc.AddNode(&n4);

  success = graph_desc.AddEdge(v1, v2, SEQ("ACGT"), kGeneSeqTypeDNA, &ed1, &e1);

  success = graph_desc.AddEdge(v2, v3, SEQ("ACGT"), kGeneSeqTypeDNA, &ed2, &e2);

  success = graph_desc.AddEdge(v3, v4, SEQ("ACGT"), kGeneSeqTypeDNA, &ed3, &e3);

  GraphCursor gcur(&graph_desc);
  gcur.set_node(v4);

  gcur.GraphvizRepr(cout, 2);

  vector<char> *cvec = gcur.ConsumeNextNNtds(/*direction_is_up*/ true, 2);

  gcur.GraphvizRepr(cout, 2);

  string msg("");
  for (int i = 0; i < cvec->size(); i++) {
    msg.push_back((*cvec)[i]);
  }

  spdlog::info("    ntds: {}, cvec->size()={}", msg, cvec->size());
}

TEST_CASE("GraphCursor FindMatchingEdge 001") {
  GraphDescriptor graph_desc;

  GraphNode *n[4];
  GraphNodeDescriptor v[4];
  GraphEdge *e[7];
  GraphEdgeDescriptor ed[7];
  bool success;

  spdlog::info("GraphCursor FindMatchingEdge 001");

  for (int i = 0; i < 4; i++) {
    v[i] = graph_desc.AddNode(&n[i]);
  }

  success = graph_desc.AddEdge(v[0], v[1], SEQ("ACGTT"), kGeneSeqTypeDNA, &ed[0], &e[0]);
  success = graph_desc.AddEdge(v[0], v[1], SEQ("TGCA"), kGeneSeqTypeDNA, &ed[5], &e[5]);
  success = graph_desc.AddEdge(v[0], v[1], kGENE_SEQ_EPSILON, kGeneSeqTypeDNA, &ed[1], &e[1]);

  success = graph_desc.AddEdge(v[1], v[2], SEQ("CGTA"), kGeneSeqTypeDNA, &ed[2], &e[2]);
  success = graph_desc.AddEdge(v[1], v[2], kGENE_SEQ_EPSILON, kGeneSeqTypeDNA, &ed[3], &e[3]);

  success = graph_desc.AddEdge(v[2], v[3], SEQ("GTCA"), kGeneSeqTypeDNA, &ed[4], &e[4]);
  success = graph_desc.AddEdge(v[2], v[3], SEQ("TGCA"), kGeneSeqTypeDNA, &ed[6], &e[6]);

  GraphCursor gcur(&graph_desc);
  gcur.set_node(v[1]);
  gcur.GraphvizRepr(cout, 2);

  bool match;
  GraphEdgeDescriptor match_edge_desc;
  int32_t match_edge_off;
  match = gcur.FindMatchingEdge('A', /*direction_is_up*/ false, v[0], &match_edge_desc,
                                &match_edge_off);
  CHECK(match == true);
  CHECK(match_edge_off == 0);
  match = gcur.FindMatchingEdge('T', /*direction_is_up*/ false, v[0], &match_edge_desc,
                                &match_edge_off);
  CHECK(match == true);
  CHECK(match_edge_off == 0);
  match = gcur.FindMatchingEdge('C', /*direction_is_up*/ false, v[0], &match_edge_desc,
                                &match_edge_off);
  CHECK(match == true);
  CHECK(match_edge_off == 0);
  match = gcur.FindMatchingEdge('G', /*direction_is_up*/ false, v[0], &match_edge_desc,
                                &match_edge_off);
  CHECK(match == true);
  CHECK(match_edge_off == 0);

  match =
      gcur.FindMatchingEdge('G', /*direction_is_up*/ true, v[2], &match_edge_desc, &match_edge_off);
  CHECK(match == false);
  match =
      gcur.FindMatchingEdge('T', /*direction_is_up*/ true, v[2], &match_edge_desc, &match_edge_off);
  CHECK(match == true);
  CHECK(match_edge_off == 4);
  match =
      gcur.FindMatchingEdge('C', /*direction_is_up*/ true, v[2], &match_edge_desc, &match_edge_off);
  CHECK(match == false);
  match =
      gcur.FindMatchingEdge('A', /*direction_is_up*/ true, v[2], &match_edge_desc, &match_edge_off);
  CHECK(match == true);
  CHECK(match_edge_off == 3);

  for (int i = 0; i < 4; i++) {
    graph_desc.RemoveNode(v[i]);
  }
}

TEST_CASE("GraphDescriptor FindPathsBetweenNodes 001") {
  GraphDescriptor graph_desc;

  GraphNode *n[4];
  GraphNodeDescriptor v[4];
  GraphEdge *e[7];
  GraphEdgeDescriptor ed[7];
  int len_dist[4] = {0};
  bool success;

  spdlog::info("GraphDescriptor FindPathsBetweenNodes 001");

  for (int i = 0; i < 4; i++) {
    v[i] = graph_desc.AddNode(&n[i]);
  }

  success = graph_desc.AddEdge(v[0], v[1], SEQ("ACGTT"), kGeneSeqTypeDNA, &ed[0], &e[0]);
  success = graph_desc.AddEdge(v[0], v[1], SEQ("TGCA"), kGeneSeqTypeDNA, &ed[5], &e[5]);
  success = graph_desc.AddEdge(v[0], v[1], kGENE_SEQ_EPSILON, kGeneSeqTypeDNA, &ed[1], &e[1]);

  success = graph_desc.AddEdge(v[1], v[2], SEQ("CGTA"), kGeneSeqTypeDNA, &ed[2], &e[2]);
  success = graph_desc.AddEdge(v[1], v[2], kGENE_SEQ_EPSILON, kGeneSeqTypeDNA, &ed[3], &e[3]);

  success = graph_desc.AddEdge(v[2], v[3], SEQ("GTCA"), kGeneSeqTypeDNA, &ed[4], &e[4]);
  success = graph_desc.AddEdge(v[2], v[3], SEQ("TGCA"), kGeneSeqTypeDNA, &ed[6], &e[6]);

  graph_desc.start_node = v[0];
  graph_desc.end_node = v[3];

  GraphCursor gcur(&graph_desc);
  gcur.set_node(v[0]);
  gcur.GraphvizRepr(cout, 3);

  graph_desc.CalculatePaths(/*already_locked*/ false);

  int path_len;
  int path_num = 0;
  for (GraphPath *path : *(graph_desc.paths)) {
    spdlog::info("    paths[{}]: gdid={}", path_num++, path->graph_desc->id);
    path_len = path->path->size();
    CHECK(path_len <= 3);
    len_dist[path_len]++;
    for (GraphEdgeDescriptor &eref : *(path->path)) {
      GraphEdge *gep = graph_desc.graph[eref];
      spdlog::info("        eid={}", gep->id);
    }
  }

  CHECK(len_dist[0] == 0);
  CHECK(len_dist[1] == 2);
  CHECK(len_dist[2] == 6);
  CHECK(len_dist[3] == 4);

  for (int i = 0; i < 4; i++) {
    graph_desc.RemoveNode(v[i]);
  }
}