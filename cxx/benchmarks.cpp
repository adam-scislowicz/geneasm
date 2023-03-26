#include <benchmark/benchmark.h>

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <fn.hpp> //! rangeless

#include "graph.h"

// Define another benchmark
static void BM_StringCopy(benchmark::State &state) {
  std::string x = "hello";
  for (auto _ : state)
    std::string copy(x);
}

BENCHMARK(BM_StringCopy);

// BENCHMARKS

// Define another benchmark
static void BM_GraphAndCursorConstructionAndConsumeN(benchmark::State &state) {
  GraphDescriptor *graph_desc;
  GraphNode *n1, *n2;
  GraphCursor *gcur;

  for (auto _ : state) {
    GraphNodeDescriptor v1, v2;
    GraphEdgeDescriptor ed1;
    GraphEdge *e1;
    bool success;

    graph_desc = new GraphDescriptor();

    v1 = graph_desc->AddNode(&n1);
    v2 = graph_desc->AddNode(&n2);

    success = graph_desc->AddEdge(v1, v2, SEQ("ACGT"), kGeneSeqTypeDNA, &ed1, &e1);

    gcur = new GraphCursor(graph_desc);
    gcur->set_node(v1);

    vector<char> *cvec = gcur->ConsumeNextNNtds(/*direction_is_up*/ false, 4);

    delete cvec;
    delete e1->seq_slices[0].seqp;
    graph_desc->RemoveEdge(ed1, e1);
    delete gcur;
    graph_desc->RemoveNode(v1);
    graph_desc->RemoveNode(v2);
    delete graph_desc;
  }
}

char kBIG_IN_SEQ[16384];

static void BM_KMerAsRangelessSlidingWindowOfGeneSeqData(benchmark::State &state) {
  using rangeless::fn::operators::operator%;

  GeneSeq gseq(kBIG_IN_SEQ, 16384, kGeneSeqTypeRNA);
  // int64_t a, c, g, u;
  int32_t l;

  // a=c=g=u=0;

  for (auto _ : state) {
    l = 0;

    gseq.data % rangeless::fn::sliding_window(/*win_size*/ 4) %
        rangeless::fn::for_each([&](const auto &group) {
          l += distance(group.begin(), group.end());
#if 0
            for(const auto &kmer : group) {
                switch(kmer) {
                    case 'A':
                        a++; break;
                    case 'C':
                        c++; break;
                    case 'G':
                        g++; break;
                    case 'U':
                        u++; break;
                }
            }
#endif
        });
  }

  assert(l == 65524);
}

static void BM_KMerAsRangeV3SlidingViewOfGeneSeqData(benchmark::State &state) {
  GeneSeq gseq(kBIG_IN_SEQ, 16384, kGeneSeqTypeRNA);
  int32_t l;

  for (auto _ : state) {
    auto kmers = ::ranges::views::sliding(gseq.data, /*win_size*/ 4);
    l = 0;

    for (auto it = kmers.begin(); it != kmers.end(); it++) {
      l += distance((*it).begin(), (*it).end());
    }
  }

  assert(l == 65524);
}

BENCHMARK(BM_GraphAndCursorConstructionAndConsumeN);
BENCHMARK(BM_KMerAsRangelessSlidingWindowOfGeneSeqData);
BENCHMARK(BM_KMerAsRangeV3SlidingViewOfGeneSeqData);

BENCHMARK_MAIN();