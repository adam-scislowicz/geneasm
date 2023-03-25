
// vim: ts=4:sw=4:et
#ifndef __GENE_ASSEMBLER_H
#define __GENE_ASSEMBLER_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tbb/concurrent_unordered_map.h"

#include <pybind11/pybind11.h>

#ifdef RANGES_V3_CORE_HPP
#error Ranges included before seqan3, but it must be included after.
#endif

#include <filesystem>

//#include <boost/core/bit.hpp>

#include <parasail.h>
#include <parasail/matrices/dnafull.h>

#include "graph.h"
#include "seqemb.h"

#include <doctest/doctest.h>

static const int kGeneSeqDuplicate = -1;

class PotentialJoin {
  public:
    PotentialJoin();
    PotentialJoin(GraphDescriptorID gdid_a, GraphDescriptorID gdid_b, float norm_score_in);
    pair<GraphDescriptorID, GraphDescriptorID> align_pair;
    float norm_score;
    bool operator<(const PotentialJoin &o) const { return norm_score < o.norm_score; }
    bool operator==(const PotentialJoin &o) const { return norm_score == o.norm_score; }
    bool operator!=(const PotentialJoin &o) const { return norm_score != o.norm_score; }
};

enum AssemblerJobType {
    kAssemblerJobInvalid,
    kAssemblerJobInferEmbedding,
    kAssemblerJobGetApproxKNN,
    kAssemblerJobScoreCandidates,
    kAssemblerJobScoreSubgraph,
    kAssemblerJobJoinSubgraphs
};

enum FastaParserState {
    kParserStateStart,
    kParserStateComment,
    kParserStateCommentNewLine,
    kParserStateSeqName,
    kParserStateSeqDescription,
    kParserStateSeqDataNewLine,
    kParserStateSeqData
};

class AssemblerJobParams {
  public:
    virtual ~AssemblerJobParams() = default;
};

class AssemblerJobInferEmbeddingParams : public AssemblerJobParams {
  public:
    AssemblerJobInferEmbeddingParams(GeneSeq *gseq_in) { gseq = gseq_in; }
    GeneSeq *gseq;
};

class AssemblerJobGetApproxKNNParams : public AssemblerJobParams {
  public:
    AssemblerJobGetApproxKNNParams(GeneSeq *gseq_in, size_t kmer_off_in) {
        gseq = gseq_in;
        kmer_off = kmer_off_in;
    }
    GeneSeq *gseq;
    size_t kmer_off;
};

class AssemblerJobScoreCandidatesParams : public AssemblerJobParams {
  public:
    AssemblerJobScoreCandidatesParams(SeqKmerID repr_kmer_in) { repr_kmer = repr_kmer_in; }
    SeqKmerID repr_kmer;
};

class AssemblerJobScoreSubgraphParams : public AssemblerJobParams {
  public:
    AssemblerJobScoreSubgraphParams(GraphDescriptorID gdid_in) { gdid = gdid_in; }
    GraphDescriptorID gdid;
};

class AssemblerJobJoinSubgraphsParams : public AssemblerJobParams {
  public:
    AssemblerJobJoinSubgraphsParams(PotentialJoin *potential_join_in) {
        potential_join = *potential_join_in;
    }
    PotentialJoin potential_join;
};

struct AssemblyJob {
    AssemblerJobType job_type;
    AssemblerJobParams *job_params;
};

struct AssemblyJobStatus {};

class AssemblerConfig {
  public:
    AssemblerConfig();
    AssemblerConfig(pybind11::object config_dict);
    int max_ntds_per_edge = 4096;
    int kmer_len = 64;
    int alphabet_len = 16;
    int work_queue_low_watermark = 4;
    int work_queue_high_watermark = 12;
    int knn_top_k = 40;
    string seqemb_onnx_path = "embnet_tf2onnx_v13.onnx";
    float subgraph_join_threshold = 200.0;
};
typedef struct AssemblerConfig AssemblerConfig;

enum AssemblyState {
    kAssemblyInProgress,
    kAssemblyAborted,
    kAssemblyError,
    kAssemblyComplete,
    kAssemblyIdle
};

enum AssemblySubState {
    kAssemblyInitializing,
    kAssemblyComputingEmbeddings,
    kAssemblyScoringCandidates,
    kAssemblyInitialSubgraphJoinScoring,
    kAssemblySubgraphJoining,
    kAssemblyCleaningUp
};

class AssemblyStatus {
  public:
    enum AssemblyState state;
    enum AssemblySubState sub_state;
    int geneseqs_expected;
    int geneseqs_embedded;
    int geneseqs_join_candidates_scored;
    int subgraphs_scored;
    int compute_embeddings_duration_ms;
    int score_candidates_duration_ms;
    int initial_subgraph_join_score_duration_ms;
    int subgraph_joining_duration_ms;
};
typedef AssemblyStatus AssemblyStatus;

struct hash_pair {
    template <class T1, class T2> size_t operator()(const pair<T1, T2> &p) const {
        auto hash1 = hash<T1>{}(p.first);
        auto hash2 = hash<T2>{}(p.second);
        return hash1 ^ hash2;
    }
};

// XXXADS TODO put everything in a namespace or something.
pair<SeqKmerID, SeqKmerID> MakeCanonicalKmerPair(SeqKmerID kmer_a, SeqKmerID kmer_b);
pair<GraphDescriptorID, GraphDescriptorID> MakeCanonicalGraphDescriptorPair(GraphDescriptorID gd_a,
                                                                            GraphDescriptorID gd_b);

class Assembler {
  public:
    int id;
    Assembler();
    Assembler(AssemblerConfig config_in);
    ~Assembler();

    int StartAsync(string fasta_path, GeneSeqType gene_seq_type_in);
    AssemblyStatus GetStatus(void);
    void Abort();

    // used primarily for testing.
    int StartSync(vector<char *> reads, GeneSeqType gene_seq_type);
    int StartSync(vector<string> reads, GeneSeqType gene_seq_type);

    bool ParseFastaFile(
        string filename, bool only_count_sequences, int *num_sequences_out,
        std::function<void(Assembler *assembler, string *name, string *description, string *data)> &onParsedSeqFn);

    friend ostream &operator<<(ostream &os, const Assembler &a);

    AssemblerConfig *config = NULL;

  private:
    void get_next_id_() {
        static int next_id_ = 1;
        id = next_id_++;
    }

    bool DetectedAbort();
    int WaitForAvailableWorker(int tid);
    int SubmitJobToWorker(int tid, AssemblyJob *job);

    void CoordinatorFromFastFile(string fasta_path);
    void CoordinatorFromFastaFile_ComputeEmbeddings(std::filesystem::path fasta_full_path);
    void CoordinatorFromFastaFile_ScoreCandidates();
    void CoordinatorFromFastaFile_InitialSubgraphJoinScoring();
    void CoordinatorFromFastaFile_SubgraphJoining();
    void WorkerThread(int idx);

    int AddGeneSeq(char *data, GeneSeqType gene_seq_type);
    char *GetKmerBySeqKmerIDView(SeqKmerID seqkmer);
    char *GetKmerBySeqKmerIDAlloced(SeqKmerID seqkmer);
    SeqKmerID GetReprKmer(SeqKmerID seqkmer);
    bool GeneSeqToGraph(GeneSeq *seq, GraphDescriptor **graph_desc_out);

    void ScoreSubgraph(AssemblerJobScoreSubgraphParams *params, int idx);
    AlignmentCandidates *KmerGetApproxKnn(float *emb, size_t k);
    int KmerCalcKnnSSWAlignments(SeqKmerID kmer_a, AlignmentCandidates *candidates, int idx);
    void ScoreCandidates(AssemblerJobScoreCandidatesParams *params, int idx);
    void JoinSubgraphs(AssemblerJobJoinSubgraphsParams *params, int idx);

    void DrainWorkerJobQueues();
    void StartWorkerThreads(unsigned int n_workers);
    void StopWorkerThreads();

    enum AssemblyState state = kAssemblyIdle;
    enum AssemblySubState sub_state = kAssemblyInitializing;
    GeneSeqType gene_seq_type;
    mutex assembler_mutex;
    thread **worker_threads;
    mutex *worker_mutexes;
    condition_variable *worker_syncs;
    bool *work_pending;
    bool *worker_busy;
    stack<AssemblyJob *> *work_queues;
    thread *coordinator_thread;
    atomic<bool> coordinator_abort{false};
    mutex done_queue_mutex;
    stack<AssemblyJobStatus *> done_queue;
    unsigned int n_worker_threads;
    atomic<bool> workers_active{false};
    atomic<int> n_workers_busy{0};
    atomic<int> n_workers_active{0};
    atomic<int> geneseqs_embedded{0};
    atomic<int> geneseqs_join_candidates_scored{0};
    atomic<int> subgraphs_scored{0};
    int geneseqs_expected = -1;
    int embeddings_expected = -1;
    condition_variable some_workers_ready;
    chrono::high_resolution_clock::time_point start_time;
    chrono::high_resolution_clock::time_point emb_start_time;
    chrono::high_resolution_clock::time_point score_start_time;
    chrono::high_resolution_clock::time_point isgjoin_score_start_time;
    chrono::high_resolution_clock::time_point sgjoin_start_time;
    chrono::high_resolution_clock::time_point end_time;

    KmerEmbeddingSpace *kmer_embedding_space = NULL;

    mutex data_mutex;
    unordered_map<string, int> unique_gene_seq_to_id_umap;
    unordered_map<int, GeneSeq *> gene_seq_umap;
    unordered_map<int, GraphDescriptor *> gene_seq_id_to_subgraph_umap;
    unordered_map<GraphDescriptorID, GraphDescriptor *> subgraph_umap;
    unordered_map<GraphDescriptorID, vector<GeneSeq *> *> subgraph_to_gene_seqs_umap;
    priority_queue<PotentialJoin> pqueue_potential_joins;

    unordered_map<SeqKmerID, AlignmentCandidates *> kmer_alignment_candidates_umap;

    unordered_map<pair<SeqKmerID, SeqKmerID>, parasail_result_t *, hash_pair> kmer_alignment_umap;
    unordered_map<pair<GraphDescriptorID, GraphDescriptorID>, PotentialJoin *, hash_pair>
        potential_join_umap;
};

#endif