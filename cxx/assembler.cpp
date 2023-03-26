// vim: ts=4:sw=4:et

#include "assembler.h"
#include "graph.h"
#include "logger.h"

#include <fcntl.h>
#include <string>
#include <stdio.h>
#include <unordered_map>

#include <boost/endian/conversion.hpp>

#include "assembler_generated.h"
#include <flatbuffers/flatbuffers.h>

using namespace Gene::Assembler;

using namespace std::chrono_literals;

namespace py = pybind11;

AssemblerConfig::AssemblerConfig() {}

AssemblerConfig::AssemblerConfig(py::object config_dict) {
    if (config_dict.contains("max_ntds_per_edge")) {
        max_ntds_per_edge = config_dict["max_ntds_per_edge"].cast<int>();
    }

    if (config_dict.contains("kmer_len")) {
        kmer_len = config_dict["kmer_len"].cast<int>();
    }

    if (config_dict.contains("alphabet_len")) {
        alphabet_len = config_dict["alphabet_len"].cast<int>();
    }

    if (config_dict.contains("work_queue_low_watermark")) {
        work_queue_low_watermark = config_dict["work_queue_low_watermark"].cast<int>();
    }

    if (config_dict.contains("work_queue_high_watermark")) {
        work_queue_high_watermark = config_dict["work_queue_high_watermark"].cast<int>();
    }

    if (config_dict.contains("knn_top_k")) {
        knn_top_k = config_dict["knn_top_k"].cast<int>();
    }

    if (config_dict.contains("seqemb_onnx_path")) {
        seqemb_onnx_path = config_dict["seqemb_onnx_path"].cast<string>();
    }

    if (config_dict.contains("subgraph_join_threshold")) {
        subgraph_join_threshold = config_dict["subgraph_join_threshold"].cast<float>();
    }
}

void Assembler::DrainWorkerJobQueues() {
    spdlog::info("DrainWorkerJobQueues start...");
    if (n_workers_busy.load() == n_worker_threads) {
        unique_lock<std::mutex> lk(assembler_mutex);
        some_workers_ready.wait(lk, [&] { return n_workers_busy.load() < n_worker_threads; });
        lk.unlock();
    }

    int num_workers_nonempty;
    do {
        num_workers_nonempty = 0;
        for (int tid = 0; tid < n_worker_threads; tid++) {
            worker_mutexes[tid].lock();
            if (work_queues[tid].size() > 0) {
                num_workers_nonempty++;
            }
            worker_mutexes[tid].unlock();
        }
    } while ((num_workers_nonempty > 0) || (n_workers_active.load() > 0));

    spdlog::info("DrainWorkerJobQueues end. n_workers_busy={}", n_workers_busy.load());
}

void LogAssemblerConfig(AssemblerConfig *config) {
    spdlog::info("AssemblerConfig:");
    spdlog::info("    max_ntds_per_edge = {}", config->max_ntds_per_edge);
    spdlog::info("    kmer_len = {}", config->kmer_len);
    spdlog::info("    alphabet_len = {}", config->alphabet_len);
    spdlog::info("    work_queue_watermakrs: {},{}", config->work_queue_low_watermark,
                 config->work_queue_high_watermark);
    spdlog::info("    knn_top_k: {}", config->knn_top_k);
    spdlog::info("    seqemb_onnx_path: {}", config->seqemb_onnx_path);
    spdlog::info("    subgraph_join_threshold: {}", config->subgraph_join_threshold);
}

PotentialJoin::PotentialJoin() {}

PotentialJoin::PotentialJoin(GraphDescriptorID gdid_a, GraphDescriptorID gdid_b,
                             float norm_score_in) {
    align_pair = MakeCanonicalGraphDescriptorPair(gdid_a, gdid_b);
    norm_score = norm_score_in;
}

Assembler::Assembler() {
    this->get_next_id_();

    if (config == NULL) {
        config = new AssemblerConfig();
    }
    LogAssemblerConfig(config);
}

Assembler::Assembler(AssemblerConfig config_in) {
    this->get_next_id_();

    config = new AssemblerConfig(config_in);
    LogAssemblerConfig(config);
}

int Assembler::StartAsync(string fasta_path, GeneSeqType gene_seq_type_in) {
    state = kAssemblyInProgress;
    sub_state = kAssemblyInitializing;
    gene_seq_type = gene_seq_type_in;
    start_time = chrono::high_resolution_clock::now();

    coordinator_thread = new thread(&Assembler::CoordinatorFromFastFile, this, fasta_path);

    return 0;
}

AssemblyStatus Assembler::GetStatus() {
    AssemblyStatus status;

    status.state = state;
    status.sub_state = sub_state;
    status.geneseqs_expected = geneseqs_expected;
    status.geneseqs_embedded = geneseqs_embedded;
    status.geneseqs_join_candidates_scored = geneseqs_join_candidates_scored;
    status.subgraphs_scored = subgraphs_scored;

    status.compute_embeddings_duration_ms = -1;
    status.score_candidates_duration_ms = -1;
    status.initial_subgraph_join_score_duration_ms = -1;
    status.subgraph_joining_duration_ms = -1;

    if (sub_state == kAssemblyComputingEmbeddings) {
        status.compute_embeddings_duration_ms =
            chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() -
                                                        emb_start_time)
                .count();
    } else if (sub_state == kAssemblyScoringCandidates) {
        status.compute_embeddings_duration_ms =
            chrono::duration_cast<chrono::milliseconds>(score_start_time - emb_start_time).count();
        status.score_candidates_duration_ms =
            chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() -
                                                        score_start_time)
                .count();
    } else if (sub_state == kAssemblyScoringCandidates) {
        status.compute_embeddings_duration_ms =
            chrono::duration_cast<chrono::milliseconds>(score_start_time - emb_start_time).count();
        status.score_candidates_duration_ms =
            chrono::duration_cast<chrono::milliseconds>(sgjoin_start_time - score_start_time)
                .count();
    } else if (sub_state == kAssemblyInitialSubgraphJoinScoring) {
        status.compute_embeddings_duration_ms =
            chrono::duration_cast<chrono::milliseconds>(score_start_time - emb_start_time).count();
        status.score_candidates_duration_ms =
            chrono::duration_cast<chrono::milliseconds>(isgjoin_score_start_time - score_start_time)
                .count();
        status.initial_subgraph_join_score_duration_ms =
            chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() -
                                                        isgjoin_score_start_time)
                .count();
    } else if (sub_state == kAssemblySubgraphJoining) {
        status.compute_embeddings_duration_ms =
            chrono::duration_cast<chrono::milliseconds>(score_start_time - emb_start_time).count();
        status.score_candidates_duration_ms =
            chrono::duration_cast<chrono::milliseconds>(isgjoin_score_start_time - score_start_time)
                .count();
        status.initial_subgraph_join_score_duration_ms =
            chrono::duration_cast<chrono::milliseconds>(sgjoin_start_time -
                                                        isgjoin_score_start_time)
                .count();
        status.subgraph_joining_duration_ms =
            chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() -
                                                        sgjoin_start_time)
                .count();
    } else if (sub_state > kAssemblySubgraphJoining) {
        status.compute_embeddings_duration_ms =
            chrono::duration_cast<chrono::milliseconds>(score_start_time - emb_start_time).count();
        status.score_candidates_duration_ms =
            chrono::duration_cast<chrono::milliseconds>(sgjoin_start_time - score_start_time)
                .count();
        status.initial_subgraph_join_score_duration_ms =
            chrono::duration_cast<chrono::milliseconds>(sgjoin_start_time -
                                                        isgjoin_score_start_time)
                .count();
        status.subgraph_joining_duration_ms =
            chrono::duration_cast<chrono::milliseconds>(end_time - sgjoin_start_time).count();
    }

    return status;
}

void Assembler::Abort() { coordinator_abort = true; }

template <typename T> inline void freeContainer(T &p_container) {
    T empty;
    using std::swap;
    swap(p_container, empty);
}

int Assembler::WaitForAvailableWorker(int tid) {
    if (n_workers_busy.load() == n_worker_threads) {
        unique_lock<std::mutex> lk(assembler_mutex);
        some_workers_ready.wait(lk, [&] { return n_workers_busy.load() < n_worker_threads; });
        lk.unlock();
    }

    // find the first worker ready to accept work
    while (work_queues[tid].size() >= config->work_queue_high_watermark) {
        tid++;
        if (tid >= n_worker_threads) {
            tid = 0;
        }
    }
    assert(tid < n_worker_threads);

    return tid;
}

int Assembler::SubmitJobToWorker(int tid, AssemblyJob *job) {
    int q_depth;

    worker_mutexes[tid].lock();
    work_queues[tid].push(job);
    work_pending[tid] = true;
    q_depth = work_queues[tid].size();
    if ((q_depth == config->work_queue_high_watermark) && (worker_busy[tid] == false)) {
        n_workers_busy++;
        worker_busy[tid] = true;
    }
    worker_mutexes[tid].unlock();
    if (q_depth == 1) {
        worker_syncs[tid].notify_one();
    }

    tid++;
    if (tid >= n_worker_threads) {
        tid = 0;
    }

    return tid;
}

bool Assembler::DetectedAbort() {
    if (coordinator_abort.load() == true) {
        DrainWorkerJobQueues();
        StopWorkerThreads();
        state = kAssemblyAborted;
        return true;
    }

    return false;
}

void Assembler::CoordinatorFromFastaFile_ComputeEmbeddings(std::filesystem::path fasta_full_path) {
    int tid = 0;

    sub_state = kAssemblyComputingEmbeddings;
    emb_start_time = chrono::high_resolution_clock::now();

    embeddings_expected = geneseqs_expected * 300; // XXXADS TODO update Fasta parser to measure this when calculating geneseqs_expected

    kmer_embedding_space =
        new KmerEmbeddingSpace(embeddings_expected, config->kmer_len, config->alphabet_len,
                               n_worker_threads, config->seqemb_onnx_path);

    std::function<void(Assembler *assembler, string * name, string * description, string * data)> onEachSeqFn =
        [&tid](Assembler *assembler, string *name, string *description, string *data) {
            int gsid;
            AssemblyJob *job;

            //spdlog::warn("onEachSeqFn name={}, descr={}, datalen={}",
            //    *name, *description, data->size());

            gsid = assembler->AddGeneSeq((char *)data->c_str(), assembler->gene_seq_type);
            if (gsid == kGeneSeqDuplicate) {
                return;
            }

            tid = assembler->WaitForAvailableWorker(tid);

            job = new AssemblyJob;
            job->job_type = kAssemblerJobInferEmbedding;
            job->job_params = new AssemblerJobInferEmbeddingParams(assembler->gene_seq_umap[gsid]);

            tid = assembler->SubmitJobToWorker(tid, job);
        };

    ParseFastaFile(fasta_full_path, /*only_count_sequences*/ false, &geneseqs_expected,
                   /*onParsedSeqFn*/ onEachSeqFn);

    freeContainer(unique_gene_seq_to_id_umap);

    DrainWorkerJobQueues();
    spdlog::info("DrainWorkerJobQueues complete: n_workers_busy={}", n_workers_busy);
    assert(n_workers_busy == 0);
}

void Assembler::CoordinatorFromFastaFile_ScoreCandidates() {
    int gsid;
    int tid = 0;
    AssemblyJob *job;

    sub_state = kAssemblyScoringCandidates;
    score_start_time = chrono::high_resolution_clock::now();

    tid = 0;
    kmer_embedding_space->kmer_instance_umap_mutex.lock();
    for (pair<string, vector<SeqKmerID>> element :
         kmer_embedding_space->kmer_instance_umap) // XXXADS efficiency, avoid this copy?
    {
        if (DetectedAbort()) {
            return;
        }

        tid = WaitForAvailableWorker(tid);

        if (spdlog::default_logger()->level() <= SPDLOG_LEVEL_DEBUG) {
            for (SeqKmerID skid : element.second) {
                int seq_id;
                int kmer_off;
                GetSeqKmerIDOff(skid, &seq_id, &kmer_off);
                spdlog::debug("       tid={}, seq_id={}, kmer_off={}", tid, seq_id, kmer_off);
            }
        }

        assert(element.second.size() > 0);

        job = new AssemblyJob;
        job->job_type = kAssemblerJobScoreCandidates;
        job->job_params = new AssemblerJobScoreCandidatesParams(element.second.front());

        tid = SubmitJobToWorker(tid, job);
    } // end of for each unique kmer
    kmer_embedding_space->kmer_instance_umap_mutex.unlock();

    DrainWorkerJobQueues();
    assert(n_workers_busy == 0);
}

void Assembler::CoordinatorFromFastaFile_InitialSubgraphJoinScoring() {
    int tid = 0;
    AssemblyJob *job;

    sub_state = kAssemblyInitialSubgraphJoinScoring;
    isgjoin_score_start_time = chrono::high_resolution_clock::now();

    for (auto &subgraph_to_gene_seqs_entry : subgraph_to_gene_seqs_umap) {
        const GraphDescriptorID gdid = subgraph_to_gene_seqs_entry.first;
        const vector<GeneSeq *> *seqs = subgraph_to_gene_seqs_entry.second;

        if (DetectedAbort()) {
            return;
        }

        tid = WaitForAvailableWorker(tid);

        job = new AssemblyJob;
        job->job_type = kAssemblerJobScoreSubgraph;
        job->job_params = new AssemblerJobScoreSubgraphParams(gdid);

        tid = SubmitJobToWorker(tid, job);
    }

    DrainWorkerJobQueues();
    assert(n_workers_busy == 0);
}

void Assembler::CoordinatorFromFastaFile_SubgraphJoining() {
    int tid = 0;
    AssemblyJob *job;

    sub_state = kAssemblySubgraphJoining;
    sgjoin_start_time = chrono::high_resolution_clock::now();

    while (!pqueue_potential_joins.empty()) {
        PotentialJoin potential_join = pqueue_potential_joins.top();

        if (DetectedAbort()) {
            return;
        }

        tid = WaitForAvailableWorker(tid);

        job = new AssemblyJob;
        job->job_type = kAssemblerJobJoinSubgraphs;
        job->job_params = new AssemblerJobJoinSubgraphsParams(&potential_join);

        spdlog::warn("    PotentialJoin {}<->{}: {}", potential_join.align_pair.first,
                     potential_join.align_pair.second, potential_join.norm_score);

        tid = SubmitJobToWorker(tid, job);

        pqueue_potential_joins.pop();
    }

    DrainWorkerJobQueues();
    assert(n_workers_busy == 0);
}

bool Assembler::ParseFastaFile(
    string filename, bool only_count_sequences, int *num_sequences_out,
    std::function<void(Assembler *assembler, string *name, string *description, string *data)> &onParsedSeqFn) {
    const int BUFFER_SIZE = 1024 * 1024; // 1 MiB

    int fd = open(filename.c_str(), O_RDONLY);
    assert(fd != -1); // XXXADS TODO add graceful error handling

    posix_fadvise(fd, 0, 0, 1); // advertise sequential access hint VFS prefetching

    char buf[BUFFER_SIZE+1];
    char *brk;
    string name;
    string description;
    string data;

    int seq_num = 0;
    enum FastaParserState parser_state = kParserStateStart;

    buf[BUFFER_SIZE]='\0';
    while (size_t bytes_size = read(fd, buf, BUFFER_SIZE)) {
        if (coordinator_abort.load() == true) {
            return false;
        }
        if (bytes_size == (size_t)-1) {
            if (errno == EINTR) {
                continue;
            }
            assert(0); // XXXADS TODO add graceful error handling
        }

        int off = 0;
        size_t len;

        while (off < bytes_size) {
            switch (parser_state) {
            case kParserStateStart:
                switch (buf[off]) {
                case '>':
                    parser_state = kParserStateSeqName;
                    off++;
                    break;
                case ';':
                    parser_state = kParserStateComment;
                    off++;
                    break;
                case '\n':
                    assert(0); // XXXADS TODO add graceful error handling.
                default:
                    brk = strpbrk(&buf[off], "\n");
                    if (brk == NULL) {
                        len = (BUFFER_SIZE - off) + 1;
                    } else {
                        len = (brk - &buf[off]) + 1;
                        parser_state = kParserStateSeqDataNewLine;
                    }
                    if (!only_count_sequences) {
                        data.insert(data.size(), &buf[off], len-1);
                    }
                    off += len;
                    break;
                }
                break;
            case kParserStateComment:
                switch (buf[off]) {
                case '\n':
                    parser_state = kParserStateCommentNewLine;
                    off++;
                    break;
                default:
                    brk = strpbrk(&buf[off], "\n");
                    if (brk == NULL) {
                        len = BUFFER_SIZE - off;
                    } else {
                        len = (brk - &buf[off]) + 1;
                        parser_state = kParserStateCommentNewLine;
                    }
                    off += len;
                    break;
                }
                break;
            case kParserStateCommentNewLine:
                switch (buf[off]) {
                case '>':
                    parser_state = kParserStateSeqName;
                    break;
                case ';':
                    parser_state = kParserStateComment;
                    break;
                default:
                    assert(0); // XXXADS TODO add graceful error handling
                }
                off++;
                break;
            case kParserStateSeqName:
                switch (buf[off]) {
                case ' ':
                    parser_state = kParserStateSeqDescription;
                    off++;
                    break;
                case '\n':
                    parser_state = kParserStateSeqDataNewLine;
                    off++;
                    break;
                default:
                    brk = strpbrk(&buf[off], " \t\n");
                    if (brk == NULL) {
                        len = (BUFFER_SIZE - off) + 1;
                    } else {
                        len = (brk - &buf[off]) + 1;
                        if (*brk == '\n') {
                            parser_state = kParserStateSeqDataNewLine;
                        } else {
                            parser_state = kParserStateSeqDescription;
                        }
                    }
                    if (!only_count_sequences) {
                        name.insert(name.size(), &buf[off], len-1);
                    }
                    off += len;
                    break;
                }
                break;
            case kParserStateSeqDescription:
                switch (buf[off]) {
                case '\n':
                    parser_state = kParserStateSeqDataNewLine;
                    off++;
                    // end of description.
                    break;
                default:
                    brk = strpbrk(&buf[off], "\n");
                    if (brk == NULL) {
                        len = (BUFFER_SIZE - off) + 1;
                    } else {
                        len = (brk - &buf[off]) + 1;
                        parser_state = kParserStateSeqDataNewLine;
                    }
                    if (!only_count_sequences) {
                        description.insert(description.size(), &buf[off], len-1);
                    }
                    off += len;
                    break;
                }
                break;
            case kParserStateSeqDataNewLine:
                switch (buf[off]) {
                case '>':
                    parser_state = kParserStateSeqName;
                    off++;
                    seq_num++;
                    if (!only_count_sequences) {
                        onParsedSeqFn(this, &name, &description, &data);
                        name.clear();
                        description.clear();
                        data.clear();
                    }
                    break;
                case ';':
                    parser_state = kParserStateComment;
                    off++;
                    seq_num++;
                    if (!only_count_sequences) {
                        onParsedSeqFn(this, &name, &description, &data);
                        name.clear();
                        description.clear();
                        data.clear();
                    }
                    break;
                default:
                    brk = strpbrk(&buf[off], "\n");
                    if (brk == NULL) {
                        len = (BUFFER_SIZE - off) + 1;
                        parser_state = kParserStateSeqData;
                    } else {
                        len = (brk - &buf[off]) + 1;
                        parser_state = kParserStateSeqDataNewLine;
                    }
                    if (!only_count_sequences) {
                        data.insert(data.size(), &buf[off], len-1);
                    }
                    off += len;
                    break;
                }
                break;
            case kParserStateSeqData:
                switch (buf[off]) {
                case '\n':
                    parser_state = kParserStateSeqDataNewLine;
                    off++;
                    break;
                default:
                    brk = strpbrk(&buf[off], "\n");
                    if (brk == NULL) {
                        len = (BUFFER_SIZE - off) + 1;
                        parser_state = kParserStateSeqData;
                    } else {
                        len = (brk - &buf[off]) + 1;
                        parser_state = kParserStateSeqDataNewLine;
                    }
                    if (!only_count_sequences) {
                        data.insert(data.size(), &buf[off], len-1);
                    }
                    off += len;
                    break;
                }
                break;
            }
        }
    }

    if (data.length() > 0) {
        if (!only_count_sequences) {
            onParsedSeqFn(this, &name, &description, &data);
            seq_num++;
        }
    }

    close(fd);

    *num_sequences_out = seq_num;
    return true;
}

void Assembler::CoordinatorFromFastFile(string fasta_path) {
    auto filename = std::filesystem::current_path() / fasta_path;

    StartWorkerThreads(/*use all hw threads*/ 0);

    chrono::high_resolution_clock::time_point scan_start_time =
        chrono::high_resolution_clock::now();

    std::function<void(Assembler *assembler, string * name, string * description, string * data)> onEachSeqFn =
        [](Assembler *assembler, string *name, string *description, string *data) {
        };

    ParseFastaFile(filename, /*only_count_sequences*/ true, &geneseqs_expected,
                   /*onParsedSeqFn*/onEachSeqFn);

    if (coordinator_abort.load() == true) {
        StopWorkerThreads();
        state = kAssemblyAborted;
        return;
    }

    auto scan_duration_ms = chrono::duration_cast<chrono::milliseconds>(
        chrono::high_resolution_clock::now() - scan_start_time);
    spdlog::info("    fasta sequence scan duration: {} ms, {} total sequences",
                 scan_duration_ms.count(), geneseqs_expected);

    CoordinatorFromFastaFile_ComputeEmbeddings(filename);
    if (coordinator_abort.load() == true) {
        return;
    }

    CoordinatorFromFastaFile_ScoreCandidates();
    if (coordinator_abort.load() == true) {
        return;
    }

    CoordinatorFromFastaFile_InitialSubgraphJoinScoring();
    if (coordinator_abort.load() == true) {
        return;
    }

    CoordinatorFromFastaFile_SubgraphJoining();
    if (coordinator_abort.load() == true) {
        return;
    }

    sub_state = kAssemblyCleaningUp;
    StopWorkerThreads();

    // XXXADS TODO report contigs

    end_time = chrono::high_resolution_clock::now();
    state = kAssemblyComplete;
}

int Assembler::StartSync(vector<char *> reads, GeneSeqType gene_seq_type) {
    spdlog::info("Assembler constructor (reads provided as a vec of char ptrs).");
    this->get_next_id_();
    if (config == NULL) {
        config = new AssemblerConfig();
    }

    StartWorkerThreads(/*use all hw threads*/ 0);

    for (size_t i = 0; i < reads.size(); i++) {
        spdlog::info("    read[{}]: {}", i, reads[i]);
        this->AddGeneSeq(reads[i], gene_seq_type);
    }

    this_thread::sleep_for(chrono::seconds(2)); // XXXADS DEPRECATE
    StopWorkerThreads();

    return 0;
}

int Assembler::StartSync(vector<string> reads, GeneSeqType gene_seq_type) {
    spdlog::info("Assembler constructor (reads provided as a vec of strings).");
    this->get_next_id_();
    if (config == NULL) {
        config = new AssemblerConfig();
    }

    StartWorkerThreads(/*use all hw threads*/ 0);

    for (size_t i = 0; i < reads.size(); i++) {
        spdlog::info("    read[{}]: {}", i, reads[i]);
        this->AddGeneSeq((char *)reads[i].c_str(), gene_seq_type);
    }

    this_thread::sleep_for(chrono::seconds(2)); // XXXADS DEPRECATE / remove later
    StopWorkerThreads();

    return 0;
}

int Assembler::AddGeneSeq(char *data, GeneSeqType gene_seq_type) {
    GeneSeq *gseq = NULL;
    GraphDescriptor *graph_desc;

    string datastr = string(data);
    auto it = unique_gene_seq_to_id_umap.find(datastr);
    if (it != unique_gene_seq_to_id_umap.end()) {
        gseq = gene_seq_umap[(*it).second];
        gseq->instances++;
        spdlog::info("    seq[{}]: not unique. {} instances.", data, gseq->instances);
        return kGeneSeqDuplicate;
    }

    gseq = new GeneSeq(data, gene_seq_type);
    unique_gene_seq_to_id_umap[datastr] = gseq->id;

    gene_seq_umap[gseq->id] = gseq;
    GeneSeqToGraph(gseq, &graph_desc);

    return gseq->id;
}

char *Assembler::GetKmerBySeqKmerIDView(SeqKmerID seqkmer) {
    int seq_id;
    int kmer_off;
    GeneSeq *seq_p = NULL;

    // using rangeless::fn::operators::operator%;

    GetSeqKmerIDOff(seqkmer, &seq_id, &kmer_off);
    spdlog::debug("    GetKmerBySeqKmerIDView: seq_id={}, kmer_off={}", seq_id, kmer_off);
    auto it = gene_seq_umap.find(seq_id);
    if (it == gene_seq_umap.end()) {
        return NULL;
    }

    seq_p = (*it).second;

#if 0
    vector<char> kmer = rangeless::fn::view(seq_p->data.begin()+(config->kmer_len*kmer_off),
        seq_p->data.begin()+(config->kmer_len*(kmer_off+1)));

    return kmer.data();
#endif

    // XXXADS optimization opportunity.

    vector<char> *kmer = slice(seq_p->data, kmer_off, config->kmer_len + kmer_off - 1);

    return kmer->data();
}

char *Assembler::GetKmerBySeqKmerIDAlloced(SeqKmerID seqkmer) {
    int seq_id;
    int kmer_off;
    GeneSeq *seq_p = NULL;
    char *kmer_data = new char[config->kmer_len + 1];

    // using rangeless::fn::operators::operator%;

    GetSeqKmerIDOff(seqkmer, &seq_id, &kmer_off);
    spdlog::debug("    GetKmerBySeqKmerIDAlloced: seq_id={}, kmer_off={}", seq_id, kmer_off);
    auto it = gene_seq_umap.find(seq_id);
    if (it == gene_seq_umap.end()) {
        return NULL;
    }

    seq_p = (*it).second;

#if 0
    vector<char> kmer = rangeless::fn::view(seq_p->data.begin()+(config->kmer_len*kmer_off),
        seq_p->data.begin()+(config->kmer_len*(kmer_off+1)));

    return kmer.data();
#endif

    // XXXADS optimization opportunity.

    vector<char> *kmer = slice(seq_p->data, kmer_off, config->kmer_len + kmer_off - 1);

    memcpy(kmer_data, kmer->data(), config->kmer_len);
    kmer_data[config->kmer_len] = '\0';

    return kmer_data;
}

SeqKmerID Assembler::GetReprKmer(SeqKmerID seqkmer) {
    char *kmer_str = GetKmerBySeqKmerIDAlloced(seqkmer);

    auto kit = kmer_embedding_space->kmer_instance_umap.find(string(kmer_str));
    if (kit == kmer_embedding_space->kmer_instance_umap.end()) {
        spdlog::warn("GetReprKmer: no reprkmer found for: {}", string(kmer_str));
        free(kmer_str);
        return seqkmer;
    }

    free(kmer_str);

    return (*kit).second.front();
}

void Assembler::JoinSubgraphs(AssemblerJobJoinSubgraphsParams *params, int idx) {
    GraphDescriptor *ga, *gb;

    auto git = subgraph_umap.find(params->potential_join.align_pair.first);
    assert(git != subgraph_umap.end());
    ga = (*git).second;

    if (!ga->subgraph_mutex.try_lock()) {
        spdlog::warn("    JoinSubgraphs: gdid={} was locked", ga->id);
        return;
    }

    git = subgraph_umap.find(params->potential_join.align_pair.second);
    assert(git != subgraph_umap.end());
    gb = (*git).second;

    if (!gb->subgraph_mutex.try_lock()) {
        spdlog::warn("    JoinSubgraphs: gdid={} was locked", gb->id);
        ga->subgraph_mutex.unlock();
        return;
    }

    ga->CalculatePaths(/*already_locked*/ true);
    gb->CalculatePaths(/*already_locked*/ true);

    char *pa_str, *pb_str;
    int pa_strlen, pb_strlen;
    parasail_result_t *swa;

    // SSW on every unique pair of paths.
    for (GraphPath *pa : *(ga->paths)) {
        for (GraphPath *pb : *(gb->paths)) {
            pa->AsCharArray(&pa_str, &pa_strlen);
            pb->AsCharArray(&pb_str, &pb_strlen);

            swa = parasail_sw(pa_str, pa_strlen, pb_str, pb_strlen, 3, 1, &parasail_dnafull);
            parasail_result_free(swa);
        }
    }

    // join subgraphs here

    ga->subgraph_mutex.unlock();
    gb->subgraph_mutex.unlock();
}

void Assembler::ScoreSubgraph(AssemblerJobScoreSubgraphParams *params, int idx) {
    GraphDescriptor *gd;
    vector<GeneSeq *> *gene_seqs;

    unordered_map<GraphDescriptorID, int> potential_join_scores;
    int max_score = 0;
    int num_kmers = 0;
    GraphDescriptorID top_target;

    auto git = subgraph_umap.find(params->gdid);
    if (git == subgraph_umap.end()) {
        return;
    }
    gd = (*git).second;

    if (!gd->subgraph_mutex.try_lock()) {
        spdlog::warn("    ScoreSubgraph: gdid={} was locked", params->gdid);
        return;
    }

    auto gsit = subgraph_to_gene_seqs_umap.find(params->gdid);
    if (gsit == subgraph_to_gene_seqs_umap.end()) {
        spdlog::warn("    ScoreSubgraph: gdid={} has no gene seqs", params->gdid);
        gd->subgraph_mutex.unlock();
        return;
    }

    gene_seqs = (*gsit).second;

    for (auto &gene_seq : *gene_seqs) {
        for (int off = 0; off < (int)gene_seq->len - config->kmer_len; off++) {
            SeqKmerID kmer_a = MakeSeqKMerID(gene_seq->id, off);
            AlignmentCandidates *candidates;

            kmer_a = GetReprKmer(kmer_a); // kmers are deduped, the repr. kmer contains the
                                          // instance-independent meta data

            auto acit = kmer_alignment_candidates_umap.find(kmer_a);
            if (acit == kmer_alignment_candidates_umap.end()) {
                spdlog::warn(
                    "    ScoreSubgraph: gdid={}, kmer_a={} +{} has no alignment candidates",
                    params->gdid, (size_t)kmer_a, off);
                continue;
            }

            candidates = (*acit).second;

            for (auto &candidate : *candidates) {
                const float score = candidate.first;
                const SeqKmerID kmer_b = candidate.second;
                int kmer_b_seq_id;
                int kmer_b_seq_off;
                pair<SeqKmerID, SeqKmerID> canonical_kmer_pair;
                parasail_result_t *swa;

                GetSeqKmerIDOff(kmer_b, &kmer_b_seq_id, &kmer_b_seq_off);

                canonical_kmer_pair = MakeCanonicalKmerPair(kmer_a, kmer_b);

                auto ait = kmer_alignment_umap.find(canonical_kmer_pair);
                if (ait == kmer_alignment_umap.end()) {
                    continue;
                }

                swa = (*ait).second;

                spdlog::info("    ScoreSubgraph: ka={} sid{}+off{}, kb={}, score={}",
                             (size_t)kmer_a, gene_seq->id, off, (size_t)kmer_b, swa->score);

                auto gstgdit = gene_seq_id_to_subgraph_umap.find(kmer_b_seq_id);
                assert(gstgdit != gene_seq_id_to_subgraph_umap.end());

                GraphDescriptor *kmer_b_subgraph = (*gstgdit).second;
                const GraphDescriptorID kbgdid = kmer_b_subgraph->id;

                auto pjsit = potential_join_scores.find(kbgdid);
                if (pjsit == potential_join_scores.end()) {
                    potential_join_scores[kbgdid] = swa->score;
                } else {
                    (*pjsit).second += swa->score;
                }
                if (potential_join_scores[kbgdid] > max_score) {
                    max_score = potential_join_scores[kbgdid];
                    top_target = kbgdid;
                }
            }

            num_kmers++;
        } // end of for each kmer
    }     // end of for each gene seq

    float norm_score = (float)max_score / num_kmers;

    if (norm_score >= config->subgraph_join_threshold) {
        data_mutex.lock();
        pqueue_potential_joins.push(PotentialJoin(params->gdid, top_target, norm_score));
        data_mutex.unlock();
    }

    gd->subgraph_mutex.unlock();
    spdlog::warn("    ScoreSubgraph: gaid={}: gbid={}, score={}", (size_t)params->gdid,
                 (int)top_target, norm_score);
}

AlignmentCandidates *Assembler::KmerGetApproxKnn(float *emb, size_t k) {
    AlignmentCandidates *knn = new AlignmentCandidates(
        kmer_embedding_space->alg_hnsw->searchKnnCloserFirst((void *)emb, k));

    (*knn).erase((*knn).begin());

    return knn;
}

pair<SeqKmerID, SeqKmerID> MakeCanonicalKmerPair(SeqKmerID kmer_a, SeqKmerID kmer_b) {
    if ((size_t)kmer_a > (size_t)kmer_b) {
        return pair<SeqKmerID, SeqKmerID>(kmer_b, kmer_a);
    }

    return pair<SeqKmerID, SeqKmerID>(kmer_a, kmer_b);
}

pair<GraphDescriptorID, GraphDescriptorID>
MakeCanonicalGraphDescriptorPair(GraphDescriptorID gd_a, GraphDescriptorID gd_b) {
    if ((int)gd_a > (int)gd_b) {
        return pair<GraphDescriptorID, GraphDescriptorID>(gd_b, gd_a);
    }

    return pair<GraphDescriptorID, GraphDescriptorID>(gd_a, gd_b);
}

int Assembler::KmerCalcKnnSSWAlignments(SeqKmerID kmer_a, AlignmentCandidates *candidates,
                                        int idx) {
    for (auto &candidate : *candidates) {
        const float &dist = candidate.first;
        const SeqKmerID &kmer_b = candidate.second;
        const char *kmer_a_str = GetKmerBySeqKmerIDView(kmer_a);
        const char *kmer_b_str = GetKmerBySeqKmerIDView(kmer_b);
        parasail_result_t *swa;

        pair<SeqKmerID, SeqKmerID> canonical_kmer_pair = MakeCanonicalKmerPair(kmer_a, kmer_b);

        spdlog::info("    SSWAlign ka={:X}, kb={:X}, config->kmer_len={}, idx={}", kmer_a, kmer_b,
                     config->kmer_len, idx);

        assert(kmer_a_str != NULL);
        assert(kmer_b_str != NULL);
        swa = parasail_sw(kmer_a_str, config->kmer_len, kmer_b_str, config->kmer_len, 3, 1,
                          &parasail_dnafull);
        spdlog::info("        idx={}, score={}", idx, swa->score);

        data_mutex.lock();
        auto rval = kmer_alignment_umap.try_emplace(canonical_kmer_pair, swa);
        data_mutex.unlock();
        if (rval.second == false) {
            parasail_result_free(swa);
        }
    }

    spdlog::info("        RET FROM KmerCalcKnnSSWAlignments, idx={}", idx);

    return 0;
}

void Assembler::ScoreCandidates(AssemblerJobScoreCandidatesParams *params, int idx) {
    AlignmentCandidates *candidates;

    int seq_id;
    int kmer_off;
    GetSeqKmerIDOff(params->repr_kmer, &seq_id, &kmer_off);

    auto it = gene_seq_umap.find(seq_id);
    spdlog::debug("    ScoreCandidates id={}, seq_id={}", this_thread::get_id(), seq_id);

    assert(it != gene_seq_umap.end());

    GeneSeq *gseq = (*it).second;

    candidates = KmerGetApproxKnn(gseq->emb_data[kmer_off], config->knn_top_k);
    if (candidates->size() == 0) {
        spdlog::warn("    ScoreCandidates id={}, seq_id={} +{} has no candidates",
                     this_thread::get_id(), seq_id, kmer_off);
    }
    data_mutex.lock();
    kmer_alignment_candidates_umap[params->repr_kmer] = candidates;
    data_mutex.unlock();

    KmerCalcKnnSSWAlignments(params->repr_kmer, candidates, idx);

    spdlog::info("        RET FROM ScoreCandidates, idx={}", idx);
}

void Assembler::WorkerThread(int idx) {
    int q_depth;
    int max_wait;

    spdlog::info("    WorkerThread INIT id={}, idx={}", this_thread::get_id(), idx);
    while (workers_active.load()) {
        if (work_queues[idx].empty()) {
            // spdlog::info("    WorkerThread WAIT id={}, idx={}", this_thread::get_id(), idx);
            unique_lock<std::mutex> lk(worker_mutexes[idx]);
            worker_syncs[idx].wait(lk, [&] { return work_pending[idx]; });
            // spdlog::info("    WorkerThread UNBLOCKED id={}, idx={}", this_thread::get_id(), idx);
            lk.unlock();
        }

        n_workers_active++;

        max_wait = 100;
        while (work_queues[idx].empty()) {
            this_thread::yield();
            max_wait--;
            if (max_wait <= 0) {
                break;
            }
        }
        if (max_wait <= 0) {
            spdlog::warn("    WorkerThread MAXWAIT id={}, idx={}", this_thread::get_id(), idx);
            n_workers_active--;
            continue;
        }

        worker_mutexes[idx].lock();
        AssemblyJob *job = work_queues[idx].top();
        work_queues[idx].pop();
        q_depth = work_queues[idx].size();
        if ((q_depth == config->work_queue_low_watermark) && worker_busy[idx]) {
            worker_busy[idx] = false;
            if (n_workers_busy.fetch_sub(1) == n_worker_threads) {
                some_workers_ready.notify_one();
            }
        }
        // avoid busy loop on empty
        else if (q_depth == 0) {
            work_pending[idx] = false;
        }
        worker_mutexes[idx].unlock();

        if (job->job_type == kAssemblerJobInferEmbedding) {
            AssemblerJobInferEmbeddingParams *job_params =
                dynamic_cast<AssemblerJobInferEmbeddingParams *>(job->job_params);
            GeneSeq *gseq = job_params->gseq;
            // spdlog::info("    InferEmbeddings for idx={}", idx);
            kmer_embedding_space->GeneSeqKmersToEmbeddings(gseq, /*worker_id*/ idx);
            geneseqs_embedded++;
        } else if (job->job_type == kAssemblerJobGetApproxKNN) {
            AssemblerJobGetApproxKNNParams *job_params =
                dynamic_cast<AssemblerJobGetApproxKNNParams *>(job->job_params);
            GeneSeq *gseq = job_params->gseq;
            // spdlog::info("    GetApproxKNN for idx={}", idx);
            KmerGetApproxKnn(gseq->emb_data[job_params->kmer_off], config->knn_top_k);
        } else if (job->job_type == kAssemblerJobScoreSubgraph) {
            AssemblerJobScoreSubgraphParams *job_params =
                dynamic_cast<AssemblerJobScoreSubgraphParams *>(job->job_params);
            ScoreSubgraph(job_params, idx);
            subgraphs_scored++;
        } else if (job->job_type == kAssemblerJobScoreCandidates) {
            AssemblerJobScoreCandidatesParams *job_params =
                dynamic_cast<AssemblerJobScoreCandidatesParams *>(job->job_params);
            ScoreCandidates(job_params, idx);
            geneseqs_join_candidates_scored++;
        } else if (job->job_type == kAssemblerJobJoinSubgraphs) {
            AssemblerJobJoinSubgraphsParams *job_params =
                dynamic_cast<AssemblerJobJoinSubgraphsParams *>(job->job_params);
            JoinSubgraphs(job_params, idx);
        }

        delete job->job_params;
        delete job;

        n_workers_active--;
        // spdlog::info("    WorkerThread Iteration id={}, idx={}", this_thread::get_id(), idx);
    }
}

//! n_workers, 0 to autodetect and use all hardware resources
void Assembler::StartWorkerThreads(unsigned int n_workers) {
    lock_guard<mutex> guard(assembler_mutex);

    unsigned int nthreads = n_workers;

    if (n_workers == 0) {
        nthreads = thread::hardware_concurrency() - 1; // save a core for the submission thread
    }

    spdlog::info("StartWorkerThreads n={}", nthreads);
    n_worker_threads = nthreads;
    worker_threads = new thread *[nthreads];
    worker_mutexes = new mutex[nthreads];
    work_pending = new bool[nthreads];
    worker_busy = new bool[nthreads];
    worker_syncs = new condition_variable[nthreads];
    work_queues = new stack<AssemblyJob *>[nthreads];

    workers_active = true;

    for (int i = 0; i < n_worker_threads; i++) {
        work_pending[i] = false;
        worker_busy[i] = false;
        worker_threads[i] = new thread(&Assembler::WorkerThread, this, i);
    }
}

void Assembler::StopWorkerThreads(void) {
    lock_guard<mutex> guard(assembler_mutex);

    if (!workers_active.load()) {
        return;
    }

    workers_active = false;

    for (int i = 0; i < n_worker_threads; i++) {
        spdlog::info("    WorkerThread NOTIFY idx={}", i);
        work_pending[i] = true;
        worker_syncs[i].notify_all();
        worker_threads[i]->join();
        delete worker_threads[i];
    }

    spdlog::info("StopWorkerThreads bottom half n={}", n_worker_threads);
    delete[] worker_threads;
    delete[] work_pending;
    delete[] worker_busy;
    delete[] worker_mutexes;
    delete[] worker_syncs;
    delete[] work_queues;
    n_worker_threads = 0;
}

bool Assembler::GeneSeqToGraph(GeneSeq *seq, GraphDescriptor **graph_desc_out) {
    // assert(*graph_desc_out != NULL);
    *graph_desc_out = NULL;

    GraphDescriptor *graph_desc = new GraphDescriptor;

    GraphNode *n1, *n2;
    GraphEdge *e1;
    GraphEdgeDescriptor ed1;
    bool success;

    graph_desc->start_node = graph_desc->AddNode(&n1);
    graph_desc->end_node = graph_desc->AddNode(&n2);

    success = graph_desc->AddEdge(graph_desc->start_node, graph_desc->end_node, &ed1, &e1);
    e1->seq_slices.push_back(GeneSeqSlice(seq));

    gene_seq_id_to_subgraph_umap[seq->id] = graph_desc;
    subgraph_umap[graph_desc->id] = graph_desc;
    vector<GeneSeq *> *seqs = new vector<GeneSeq *>;
    seqs->push_back(seq);
    subgraph_to_gene_seqs_umap[graph_desc->id] = seqs;
    *graph_desc_out = graph_desc;
    return true;
}

ostream &operator<<(ostream &os, const Assembler &a) {
    os << "Assembler{id=" << a.id;

    if (!a.gene_seq_umap.empty()) {
        os << ",gene_seq_umap={";
        for_each(a.gene_seq_umap.begin(), a.gene_seq_umap.end(),
                 [&](pair<int, GeneSeq *> el) { os << el.first << ","; });
        os << "}";
    }

    if (!a.gene_seq_id_to_subgraph_umap.empty()) {
        os << ",gene_seq_to_subgraph_umap={";
        for_each(a.gene_seq_id_to_subgraph_umap.begin(), a.gene_seq_id_to_subgraph_umap.end(),
                 [&](pair<int, GraphDescriptor *> el) {
                     os << el.first << "->" << el.second->id << ",";
                 });
        os << "}";
    }

    os << "}";

    return os;
}

Assembler::~Assembler() {
    spdlog::info("Assembler destructor.");

    if (config != NULL) {
        delete config;
        config = NULL;
    }

    if (kmer_embedding_space != NULL) {
        delete kmer_embedding_space;
        kmer_embedding_space = NULL;
    }

    for (auto element : kmer_alignment_umap) {
        parasail_result_free(element.second);
    }
    kmer_alignment_umap.clear();
}

//*****************************************************************************
// TEST CASES

TEST_CASE("Assembler no param constructor") {
    init_logger(131072, 3);

    Assembler asmblr;

    spdlog::info("Assembler no param constructor:");
}

TEST_CASE("Assembler with reads provided") {
    vector<char *> reads = {SEQ("GCATGCATGCAT"), SEQ("ATCGATCGATCG")};

    init_logger(131072, 3);
    Assembler asmblr;

    asmblr.StartSync(reads, kGeneSeqTypeDNA);

    spdlog::info("    asmblr: {}", asmblr);
}

TEST_CASE("EMBNet Produce Embeddings") {

    init_logger(131072, 3);
    Assembler asmblr;

    asmblr.StartAsync("./testdata/simsample.fasta", kGeneSeqTypeDNA);

    spdlog::info("    asmblr: {}", asmblr);

    AssemblyStatus asmStatus;

    do {
        asmStatus = asmblr.GetStatus(); // XXXADS need to add a pretty printer here so we can watch progress.a
        fprintf(stderr, "status: %d.%d %d %d %d %d %d %d %d %d\n", asmStatus.state, asmStatus.sub_state,
            asmStatus.geneseqs_expected,
            asmStatus.geneseqs_embedded,
            asmStatus.geneseqs_join_candidates_scored,
            asmStatus.subgraphs_scored,
            asmStatus.compute_embeddings_duration_ms,
            asmStatus.score_candidates_duration_ms,
            asmStatus.initial_subgraph_join_score_duration_ms,
            asmStatus.subgraph_joining_duration_ms);
        spdlog::info("    satutus: {}.{}:  {} {} {} {} {} {} {} {}", asmStatus.state, asmStatus.sub_state,
            asmStatus.geneseqs_expected,
            asmStatus.geneseqs_embedded,
            asmStatus.geneseqs_join_candidates_scored,
            asmStatus.subgraphs_scored,
            asmStatus.compute_embeddings_duration_ms,
            asmStatus.score_candidates_duration_ms,
            asmStatus.initial_subgraph_join_score_duration_ms,
            asmStatus.subgraph_joining_duration_ms);
        std::this_thread::sleep_for(3s);

    } while (asmStatus.state == kAssemblyInProgress);

    CHECK(asmStatus.state == kAssemblyComplete);

    // XXXADS
    //  - cross-check embedding values with those from the jupyter notebook.
    //    include some expected values for test along with the trained model.
}
