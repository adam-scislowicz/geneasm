#ifndef _GENEASM_SEQEMB_H
#define _GENEASM_SEQEMB_H

#include <bit>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <utility> // for std::pair
#include <vector>

#include <boost/exception/diagnostic_information.hpp>

#include <fn.hpp> //! rangeless. XXXADS move to rangeless/fn.hp
#include <range/v3/core.hpp>
#include <range/v3/view/sliding.hpp>

#include <parasail.h>

#include "hnswlib/hnswlib.h"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"

#include <doctest/doctest.h>

#include "logger.h"

using namespace std;

enum GeneSeqType {
  kGeneSeqTypeUninitilized,
  kGeneSeqTypeRNA,
  kGeneSeqTypeDNA,
  kGeneSeqTypeAminoAcid
};

typedef map<char, string> CharToStringMap;

#define SEQ(data) ((char *)data)
extern char *kGENE_SEQ_EPSILON;

enum GeneSeqExtendedDataType {
  kGeneSeqExtendedDataTypeUninitilized,
  kGeneSeqExtendedDataTypePhredQuality
};
class ExtendedData {
public:
  GeneSeqExtendedDataType data_type;
};

class PhredQualityData : public ExtendedData {
public:
  vector<uint8_t> qual_vec;
};

template <typename T> vector<T> *slice(vector<T> const &v, int m, int n) {
  auto first = v.cbegin() + m;
  auto last = v.cbegin() + n + 1;

  vector<T> *vec = new vector<T>(first, last);
  return vec;
}

class GeneSeq {
public:
  int id;
  GeneSeq();
  //! constructor using null-terminated char array as input
  GeneSeq(char *in_seq_p, GeneSeqType in_seq_type);
  //! constructor using run-length char array as input
  GeneSeq(char *in_seq_p, size_t in_seq_len, GeneSeqType in_seq_type);
  GeneSeq(const GeneSeq &seq) {
    SPDLOG_ERROR("XXX GeneSeq Copy Constructor Called.\n{}", boost::stacktrace::stacktrace());
  };
  ~GeneSeq();

  friend ostream &operator<<(ostream &os, const GeneSeq &seq);

  // XXXADS we may want a slice reference counter.

  size_t len = 0;
  size_t instances = 1;
  vector<char> data;
  vector<ExtendedData> ext_data;
  float **emb_data = NULL;
  GeneSeqType type = kGeneSeqTypeUninitilized;

private:
  void get_next_id_() {
    static int next_id_ = 1;
    id = next_id_++;
  }
};

class GeneSeqSlice {
public:
  int id;
  GeneSeqSlice();
  GeneSeqSlice(GeneSeq *in_gene_seq);
  GeneSeqSlice(GeneSeq *in_gene_seq, size_t off_in, size_t len_in);
  ~GeneSeqSlice();

  vector<char> *GetData();
  char GetData(int32_t relative_off);
  float **GetEmbData();

  GeneSeq *seqp;
  size_t off;
  size_t len;

private:
  void get_next_id_() {
    static int next_id_ = 1;
    id = next_id_++;
  }
};

#define ONEHOT_SIZE (16)

typedef map<char, array<float, ONEHOT_SIZE>> CharToOnehotMap;

#define CHAR2ONEHOT(CVAL, ...)                                                                     \
  { CVAL, array<float, ONEHOT_SIZE> __VA_ARGS__ }

struct KmerEmbeddingRecord {
  float *emb_data;
  uint32_t weight;
};
typedef struct KmerEmbeddingRecord KmerEmbeddingRecord;
typedef hnswlib::labeltype SeqKmerID;
static_assert(sizeof(SeqKmerID) == sizeof(size_t), "sizes do not match");

typedef unordered_map<string, KmerEmbeddingRecord> KmerToEmbeddingUMap;
typedef unordered_map<string, vector<SeqKmerID>> KmerInstanceUMap;
typedef vector<pair<float, SeqKmerID>> AlignmentCandidates;

class KmerToEmbeddingCache {
public:
  int id;
  int max_entries;
  int entry_dim;
  int n_new_entries;
  int save_every_n_entries;
  string save_path;

  KmerToEmbeddingCache(int max_entries, int entry_dim, int save_every_n_entries,
                       string save_path_in);
  void InitKmerEmbeddingRecord(KmerEmbeddingRecord *rec);
  void SaveToFile(string path);
  void LoadFromFile(string path);

  bool UpdateIfNotPresent(string kmer_str, float *emb_vec);
  KmerEmbeddingRecord *Lookup(string kmer);

private:
  void get_next_id_() {
    static int next_id_ = 1;
    id = next_id_++;
  }
  void Reset(void);
  char onnx_sha384[96];

  mutex save_file_write_mutex;
  mutex cache_mutex; // for the FIFO and the UMAP
  queue<string> eviction_fifo;
  KmerToEmbeddingUMap kmer_embedding_umap;
};

void NormalizeEmbedding(float *data, int knn_dim, float *norm_array);
void NormalizeEmbeddingInPlace(float *data, int knn_dim);

static SeqKmerID MakeSeqKMerID(int seqid, int kmer_off) {
  if constexpr (std::endian::native == std::endian::big) {
    return ((seqid & 0xffff) >> 16) | (kmer_off & 0xffff);
  } else if constexpr (std::endian::native == std::endian::little) {
    return ((seqid & 0xffff) << 16) | (kmer_off & 0xffff);
  }
}

static void GetSeqKmerIDOff(SeqKmerID seqkmer, int *seqid_out, int *kmer_off_out) {
  if constexpr (std::endian::native == std::endian::big) {
    *seqid_out = ((seqkmer << 16) & 0xffff);
  } else if constexpr (std::endian::native == std::endian::little) {
    *seqid_out = ((seqkmer >> 16) & 0xffff);
  }
  *kmer_off_out = (seqkmer & 0xffff);
  return;
}

class KmerEmbeddingSpace {
public:
  void GeneSeqKmersToEmbeddings(GeneSeq *seq, int worker_id);

  bool LoadONNXModel(string model_path, int n_worker_threads);
  void FreeONNXModel();
  void InitKnnSpace(int dim, size_t max_elements, size_t m, size_t ef, size_t rand_seed);
  void FreeKnnSpace(void);

  void AddKmerInstance(string kmer_str, SeqKmerID seq_kmer_id);
  KmerEmbeddingSpace(int expected_entries, int kmer_len_in, int alphabet_len_in,
                     int n_worker_threads_in, string seqemb_onnx_path);
  ~KmerEmbeddingSpace();

private:
  mutex emb_mutex;
  Ort::Session **onnx_sessions = NULL;
  Ort::MemoryInfo *memory_info = NULL;
  vector<int64_t> input_node_dims;
  vector<const char *> input_node_names;
  int64_t output_node_dim;
  vector<const char *> output_node_names;

  int knn_dim;
  int alphabet_len;
  int kmer_len;
  int n_worker_threads;
  hnswlib::InnerProductSpace *knn_space = NULL;
  hnswlib::AlgorithmInterface<float> *alg_hnsw = NULL;

  KmerToEmbeddingCache *kmer_embedding_cache = NULL;

  friend class Assembler;

protected:
  mutex kmer_instance_umap_mutex;
  KmerInstanceUMap kmer_instance_umap;
};

#endif // _GENEASM_SEQEMB_H