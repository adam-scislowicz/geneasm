#include "seqemb.h"

CharToStringMap kRNACodeMap = {
    {'A', "Adenine"},     {'C', "Cytosine"},    {'G', "Guanine"},     {'U', "Uracil"},
    {'R', "A or G"},      {'Y', "C or U"},      {'S', "G or C"},      {'W', "A or U"},
    {'K', "G or U"},      {'M', "A or C"},      {'B', "C or G or U"}, {'D', "A or G or U"},
    {'H', "A or C or U"}, {'V', "A or C or G"}, {'N', "any base"},    {'.', "gap"},
    {'-', "gap"},
};

CharToStringMap kDNACodeMap = {
    {'A', "Adenine"},     {'C', "Cytosine"},    {'G', "Guanine"},     {'T', "Thymine"},
    {'R', "A or G"},      {'Y', "C or T"},      {'S', "G or C"},      {'W', "A or T"},
    {'K', "G or T"},      {'M', "A or C"},      {'B', "C or G or T"}, {'D', "A or G or T"},
    {'H', "A or C or T"}, {'V', "A or C or G"}, {'N', "any base"},    {'.', "gap"},
    {'-', "gap"},
};

CharToStringMap kAminoAcidCodeMap = {
    {'A', "Alanine"},       {'C', "Cysteine"},       {'D', "Aspartic Acid"}, {'E', "Glutamic Acid"},
    {'F', "Phenylalanine"}, {'G', "Glycine"},        {'H', "Histidine"},     {'I', "Isoleucine"},
    {'K', "Lysine"},        {'L', "Leucine"},        {'M', "Methionine"},    {'N', "Asparagine"},
    {'P', "Proline"},       {'Q', "Glutamine"},      {'R', "Arginine"},      {'S', "Serine"},
    {'T', "Threonine"},     {'U', "Selenocysteine"}, {'V', "Valine"},        {'W', "Tryptophan"},
    {'Y', "Tyrosine"},
};

CharToStringMap kAminoAcidShortCodeMap = {
    {'A', "Ala"}, {'C', "Cys"}, {'D', "Asp"}, {'E', "Glu"}, {'F', "Phe"}, {'G', "Gly"},
    {'H', "His"}, {'I', "Ile"}, {'K', "Lys"}, {'L', "Leu"}, {'M', "Met"}, {'N', "Asn"},
    {'P', "Pro"}, {'Q', "Gln"}, {'R', "Arg"}, {'S', "Ser"}, {'T', "Thr"}, {'U', "Sec"},
    {'V', "Val"}, {'W', "Trp"}, {'Y', "Tyr"},
};

char *kGENE_SEQ_EPSILON = (char *)NULL;

template <typename OStream> OStream &operator<<(OStream &os, const GeneSeqType &seq_type) {
  switch (seq_type) {
  case kGeneSeqTypeUninitilized:
    os << "kGeneSeqTypeUninitialied";
    break;
  case kGeneSeqTypeRNA:
    os << "kGeneSeqTypeRNA";
    break;
  case kGeneSeqTypeDNA:
    os << "kGeneSeqTypeDNA";
    break;
  case kGeneSeqTypeAminoAcid:
    os << "kGeneSeqTypeAminoAcid";
    break;
  }

  return os;
}

GeneSeq::GeneSeq() { this->get_next_id_(); }

GeneSeq::GeneSeq(char *in_seq_p, GeneSeqType in_seq_type) {
  assert(in_seq_p != NULL);

  this->get_next_id_();

  this->len = strlen(in_seq_p);
  this->data.insert(this->data.begin(), in_seq_p, in_seq_p + this->len);
  this->type = in_seq_type;
}

GeneSeq::GeneSeq(char *in_seq_p, size_t in_seq_len, GeneSeqType in_seq_type) {
  assert(in_seq_p != NULL);

  this->get_next_id_();

  this->len = in_seq_len;
  this->data.insert(this->data.begin(), in_seq_p, in_seq_p + in_seq_len);
  this->type = in_seq_type;
}

ostream &operator<<(ostream &os, const GeneSeq &seq) {
  os << "GeneSeq{id=" << seq.id << ", len=" << seq.len << ", type=" << seq.type << "}";

  return os;
}

GeneSeq::~GeneSeq() {
  if (emb_data != NULL) {
    spdlog::debug("GeneSeq destructor id={}", id);
    delete[] emb_data;
    emb_data = NULL;
  }
}

GeneSeqSlice::GeneSeqSlice() { this->get_next_id_(); }

GeneSeqSlice::GeneSeqSlice(GeneSeq *in_gene_seq) {
  assert(in_gene_seq != NULL);

  this->get_next_id_();

  seqp = in_gene_seq;
  off = 0;
  len = seqp->len;
}

GeneSeqSlice::GeneSeqSlice(GeneSeq *in_gene_seq, size_t off_in, size_t len_in) {
  assert(in_gene_seq != NULL);
  assert(off_in + len_in <= in_gene_seq->len);

  this->get_next_id_();

  seqp = in_gene_seq;
  off = off_in;
  len = len_in;
}

vector<char> *GeneSeqSlice::GetData() { return slice(seqp->data, off, len + off - 1); }

char GeneSeqSlice::GetData(int32_t relative_off) { return seqp->data[off + relative_off]; }

float **GeneSeqSlice::GetEmbData() {
  // XXXADS TODO

  return NULL;
}

GeneSeqSlice::~GeneSeqSlice() {}

KmerToEmbeddingCache::KmerToEmbeddingCache(int max_entries_in, int entry_dim_in,
                                           int save_every_n_entries_in, string save_path_in) {
  this->get_next_id_();

  max_entries = max_entries_in;
  save_every_n_entries = save_every_n_entries_in;
  n_new_entries = 0;
  save_path = save_path_in;
  entry_dim = entry_dim_in;

  kmer_embedding_umap.reserve(max_entries);
}

void KmerToEmbeddingCache::SaveToFile(string path) {}

void KmerToEmbeddingCache::LoadFromFile(string path) {}

#if 0
void NormalizeEmbedding(float *data, int knn_dim, float *norm_array) {
    float norm=0.0f;
    for(int i=0; i < knn_dim; i++)
        norm += data[i]*data[i];
    norm = 1.0f / (sqrtf(norm) + 1e-30f);

    for(int i=0; i < knn_dim; i++)
        norm_array[i]=data[i]*norm;
}
#endif

void NormalizeEmbeddingInPlace(float *data, int knn_dim) {
  float norm = 0.0f;
  for (int i = 0; i < knn_dim; i++)
    norm += data[i] * data[i];
  norm = 1.0f / (sqrtf(norm) + 1e-30f);

  for (int i = 0; i < knn_dim; i++)
    data[i] *= norm;
}

void KmerToEmbeddingCache::InitKmerEmbeddingRecord(KmerEmbeddingRecord *rec) {
  rec->emb_data = new float[entry_dim];
  rec->weight = 0;
}

bool KmerToEmbeddingCache::UpdateIfNotPresent(string kmer_str, float *emb_vec_in) {
  cache_mutex.lock();

  auto it = kmer_embedding_umap.find(kmer_str);
  if (it != kmer_embedding_umap.end()) {
    (*it).second.weight++;
    cache_mutex.unlock();
    return false; // return early, already present. could assert equality but shouldn't be necessary
  }

  InitKmerEmbeddingRecord(&kmer_embedding_umap[kmer_str]);

  // expect pre-normalized data. we want to perform normalization outside of the mutex
  memcpy(kmer_embedding_umap[kmer_str].emb_data, emb_vec_in, sizeof(float) * entry_dim);
  kmer_embedding_umap[kmer_str].weight = 1;

  n_new_entries++;

  cache_mutex.unlock();

  if ((n_new_entries % save_every_n_entries) != 0)
    return true;

  if (save_path.length() == 0) {
    return true;
  }

  save_file_write_mutex.lock();
  // XXXADS TODO save file here.
  save_file_write_mutex.unlock();

  return true;
}

KmerEmbeddingRecord *KmerToEmbeddingCache::Lookup(string kmer_str) {
  cache_mutex.lock();

  auto it = kmer_embedding_umap.find(kmer_str);
  if (it != kmer_embedding_umap.end()) {
    cache_mutex.unlock();
    return &kmer_embedding_umap[kmer_str];
  }

  cache_mutex.unlock();

  return NULL;
}

// clear contents.
void KmerToEmbeddingCache::Reset() {
  eviction_fifo = queue<string>();
  kmer_embedding_umap.clear();

  if (save_path.length() != 0) {
    // delete the save file
  }
}

CharToOnehotMap kDNAToOnehotMap = {
    CHAR2ONEHOT('A', {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
    CHAR2ONEHOT('C', {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
    CHAR2ONEHOT('G', {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
    CHAR2ONEHOT('T', {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
    CHAR2ONEHOT('R', {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
    CHAR2ONEHOT('Y', {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
    CHAR2ONEHOT('S', {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
    CHAR2ONEHOT('W', {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}),
    CHAR2ONEHOT('K', {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0}),
    CHAR2ONEHOT('M', {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0}),
    CHAR2ONEHOT('B', {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}),
    CHAR2ONEHOT('D', {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0}),
    CHAR2ONEHOT('H', {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0}),
    CHAR2ONEHOT('V', {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0}),
    CHAR2ONEHOT('N', {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0}),
    CHAR2ONEHOT('.', {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}),
    CHAR2ONEHOT('-', {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}),
};

CharToOnehotMap kRNAToOnehotMap = {
    CHAR2ONEHOT('A', {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
    CHAR2ONEHOT('C', {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
    CHAR2ONEHOT('G', {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
    CHAR2ONEHOT('U', {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
    CHAR2ONEHOT('R', {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
    CHAR2ONEHOT('Y', {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
    CHAR2ONEHOT('S', {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
    CHAR2ONEHOT('W', {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}),
    CHAR2ONEHOT('K', {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0}),
    CHAR2ONEHOT('M', {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0}),
    CHAR2ONEHOT('B', {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}),
    CHAR2ONEHOT('D', {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0}),
    CHAR2ONEHOT('H', {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0}),
    CHAR2ONEHOT('V', {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0}),
    CHAR2ONEHOT('N', {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0}),
    CHAR2ONEHOT('.', {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}),
    CHAR2ONEHOT('-', {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}),
};

bool KmerEmbeddingSpace::LoadONNXModel(string model_path, int n_worker_threads) {
  lock_guard<mutex> guard(emb_mutex);

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test"); // XXXADS how to integrate with SPDLOG
  Ort::SessionOptions session_options;
  Ort::AllocatorWithDefaultOptions allocator;

#if 0
    OrtCUDAProviderOptions options;
    options.device_id = 0;
    options.arena_extend_strategy = 0; // use -1 to allow ORT to choose the default, 0 = kNextPowerOfTwo, 1 = kSameAsRequested
    //options.gpu_mem_limit = 1L * 1024 * 1024 * 1024;
    options.gpu_mem_limit = 256L * 1024 * 1024;
    options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE;
    options.do_copy_in_default_stream = 1;
    options.user_compute_stream = nullptr;
    options.default_memory_arena_cfg = nullptr;

    //session_options.AppendExecutionProvider_CUDA(options);
#endif

  session_options.SetIntraOpNumThreads(1);

#if 0
    OrtTensorRTProviderOptions trt_options{};
    trt_options.device_id = 0;
    trt_options.trt_max_workspace_size = 1L * 1024 * 1024 *1024;
    trt_options.trt_max_partition_iterations = 10;
    trt_options.trt_min_subgraph_size = 5;
    trt_options.trt_fp16_enable = 0;
    trt_options.trt_int8_enable = 0;
    trt_options.trt_int8_use_native_calibration_table = 1;
    trt_options.trt_engine_cache_enable = 0;
    trt_options.trt_engine_cache_path = "/run";
    trt_options.trt_dump_subgraphs = 1;
    //session_options.AppendExecutionProvider_TensorRT(trt_options);
#endif

  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  assert(n_worker_threads > 0);
  onnx_sessions = new Ort::Session *[n_worker_threads];
  for (int i = 0; i < n_worker_threads; i++) {
    spdlog::info("Ort::Session initialization for idx={}", i);
    try {
      onnx_sessions[i] = new Ort::Session(env, model_path.c_str(), session_options);
    } catch (...) {
      cerr << "XXXADS Ort:Session(): " << boost::current_exception_diagnostic_information() << endl;
      cerr << "SESSION IDX:" << i << endl;
    }
  }

  size_t num_input_nodes = onnx_sessions[0]->GetInputCount();

  for (int i = 0; i < num_input_nodes; i++) {
    char *input_name = onnx_sessions[0]->GetInputName(i, allocator);
    Ort::TypeInfo type_info = onnx_sessions[0]->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    spdlog::info("Input[{}]: type={}\n", i, type);

    input_node_dims = tensor_info.GetShape();
    spdlog::info("Input[{}]: num_dims={}\n", i, input_node_dims.size());
    for (int j = 0; j < input_node_dims.size(); j++) {
      spdlog::info("    dim {}={}\n", j, input_node_dims[j]);
    }

    assert(input_node_dims[1] == kmer_len);
    assert(input_node_dims[2] == alphabet_len);
  }

  memory_info =
      new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

  input_node_dims[0] = 1;
  input_node_names = {"input_1"};
  output_node_names = {"output_1"};

  size_t num_output_nodes = onnx_sessions[0]->GetOutputCount();

  assert(num_output_nodes == 1);
  Ort::TypeInfo type_info = onnx_sessions[0]->GetOutputTypeInfo(0);
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  output_node_dim = tensor_info.GetShape()[1];
  spdlog::info("Output: dim={}\n", output_node_dim);

  return true;
}

void KmerEmbeddingSpace::FreeONNXModel() {
  lock_guard<mutex> guard(emb_mutex);

  if (onnx_sessions != NULL) {
    for (int i = 0; i < n_worker_threads; i++) {
      if (onnx_sessions[i] != NULL) {
        delete onnx_sessions[i];
        onnx_sessions[i] = NULL;
      }
    }
    delete onnx_sessions;
    onnx_sessions = NULL;
  }
  if (memory_info != NULL) {
    delete memory_info;
    memory_info = NULL;
  }
}

//! m = 48-64, ef=?
void KmerEmbeddingSpace::InitKnnSpace(int dim, size_t max_elements, size_t m, size_t ef,
                                      size_t rand_seed) {
  lock_guard<mutex> guard(emb_mutex);

  knn_dim = dim;

  knn_space = new hnswlib::InnerProductSpace(dim);
  alg_hnsw = new hnswlib::HierarchicalNSW<float>(knn_space, max_elements, m, ef, rand_seed);
}

void KmerEmbeddingSpace::FreeKnnSpace(void) {
  lock_guard<mutex> guard(emb_mutex);

  if (knn_space != NULL) {
    delete knn_space;
    knn_space = NULL;
  }

  if (alg_hnsw != NULL) {
    delete alg_hnsw;
    alg_hnsw = NULL;
  }
}

/// XXXADS
/// additional case. if GeneSeq is identical to another geneseq. deep copy
// correct values and update the initial GeneSeq subgraphs list of IDs to
// include self.
void KmerEmbeddingSpace::GeneSeqKmersToEmbeddings(GeneSeq *seq, int worker_id) {
  using rangeless::fn::operators::operator%;

  size_t input_tensor_size = kmer_len * alphabet_len;
  vector<float> input_tensor_values(kmer_len * alphabet_len);
  array<float, ONEHOT_SIZE> *fv;
  Ort::Session *onnx_session = onnx_sessions[worker_id];
  int idx = 0;

  seq->emb_data = new float *[seq->len];
  for (int i = 0; i < seq->len; i++) {
    seq->emb_data[i] = new float[knn_dim];
  }

  // kmer view via rangeless
  seq->data % rangeless::fn::sliding_window(/*win_size*/ kmer_len) %
      rangeless::fn::for_each([&](const auto &kmer) {
        string kmer_str;

        input_tensor_values.clear();

        // kmer to ONEHOT input vector. (functionalize)
        for (const auto &elem : kmer) {
          if (seq->type == kGeneSeqTypeDNA) {
            fv = &kDNAToOnehotMap[elem];
          } else if (seq->type == kGeneSeqTypeRNA) {
            fv = &kRNAToOnehotMap[elem];
          } else {
            spdlog::error("GeneSeqKmersToEmbeddings: unsupported seq type = {}", seq->type);
            throw;
          }

          kmer_str += elem;
          input_tensor_values.insert(input_tensor_values.end(), begin(*fv), end(*fv));
        }

        KmerEmbeddingRecord *rec = kmer_embedding_cache->Lookup(kmer_str);

        AddKmerInstance(kmer_str, MakeSeqKMerID(seq->id, idx));
        spdlog::debug("AddKmerInstance: seq->id={}, kmer_off={}", seq->id, idx);

        // compute if not found.
        if (rec == NULL) {
          Ort::Value input_tensor =
              Ort::Value::CreateTensor<float>(*memory_info, input_tensor_values.data(),
                                              input_tensor_size, input_node_dims.data(), 3);
          assert(input_tensor.IsTensor());

          auto output_tensors = onnx_session->Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                                                  &input_tensor, 1, output_node_names.data(), 1);

          assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

          float *floatarr = output_tensors.front().GetTensorMutableData<float>();

          NormalizeEmbeddingInPlace(floatarr, knn_dim);
          bool first_writer = kmer_embedding_cache->UpdateIfNotPresent(kmer_str, floatarr);
          memcpy(seq->emb_data[idx], floatarr, sizeof(float) * knn_dim);
          seq->emb_data[idx] = floatarr;
          if (first_writer) {
            // only the first writer adds the point to the KNN-space
            alg_hnsw->addPoint(seq->emb_data[idx], MakeSeqKMerID(seq->id, idx));
          }
        } else {
          // found. use cached value
          memcpy(seq->emb_data[idx], rec->emb_data, sizeof(float) * knn_dim);
        }

        idx++;
      });
}

void KmerEmbeddingSpace::AddKmerInstance(string kmer_str, SeqKmerID seq_kmer_id) {
  kmer_instance_umap_mutex.lock();
  kmer_instance_umap[kmer_str].push_back(seq_kmer_id);
  kmer_instance_umap_mutex.unlock();
}

KmerEmbeddingSpace::KmerEmbeddingSpace(int expected_entries, int kmer_len_in, int alphabet_len_in,
                                       int n_worker_threads_in, string seqemb_onnx_path) {

  kmer_len = kmer_len_in;
  alphabet_len = alphabet_len_in;
  n_worker_threads = n_worker_threads_in;

  LoadONNXModel(seqemb_onnx_path.c_str(), n_worker_threads);

  InitKnnSpace(/*dim*/ output_node_dim, /*max_elems*/ expected_entries * 2, /*m*/ 48, /*ef*/ 64,
               /*rand_seed*/ 0);

  kmer_embedding_cache = new KmerToEmbeddingCache(/*max_entries*/ 1024 * 100,
                                                  /*entry_dim*/ knn_dim,
                                                  /*save_every_n_entries*/ 1024,
                                                  /*save_path*/ "embedding_cache.fb");
}

KmerEmbeddingSpace::~KmerEmbeddingSpace() {

  if (kmer_embedding_cache) {
    spdlog::info("New unique embeddings calculate: {}", kmer_embedding_cache->n_new_entries);

    delete kmer_embedding_cache;
    kmer_embedding_cache = NULL;
  }

  FreeKnnSpace();
  FreeONNXModel();
}

//*****************************************************************************
// TEST CASES

TEST_CASE("GeneSeq no param constructor") {
  GeneSeq gseq;

  CHECK(gseq.len == 0);
  CHECK(gseq.type == kGeneSeqTypeUninitilized);

  spdlog::info("GeneSeq no param constructor: {}", gseq);
}

TEST_CASE("GeneSeq null-terminated constructor") {
  char in_seq[] = "ACGU";
  GeneSeq gseq(in_seq, kGeneSeqTypeRNA);

  CHECK(gseq.len == 4);
  CHECK(gseq.type == kGeneSeqTypeRNA);

  spdlog::info("GeneSeq null-terminated constructor: {}", gseq);
}

TEST_CASE("GeneSeq run-length constructor") {
  char in_seq[] = "ACGT";
  size_t in_seq_len = strlen(in_seq);
  GeneSeq gseq(in_seq, in_seq_len, kGeneSeqTypeDNA);

  CHECK(gseq.len == 4);
  CHECK(gseq.type == kGeneSeqTypeDNA);

  spdlog::info("GeneSeq run-length constructor: {}", gseq);
}

TEST_CASE("GeneSeq run-length constructor amino acid type") {
  char in_seq[] = "MAAT";
  size_t in_seq_len = strlen(in_seq);
  GeneSeq gseq(in_seq, in_seq_len, kGeneSeqTypeAminoAcid);

  CHECK(gseq.len == 4);
  CHECK(gseq.type == kGeneSeqTypeAminoAcid);

  string msg("");
  for (int i = 0; i < (gseq.data.size()); i++) {
    msg.push_back((gseq.data)[i]);
  }
  CHECK(msg == "MAAT");

  spdlog::info("GeneSeq run-length constructor amino acid type: {}", gseq);
}

TEST_CASE("GeneSeq kmers as a rangeless sliding_window") {
  spdlog::info("GeneSeq kmers as a rangeless sliding_window");

  using rangeless::fn::operators::operator%;

  char in_seq[] = "ACGUACGUACGUACGUACGU";
  GeneSeq gseq(in_seq, kGeneSeqTypeRNA);

  CHECK(gseq.len == 20);
  CHECK(gseq.type == kGeneSeqTypeRNA);

  gseq.data % rangeless::fn::sliding_window(/*win_size*/ 4) %
      rangeless::fn::for_each([&](const auto &group) {
        string msg("");
        for (const auto &kmer : group) {
          msg.push_back(kmer);
        }
        spdlog::info("    : {}", msg);
      });
}

TEST_CASE("GeneSeq kmers as a range-v3 sliding view") {
  spdlog::info("GeneSeq kmers as a range-v3 sliding view");

  char in_seq[] = "ACGUACGUACGUACGUACGU";
  GeneSeq gseq(in_seq, kGeneSeqTypeRNA);

  CHECK(gseq.len == 20);
  CHECK(gseq.type == kGeneSeqTypeRNA);

  auto kmers = ::ranges::views::sliding(gseq.data, /*win_size*/ 4);
  auto it = kmers.begin();

  for (auto it = kmers.begin(); it != kmers.end(); it++) {
    string msg("");
    for (const auto &kmer : *it) {
      msg.push_back(kmer);
    }
    spdlog::info("    : {}", msg);
  }
}

TEST_CASE("Validate SeqKMerID encode/decode function") {
  spdlog::info("Validate SeqKMerID encode/decode function");

  SeqKmerID skid;
  int seqid;
  int kmer_off;

  skid = MakeSeqKMerID(0, 0);
  GetSeqKmerIDOff(skid, &seqid, &kmer_off);
  CHECK(seqid == 0);
  CHECK(kmer_off == 0);

  skid = MakeSeqKMerID(0, 65535);
  GetSeqKmerIDOff(skid, &seqid, &kmer_off);
  CHECK(seqid == 0);
  CHECK(kmer_off == 65535);

  skid = MakeSeqKMerID(65535, 0);
  GetSeqKmerIDOff(skid, &seqid, &kmer_off);
  CHECK(seqid == 65535);
  CHECK(kmer_off == 0);

  skid = MakeSeqKMerID(65535, 65535);
  GetSeqKmerIDOff(skid, &seqid, &kmer_off);
  CHECK(seqid == 65535);
  CHECK(kmer_off == 65535);
}
