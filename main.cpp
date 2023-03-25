#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <spdlog/fmt/bundled/format.h>
#include <spdlog/spdlog.h>

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include "assembler.h"
#include "graph.h"
#include "logger.h"

#include <string>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

int run_unit_tests(void) {
  doctest::Context context;

  context.setOption("abort-after", 5);
  context.setOption("order-by", "name");
  context.setOption("no-breaks", false);
  int res = context.run();

  spdlog::info("doctest result={}.", res);
  return res;
}

PYBIND11_MODULE(native, m) {

  py::object config_dict = (py::object)py::module::import("geneasm").attr("CONFIG").attr("dict");

  // initialize to defaults if not defined.
  size_t max_file_size = 131072;
  size_t max_files = 3;

  if (config_dict.contains("native_log_max_size")) {
    max_file_size = config_dict["native_log_max_size"].cast<size_t>();
  }

  if (config_dict.contains("native_log_max_files")) {
    max_files = config_dict["native_log_max_files"].cast<size_t>();
  }

  init_logger(max_file_size, max_files);

  m.doc() = "native objects";

  m.def("set_spdlog_level", &set_log_level, "set the spdlog level", py::arg("log_level"));
  m.def("get_spdlog_level", &get_log_level, "get the spdlog level");

  m.def("run_unit_tests", &run_unit_tests, "Run the GeneAssembler native unit tests.");

  py::enum_<GeneSeqType>(m, "GeneSeqType")
      .value("kGeneSeqTypeUninitilized", GeneSeqType::kGeneSeqTypeUninitilized)
      .value("kGeneSeqTypeRNA", GeneSeqType::kGeneSeqTypeRNA)
      .value("kGeneSeqTypeDNA", GeneSeqType::kGeneSeqTypeDNA)
      .value("kGeneSeqTypeAminoAcid", GeneSeqType::kGeneSeqTypeAminoAcid)
      .export_values();

  py::class_<AssemblerConfig>(m, "AssemblerConfig")
      .def(py::init<>())
      .def(py::init<py::object>())
      .def_readwrite("max_ntds_per_edge", &AssemblerConfig::max_ntds_per_edge)
      .def_readwrite("kmer_len", &AssemblerConfig::kmer_len)
      .def_readwrite("alphabet_len", &AssemblerConfig::alphabet_len)
      .def_readwrite("work_queue_low_watermark", &AssemblerConfig::work_queue_low_watermark)
      .def_readwrite("work_queue_high_watermark", &AssemblerConfig::work_queue_high_watermark)
      .def_readwrite("knn_top_k", &AssemblerConfig::knn_top_k)
      .def_readwrite("seqemb_onnx_path", &AssemblerConfig::seqemb_onnx_path)
      .def_readwrite("subgraph_join_threshold", &AssemblerConfig::subgraph_join_threshold)
      .def("__repr__", [](const AssemblerConfig &asmCfg) {
        return "<AssemblerConfig: kmer_len='" + to_string(asmCfg.kmer_len) +
               "' work_queue_low_watermark='" + to_string(asmCfg.work_queue_low_watermark) + "'>";
      });

  py::enum_<AssemblyState>(m, "AssemblyState")
      .value("kAssemblyInProgress", AssemblyState::kAssemblyInProgress)
      .value("kAssemblyAborted", AssemblyState::kAssemblyAborted)
      .value("kAssemblyError", AssemblyState::kAssemblyError)
      .value("kAssemblyComplete", AssemblyState::kAssemblyComplete)
      .value("kAssemblyIdle", AssemblyState::kAssemblyIdle)
      .export_values();

  py::enum_<AssemblySubState>(m, "AssemblySubState")
      .value("kAssemblyInitializing", AssemblySubState::kAssemblyInitializing)
      .value("kAssemblyComputingEmbeddings", AssemblySubState::kAssemblyComputingEmbeddings)
      .value("kAssemblyScoringCandidates", AssemblySubState::kAssemblyScoringCandidates)
      .value("kAssemblyInitialSubgraphJoinScoring",
             AssemblySubState::kAssemblyInitialSubgraphJoinScoring)
      .value("kAssemblySubgraphJoining", AssemblySubState::kAssemblySubgraphJoining)
      .value("kAssemblyCleaningUp", AssemblySubState::kAssemblyCleaningUp)
      .export_values();

  py::class_<AssemblyStatus>(m, "AssemblyStatus")
      .def(py::init<>())
      .def_readonly("state", &AssemblyStatus::state)
      .def_readonly("sub_state", &AssemblyStatus::sub_state)
      .def_readonly("geneseqs_expected", &AssemblyStatus::geneseqs_expected)
      .def_readonly("geneseqs_embedded", &AssemblyStatus::geneseqs_embedded)
      .def_readonly("geneseqs_join_candidates_scored",
                    &AssemblyStatus::geneseqs_join_candidates_scored)
      .def_readonly("subgraphs_scored", &AssemblyStatus::subgraphs_scored)
      .def_readonly("compute_embeddings_duration_ms",
                    &AssemblyStatus::compute_embeddings_duration_ms)
      .def_readonly("score_candidates_duration_ms", &AssemblyStatus::score_candidates_duration_ms)
      .def_readonly("initial_subgraph_join_score_duration_ms",
                    &AssemblyStatus::initial_subgraph_join_score_duration_ms)
      .def_readonly("subgraph_joining_duration_ms", &AssemblyStatus::subgraph_joining_duration_ms);

  py::class_<Assembler>(m, "Assembler")
      .def(py::init<>())
      .def(py::init<class AssemblerConfig>())
      .def("StartAsync", py::overload_cast<string, GeneSeqType>(&Assembler::StartAsync))
      .def("Abort", &Assembler::Abort)
      .def("GetStatus", &Assembler::GetStatus)
      .def_readonly("id", &Assembler::id);

  return;

  // m.attr("the_answer") = 42;
  // py::object world = py::cast("World");
  // m.attr("what") = world;

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}