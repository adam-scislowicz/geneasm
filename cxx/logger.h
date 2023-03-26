#ifndef __GENEASM_LOGGER_H
#define __GENEASM_LOGGER_H

#include <spdlog/fmt/bundled/format.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/spdlog.h>

#include <boost/stacktrace.hpp>

void init_logger(size_t max_file_size, size_t max_files);

void set_log_level(int log_level);
int get_log_level(void);

#endif