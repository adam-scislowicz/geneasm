#include "logger.h"
#include <iostream>
#include <string>

void init_logger(size_t max_file_size, size_t max_files) {

  // return early if the default logger has already been set to interoperate with applications
  // calling gataca.native.
  if (spdlog::default_logger()->name().length() != 0) {
    std::cout << "Gataca Native: not reinitializing logging.\n";
    return;
  }

  auto logger =
      spdlog::rotating_logger_mt("gataca_cpp", "logs/rotating.txt", max_file_size, max_files);

  spdlog::set_default_logger(logger);
  spdlog::set_level(spdlog::level::warn);
}

void set_log_level(int log_level) {
  auto logger = spdlog::get("gataca_cpp");
  std::cout << "set_log_level: " << std::to_string(log_level) << "\n";
  logger->set_level((enum spdlog::level::level_enum)log_level);
}

int get_log_level(void) {
  auto logger = spdlog::get("gataca_cpp");
  return (int)logger->level();
}