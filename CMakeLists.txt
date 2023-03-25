cmake_minimum_required(VERSION 3.4...3.18)

project(gene.native LANGUAGES C CXX)
if (NOT APPLE)
    set(CMAKE_C_STANDARD 17 CACHE STRING "C version selection")
    set(CMAKE_C_STANDARD_REQUIRED ON)
endif()
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ version selection")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	set(DEBUG True)
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
	unset(DEBUG)
else()
	message( FATAL_ERROR "CMAKE_BUILD_TYPE must be defined as either Release or Debug.")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror -D_GNU_SOURCE")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -D_GNU_SOURCE")

if (DEBUG)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
else()
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
	if (PROFILING)
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
	endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")
	if (PROFILING)
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
	endif()
endif()

IF(APPLE)
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
    set(CMAKE_USE_PTHREADS_INIT 1)
ENDIF()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)
find_package(TBB)
find_package(benchmark REQUIRED)
find_package(pybind11 CONFIG)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/parasail)

# XXXADS TODO fork out common ONNX code in the Apple and Linux sections below. Throw error on unsupported platform.
# e.g. the add_custom_command is generic, the setting fo variables in tagret specific.
if (APPLE)
	set(ONNX_LIBS
       	${PROJECT_SOURCE_DIR}/onnxruntime-bin/onnxruntime-osx-x64-1.8.0/lib/libonnxruntime.dylib)
	set(ONNX_INCLUDE_PATH
		${PROJECT_SOURCE_DIR}/onnxruntime-bin/onnxruntime-osx-x64-1.8.0/include)

	add_custom_command(
	OUTPUT ${ONNX_LIBS} ${ONNX_INCLUDE_PATH}
	COMMAND ${CMAKE_COMMAND} -E remove_directory ${PROJECT_SOURCE_DIR}/onnxruntime-bin/onnxruntime-osx-x64-1.8.0
	COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_SOURCE_DIR}/onnxruntime-bin/onnxruntime-osx-x64-1.8.0.tgz
	COMMAND sync
 	WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/onnxruntime-bin
 	DEPENDS ${PROJECT_SOURCE_DIR}/onnxruntime-bin/onnxruntime-osx-x64-1.8.0.tgz
 	COMMENT "unarchiving onnxruntime-osx-x64-1.8.0.tgz"
 	VERBATIM)

	set(TOOL_PREFIX xcrun)
	set(DLIB_PREFIX
		DYLD_LIBRARY_PATH=${PROJECT_SOURCE_DIR}/onnxruntime-bin/onnxruntime-osx-x64-1.8.0/lib)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	set(ONNX_VERSION 1.10.0)
	set(ONNX_BASENAME onnxruntime-linux-x64-${ONNX_VERSION})
	set(ONNX_BINARCHIVE_URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${ONNX_BASENAME}.tgz)
	set(ONNX_LIBS
       	${PROJECT_SOURCE_DIR}/../../../../third-party/${ONNX_BASENAME}/lib/libonnxruntime.so)
	set(ONNX_INCLUDE_PATH
		${PROJECT_SOURCE_DIR}/../../../../third-party/${ONNX_BASENAME}/include)

	file(GLOB ONNX_INCLUDE_FILES
		${ONNX_INCLUDE_PATH}/cpu_provider_factory.h
		${ONNX_INCLUDE_PATH}/onnxruntime_c_api.h
		${ONNX_INCLUDE_PATH}/onnxruntime_cxx_api.h
		${ONNX_INCLUDE_PATH}/onnxruntime_cxx_inline.h
		${ONNX_INCLUDE_PATH}/onnxruntime_run_options_config_keys.h
		${ONNX_INCLUDE_PATH}/onnxruntime_session_options_config_keys.h
		${ONNX_INCLUDE_PATH}/provider_options.h
	)

	add_custom_command(
	COMMAND ${CMAKE_COMMAND} -E remove_directory ${PROJECT_SOURCE_DIR}/../../../../third-party/${ONNX_BASENAME}
	COMMAND wget -c --show-progress ${ONNX_BINARCHIVE_URL}
	COMMAND grep ${ONNX_BASENAME} ${PROJECT_SOURCE_DIR}/../../../../third-party/manifest.sha256 | sha256sum -c --status || exit -1
	COMMAND ${CMAKE_COMMAND} -E tar xzf ${ONNX_BASENAME}.tgz
	COMMAND sync
 	WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/../../../../third-party
 	DEPENDS ${PROJECT_SOURCE_DIR}/../../../../third-party/manifest.sha256
	OUTPUT ${ONNX_LIBS} ${ONNX_INCLUDE_FILES} ${ONNX_INCLUDE_PATH}
 	COMMENT "Fetching and unarchiving ${ONNX_BINARCHIVE_URL}.tgz"
 	VERBATIM)

	set(TOOL_PREFIX)
	set(DLIB_PREFIX
		LD_LIBRARY_PATH=${PROJECT_SOURCE_DIR}/../../../../third-party/${ONNX_BASENAME}/lib)
endif()

add_custom_command(
	COMMAND flatc -c assembler.fbs
	WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
	DEPENDS assembler.fbs
	COMMENT "Flatbuffer compiler building assembler_generated.h from assembler.fbs"
	OUTPUT assembler_generated.h
)

set(COMMON_SOURCES assembler.cpp seqemb.cpp graph.cpp logger.cpp)
set(COMMON_HEADERS assembler.h assembler_generated.h seqemb.h graph.h logger.h ${ONNX_INCLUDE_PATH})
set(INCLUDE_DIRS
    ${Python3_INCLUDE_DIRS}
	${PROJECT_SOURCE_DIR}/rangeless/include
	${PROJECT_SOURCE_DIR}/range-v3/include
	${PROJECT_SOURCE_DIR}/hnswlib
	${PROJECT_SOURCE_DIR}/parasail
	${ONNX_INCLUDE_PATH}
	$ENV{CONDA_PREFIX}/include)

link_directories($ENV{CONDA_PREFIX}/lib)

message("*** START OF ENVIRONMENT DUMP")
execute_process(COMMAND "${CMAKE_COMMAND}" "-E" "environment")

get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()
message("*** END OF ENVIRONMENT DUMP")

pybind11_add_module(native SHARED main.cpp ${COMMON_SOURCES} ${ONNX_INCLUDE_PATH})
target_include_directories(native PRIVATE ${INCLUDE_DIRS})
target_link_libraries(native PRIVATE Threads::Threads tbb ${ONNX_LIBS} spdlog::spdlog parasail fmt::fmt-header-only)

add_executable(tests tests.cpp ${COMMON_SOURCES} ${COMMON_HEADERS})
target_include_directories(tests PRIVATE ${INCLUDE_DIRS})
target_link_libraries(tests Threads::Threads tbb ${ONNX_LIBS} spdlog::spdlog parasail ${Python3_LIBRARIES} fmt::fmt-header-only)

add_custom_target(generate_coverage_report
	COMMENT "Executing benchmarks then tests and generating a coverage report. see *_log.txt and tests-llvm-cov/index.html"
	COMMAND ${DLIB_PREFIX} ./benchmarks | tee benchmarks_log.txt
	COMMAND ${DLIB_PREFIX} ./tests | tee tests_log.txt
	COMMAND ${TOOL_PREFIX} llvm-profdata merge -sparse default.profraw -o default.profdata
	COMMAND rm -rf ./tests-llvm-cov
	COMMAND mkdir ./tests-llvm-cov
	COMMAND ${TOOL_PREFIX} llvm-cov show tests -instr-profile=default.profdata -show-line-counts-or-regions -output-dir=./tests-llvm-cov -format=html
	DEPENDS benchmarks tests)

add_executable(benchmarks benchmarks.cpp ${COMMON_SOURCES} ${COMMON_HEADERS})
target_include_directories(benchmarks PRIVATE ${INCLUDE_DIRS})
target_link_libraries(benchmarks PRIVATE benchmark Threads::Threads tbb ${ONNX_LIBS} spdlog::spdlog parasail ${Python3_LIBRARIES} fmt::fmt-header-only)

# examples
#COMMAND xcrun llvm-cov show tests -instr-profile=default.profdata -show-line-counts-or-regions > cov-annotated.txt
#COMMAND xcrun llvm-cov report tests -instr-profile=default.profdata
