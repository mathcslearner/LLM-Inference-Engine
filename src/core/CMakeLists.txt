add_library(engine_core STATIC logging.cpp)
add_library(engine::core ALIAS engine_core)
target_include_directories(engine_core PUBLIC ${PROJECT_SOURCE_DIR}/src)
# fmt is PUBLIC because logging.h formats messages at the call site; spdlog
# is PRIVATE — it must stay an implementation detail of logging.cpp.
target_link_libraries(engine_core
  PUBLIC fmt::fmt
  PRIVATE engine_warnings spdlog::spdlog)
