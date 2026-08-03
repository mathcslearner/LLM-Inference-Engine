add_library(engine_quant STATIC quant.cpp)
add_library(engine::quant ALIAS engine_quant)
target_include_directories(engine_quant PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_quant PRIVATE engine_warnings)
