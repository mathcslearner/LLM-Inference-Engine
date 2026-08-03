add_library(engine_model STATIC model.cpp)
add_library(engine::model ALIAS engine_model)
target_include_directories(engine_model PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_model PRIVATE engine_warnings)
