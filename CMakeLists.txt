OPTION(SB_ENABLE_SAMPLE_VK_BASIC "Enable Sunburst Vulkan basic sample" OFF)

if(SB_ENABLE_SAMPLE_VK_BASIC)
    project(sb_vk_basic LANGUAGES CXX)
    
    add_executable(sb_vk_basic
        src/main.cpp
        ${SB_ENGINE_MEMORY_HOOK_FILE_PATH})

    target_link_libraries(sb_vk_basic 
        PRIVATE
            sb::core
            sb::cxx_common_private)

    sb_copy_target_shared_dependencies(sb_vk_basic)
endif()


# project(sb_vk_basic LANGUAGES CXX)

# sbAddExecutable(sb_vk_basic
#     src/main.cpp)

# target_link_libraries(sb_vk_basic 
#     PRIVATE
#         sb::core)

# target_include_directories(sb_vk_basic
#     PRIVATE
#         ${CMAKE_CURRENT_LIST_DIR}/thirdparty/inc)

# target_compile_definitions(sb_vk_basic
#     PRIVATE 
#         GLFW_INCLUDE_VULKAN
#         GLM_FORCE_DEPTH_ZERO_TO_ONE)
