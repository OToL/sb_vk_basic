OPTION(SB_ENABLE_SAMPLE_VK_BASIC "Enable Sunburst Vulkan basic sample" OFF)

if(SB_ENABLE_SAMPLE_VK_BASIC)
    project(sb_vk_basic LANGUAGES CXX)

    find_package(Vulkan REQUIRED FATAL_ERROR)

    add_executable(sb_vk_basic
        src/main.cpp
        src/utility_vulkan.cpp
        src/utility.cpp
        ${SB_ENGINE_MEMORY_HOOK_FILE_PATH})
    target_include_directories(sb_vk_basic
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/extern/inc)
    target_compile_options(sb_vk_basic
        PRIVATE
            "/wd5039" 
            "/wd4464"
            "/wd4996"
            "/wd4191"
            "/wd4365"
            "/wd5219"
            "/wd4201"
            "/wd4324"
            "/wd4626"
            "/wd5214")
    target_link_options(sb_vk_basic
        PRIVATE
            "/NODEFAULTLIB:library")
    target_link_libraries(sb_vk_basic 
        PRIVATE
            sb::core
            sb::cxx_common_private
            sbext::glfw
            sbext::glm
            sbext::stb
            sbext::tinyobjloader
            Vulkan::Vulkan)

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
