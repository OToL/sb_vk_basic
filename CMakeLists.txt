project(sb_vk_basic LANGUAGES CXX)

sbAddExecutable(sb_vk_basic
    src/main.cpp)

target_link_libraries(sb_vk_basic 
    PRIVATE
        sb::core)

target_include_directories(sb_vk_basic
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/thirdparty/inc)

target_compile_definitions(sb_vk_basic
    PRIVATE 
        GLFW_INCLUDE_VULKAN
        GLM_FORCE_DEPTH_ZERO_TO_ONE)

if(SB_TARGET_MACOS)
    find_package(vulkan REQUIRED)
    find_library(carbon Carbon)
    find_library(cocoa Cocoa)
    find_library(iokit IOKit)
    find_library(quartzCore QuartzCore)

    target_compile_options(sb_vk_basic
        PRIVATE 
            -Wno-zero-as-null-pointer-constant
            -Wno-unused-parameter
            -Wno-unused-variable)

    target_link_libraries(sb_vk_basic 
        PRIVATE
            sb::core
            Vulkan::Vulkan
            ${CMAKE_CURRENT_LIST_DIR}/thirdparty/lib/libglfw3.a
            ${carbon}
            ${cocoa}
            ${iokit}
            ${quartzCore})

elseif(SB_TARGET_WINDOWS)

    find_package(vulkan REQUIRED)

    target_include_directories(sb_vk_basic
        PRIVATE
            ${Vulkan_INCLUDE_DIRS})

    target_link_libraries(sb_vk_basic 
        PRIVATE
            sb::core
            Vulkan::Vulkan
            ${CMAKE_CURRENT_LIST_DIR}/thirdparty/lib/glfw3.lib)
else()
    message(FATAL_ERROR "Unsupported platform: ${SB_TARGET_OS}")
endif()
