#include "utility_vulkan.h"
#include "utility.h"

#include <sb_core/core.h>
#include <sb_core/error/error.h>
#include <sb_core/log.h>
#include <sb_core/string/string_format.h>
#include <sb_core/container/small_array.h>
#include <sb_core/container/fix_array.h>
#include <sb_core/io/virtual_file_system.h>
#include <sb_core/io/file_stream.h>
#include <sb_core/io/path.h>
#include <sb_core/os.h>
#include <sb_core/memory/global_heap.h>

#include <sb_std/cstdlib>
#include <sb_std/algorithm>
#include <sb_std/span>
#include <sb_std/iterator>

#include <glm/glm.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>

#include <glfw/glfw3.h>

#include <tiny_obj_loader.h>

#include <stb_image.h>

#include <vulkan/vulkan.h>

#include <chrono>

using namespace sb;

class VulkanApp
{
public:
    enum class DemoMode
    {
        TRIANGLE,
        QUAD,
        MODEL
    };

    VulkanApp() = default;
    ~VulkanApp() = default;

    b8 initialize(b8 enable_dbg_layers, GLFWwindow * wnd, DemoMode mode);
    void terminate();
    b8 render();

    void notifyTargetFrameBufferResized(VkExtent2D frame_buffer_ext);

private:
    struct Vertex
    {
        glm::vec3 position;
        glm::vec3 color;
        glm::vec2 tex_coords;
    };

    struct DemoModel
    {
        VkImageMem image;
        VkImageView image_view;
        VkBufferMem vb;
        VkBufferMem ib;
        usize vtx_cnt;
        usize idx_cnt;
        u32 mip_cnt;
    };

    struct UniformMVP
    {
        glm::mat4 model;
        glm::mat4 view;
        glm::mat4 projection;
    };

    b8 initializeVulkanCore(GLFWwindow * wnd);
    void terminateVulkanCore();

    b8 createSwapChain(VkExtent2D frame_buffer_ext);
    b8 createGraphicsPipeline();

    b8 createTriangle();
    void destroyTriangle();

    b8 createQuad();
    void destroyQuad();

    b8 isDeviceSuitable(VkPhysicalDevice device, sbstd::span<char const * const> required_exts,
                        EnumMask<VkQueueFamilyFeature> queue_features);

    b8 createUniformBuffers();
    void destroyUniformBuffers();

    b8 loadTestTexture();
    void unloadTestTexture();

    b8 loadModel();
    void unloadModel();

    b8 createDepthImage();
    void destroyDepthImage();

    b8 createColorImage();
    void destroyColorImage();

    b8 createDescriptors();
    void destroyDescriptors();

    b8 createCommandBuffers();
    b8 createFrameBuffers();

    void cleanupSwapChainRelatedData();
    void recreateSwapChainRelatedData(VkExtent2D frame_buffer_ext);

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugVulkanCallback(VkDebugUtilsMessageSeverityFlagBitsEXT msg_severity,
                                                              VkDebugUtilsMessageTypeFlagsEXT msg_type,
                                                              VkDebugUtilsMessengerCallbackDataEXT const * data,
                                                              void * user_data);

    static constexpr u32 MAX_INFLIGHT_FRAMES = 2;

    b8 _enable_dbg_layers = false;
    VkSampleCountFlagBits _vk_sample_count = VK_SAMPLE_COUNT_1_BIT;
    VkInstance _vk_instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT _vk_messenger = VK_NULL_HANDLE;
    VkPhysicalDevice _vk_phys_device = VK_NULL_HANDLE;
    VkDevice _vk_device = VK_NULL_HANDLE;
    VkQueueFamilyIndices _queue_families = {};
    VkQueue _vk_graphics_queue = VK_NULL_HANDLE;
    VkQueue _vk_present_queue = VK_NULL_HANDLE;
    VkSurfaceKHR _vk_wnd_surface = VK_NULL_HANDLE;
    VkSwapchainKHR _vk_swapchain = VK_NULL_HANDLE;
    DArray<VkImage> _vk_swapchain_imgs;
    DArray<VkImageView> _vk_swapchain_imgs_view;
    VkExtent2D _vk_swapchain_ext = {};
    VkFormat _vk_swapchain_fmt = VK_FORMAT_UNDEFINED;
    VkPipelineLayout _vk_pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout _vk_desc_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool _vk_desc_pool = VK_NULL_HANDLE;
    DArray<VkDescriptorSet> _vk_desc_sets;
    VkRenderPass _vk_render_pass = VK_NULL_HANDLE;
    VkPipeline _vk_graphics_pipeline = VK_NULL_HANDLE;
    DArray<VkFramebuffer> _vk_frame_buffers;
    VkCommandPool _vk_graphics_cmd_pool = VK_NULL_HANDLE;
    DArray<VkCommandBuffer> _vk_cmd_buffers;
    VkSemaphore _vk_image_available_sems[MAX_INFLIGHT_FRAMES] = {};
    VkSemaphore _vk_render_finished_sems[MAX_INFLIGHT_FRAMES] = {};
    VkFence _vk_inflight_fences[MAX_INFLIGHT_FRAMES] = {};
    DArray<VkFence> _vk_inuse_fences;

    VkImageMem _vk_color_image = {};
    VkImageView _vk_color_image_view = VK_NULL_HANDLE;

    VkImageMem _vk_test_texture = {};
    VkImageView _vk_test_texture_view = VK_NULL_HANDLE;
    VkSampler _vk_test_sampler = VK_NULL_HANDLE;

    DemoModel _model = {};

    VkFormat _vk_depth_fmt = VK_FORMAT_UNDEFINED;
    VkImageMem _vk_depth_image = {};
    VkImageView _vk_depth_image_view = VK_NULL_HANDLE;

    VkBuffer _vk_triangle_vb = VK_NULL_HANDLE;
    VkDeviceMemory _vk_triangle_vb_mem = VK_NULL_HANDLE;

    VkBuffer _vk_quad_vb = VK_NULL_HANDLE;
    VkDeviceMemory _vk_quad_vb_mem = VK_NULL_HANDLE;
    VkBuffer _vk_quad_ib = VK_NULL_HANDLE;
    VkDeviceMemory _vk_quad_ib_mem = VK_NULL_HANDLE;

    DArray<VkBufferMem> _vk_mvp_buffers;

    VkExtent2D _target_frame_buffer_ext = {};
    u32 _current_frame = 0U;
    DemoMode _demo_mode = DemoMode::TRIANGLE;
    std::chrono::high_resolution_clock::time_point _start_time;

    VkVertexInputBindingDescription _vk_vertex_binding_desc = {};
    VkVertexInputAttributeDescription _vk_vertex_attributes_desc[3] = {};
};
VKAPI_ATTR VkBool32 VKAPI_CALL VulkanApp::debugVulkanCallback(VkDebugUtilsMessageSeverityFlagBitsEXT msg_severity,
                                                              VkDebugUtilsMessageTypeFlagsEXT msg_type,
                                                              VkDebugUtilsMessengerCallbackDataEXT const * data, void *)
{
    char const * type_str = "N/A";

    if (msg_type & VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT)
    {
        type_str = "VK_GENERAL";
    }
    else if (msg_type & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT)
    {
        type_str = "VK_PERFORMANCE";
    }
    else if (msg_type & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT)
    {
        type_str = "VK_VALIDATION";
    }

    if (msg_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT)
    {
        sbLogD("[{}] {}", type_str, data->pMessage);
    }
    else if (msg_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
    {
        sbLogI("[{}] {}", type_str, data->pMessage);
    }
    else if (msg_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    {
        sbLogW("[{}] {}", type_str, data->pMessage);
    }
    else if (msg_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    {
        sbLogE(data->pMessage);
    }
    else
    {
        sbWarn(false, "Unknown message severity");
        sbLogI(data->pMessage);
    }

    return VK_FALSE;
}

void VulkanApp::notifyTargetFrameBufferResized(VkExtent2D frame_buffer_ext)
{
    _target_frame_buffer_ext = frame_buffer_ext;
}

b8 VulkanApp::initializeVulkanCore(GLFWwindow * wnd)
{
    SArray<VkExtensionProperties, 20> exts;
    u32 ext_cnt = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_cnt, nullptr);
    exts.resize(ext_cnt);
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_cnt, exts.data());

    sbLogI("Vulkan Instance extensions:");
    for (auto const & ext_props : exts)
    {
        sbLogI("\t- {}", ext_props.extensionName);
    }

    char const * const REQUIRED_VK_LAYERS[] = {"VK_LAYER_KHRONOS_validation"};
    u32 layer_cnt = 0;
    SArray<VkLayerProperties, 20> layers;

    vkEnumerateInstanceLayerProperties(&layer_cnt, nullptr);
    layers.resize(layer_cnt);
    vkEnumerateInstanceLayerProperties(&layer_cnt, layers.data());

    sbLogI("Vulkan layers:");
    for (auto const & layer_props : layers)
    {
        sbLogI("\t- {}", layer_props.layerName);
    }

    for (char const * req_layer : REQUIRED_VK_LAYERS)
    {
        auto const layer_iter = sbstd::find_if(begin(layers), end(layers), [req_layer](auto const & layer_props) {
            return strcmpi(req_layer, layer_props.layerName) == 0;
        });

        if (layer_iter == end(layers))
        {
            sbLogE("Cannot find Vulkan layer {}", req_layer);
            return false;
        }
    }

    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Vulkan";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "Sunburst";
    app_info.engineVersion = VK_MAKE_VERSION(0, 0, 1);
    app_info.apiVersion = VK_API_VERSION_1_1;

    u32 glfw_ext_cnt = 0;
    const char ** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_cnt);
    SArray<char const *, 20> req_exts(glfw_exts, glfw_exts + glfw_ext_cnt);

    if (_enable_dbg_layers)
    {
        req_exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkDebugUtilsMessengerCreateInfoEXT messenger_info = {};
    messenger_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    messenger_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    messenger_info.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    messenger_info.pfnUserCallback = &debugVulkanCallback;
    messenger_info.pUserData = nullptr;

    VkInstanceCreateInfo instance_info = {};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;
    instance_info.enabledExtensionCount = numericConv<u32>(req_exts.size());
    instance_info.ppEnabledExtensionNames = req_exts.data();

    if (_enable_dbg_layers)
    {
        instance_info.pNext = &messenger_info;
        instance_info.enabledLayerCount = numericConv<u32>(sbstd::size(REQUIRED_VK_LAYERS));
        instance_info.ppEnabledLayerNames = sbstd::data(REQUIRED_VK_LAYERS);
    }
    else
    {
        instance_info.enabledLayerCount = 0;
    }

    VkResult vk_res = vkCreateInstance(&instance_info, nullptr, &_vk_instance);
    if (vk_res != VK_SUCCESS)
    {
        sbLogE("Failed to create Vulkan instance (error = {})", getEnumValue(vk_res));
        return false;
    }

    if (_enable_dbg_layers)
    {
        if (VK_SUCCESS != sb::createVkDebugUtilsMessenger(_vk_instance, &messenger_info, nullptr, &_vk_messenger))
        {
            return false;
        }
    }

    vk_res = glfwCreateWindowSurface(_vk_instance, wnd, nullptr, &_vk_wnd_surface);
    if (VK_SUCCESS != vk_res)
    {
        sbLogE("Failed to create Vulkan Window Surface (error = '{}'", getEnumValue(vk_res));
    }

    u32 phys_device_cnt = 0;
    SArray<VkPhysicalDevice, 5> phys_devices;
    vkEnumeratePhysicalDevices(_vk_instance, &phys_device_cnt, nullptr);
    sbAssert(0 != phys_device_cnt);

    phys_devices.resize(phys_device_cnt);
    vkEnumeratePhysicalDevices(_vk_instance, &phys_device_cnt, phys_devices.data());

    VkQueueFamilyIndices best_queue_desc = {};
    VkPhysicalDeviceProperties best_props;
    char const * const required_device_extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    EnumMask<VkQueueFamilyFeature> required_queue_features =
        makeEnumMask(VkQueueFamilyFeature::GRAPHICS, VkQueueFamilyFeature::PRESENT);

    sbLogI("Vulkan physical devices:");
    if (1 == phys_device_cnt)
    {
        if (!isDeviceSuitable(phys_devices[0], required_device_extensions, required_queue_features))
        {
            sbLogE("The Physical Device does not support required features");
            return false;
        }

        _vk_phys_device = phys_devices[0];
        vkGetPhysicalDeviceProperties(_vk_phys_device, &best_props);

        best_queue_desc = getVkQueueFamilyIndicies(_vk_phys_device, &_vk_wnd_surface);
    }
    else
    {
        int best_score = -1;

        for (auto const phys_device : phys_devices)
        {
            int score = 0;

            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(phys_device, &props);

            sbLogI("\t- {}", props.deviceName);

            if (!isDeviceSuitable(phys_device, required_device_extensions, required_queue_features))
            {
                continue;
            }

            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            {
                score += 1000;
            }

            score += props.limits.maxImageDimension2D;

            if (score > best_score)
            {
                auto const queue_families = getVkQueueFamilyIndicies(phys_device, &_vk_wnd_surface);
                _vk_phys_device = phys_device;

                best_score = score;
                best_props = props;
                best_queue_desc = queue_families;
            }
        }
    }

    sbAssert(_vk_phys_device != VK_NULL_HANDLE);
    sbLogI("Physical device '{}' has been selected", best_props.deviceName);

    _queue_families = best_queue_desc;

    f32 queue_priority = 1.f;
    SArray<VkDeviceQueueCreateInfo, 5> queues_info;

    u32 queue_create_mask = 0;
    VkQueueFamilyIndex const family_indices[2] = {best_queue_desc.graphics, best_queue_desc.present};

    for (auto const queue_family_idx : family_indices)
    {
        if (0 == ((1 << queue_family_idx) & queue_create_mask))
        {
            VkDeviceQueueCreateInfo queue_info = {};

            queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queue_info.queueCount = 1;
            queue_info.queueFamilyIndex = queue_family_idx;
            queue_info.pQueuePriorities = &queue_priority;
            queues_info.push_back(queue_info);

            queue_create_mask |= 1 << queue_family_idx;
        }
    }

    VkPhysicalDeviceFeatures device_features = {};
    device_features.samplerAnisotropy = VK_TRUE;

    VkDeviceCreateInfo device_info = {};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pQueueCreateInfos = sbstd::data(queues_info);
    device_info.queueCreateInfoCount = numericConv<u32>(queues_info.size());
    device_info.pEnabledFeatures = &device_features;
    device_info.ppEnabledExtensionNames = required_device_extensions;
    device_info.enabledExtensionCount = numericConv<u32>(sbstd::size(required_device_extensions));

    if (_enable_dbg_layers)
    {
        // this is deperecated in new Vulkan versions
        // debug layers can be only specified at VkInstance creation
        device_info.enabledLayerCount = numericConv<u32>(sbstd::size(REQUIRED_VK_LAYERS));
        device_info.ppEnabledLayerNames = sbstd::data(REQUIRED_VK_LAYERS);
    }
    else
    {
        instance_info.enabledLayerCount = 0;
    }

    vk_res = vkCreateDevice(_vk_phys_device, &device_info, nullptr, &_vk_device);
    if (VK_SUCCESS != vk_res)
    {
        sbLogE("Failed to create Vulkan Device (error = '{}')", getEnumValue(vk_res));
        return false;
    }

    vkGetDeviceQueue(_vk_device, best_queue_desc.graphics, 0, &_vk_graphics_queue);
    if (VK_NULL_HANDLE == _vk_graphics_queue)
    {
        sbLogE("Failed to acquire graphics queue from the Vulkan Device");
        return false;
    }

    vkGetDeviceQueue(_vk_device, best_queue_desc.present, 0, &_vk_present_queue);
    if (VK_NULL_HANDLE == _vk_present_queue)
    {
        sbLogE("Failed to acquire present queue from the Vulkan Device");
        return false;
    }

    VkPhysicalDeviceProperties phys_device_props = {};
    vkGetPhysicalDeviceProperties(_vk_phys_device, &phys_device_props);
    auto const sample_cnt =
        phys_device_props.limits.framebufferColorSampleCounts & phys_device_props.limits.framebufferDepthSampleCounts;

    if (sample_cnt & VK_SAMPLE_COUNT_64_BIT)
    {
        _vk_sample_count = VK_SAMPLE_COUNT_64_BIT;
    }
    else if (sample_cnt & VK_SAMPLE_COUNT_32_BIT)
    {
        _vk_sample_count = VK_SAMPLE_COUNT_32_BIT;
    }
    else if (sample_cnt & VK_SAMPLE_COUNT_16_BIT)
    {
        _vk_sample_count = VK_SAMPLE_COUNT_16_BIT;
    }
    else if (sample_cnt & VK_SAMPLE_COUNT_8_BIT)
    {
        _vk_sample_count = VK_SAMPLE_COUNT_8_BIT;
    }
    else if (sample_cnt & VK_SAMPLE_COUNT_4_BIT)
    {
        _vk_sample_count = VK_SAMPLE_COUNT_4_BIT;
    }
    else if (sample_cnt & VK_SAMPLE_COUNT_2_BIT)
    {
        _vk_sample_count = VK_SAMPLE_COUNT_2_BIT;
    }
    else
    {
        _vk_sample_count = VK_SAMPLE_COUNT_1_BIT;
    }

    int width, height = 0;
    glfwGetFramebufferSize(wnd, &width, &height);

    if (!createSwapChain({numericConv<u32>(width), numericConv<u32>(height)}))
    {
        sbLogE("Failed to create Vulkan swapchain");
        return false;
    }

    VkCommandPoolCreateInfo cmd_pool_info = {};
    cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmd_pool_info.queueFamilyIndex = best_queue_desc.graphics;
    cmd_pool_info.flags = 0;

    vk_res = vkCreateCommandPool(_vk_device, &cmd_pool_info, nullptr, &_vk_graphics_cmd_pool);
    if (VK_SUCCESS != vk_res)
    {
        sbLogE("Failed to create Vulkan graphics command queue (error = '{}')", getEnumValue(vk_res));
        return false;
    }

    VkSemaphoreCreateInfo sem_info = {};
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sem_info.flags = 0;

    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (u32 sem_idx = 0; MAX_INFLIGHT_FRAMES != sem_idx; ++sem_idx)
    {
        if ((VK_SUCCESS != vkCreateSemaphore(_vk_device, &sem_info, nullptr, &_vk_image_available_sems[sem_idx])) ||
            (VK_SUCCESS != vkCreateSemaphore(_vk_device, &sem_info, nullptr, &_vk_render_finished_sems[sem_idx])))
        {
            sbLogE("Failed to create Vulkan sync semaphores");
            return false;
        }

        if (VK_SUCCESS != vkCreateFence(_vk_device, &fence_info, nullptr, &_vk_inflight_fences[sem_idx]))
        {
            sbLogE("Failed to create Vulkan sync fences");
            return false;
        }
    }

    _vk_inuse_fences.resize(_vk_swapchain_imgs.size());

    return true;
}

b8 VulkanApp::isDeviceSuitable(VkPhysicalDevice phys_device, sbstd::span<char const * const> required_exts,
                               EnumMask<VkQueueFamilyFeature> queue_features)
{
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(phys_device, &features);
    if (!features.geometryShader || !features.samplerAnisotropy)
    {
        return false;
    }

    if (!checkDeviceExtensionsSupport(phys_device, required_exts))
    {
        return false;
    }

    auto const queue_families = getVkQueueFamilyIndicies(phys_device, &_vk_wnd_surface);
    if (enummask_checkValues(queue_families.families, queue_features))
    {
        return false;
    }

    auto const surface_swapchain_props = getVkSurfaceSwapChainProperties(phys_device, _vk_wnd_surface);
    if (surface_swapchain_props.formats.empty() || surface_swapchain_props.present_modes.empty())
    {
        return false;
    }

    return true;
}

void VulkanApp::terminateVulkanCore()
{
    // vkFreeCommandBuffers(_vk_device, _vk_graphics_cmd_pool, (u32)_vk_cmd_buffers.size(), _vk_cmd_buffers.data());
    // _vk_cmd_buffers.clear();

    vkDestroyPipeline(_vk_device, _vk_graphics_pipeline, nullptr);
    _vk_graphics_pipeline = VK_NULL_HANDLE;
    vkDestroyPipelineLayout(_vk_device, _vk_pipeline_layout, nullptr);
    _vk_pipeline_layout = VK_NULL_HANDLE;
    vkDestroyRenderPass(_vk_device, _vk_render_pass, nullptr);
    _vk_render_pass = VK_NULL_HANDLE;

    if (_vk_desc_set_layout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(_vk_device, _vk_desc_set_layout, nullptr);
        _vk_desc_set_layout = VK_NULL_HANDLE;
    }

    for (u32 sem_idx = 0; sem_idx != MAX_INFLIGHT_FRAMES; ++sem_idx)
    {
        if (VK_NULL_HANDLE != _vk_image_available_sems[sem_idx])
        {
            vkDestroySemaphore(_vk_device, _vk_image_available_sems[sem_idx], nullptr);
        }

        if (VK_NULL_HANDLE != _vk_render_finished_sems[sem_idx])
        {
            vkDestroySemaphore(_vk_device, _vk_render_finished_sems[sem_idx], nullptr);
        }

        if (VK_NULL_HANDLE != _vk_inflight_fences[sem_idx])
        {
            vkDestroyFence(_vk_device, _vk_inflight_fences[sem_idx], nullptr);
        }

        _vk_image_available_sems[sem_idx] = VK_NULL_HANDLE;
        _vk_render_finished_sems[sem_idx] = VK_NULL_HANDLE;
        _vk_inflight_fences[sem_idx] = VK_NULL_HANDLE;
    }

    if (VK_NULL_HANDLE != _vk_graphics_cmd_pool)
    {
        vkDestroyCommandPool(_vk_device, _vk_graphics_cmd_pool, nullptr);
        _vk_graphics_cmd_pool = VK_NULL_HANDLE;
    }

    _vk_graphics_queue = VK_NULL_HANDLE;
    _vk_present_queue = VK_NULL_HANDLE;

    if (VK_NULL_HANDLE != _vk_device)
    {
        vkDestroyDevice(_vk_device, nullptr);
        _vk_device = VK_NULL_HANDLE;
    }

    if (VK_NULL_HANDLE != _vk_wnd_surface)
    {
        vkDestroySurfaceKHR(_vk_instance, _vk_wnd_surface, nullptr);
        _vk_wnd_surface = VK_NULL_HANDLE;
    }

    if (VK_NULL_HANDLE != _vk_messenger)
    {
        destroyVkDebugUtilsMessenger(_vk_instance, _vk_messenger, nullptr);
        _vk_messenger = VK_NULL_HANDLE;
    }

    if (VK_NULL_HANDLE != _vk_instance)
    {
        vkDestroyInstance(_vk_instance, nullptr);
        _vk_instance = VK_NULL_HANDLE;
    }
}

void VulkanApp::cleanupSwapChainRelatedData()
{
    for (auto frame_buffer : _vk_frame_buffers)
    {
        vkDestroyFramebuffer(_vk_device, frame_buffer, nullptr);
    }
    _vk_frame_buffers.clear();

    for (auto image_view : _vk_swapchain_imgs_view)
    {
        vkDestroyImageView(_vk_device, image_view, nullptr);
    }
    _vk_swapchain_imgs_view.clear();
    _vk_swapchain_imgs.clear();

    vkDestroySwapchainKHR(_vk_device, _vk_swapchain, nullptr);

    destroyColorImage();
    destroyDepthImage();
}

b8 VulkanApp::createQuad()
{
    Vertex const quad_data[] = {{{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
                                {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
                                {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
                                {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
                                {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
                                {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
                                {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
                                {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}}};
    u32 const quad_indices[] = {0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4};
    VkDeviceSize const ib_size = sizeof(quad_indices);
    VkDeviceSize const vb_size = sizeof(quad_data);

    {
        VkBufferMem final_ib_mem;
        auto vk_res = createVkBuffer(_vk_phys_device, _vk_device, ib_size,
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &final_ib_mem);

        if (VK_SUCCESS != vk_res)
        {
            sbLogE("Failed to create Vulkan final quad IB (error = '{}')", getEnumValue(vk_res));
            return false;
        }

        _vk_quad_ib = final_ib_mem.buffer;
        _vk_quad_ib_mem = final_ib_mem.memory;

        vk_res = uploadVkBufferDataToDevice(_vk_phys_device, _vk_device, (void *)sbstd::data(quad_indices), ib_size,
                                            _vk_graphics_cmd_pool, _vk_graphics_queue, _vk_quad_ib);
        if (VK_SUCCESS != vk_res)
        {
            sbLogE("Failed to upload Vulkan quad data (error = '{}')", getEnumValue(vk_res));
            return false;
        }
    }

    {
        VkBufferMem final_vb_mem;
        auto vk_res = createVkBuffer(_vk_phys_device, _vk_device, vb_size,
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &final_vb_mem);

        if (VK_SUCCESS != vk_res)
        {
            sbLogE("Failed to create Vulkan final quad VB (error = '{}')", getEnumValue(vk_res));
            return false;
        }

        _vk_quad_vb = final_vb_mem.buffer;
        _vk_quad_vb_mem = final_vb_mem.memory;

        vk_res = uploadVkBufferDataToDevice(_vk_phys_device, _vk_device, (void *)sbstd::data(quad_data), vb_size,
                                            _vk_graphics_cmd_pool, _vk_graphics_queue, _vk_quad_vb);
        if (VK_SUCCESS != vk_res)
        {
            sbLogE("Failed to upload Vulkan quad data (error = '{}')", getEnumValue(vk_res));
            return false;
        }
    }

    return true;
}

void VulkanApp::destroyQuad()
{
    if (_vk_quad_ib_mem != VK_NULL_HANDLE)
    {
        vkFreeMemory(_vk_device, _vk_quad_ib_mem, nullptr);
        _vk_quad_ib_mem = VK_NULL_HANDLE;
    }

    if (_vk_quad_vb_mem != VK_NULL_HANDLE)
    {
        vkFreeMemory(_vk_device, _vk_quad_vb_mem, nullptr);
        _vk_quad_vb_mem = VK_NULL_HANDLE;
    }

    if (_vk_quad_ib != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(_vk_device, _vk_quad_ib, nullptr);
        _vk_quad_ib = VK_NULL_HANDLE;
    }

    if (_vk_quad_vb != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(_vk_device, _vk_quad_vb, nullptr);
        _vk_quad_vb = VK_NULL_HANDLE;
    }
}

b8 VulkanApp::createTriangle()
{
    Vertex const triangle_data[] = {
        {{-0.5f, 0.5f, 0.f}, {1.f, 0.f, 0.f}, {0.0, 1.0}},  {{0.f, -0.5f, 0.f}, {0.f, 1.f, 0.f}, {0.5, 0.0}},
        {{0.5f, 0.5f, 0.f}, {0.f, 0.f, 1.f}, {1.0, 1.0}},   {{-0.5f, 0.5f, -0.5f}, {1.f, 0.f, 0.f}, {0.0, 1.0}},
        {{0.f, -0.5f, -0.5f}, {0.f, 1.f, 0.f}, {0.5, 0.0}}, {{0.5f, 0.5f, -0.5f}, {0.f, 0.f, 1.f}, {1.0, 1.0}}};

    auto const triangle_data_size = sizeof(triangle_data);

    VkBufferMem final_buffer_mem = {};
    auto vk_res = createVkBuffer(_vk_phys_device, _vk_device, triangle_data_size,
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &final_buffer_mem);

    if (VK_SUCCESS != vk_res)
    {
        sbLogE("Failed to create Vulkan final buffer for test triangle (error = '{}')", getEnumValue(vk_res));
        return false;
    }

    _vk_triangle_vb = final_buffer_mem.buffer;
    _vk_triangle_vb_mem = final_buffer_mem.memory;

    vk_res = uploadVkBufferDataToDevice(_vk_phys_device, _vk_device, (void *)sbstd::data(triangle_data),
                                        triangle_data_size, _vk_graphics_cmd_pool, _vk_graphics_queue, _vk_triangle_vb);

    if (VK_SUCCESS != vk_res)
    {
        sbLogE("Failed to upload Vulkan triangle data to the device (error = '{}')", getEnumValue(vk_res));
        return false;
    }

    return true;
}

void VulkanApp::destroyTriangle()
{
    if (_vk_triangle_vb_mem != VK_NULL_HANDLE)
    {
        vkFreeMemory(_vk_device, _vk_triangle_vb_mem, nullptr);
        _vk_triangle_vb_mem = VK_NULL_HANDLE;
    }

    if (_vk_triangle_vb != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(_vk_device, _vk_triangle_vb, nullptr);
        _vk_triangle_vb = VK_NULL_HANDLE;
    }
}

b8 VulkanApp::createSwapChain(VkExtent2D frame_buffer_ext)
{
    auto const surface_swapchain_props = getVkSurfaceSwapChainProperties(_vk_phys_device, _vk_wnd_surface);

    VkSurfaceFormatKHR swapchain_surface_fmt = {};
    for (auto const & fmt : surface_swapchain_props.formats)
    {
        if ((fmt.format == VK_FORMAT_B8G8R8_SRGB) && (fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR))
        {
            swapchain_surface_fmt = fmt;
            break;
        }
    }
    if (swapchain_surface_fmt.format == VK_FORMAT_UNDEFINED)
    {
        swapchain_surface_fmt = surface_swapchain_props.formats[0];
    }

    VkPresentModeKHR swapchain_present_mode = VK_PRESENT_MODE_FIFO_KHR;
    for (auto mode_iter : surface_swapchain_props.present_modes)
    {
        if (mode_iter == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            swapchain_present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
            break;
        }
    }

    VkExtent2D swapchain_ext = {};
    if (surface_swapchain_props.caps.currentExtent.width != UINT32_MAX)
    {
        swapchain_ext = surface_swapchain_props.caps.currentExtent;
    }
    else
    {
        swapchain_ext.width =
            sbstd::max(surface_swapchain_props.caps.minImageExtent.width,
                       sbstd::min(frame_buffer_ext.width, surface_swapchain_props.caps.maxImageExtent.width));
        swapchain_ext.height =
            sbstd::max(surface_swapchain_props.caps.minImageExtent.height,
                       sbstd::min(frame_buffer_ext.height, surface_swapchain_props.caps.maxImageExtent.height));
    }

    u32 swapchain_img_cnt = surface_swapchain_props.caps.minImageCount + 1;
    if ((surface_swapchain_props.caps.minImageCount > 0) &&
        (swapchain_img_cnt > surface_swapchain_props.caps.maxImageCount))
    {
        swapchain_img_cnt = surface_swapchain_props.caps.maxImageCount;
    }

    if (swapchain_img_cnt < MAX_INFLIGHT_FRAMES)
    {
        sbLogE("Failed to create swapchain because the minimum of {} images cannot be fulfilled", MAX_INFLIGHT_FRAMES);
        return false;
    }

    VkSwapchainCreateInfoKHR swapchain_info = {};
    swapchain_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchain_info.surface = _vk_wnd_surface;
    swapchain_info.minImageCount = swapchain_img_cnt;
    swapchain_info.imageFormat = swapchain_surface_fmt.format;
    swapchain_info.imageColorSpace = swapchain_surface_fmt.colorSpace;
    swapchain_info.imageExtent = swapchain_ext;
    swapchain_info.imageArrayLayers = 1;
    swapchain_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkQueueFamilyIndex const family_indices[2] = {_queue_families.graphics, _queue_families.present};

    if (_queue_families.graphics != _queue_families.present)
    {
        swapchain_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapchain_info.queueFamilyIndexCount = 2;
        swapchain_info.pQueueFamilyIndices = sbstd::data(family_indices);
    }
    else
    {
        swapchain_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        swapchain_info.queueFamilyIndexCount = 0;
        swapchain_info.pQueueFamilyIndices = nullptr;
    }

    swapchain_info.preTransform = surface_swapchain_props.caps.currentTransform;
    swapchain_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchain_info.presentMode = swapchain_present_mode;
    swapchain_info.clipped = VK_TRUE;
    swapchain_info.oldSwapchain = VK_NULL_HANDLE;

    VkResult vk_res = vkCreateSwapchainKHR(_vk_device, &swapchain_info, nullptr, &_vk_swapchain);
    if (VK_SUCCESS != vk_res)
    {
        sbLogE("Failed to create Vulkan swapchain (error = '{}')", getEnumValue(vk_res));
        return false;
    }

    _vk_swapchain_ext = swapchain_ext;
    _target_frame_buffer_ext = swapchain_ext;
    _vk_swapchain_fmt = swapchain_surface_fmt.format;

    //_vk_swapchain_imgs
    u32 img_cnt = 0;
    vkGetSwapchainImagesKHR(_vk_device, _vk_swapchain, &img_cnt, nullptr);
    sbAssert(img_cnt == swapchain_img_cnt);
    _vk_swapchain_imgs.resize(img_cnt);
    vkGetSwapchainImagesKHR(_vk_device, _vk_swapchain, &img_cnt, _vk_swapchain_imgs.data());

    _vk_swapchain_imgs_view.reserve(img_cnt);
    for (auto const img : _vk_swapchain_imgs)
    {
        VkImageViewCreateInfo img_view_info = {};
        img_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        img_view_info.image = img;
        img_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        img_view_info.format = _vk_swapchain_fmt;
        img_view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        img_view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        img_view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        img_view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        img_view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        img_view_info.subresourceRange.baseMipLevel = 0;
        img_view_info.subresourceRange.levelCount = 1;
        img_view_info.subresourceRange.baseArrayLayer = 0;
        img_view_info.subresourceRange.layerCount = 1;

        VkImageView img_view;
        vk_res = vkCreateImageView(_vk_device, &img_view_info, nullptr, &img_view);
        if (VK_SUCCESS != vk_res)
        {
            sbLogE("Failed to create Vulkan swapchain image view (error = '{}')");
            return false;
        }

        _vk_swapchain_imgs_view.push_back(img_view);
    }

    return true;
}

void VulkanApp::recreateSwapChainRelatedData(VkExtent2D frame_buffer_ext)
{
    vkDeviceWaitIdle(_vk_device);

    cleanupSwapChainRelatedData();

    createSwapChain(frame_buffer_ext);
    createColorImage();
    createDepthImage();
    createFrameBuffers();
    // createGraphicsPipeline();
    // createCommandBuffers();
}

b8 VulkanApp::createDescriptors()
{
    VkSamplerCreateInfo sampler_info = {};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.anisotropyEnable = VK_TRUE;
    sampler_info.maxAnisotropy = 16;
    sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    sampler_info.compareOp = VK_COMPARE_OP_EQUAL;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_info.mipLodBias = 0.f;
    if (_demo_mode == DemoMode::MODEL)
    {
        // sampler_info.minLod = (float)_model.mip_cnt/2.f;
        sampler_info.maxLod = (float)_model.mip_cnt;
    }
    else
    {
        sampler_info.minLod = 0.f;
        sampler_info.maxLod = 0;
    }

    auto vk_res = vkCreateSampler(_vk_device, &sampler_info, nullptr, &_vk_test_sampler);
    if (VK_SUCCESS != vk_res)
    {
        sbLogE("Failed to create test texture sampler (error = '{}')", getEnumValue(vk_res));
        return false;
    }

    VkDescriptorPoolSize pool_sizes[2] = {};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[0].descriptorCount = MAX_INFLIGHT_FRAMES;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[1].descriptorCount = MAX_INFLIGHT_FRAMES;

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = numericConv<u32>(sbstd::size(pool_sizes));
    pool_info.pPoolSizes = sbstd::data(pool_sizes);
    pool_info.maxSets = MAX_INFLIGHT_FRAMES;

    vk_res = vkCreateDescriptorPool(_vk_device, &pool_info, nullptr, &_vk_desc_pool);
    if (VK_SUCCESS != vk_res)
    {
        sbLogE("Failed to create Vulkan Descriptor Pool (error = '{}')", getEnumValue(vk_res));
        return false;
    }

    FArray<VkDescriptorSetLayout, MAX_INFLIGHT_FRAMES> layouts(MAX_INFLIGHT_FRAMES, _vk_desc_set_layout);

    VkDescriptorSetAllocateInfo desc_set_alloc_info = {};
    desc_set_alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    desc_set_alloc_info.descriptorSetCount = (u32)layouts.size();
    desc_set_alloc_info.pSetLayouts = layouts.data();
    desc_set_alloc_info.descriptorPool = _vk_desc_pool;

    // descriptor sets are cleaned up automatically with the pool
    _vk_desc_sets.resize(MAX_INFLIGHT_FRAMES);
    vk_res = vkAllocateDescriptorSets(_vk_device, &desc_set_alloc_info, _vk_desc_sets.data());
    if (VK_SUCCESS != vk_res)
    {
        sbLogE("Failed to allocate Vulkan descriptor sets (error = '{}')", getEnumValue(vk_res));
        return false;
    }

    for (u32 frame_idx = 0; frame_idx != MAX_INFLIGHT_FRAMES; ++frame_idx)
    {
        VkDescriptorBufferInfo buffer_info = {};
        buffer_info.buffer = _vk_mvp_buffers[frame_idx].buffer;
        buffer_info.offset = 0;
        buffer_info.range = VK_WHOLE_SIZE;

        VkDescriptorImageInfo img_info = {};
        img_info.imageView = _demo_mode == DemoMode::MODEL ? _model.image_view : _vk_test_texture_view;
        img_info.sampler = _vk_test_sampler;
        img_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet descs_write_info[2] = {};
        descs_write_info[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descs_write_info[0].dstSet = _vk_desc_sets[frame_idx];
        descs_write_info[0].dstBinding = 0;
        descs_write_info[0].dstArrayElement = 0;
        descs_write_info[0].descriptorCount = 1;
        descs_write_info[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descs_write_info[0].pBufferInfo = &buffer_info;

        descs_write_info[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descs_write_info[1].dstSet = _vk_desc_sets[frame_idx];
        descs_write_info[1].dstBinding = 1;
        descs_write_info[1].dstArrayElement = 0;
        descs_write_info[1].descriptorCount = 1;
        descs_write_info[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descs_write_info[1].pImageInfo = &img_info;

        vkUpdateDescriptorSets(_vk_device, numericConv<u32>(sbstd::size(descs_write_info)),
                               sbstd::data(descs_write_info), 0, nullptr);
    }

    return true;
}

void VulkanApp::destroyDescriptors()
{
    if (VK_NULL_HANDLE != _vk_desc_pool)
    {
        vkDestroyDescriptorPool(_vk_device, _vk_desc_pool, nullptr);
        _vk_desc_pool = VK_NULL_HANDLE;
    }

    _vk_desc_sets.clear();

    if (VK_NULL_HANDLE != _vk_test_sampler)
    {
        vkDestroySampler(_vk_device, _vk_test_sampler, nullptr);
        _vk_test_sampler = VK_NULL_HANDLE;
    }
}

b8 VulkanApp::createUniformBuffers()
{
    VkDeviceSize const uni_mvp_size = sizeof(UniformMVP);

    _vk_mvp_buffers.resize(MAX_INFLIGHT_FRAMES);
    for (auto & buffer : _vk_mvp_buffers)
    {
        VkResult vk_res =
            createVkBuffer(_vk_phys_device, _vk_device, uni_mvp_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &buffer);
        if (VK_SUCCESS != vk_res)
        {
            sbLogE("Failed to create Vulkan uniform buffer (error = '{}')", getEnumValue(vk_res));
            return false;
        }
    }

    return true;
}

void VulkanApp::destroyUniformBuffers()
{
    for (auto buffer : _vk_mvp_buffers)
    {
        destroyVkBuffer(_vk_device, buffer);
    }
    _vk_mvp_buffers.clear();
}

b8 VulkanApp::createGraphicsPipeline()
{
    VkDescriptorSetLayoutBinding desc_set_binding[2] = {};

    desc_set_binding[0].binding = 0; // binding index in the sader
    desc_set_binding[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // same type as is shader (uniform)
    desc_set_binding[0].descriptorCount = 1; // > 1 when specifying an array of uniform e.g. skinning matrices
    desc_set_binding[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT; // uniform used by vertex shader
    desc_set_binding[0].pImmutableSamplers = nullptr; // only useful for images

    desc_set_binding[1].binding = 1;
    desc_set_binding[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    desc_set_binding[1].descriptorCount = 1;
    desc_set_binding[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    desc_set_binding[1].pImmutableSamplers = nullptr;

    // Describe the descriptors binding for the whole pipeline
    VkDescriptorSetLayoutCreateInfo desc_set_layout_info = {};
    desc_set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    desc_set_layout_info.bindingCount = numericConv<u32>(sbstd::size(desc_set_binding));
    desc_set_layout_info.pBindings = sbstd::data(desc_set_binding);

    VkResult vk_res = vkCreateDescriptorSetLayout(_vk_device, &desc_set_layout_info, nullptr, &_vk_desc_set_layout);
    if (VK_SUCCESS != vk_res)
    {
        sbLogE("Failed to create Vulkan descriptor set (error = '{}')", getEnumValue(vk_res));
        return false;
    }

    VkShaderModule vert_shader = VK_NULL_HANDLE;
    VkShaderModule frag_shader = VK_NULL_HANDLE;

    DArray<u8> shader_byte_code;
    FileStream shader_file(VFS::openFileRead("/basic.vert", FileFormat::BIN));
    if (!shader_file.isValid())
    {
        sbLogE("Failed to open vertex shader 'basic_vert'");
        return false;
    }
    shader_byte_code.resize(shader_file.getLength());
    shader_file.read(shader_byte_code);
    vk_res = createVkShaderModule(_vk_device, shader_byte_code, &vert_shader);
    if (VK_SUCCESS != vk_res)
    {
        sbLogE("Failed to create vertex shader (error = '{}')", getEnumValue(vk_res));
        return false;
    }

    shader_file.reset(VFS::openFileRead("/basic.frag", FileFormat::BIN));
    if (!shader_file.isValid())
    {
        sbLogE("Failed to open vertex shader 'basic_vert'");
        return false;
    }
    shader_byte_code.resize(shader_file.getLength());
    shader_file.read(shader_byte_code);
    vk_res = createVkShaderModule(_vk_device, shader_byte_code, &frag_shader);
    if (VK_SUCCESS != vk_res)
    {
        sbLogE("Failed to create fragment shader (error = '{}')", getEnumValue(vk_res));
        return false;
    }

    shader_file.reset();

    VkPipelineShaderStageCreateInfo prog_shaders_info[2] = {};

    prog_shaders_info[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    prog_shaders_info[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    prog_shaders_info[0].module = vert_shader;
    prog_shaders_info[0].pName = "main"; // shader entry point
    prog_shaders_info[0].pSpecializationInfo = nullptr; // used to define pipeline creation time shader constants

    prog_shaders_info[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    prog_shaders_info[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    prog_shaders_info[1].module = frag_shader;
    prog_shaders_info[1].pName = "main"; // shader entry point
    prog_shaders_info[1].pSpecializationInfo = nullptr; // used to define pipeline creation time shader constants

    VkPipelineVertexInputStateCreateInfo vertex_input_info = {};
    vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input_info.vertexBindingDescriptionCount = 0;
    vertex_input_info.pVertexBindingDescriptions = nullptr;
    vertex_input_info.vertexAttributeDescriptionCount = 0;
    vertex_input_info.pVertexAttributeDescriptions = nullptr;
    vertex_input_info.vertexBindingDescriptionCount = 1;
    vertex_input_info.pVertexBindingDescriptions = &_vk_vertex_binding_desc;
    vertex_input_info.vertexAttributeDescriptionCount = numericConv<u32>(sbstd::size(_vk_vertex_attributes_desc));
    vertex_input_info.pVertexAttributeDescriptions = sbstd::data(_vk_vertex_attributes_desc);

    VkPipelineInputAssemblyStateCreateInfo input_assembly_info = {};
    input_assembly_info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineDepthStencilStateCreateInfo depth_stencil_info = {};
    depth_stencil_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil_info.depthTestEnable = VK_TRUE;
    depth_stencil_info.depthWriteEnable = VK_TRUE;
    depth_stencil_info.depthCompareOp = VK_COMPARE_OP_LESS;
    depth_stencil_info.depthBoundsTestEnable = VK_FALSE;

    // Defines render image transformation
    VkViewport view_port = {};
    view_port.width = (float)_vk_swapchain_ext.width;
    view_port.height = (float)_vk_swapchain_ext.height;
    view_port.x = 0;
    view_port.y = 0;
    view_port.minDepth = 0.f;
    view_port.maxDepth = 1.f;

    // Defines render image cut
    VkRect2D scissor = {};
    scissor.extent = _vk_swapchain_ext;
    scissor.offset = {0, 0};

    VkPipelineViewportStateCreateInfo view_port_info = {};
    view_port_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    view_port_info.viewportCount = 1;
    view_port_info.pViewports = &view_port;
    view_port_info.scissorCount = 1;
    view_port_info.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer_info = {};
    rasterizer_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer_info.depthClampEnable = VK_FALSE;
    rasterizer_info.rasterizerDiscardEnable = VK_FALSE; // if VK_TRUE, nothing is rasterized
    rasterizer_info.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer_info.lineWidth = 1.f;
    rasterizer_info.cullMode = VK_CULL_MODE_BACK_BIT; // cull back faces
    rasterizer_info.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; // defines how front face are identified/defined
    rasterizer_info.depthBiasEnable = VK_FALSE;
    rasterizer_info.depthBiasClamp = 0.f;
    rasterizer_info.depthBiasConstantFactor = 0.f;
    rasterizer_info.depthBiasSlopeFactor = 0.f;

    VkPipelineMultisampleStateCreateInfo multi_sample_info = {};
    multi_sample_info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multi_sample_info.sampleShadingEnable = VK_FALSE;
    multi_sample_info.rasterizationSamples = _vk_sample_count;
    multi_sample_info.minSampleShading = 1.f;
    multi_sample_info.pSampleMask = nullptr;
    multi_sample_info.alphaToCoverageEnable = VK_FALSE;
    multi_sample_info.alphaToOneEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState attach_blend = {};
    attach_blend.colorWriteMask =
        VK_COLOR_COMPONENT_A_BIT | VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT;
    attach_blend.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo blend_info = {};
    blend_info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend_info.logicOpEnable = VK_FALSE;
    blend_info.logicOp = VK_LOGIC_OP_COPY;
    blend_info.attachmentCount = 1;
    blend_info.pAttachments = &attach_blend;

    // Values which can be updated without re-creating the pipeline
    VkDynamicState const dyn_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT
        //    , VK_DYNAMIC_STATE_LINE_WIDTH
    };

    VkPipelineDynamicStateCreateInfo dyn_info = {};
    dyn_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn_info.dynamicStateCount = (u32)sbstd::size(dyn_states);
    dyn_info.pDynamicStates = sbstd::data(dyn_states);

    VkPipelineLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.pushConstantRangeCount = 0;
    layout_info.pPushConstantRanges = nullptr;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &_vk_desc_set_layout;

    vk_res = vkCreatePipelineLayout(_vk_device, &layout_info, nullptr, &_vk_pipeline_layout);
    if (VK_SUCCESS != vk_res)
    {
        sbLogE("Failed to create Vulkan pipeline (error = '{}')", getEnumValue(vk_res));
        return false;
    }

    VkAttachmentDescription attachments[3] = {};

    VkAttachmentDescription & color_attach_desc = attachments[0];
    color_attach_desc.format = _vk_swapchain_fmt;
    color_attach_desc.samples = _vk_sample_count;
    color_attach_desc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attach_desc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attach_desc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attach_desc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attach_desc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // We render directlry in the swapchin surface for prensent
    color_attach_desc.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription & depth_attach_desc = attachments[1];
    depth_attach_desc.format = findVkDepthImageFormat(_vk_phys_device);
    depth_attach_desc.samples = _vk_sample_count;
    depth_attach_desc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attach_desc.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attach_desc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth_attach_desc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attach_desc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth_attach_desc.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription & color_resolve_attach_desc = attachments[2];
    color_resolve_attach_desc.format = _vk_swapchain_fmt;
    color_resolve_attach_desc.samples = VK_SAMPLE_COUNT_1_BIT;
    color_resolve_attach_desc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_resolve_attach_desc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_resolve_attach_desc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_resolve_attach_desc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_resolve_attach_desc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // We render directlry in the swapchin surface for prensent
    color_resolve_attach_desc.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference attach_ref = {};
    // Index of the attachment in the list of attachments from the parent render pass
    attach_ref.attachment = 0;
    // We render to it
    attach_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depth_attach_ref = {};
    depth_attach_ref.attachment = 1;
    depth_attach_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference color_resolve_attach_ref = {};
    color_resolve_attach_ref.attachment = 2;
    color_resolve_attach_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription sub_pass_desc = {};
    sub_pass_desc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub_pass_desc.colorAttachmentCount = 1;
    // Index in this array represents the location in the shader
    sub_pass_desc.pColorAttachments = &attach_ref;
    sub_pass_desc.pDepthStencilAttachment = &depth_attach_ref;
    sub_pass_desc.pResolveAttachments = &color_resolve_attach_ref;

    VkRenderPassCreateInfo rndr_pass_info = {};
    rndr_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rndr_pass_info.attachmentCount = (u32)sbstd::size(attachments);
    rndr_pass_info.pAttachments = sbstd::data(attachments);
    rndr_pass_info.subpassCount = 1;
    rndr_pass_info.pSubpasses = &sub_pass_desc;

    // Dependency takes care of the attachment layout transition i.e. from opimal KHR destimation to collor attchment
    VkSubpassDependency subpass_dep = {};
    subpass_dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    subpass_dep.dstSubpass = 0; // our first and only sub-pass
    // Wait for the swapchain to read from the image
    subpass_dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subpass_dep.srcAccessMask = 0;
    // Wait when the pipeline is about to write to destingation image
    subpass_dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subpass_dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    rndr_pass_info.dependencyCount = 1;
    rndr_pass_info.pDependencies = &subpass_dep;

    vk_res = vkCreateRenderPass(_vk_device, &rndr_pass_info, nullptr, &_vk_render_pass);
    if (VK_SUCCESS != vk_res)
    {
        sbLogE("Failed to create Vulkan Render Pass (error = '{}')", getEnumValue(vk_res));
        return false;
    }

    VkGraphicsPipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    // Programmable stages
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = sbstd::data(prog_shaders_info);
    // Fix function stages
    pipeline_info.pInputAssemblyState = &input_assembly_info;
    pipeline_info.pVertexInputState = &vertex_input_info;
    pipeline_info.pDepthStencilState = &depth_stencil_info;
    pipeline_info.pViewportState = &view_port_info;
    pipeline_info.pRasterizationState = &rasterizer_info;
    pipeline_info.pMultisampleState = &multi_sample_info;
    pipeline_info.pColorBlendState = &blend_info;
    pipeline_info.pDynamicState = nullptr;
    pipeline_info.pDynamicState = &dyn_info;
    // Defines in which sub render pass this pipeline will be used
    pipeline_info.renderPass = _vk_render_pass;
    pipeline_info.subpass = 0;
    // Inheritance
    pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
    pipeline_info.basePipelineIndex = -1;

    pipeline_info.layout = _vk_pipeline_layout;

    vk_res = vkCreateGraphicsPipelines(_vk_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &_vk_graphics_pipeline);
    if (VK_SUCCESS != vk_res)
    {
        sbLogE("Failed to create Vulkan graphicd pipeline (error = '{}')", getEnumValue(vk_res));
        return false;
    }

    createFrameBuffers();

    // shader module are 'copied' by the pipeline
    vkDestroyShaderModule(_vk_device, vert_shader, nullptr);
    vkDestroyShaderModule(_vk_device, frag_shader, nullptr);

    return true;
}

b8 VulkanApp::createFrameBuffers()
{
    _vk_frame_buffers.resize(_vk_swapchain_imgs_view.size());
    usize frame_buffer_idx = 0;
    for (auto & frame_bufer : _vk_frame_buffers)
    {
        VkImageView const attachments[] = {_vk_color_image_view, _vk_depth_image_view, _vk_swapchain_imgs_view[frame_buffer_idx]};

        VkFramebufferCreateInfo frame_buffer_info = {};
        frame_buffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        frame_buffer_info.renderPass = _vk_render_pass;
        frame_buffer_info.attachmentCount = (u32)sbstd::size(attachments);
        frame_buffer_info.pAttachments = sbstd::data(attachments);
        frame_buffer_info.width = _vk_swapchain_ext.width;
        frame_buffer_info.height = _vk_swapchain_ext.height;
        frame_buffer_info.layers = 1;

        auto vk_res = vkCreateFramebuffer(_vk_device, &frame_buffer_info, nullptr, &frame_bufer);
        if (VK_SUCCESS != vk_res)
        {
            sbLogE("Failed to create Vulkan frame buffer (error = '{}'", getEnumValue(vk_res));
            return false;
        }

        ++frame_buffer_idx;
    }

    return true;
}

b8 VulkanApp::createCommandBuffers()
{
    _vk_cmd_buffers.resize(MAX_INFLIGHT_FRAMES, VK_NULL_HANDLE);
    /*
        _vk_cmd_buffers.resize(_vk_swapchain_imgs.size(), VK_NULL_HANDLE);
        VkCommandBufferAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = _vk_graphics_cmd_pool;
        alloc_info.commandBufferCount = numericConv<u32>(_vk_cmd_buffers.size());
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

        VkResult vk_res = vkAllocateCommandBuffers(_vk_device, &alloc_info, _vk_cmd_buffers.data());
        if (VK_SUCCESS != vk_res)
        {
            sbLogE("Failed to allocate Vulkan Command Buffers (error = '{}')", getEnumValue(vk_res));
            return false;
        }

        u32 buffer_idx = 0;
        for (auto cmd_buffer : _vk_cmd_buffers)
        {
            VkCommandBufferBeginInfo cmd_begin_info = {};
            cmd_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            cmd_begin_info.flags = 0;
            cmd_begin_info.pInheritanceInfo = nullptr;
            vk_res = vkBeginCommandBuffer(cmd_buffer, &cmd_begin_info);
            if (VK_SUCCESS != vk_res)
            {
                sbLogE("Failed to record Vulkan begin command (error = '{}')", getEnumValue(vk_res));
                return false;
            }

            VkClearValue const clear_value = {0.f, 0.f, 0.f, 1.f};
            VkRenderPassBeginInfo cmd_pass_begin_info = {};
            cmd_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            cmd_pass_begin_info.renderPass = _vk_render_pass;
            cmd_pass_begin_info.framebuffer = _vk_frame_buffers[buffer_idx];
            cmd_pass_begin_info.renderArea.offset = {0, 0};
            cmd_pass_begin_info.renderArea.extent = _vk_swapchain_ext;
            cmd_pass_begin_info.clearValueCount = 1;
            cmd_pass_begin_info.pClearValues = &clear_value;
            vkCmdBeginRenderPass(cmd_buffer, &cmd_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

            vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _vk_graphics_pipeline);

            vkCmdDraw(cmd_buffer, 3, 1, 0, 0);

            vkCmdEndRenderPass(cmd_buffer);

            vkEndCommandBuffer(cmd_buffer);

            ++buffer_idx;
        }
    */
    return true;
}

b8 VulkanApp::initialize(b8 enable_dbg_layers, GLFWwindow * wnd, DemoMode mode)
{
    _enable_dbg_layers = enable_dbg_layers;
    _current_frame = 0;
    _demo_mode = mode;

    _vk_vertex_binding_desc.binding = 0;
    _vk_vertex_binding_desc.stride = sizeof(Vertex);
    _vk_vertex_binding_desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    _vk_vertex_attributes_desc[0].binding = 0;
    _vk_vertex_attributes_desc[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    _vk_vertex_attributes_desc[0].location = 0; // location from the vertex shader code
    _vk_vertex_attributes_desc[0].offset = offsetof(Vertex, position);

    _vk_vertex_attributes_desc[1].binding = 0;
    _vk_vertex_attributes_desc[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    _vk_vertex_attributes_desc[1].location = 1; // location from the vertex shader code
    _vk_vertex_attributes_desc[1].offset = offsetof(Vertex, color);

    _vk_vertex_attributes_desc[2].binding = 0;
    _vk_vertex_attributes_desc[2].format = VK_FORMAT_R32G32_SFLOAT;
    _vk_vertex_attributes_desc[2].location = 2; // location from the vertex shader code
    _vk_vertex_attributes_desc[2].offset = offsetof(Vertex, tex_coords);

    if (sbDontExpect(!initializeVulkanCore(wnd), "failed to initialize Vulkan"))
    {
        return false;
    }

    if (sbDontExpect(!createColorImage()))
    {
        return false;
    }

    if (sbDontExpect(!createDepthImage()))
    {
        return false;
    }

    if (sbDontExpect(!createGraphicsPipeline()))
    {
        return false;
    }

    if (sbDontExpect(!createCommandBuffers()))
    {
        return false;
    }

    if (sbDontExpect(!loadTestTexture()))
    {
        return false;
    }

    if (sbDontExpect(!loadModel()))
    {
        return false;
    }

    if (sbDontExpect(!createTriangle()))
    {
        return false;
    }

    if (sbDontExpect(!createQuad()))
    {
        return false;
    }

    if (sbDontExpect(!createUniformBuffers()))
    {
        return false;
    }

    if (sbDontExpect(!createDescriptors()))
    {
        return false;
    }

    _start_time = std::chrono::high_resolution_clock::now();

    return true;
}

void VulkanApp::terminate()
{
    if (VK_NULL_HANDLE != _vk_device)
    {
        vkDeviceWaitIdle(_vk_device);
    }

    destroyDescriptors();
    destroyUniformBuffers();
    unloadTestTexture();
    unloadModel();
    destroyQuad();
    destroyTriangle();
    cleanupSwapChainRelatedData();
    terminateVulkanCore();
}

b8 VulkanApp::render()
{
    if ((_target_frame_buffer_ext.width == 0) || (_target_frame_buffer_ext.height == 0))
    {
        return false;
    }

    vkWaitForFences(_vk_device, 1, &_vk_inflight_fences[_current_frame], VK_TRUE, UINT64_MAX);

    u32 img_idx = 0;
    VkResult vk_res = vkAcquireNextImageKHR(_vk_device, _vk_swapchain, UINT64_MAX,
                                            _vk_image_available_sems[_current_frame], VK_NULL_HANDLE, &img_idx);

    if (vk_res == VK_ERROR_OUT_OF_DATE_KHR)
    {
        recreateSwapChainRelatedData(_vk_swapchain_ext);
    }
    else if ((vk_res != VK_SUCCESS) && (vk_res != VK_SUBOPTIMAL_KHR))
    {
        return false;
    }

    if (_vk_inuse_fences[img_idx] != VK_NULL_HANDLE)
    {
        vkWaitForFences(_vk_device, 1, &_vk_inuse_fences[img_idx], VK_TRUE, UINT64_MAX);
    }

    _vk_inuse_fences[img_idx] = _vk_inflight_fences[_current_frame];

    auto const time_from_start = std::chrono::duration<float, std::chrono::seconds::period>(
                                     std::chrono::high_resolution_clock::now() - _start_time)
                                     .count();
    UniformMVP mvp;
    mvp.model = glm::rotate(glm::mat4(1.f), time_from_start * glm::radians(time_from_start), glm::vec3(0.f, 0.f, 1.f));
    mvp.view = glm::lookAt(glm::vec3(2.f, 2.f, 2.f), glm::vec3(0.f, 0.f, 0.f), glm::vec3(0.f, 0.f, 1.f));
    mvp.projection =
        glm::perspective(glm::radians(45.f), _vk_swapchain_ext.width / ((float)_vk_swapchain_ext.height), 0.1f, 100.f);
    mvp.projection[1][1] *= -1.f;

    auto & curr_mvp_buffer = _vk_mvp_buffers[_current_frame];
    void * mvp_data = nullptr;
    vkMapMemory(_vk_device, curr_mvp_buffer.memory, 0, sizeof(UniformMVP), 0, &mvp_data);
    memcpy(mvp_data, &mvp, sizeof(mvp));
    vkUnmapMemory(_vk_device, curr_mvp_buffer.memory);

    VkCommandBuffer & cmd_buffer = _vk_cmd_buffers[_current_frame];

    {
        if (cmd_buffer != VK_NULL_HANDLE)
        {
            vkFreeCommandBuffers(_vk_device, _vk_graphics_cmd_pool, 1, &cmd_buffer);
            cmd_buffer = VK_NULL_HANDLE;
        }

        VkCommandBufferAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = _vk_graphics_cmd_pool;
        alloc_info.commandBufferCount = 1;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

        vk_res = vkAllocateCommandBuffers(_vk_device, &alloc_info, &cmd_buffer);
        if (VK_SUCCESS != vk_res)
        {
            sbLogE("Failed to allocate Vulkan Command Buffers (error = '{}')", getEnumValue(vk_res));
            return false;
        }

        VkCommandBufferBeginInfo cmd_begin_info = {};
        cmd_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cmd_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        cmd_begin_info.pInheritanceInfo = nullptr;
        vk_res = vkBeginCommandBuffer(cmd_buffer, &cmd_begin_info);
        if (VK_SUCCESS != vk_res)
        {
            sbLogE("Failed to record Vulkan begin command (error = '{}')", getEnumValue(vk_res));
            return false;
        }

        VkClearValue clear_values[3];
        clear_values[0].color = {0.f, 0.f, 0.f, 1.f};
        clear_values[1].depthStencil = {1.f, 0};
        clear_values[2].color = {0.f, 0.f, 0.f, 1.f};

        VkRenderPassBeginInfo cmd_pass_begin_info = {};
        cmd_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        cmd_pass_begin_info.renderPass = _vk_render_pass;
        cmd_pass_begin_info.framebuffer = _vk_frame_buffers[img_idx];
        cmd_pass_begin_info.renderArea.offset = {0, 0};
        cmd_pass_begin_info.renderArea.extent = _vk_swapchain_ext;
        cmd_pass_begin_info.clearValueCount = (u32)sbstd::size(clear_values);
        cmd_pass_begin_info.pClearValues = sbstd::data(clear_values);
        vkCmdBeginRenderPass(cmd_buffer, &cmd_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _vk_graphics_pipeline);

        VkViewport view_port = {};
        view_port.width = (float)_vk_swapchain_ext.width;
        view_port.height = (float)_vk_swapchain_ext.height;
        view_port.x = 0;
        view_port.y = 0;
        view_port.minDepth = 0.f;
        view_port.maxDepth = 1.f;
        vkCmdSetViewport(cmd_buffer, 0, 1, &view_port);

        VkRect2D scissor = {};
        scissor.extent = _vk_swapchain_ext;
        scissor.offset = {0, 0};
        vkCmdSetScissor(cmd_buffer, 0, 1, &scissor);

        VkDeviceSize offsets = 0;

        vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _vk_pipeline_layout, 0, 1,
                                &_vk_desc_sets[_current_frame], 0, nullptr);

        switch (_demo_mode)
        {
            case DemoMode::TRIANGLE:
            {
                vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &_vk_triangle_vb, &offsets);
                vkCmdDraw(cmd_buffer, 3, 1, 0, 0);

                break;
            }

            case DemoMode::QUAD:
            {
                vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &_vk_quad_vb, &offsets);
                vkCmdBindIndexBuffer(cmd_buffer, _vk_quad_ib, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(cmd_buffer, 12, 1, 0, 0, 0);

                break;
            }
            case DemoMode::MODEL:
            {
                vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &_model.vb.buffer, &offsets);
                vkCmdBindIndexBuffer(cmd_buffer, _model.ib.buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(cmd_buffer, (u32)_model.idx_cnt, 1, 0, 0, 0);
                break;
            }
            default:
            {
                sbAssert(false, "Unsupported demo mode");
                break;
            }
        };

        vkCmdEndRenderPass(cmd_buffer);

        vkEndCommandBuffer(cmd_buffer);
    }

    // The command has to wait for the image to be ready when we start wrinting to the image
    VkPipelineStageFlags const wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    // There is a 1:1 correspondance between wait stages and semaphores
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &_vk_image_available_sems[_current_frame];
    submit_info.pWaitDstStageMask = &wait_stage;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buffer;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &_vk_render_finished_sems[_current_frame];

    vkResetFences(_vk_device, 1, &_vk_inflight_fences[_current_frame]);

    vk_res = vkQueueSubmit(_vk_graphics_queue, 1, &submit_info, _vk_inflight_fences[_current_frame]);
    if (VK_SUCCESS != vk_res)
    {
        sbLogE("Failed to submit the Vulkan command buffer to the graphics queue (error = '{}'", getEnumValue(vk_res));
        return false;
    }

    VkPresentInfoKHR present_info = {};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &_vk_swapchain;
    present_info.pImageIndices = &img_idx;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &_vk_render_finished_sems[_current_frame];
    present_info.pResults = nullptr;

    vk_res = vkQueuePresentKHR(_vk_present_queue, &present_info);
    if ((vk_res == VK_ERROR_OUT_OF_DATE_KHR) || (vk_res == VK_SUBOPTIMAL_KHR) ||
        (_target_frame_buffer_ext.width != _vk_swapchain_ext.width) ||
        (_target_frame_buffer_ext.height != _vk_swapchain_ext.height))
    {
        recreateSwapChainRelatedData(_target_frame_buffer_ext);
    }
    else if (vk_res != VK_SUCCESS)
    {
        sbLogE("Failed to present Vulkan frame buffer (error = '{}')", getEnumValue(vk_res));
        return false;
    }

    // vkQueueWaitIdle(_vk_present_queue);

    _current_frame = (_current_frame + 1) % MAX_INFLIGHT_FRAMES;

    return true;
}

b8 VulkanApp::loadModel()
{
    char model_abs_path[sb::LOCAL_PATH_MAX_LEN];

    getWorkingDirectory(model_abs_path);
    concatLocalPath(model_abs_path, "viking_room.obj");

    {
        auto file_content = VFS::readFile("/viking_room.png", GHEAP);

        if (file_content.size() == 0)
        {
            sbLogE("Failed to load test texture content");
            return false;
        }

        int width, height, channel_cnt;
        auto const pixels = stbi_load_from_memory(file_content.data(), (int)file_content.size(), &width, &height,
                                                  &channel_cnt, STBI_rgb_alpha);

        GHEAP.deallocate(file_content.data());

        _model.mip_cnt = getMipLevelCount(width, height);

        if (nullptr == pixels)
        {
            sbLogE("Failed to load model texture");
            return false;
        }

        VkDeviceSize const image_size = width * height * 4;

        VkBufferMem staging_buffer = {};
        VkResult vk_res =
            createVkBuffer(_vk_phys_device, _vk_device, image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &staging_buffer);

        if (VK_SUCCESS != vk_res)
        {
            sbLogE("Failed to create model texture staging buffer (error = '{}')", getEnumValue(vk_res));
            return false;
        }

        void * data = nullptr;
        vkMapMemory(_vk_device, staging_buffer.memory, 0, image_size, 0, &data);
        sbAssert(nullptr != data);
        memcpy(data, pixels, image_size);
        vkUnmapMemory(_vk_device, staging_buffer.memory);

        stbi_image_free(pixels);

        vk_res = createVkImage(_vk_phys_device, _vk_device, width, height, _model.mip_cnt, VK_SAMPLE_COUNT_1_BIT,
                               VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
                               VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                   VK_IMAGE_USAGE_SAMPLED_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &_model.image);

        if (VK_SUCCESS != vk_res)
        {
            sbLogE("Failed to create Vulkan test image (error = '{}')", getEnumValue(vk_res));
            return false;
        }

        transitionVkImageLayout(_vk_device, _vk_graphics_queue, _vk_graphics_cmd_pool, _model.image.image,
                                VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, _model.mip_cnt);
        copyVkBufferToImage(_vk_device, _vk_graphics_cmd_pool, _vk_graphics_queue, staging_buffer.buffer,
                            _model.image.image, {(u32)width, (u32)height, 1});
        // transitionVkImageLayout(_vk_device, _vk_graphics_queue, _vk_graphics_cmd_pool, _model.image.image,
        //                         VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        //                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, _model.mip_cnt);
        generateMipmaps(_vk_phys_device, _vk_device, _vk_graphics_cmd_pool, _vk_graphics_queue, width, height,
                        _model.mip_cnt, _model.image.image, VK_FORMAT_R8G8B8A8_SRGB);

        destroyVkBuffer(_vk_device, staging_buffer);

        VkImageViewCreateInfo view_info = {};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = _model.image.image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = VK_FORMAT_R8G8B8A8_SRGB;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = _model.mip_cnt;

        vk_res = vkCreateImageView(_vk_device, &view_info, nullptr, &_model.image_view);
        if (VK_SUCCESS != vk_res)
        {
            sbLogE("Failed to create model texture image view (error = '{}')", getEnumValue(vk_res));
            return false;
        }
    }

    {
        tinyobj::attrib_t model_attrs;
        std::vector<tinyobj::shape_t> model_shapes;
        std::vector<tinyobj::material_t> model_materials;
        std::string error_str;

        if (!tinyobj::LoadObj(&model_attrs, &model_shapes, &model_materials, &error_str,
                              sbstd::data(model_abs_path)))
        {
            sbLogE("Failed to load demo model : '{}'", error_str.c_str());
            return false;
        }

        DArray<Vertex> vertices;
        DArray<u32> indices;

        sbAssert(model_shapes.size() == 1);

        vertices.resize(model_attrs.vertices.size());
        for (auto const & idx : model_shapes.front().mesh.indices)
        {
            Vertex & curr_vert = vertices[idx.vertex_index];
            curr_vert.position = {
                model_attrs.vertices[3 * idx.vertex_index + 0],
                model_attrs.vertices[3 * idx.vertex_index + 1],
                model_attrs.vertices[3 * idx.vertex_index + 2],
            };

            curr_vert.tex_coords = {model_attrs.texcoords[2 * idx.texcoord_index + 0],
                                    1.f - model_attrs.texcoords[2 * idx.texcoord_index + 1]};

            curr_vert.color = {1.f, 1.f, 1.f};

            indices.push_back(idx.vertex_index);
        }

        _model.idx_cnt = indices.size();
        _model.vtx_cnt = vertices.size();

        {
            VkDeviceSize const ib_size = indices.size() * sizeof(u32);

            VkBufferMem final_ib_mem;
            auto vk_res = createVkBuffer(_vk_phys_device, _vk_device, ib_size,
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &final_ib_mem);

            if (VK_SUCCESS != vk_res)
            {
                sbLogE("Failed to create Vulkan final model IB (error = '{}')", getEnumValue(vk_res));
                return false;
            }

            _model.ib = final_ib_mem;

            vk_res = uploadVkBufferDataToDevice(_vk_phys_device, _vk_device, (void *)sbstd::data(indices), ib_size,
                                                _vk_graphics_cmd_pool, _vk_graphics_queue, _model.ib.buffer);
            if (VK_SUCCESS != vk_res)
            {
                sbLogE("Failed to upload Vulkan model data (error = '{}')", getEnumValue(vk_res));
                return false;
            }
        }

        {
            VkDeviceSize const vb_size = vertices.size() * sizeof(Vertex);

            VkBufferMem final_vb_mem;
            auto vk_res = createVkBuffer(_vk_phys_device, _vk_device, vb_size,
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &final_vb_mem);

            if (VK_SUCCESS != vk_res)
            {
                sbLogE("Failed to create Vulkan final model VB (error = '{}')", getEnumValue(vk_res));
                return false;
            }

            _model.vb = final_vb_mem;

            vk_res = uploadVkBufferDataToDevice(_vk_phys_device, _vk_device, (void *)sbstd::data(vertices), vb_size,
                                                _vk_graphics_cmd_pool, _vk_graphics_queue, _model.vb.buffer);
            if (VK_SUCCESS != vk_res)
            {
                sbLogE("Failed to upload Vulkan model data (error = '{}')", getEnumValue(vk_res));
                return false;
            }
        }
    }

    return true;
}

void VulkanApp::unloadModel()
{
    if (VK_NULL_HANDLE != _model.image_view)
    {
        vkDestroyImageView(_vk_device, _model.image_view, nullptr);
    }

    destroyVkImage(_vk_device, _model.image);

    destroyVkBuffer(_vk_device, _model.ib);
    destroyVkBuffer(_vk_device, _model.vb);

    _model = {};
}

b8 VulkanApp::loadTestTexture()
{
    auto file_content = VFS::readFile("/texture.jpg", GHEAP);

    if (file_content.size() == 0)
    {
        sbLogE("Failed to load test texture content");
        return false;
    }

    int width, height, channel_cnt;
    auto const pixels = stbi_load_from_memory(file_content.data(), (int)file_content.size(), &width, &height,
                                              &channel_cnt, STBI_rgb_alpha);

    GHEAP.deallocate(file_content.data());

    if (nullptr == pixels)
    {
        sbLogE("Failed to load test texture");
        return false;
    }

    VkDeviceSize const image_size = width * height * 4;

    VkBufferMem staging_buffer = {};
    VkResult vk_res =
        createVkBuffer(_vk_phys_device, _vk_device, image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &staging_buffer);

    if (VK_SUCCESS != vk_res)
    {
        sbLogE("Failed to create test texture staging buffer (error = '{}')", getEnumValue(vk_res));
        return false;
    }

    void * data = nullptr;
    vkMapMemory(_vk_device, staging_buffer.memory, 0, image_size, 0, &data);
    sbAssert(nullptr != data);
    memcpy(data, pixels, image_size);
    vkUnmapMemory(_vk_device, staging_buffer.memory);

    stbi_image_free(pixels);

    vk_res =
        createVkImage(_vk_phys_device, _vk_device, width, height, 1, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB,
                      VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &_vk_test_texture);

    if (VK_SUCCESS != vk_res)
    {
        sbLogE("Failed to create Vulkan test image (error = '{}')", getEnumValue(vk_res));
        return false;
    }

    transitionVkImageLayout(_vk_device, _vk_graphics_queue, _vk_graphics_cmd_pool, _vk_test_texture.image,
                            VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            1);
    copyVkBufferToImage(_vk_device, _vk_graphics_cmd_pool, _vk_graphics_queue, staging_buffer.buffer,
                        _vk_test_texture.image, {(u32)width, (u32)height, 1});
    transitionVkImageLayout(_vk_device, _vk_graphics_queue, _vk_graphics_cmd_pool, _vk_test_texture.image,
                            VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);

    destroyVkBuffer(_vk_device, staging_buffer);

    VkImageViewCreateInfo view_info = {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = _vk_test_texture.image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_R8G8B8A8_SRGB;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;

    vk_res = vkCreateImageView(_vk_device, &view_info, nullptr, &_vk_test_texture_view);
    if (VK_SUCCESS != vk_res)
    {
        sbLogE("Failed to create test texture image view (error = '{}')", getEnumValue(vk_res));
        return false;
    }

    return true;
}

void VulkanApp::unloadTestTexture()
{
    if (VK_NULL_HANDLE != _vk_test_texture_view)
    {
        vkDestroyImageView(_vk_device, _vk_test_texture_view, nullptr);
        _vk_test_texture_view = VK_NULL_HANDLE;
    }

    destroyVkImage(_vk_device, _vk_test_texture);
    _vk_test_texture = {};
}

b8 VulkanApp::createDepthImage()
{
    _vk_depth_fmt = findVkDepthImageFormat(_vk_phys_device);
    if (VK_FORMAT_UNDEFINED == _vk_depth_fmt)
    {
        sbLogE("Unable to find suitable depth format");
        return false;
    }

    VkResult vk_res = createVkImage(_vk_phys_device, _vk_device, _vk_swapchain_ext.width, _vk_swapchain_ext.height, 1,
                                    _vk_sample_count, _vk_depth_fmt, VK_IMAGE_TILING_OPTIMAL,
                                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                    &_vk_depth_image);
    if (VK_SUCCESS != vk_res)
    {
        sbLogE("Failed to create depth buffer image (error = '{}')", getEnumValue(vk_res));
        return false;
    }

    VkImageViewCreateInfo view_info = {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = _vk_depth_image.image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = _vk_depth_fmt;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;

    vk_res = vkCreateImageView(_vk_device, &view_info, nullptr, &_vk_depth_image_view);
    if (VK_SUCCESS != vk_res)
    {
        sbLogE("Failed to create depth buffer image view (error = '{}')", getEnumValue(vk_res));
        return false;
    }
    return true;
}

void VulkanApp::destroyDepthImage()
{
    if (_vk_depth_image_view != VK_NULL_HANDLE)
    {
        vkDestroyImageView(_vk_device, _vk_depth_image_view, nullptr);
        _vk_depth_image_view = VK_NULL_HANDLE;
    }

    destroyVkImage(_vk_device, _vk_depth_image);
    _vk_depth_fmt = VK_FORMAT_UNDEFINED;
}

b8 VulkanApp::createColorImage()
{
    VkResult vk_res = createVkImage(_vk_phys_device, _vk_device, _vk_swapchain_ext.width, _vk_swapchain_ext.height, 1,
                                    _vk_sample_count, _vk_swapchain_fmt, VK_IMAGE_TILING_OPTIMAL,
                                    VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &_vk_color_image);
    if (VK_SUCCESS != vk_res)
    {
        sbLogE("Failed to create color image (error = '{}')", getEnumValue(vk_res));
        return false;
    }

    VkImageViewCreateInfo view_info = {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = _vk_color_image.image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = _vk_swapchain_fmt;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;

    vk_res = vkCreateImageView(_vk_device, &view_info, nullptr, &_vk_color_image_view);
    if (VK_SUCCESS != vk_res)
    {
        sbLogE("Failed to create color image view (error = '{}')", getEnumValue(vk_res));
        return false;
    }

    return true;
}

void VulkanApp::destroyColorImage()
{
    if (_vk_color_image_view != VK_NULL_HANDLE)
    {
        vkDestroyImageView(_vk_device, _vk_color_image_view, nullptr);
        _vk_color_image_view = VK_NULL_HANDLE;
    }

    destroyVkImage(_vk_device, _vk_color_image);
    _vk_color_image = {};
}

void glfwErrorHandler(int code, const char * description)
{
    sbLogE("GLFW - {} (err:{})", description, code);
}

static void glfwFrameBufferResized(GLFWwindow * wnd, int width, int height)
{
    VulkanApp * sample_app = (VulkanApp *)glfwGetWindowUserPointer(wnd);
    sbAssert(nullptr != sample_app);
    sample_app->notifyTargetFrameBufferResized({(u32)width, (u32)height});
}

int main()
{
    char working_dir[LOCAL_PATH_MAX_LEN];
    getWorkingDirectory(working_dir);

    VFS::LayerInitDesc layer_desc = {makeHashStr("root"), "/", working_dir};

    VFS::InitDesc vfs_init = {.layers = {&layer_desc, 1}};

    VFS::initialize(vfs_init);

    constexpr u32 WINDOW_WIDTH = 800;
    constexpr u32 WINDOW_HEIGHT = 600;

    glfwSetErrorCallback(&glfwErrorHandler);

    if (sbDontExpect(!glfwInit(), "Failed to initialize glfw"))
    {
        return EXIT_FAILURE;
    }

    // informs glfw to not create opengl context which is its default behavior
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    GLFWwindow * const wnd = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Vulkan", nullptr, nullptr);

    if (sbDontExpect(nullptr == wnd, "Failed to create window"))
    {
        return EXIT_FAILURE;
    }

    VulkanApp sample_app;

    glfwSetFramebufferSizeCallback(wnd, &glfwFrameBufferResized);
    glfwSetWindowUserPointer(wnd, &sample_app);

    if (sbDontExpect(!sample_app.initialize(true, wnd, VulkanApp::DemoMode::MODEL), "Failed to initialize sample app"))
    {
        return EXIT_FAILURE;
    }

    while (!glfwWindowShouldClose(wnd))
    {
        sample_app.render();
        glfwPollEvents();
    }

    sample_app.terminate();

    glfwDestroyWindow(wnd);
    glfwTerminate();

    VFS::terminate();

    return 0;
}
