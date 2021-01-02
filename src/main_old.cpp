#include <glfw/glfw3.h>

#include <glm/glm.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan/vulkan.h>
#include <vulkan/vk_sdk_platform.h>

#include <core/string/string_format.h>
#include <core/error.h>
#include <core/log.h>
#include <core/platform.h>
#include <core/enum.h>
#include <core/string/utility.h>
#include <core/container/vector.h>
#include <core/conversion.h>
#include <core/io/file_system.h>
#include <core/io/file.h>
#include <core/io/path.h>
#include <core/io/file_system_layer.h>
#include <core/container/small_vector.h>
#include <core/memory/memory_arena.h>
#include <core/memory/memory.h>
#include <core/timer.h>

// TODO: create wrapper/ext
#define STB_IMAGE_IMPLEMENTATION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-id-macro"
#pragma clang diagnostic ignored "-Wcast-qual"
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wcomma"
#pragma clang diagnostic ignored "-Wdouble-promotion"
#pragma clang diagnostic ignored "-Wconversion"
#include <stb/stb_image.h>
#pragma clang diagnostic pop

#define TINYOBJLOADER_IMPLEMENTATION
#include <tinyobjloader/tiny_obj_loader.h>

#include <libc++/algorithm>
#include <libc++/limits>
#include <libc++/span>
#include <libc++/iterator>

// TODO: do we need these
#include <chrono>
#include <array>

using namespace sb;

static char const * const DEFAULT_VALIDATION_LAYERS[] = {
    "VK_LAYER_LUNARG_standard_validation", "VK_LAYER_LUNARG_parameter_validation", "VK_LAYER_LUNARG_core_validation"};

static char const * const REQUIRED_PHYSICAL_DEVICE_EXTENSIONS[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

struct DeviceSwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR m_caps;
    Vector<VkSurfaceFormatKHR> m_formats;
    Vector<VkPresentModeKHR> m_present_modes;
};

struct UniforBufferObject
{
    glm::mat4 m_model;
    glm::mat4 m_view;
    glm::mat4 m_proj;
};

struct Vertex
{
    glm::vec3 m_pos;
    glm::vec3 m_color;
    glm::vec2 m_text;

    static std::array<VkVertexInputAttributeDescription, 3> getInputAttrDesc()
    {
        std::array<VkVertexInputAttributeDescription, 3> vert_attr_desc = {};

        vert_attr_desc[0].binding = 0;
        vert_attr_desc[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        vert_attr_desc[0].location = 0;
        vert_attr_desc[0].offset = offsetof(Vertex, m_pos);
        vert_attr_desc[1].binding = 0;
        vert_attr_desc[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        vert_attr_desc[1].location = 1;
        vert_attr_desc[1].offset = offsetof(Vertex, m_color);
        vert_attr_desc[2].binding = 0;
        vert_attr_desc[2].format = VK_FORMAT_R32G32_SFLOAT;
        vert_attr_desc[2].location = 2;
        vert_attr_desc[2].offset = offsetof(Vertex, m_text);

        return vert_attr_desc;
    }
};

const Vertex TEST_MESH_VERTS[] = {
    {{-0.5f, -0.5f, 0.f}, {1.0f, 0.0f, 0.0f}, {1.f, 0.f}},   {{0.5f, -0.5f, 0.f}, {0.0f, 1.0f, 0.0f}, {0.f, 0.f}},
    {{0.5f, 0.5f, 0.f}, {0.0f, 0.0f, 1.0f}, {0.f, 1.f}},     {{-0.5f, 0.5f, 0.f}, {1.0f, 1.0f, 1.0f}, {1.f, 1.f}},

    {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.f, 0.f}}, {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.f, 0.f}},
    {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {0.f, 1.f}},   {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {1.f, 1.f}}};

const ui32 TEST_MESH_INDICES[] = {0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4};

class VkTestApp
{
    static constexpr ui32 MAX_FRAMES_IN_FLIGHT = 2;

    b8 m_verbose = true;
    b8 m_validation_enabled = true;
    VkInstance m_instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_dbg_cb = VK_NULL_HANDLE;
    VkPhysicalDevice m_phys_device = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_graphics_queue = VK_NULL_HANDLE;
    VkQueue m_present_queue = VK_NULL_HANDLE;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkSwapchainKHR m_swap_chain = VK_NULL_HANDLE;
    VkFormat m_swap_chain_img_fmt;
    VkExtent2D m_swap_chain_img_ext;
    Vector<VkImage> m_swap_chain_images;
    Vector<VkImageView> m_swap_chain_image_views;
    VkDescriptorSetLayout m_descriptor_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
    VkRenderPass m_render_pass = VK_NULL_HANDLE;
    VkPipeline m_graphics_pipeline;
    Vector<VkFramebuffer> m_swap_chain_frame_buffers;
    VkCommandPool m_cmd_pool = VK_NULL_HANDLE;
    Vector<VkCommandBuffer> m_cmd_buffers;
    VkSemaphore m_img_available_sems[MAX_FRAMES_IN_FLIGHT];
    VkSemaphore m_render_finished_sems[MAX_FRAMES_IN_FLIGHT];
    VkFence m_in_flight_fences[MAX_FRAMES_IN_FLIGHT];
    usize m_curr_frame = 0;
    b8 m_frame_buffer_resized = false;

    VkDescriptorPool m_desc_pool = VK_NULL_HANDLE;
    Vector<VkDescriptorSet> m_desc_sets;

    VkBuffer m_vertex_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_vertex_buffer_mem = VK_NULL_HANDLE;
    VkBuffer m_index_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_index_buffer_mem = VK_NULL_HANDLE;
    Vector<VkBuffer> m_uniform_buffers;
    Vector<VkDeviceMemory> m_uniform_buffers_mem;

    VkImage m_depth_img = VK_NULL_HANDLE;
    VkDeviceMemory m_depth_img_mem = VK_NULL_HANDLE;
    VkImageView m_depth_img_view = VK_NULL_HANDLE;

    VkImage m_texture_img = VK_NULL_HANDLE;
    VkDeviceMemory m_texture_mem = VK_NULL_HANDLE;
    VkImageView m_texture_img_view = VK_NULL_HANDLE;
    VkSampler m_texture_sampler = VK_NULL_HANDLE;

    usize m_model_indices_cnt = 0;

    float m_camera_x = 30.f;
    float m_camera_y = 0.f;
    float m_camera_z = 0.f;

    struct QueueFamiliesDesc
    {
        si32 m_graphics_idx = -1;
        si32 m_present_idx = -1;

        b8 isValid()
        {
            return (-1 != m_graphics_idx) && (-1 != m_present_idx);
        }
    };

public:
    b8 initialize(GLFWwindow * wnd_hdl, b8 enable_validation, b8 verbose)
    {
        m_verbose = verbose;
        m_validation_enabled = enable_validation;
        m_curr_frame = 0;
        m_frame_buffer_resized = false;

        if (!createInstance())
        {
            return false;
        }

        if (m_validation_enabled)
        {
            setupDebugCallback();
        }

        if (!createWindowSurface(wnd_hdl))
        {
            return false;
        }

        if (!selectPhysicalDevice())
        {
            return false;
        }

        if (!createLogicalDevice())
        {
            return false;
        }

        if (!createSwapChain(wnd_hdl))
        {
            return false;
        }

        if (!createSwapChainImageViews())
        {
            return false;
        }

        if (!createRenderPass())
        {
            return false;
        }

        if (!createDescriptorSetLayout())
        {
            return false;
        }

        if (!createGraphicsPipeline())
        {
            return false;
        }

        if (!createFrameBuffers())
        {
            return false;
        }

        if (!createCommandPool())
        {
            return false;
        }

        if (!createDepthBuffer())
        {
            return false;
        }

        if (!createTextureImage())
        {
            return false;
        }

        if (!createTextureImageVew())
        {
            return false;
        }

        if (!createTextureSampler())
        {
            return false;
        }

        Vector<Vertex> vertices;
        Vector<ui32> indices;
        if (!loadModel(vertices, indices))
        {
            return false;
        }

        if (!createVertexBuffer(vertices))
        {
            return false;
        }

        if (!createIndexBuffer(indices))
        {
            return false;
        }

        if (!createUniformBuffers())
        {
            return false;
        }

        if (!createDescriptorPool())
        {
            return false;
        }

        if (!createDescriptorSets())
        {
            return false;
        }

        if (!createCommandBuffers())
        {
            return false;
        }

        if (!createSyncObjects())
        {
            return false;
        }

        return true;
    }

    void terminate()
    {
        vkDeviceWaitIdle(m_device);

        destroySyncObjects();

        destroyCommandBuffers();

        destroyDescriptorSets();

        destroyDescriptorPool();

        destroyUniformBuffers();

        destroyIndexBuffer();

        destroyVertexBuffer();

        destroyTextureSampler();

        destroyTextureImageView();

        destroyTextureImage();

        destroyDepthBuffer();

        destroyCommandPool();

        destroyFrameBuffers();

        destroyGraphicsPipeline();

        destroyDescriptorSetLayout();

        destroyRenderPass();

        destroySwapChainImageViews();

        destroySwapChain();

        m_graphics_queue = VK_NULL_HANDLE;
        m_present_queue = VK_NULL_HANDLE;

        destroyLogicalDevice();

        destroyWindowSurface();

        if (m_validation_enabled)
        {
            removeDebugCallback();
        }

        m_phys_device = VK_NULL_HANDLE;

        destroyInstance();
    }

    b8 render(GLFWwindow * wnd_hdl)
    {
        vkWaitForFences(m_device, 1, &m_in_flight_fences[m_curr_frame], VK_TRUE, wstd::numeric_limits<ui64>::max());

        ui32 img_index = 0;
        auto const acq_res = vkAcquireNextImageKHR(m_device, m_swap_chain, wstd::numeric_limits<ui64>::max(),
                                                   m_img_available_sems[m_curr_frame], VK_NULL_HANDLE, &img_index);

        if (acq_res == VK_ERROR_OUT_OF_DATE_KHR)
        {
            recreateSwapChain(wnd_hdl);
            m_frame_buffer_resized = false;
            return true;
        }
        else if ((acq_res != VK_SUCCESS) && (acq_res != VK_SUBOPTIMAL_KHR))
        {
            sbLogE("vkAcquireNextImageKHR failed");
            return false;
        }

        updateUniformBuffer(img_index);

        VkSubmitInfo sub_info = {};
        sub_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore signal_sems[] = {m_render_finished_sems[m_curr_frame]};
        VkSemaphore wait_sems[] = {m_img_available_sems[m_curr_frame]};
        VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

        sub_info.waitSemaphoreCount = 1;
        sub_info.pWaitSemaphores = wait_sems;
        sub_info.pWaitDstStageMask = wait_stages;
        sub_info.commandBufferCount = 1;
        sub_info.pCommandBuffers = &m_cmd_buffers[img_index];
        sub_info.signalSemaphoreCount = 1;
        sub_info.pSignalSemaphores = signal_sems;

        vkResetFences(m_device, 1, &m_in_flight_fences[m_curr_frame]);

        if (VK_SUCCESS != vkQueueSubmit(m_graphics_queue, 1, &sub_info, m_in_flight_fences[m_curr_frame]))
        {
            sbLogE("vkQueueSubmit failed");

            return false;
        }

        VkPresentInfoKHR present_info = {};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = signal_sems;
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &m_swap_chain;
        present_info.pImageIndices = &img_index;
        present_info.pResults = nullptr;

        auto const present_res = vkQueuePresentKHR(m_present_queue, &present_info);

        if ((present_res == VK_ERROR_OUT_OF_DATE_KHR) || (present_res == VK_SUBOPTIMAL_KHR) || m_frame_buffer_resized)
        {
            recreateSwapChain(wnd_hdl);
            m_frame_buffer_resized = false;
        }
        else if (present_res != VK_SUCCESS)
        {
            sbLogE("vkQueuePresentKHR failed");
            return false;
        }

        vkQueueWaitIdle(m_present_queue);

        m_curr_frame = (m_curr_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        return true;
    }

    void surfaceResized()
    {
        m_frame_buffer_resized = true;
    }

private:
    void recreateSwapChain(GLFWwindow * wnd_hdl)
    {
        si32 width = 0, height = 0;

        // When minimized, the window size is {0,0}
        // which is an invalid frame buffer size
        // This solution is really hacky bug good enough
        // for a sample app
        glfwGetFramebufferSize(wnd_hdl, &width, &height);
        while (width == 0 || height == 0)
        {
            glfwWaitEvents();
            glfwGetFramebufferSize(wnd_hdl, &width, &height);
        }

        // Wait for the GPU to be done with resources
        vkDeviceWaitIdle(m_device);

        destroyFrameBuffers();
        vkFreeCommandBuffers(m_device, m_cmd_pool, (ui32)m_cmd_buffers.size(),
                             m_cmd_buffers.data()); // We have to free command buffers manually because we are not
                                                    // destroying the associated command pool which is owning them
        destroyCommandBuffers();
        destroyGraphicsPipeline();
        destroyRenderPass();
        destroySwapChainImageViews();
        destroySwapChain();

        createSwapChain(wnd_hdl);
        createSwapChainImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFrameBuffers();
        createCommandBuffers();
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity_flags,
                                                        VkDebugUtilsMessageTypeFlagsEXT /*msg_type*/,
                                                        VkDebugUtilsMessengerCallbackDataEXT const * cb_data,
                                                        void * /*user_data*/)
    {
        if (severity_flags & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        {
            sbLogE("[Vulkan] {}", cb_data->pMessage);
        }
        else if (severity_flags & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        {
            sbLogW("[Vulkan] {}", cb_data->pMessage);
        }
        else if (severity_flags & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
        {
            sbLogI("[Vulkan] {}", cb_data->pMessage);
        }
        else if (severity_flags & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT)
        {
            sbLogD("[Vulkan] {}", cb_data->pMessage);
        }

        return VK_FALSE; // Do not crash the vulkan driver
    }

    static Vector<char const *> getRequiredExtensions(b8 enable_validation)
    {
        Vector<char const *> required_exts;

        ui32 glfw_ext_cnt = 0;
        char const ** glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_ext_cnt);

        required_exts.insert(required_exts.begin(), glfw_extensions, glfw_extensions + glfw_ext_cnt);

        if (enable_validation)
        {
            required_exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return required_exts;
    }

    static QueueFamiliesDesc findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface)
    {
        QueueFamiliesDesc queues_desc;

        ui32 family_cnt = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &family_cnt, nullptr);

        if (0 != family_cnt)
        {
            SmallVector<VkQueueFamilyProperties, 10> props(family_cnt);
            vkGetPhysicalDeviceQueueFamilyProperties(device, &family_cnt, props.data());

            si32 curr_idx = 0;
            for (auto const & queue_props : props)
            {
                if (queue_props.queueCount > 0)
                {
                    if (queue_props.queueFlags & VK_QUEUE_GRAPHICS_BIT)
                    {
                        queues_desc.m_graphics_idx = curr_idx;
                    }

                    VkBool32 support_present = false;
                    vkGetPhysicalDeviceSurfaceSupportKHR(device, numericCast<ui32>(curr_idx), surface,
                                                         &support_present);

                    if (support_present)
                    {
                        queues_desc.m_present_idx = curr_idx;
                    }
                }

                if (queues_desc.isValid())
                {
                    break;
                }

                ++curr_idx;
            }
        }

        return queues_desc;
    }

    static b8 checkDeviceExtensions(VkPhysicalDevice device, wstd::span<char const * const> extensions)
    {
        ui32 device_ext_cnt = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &device_ext_cnt, nullptr);

        if (device_ext_cnt != 0)
        {
            SmallVector<VkExtensionProperties, 15> available_exts(device_ext_cnt);
            vkEnumerateDeviceExtensionProperties(device, nullptr, &device_ext_cnt, available_exts.data());

            for (auto const ext_name : extensions)
            {
                auto ext_iter = wstd::find_if(wstd::begin(available_exts), wstd::end(available_exts),
                                              [ext_name](VkExtensionProperties const & props) {
                                                  return 0 == strcmp(props.extensionName, ext_name);
                                              });

                if (ext_iter == wstd::end(available_exts))
                {
                    return false;
                }
            }
        }

        return true;
    }

    static DeviceSwapChainSupportDetails getDeviceSwapChainSupportDetails(VkPhysicalDevice device, VkSurfaceKHR surface)
    {
        DeviceSwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.m_caps);

        ui32 format_cnt = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_cnt, nullptr);
        details.m_formats.resize(format_cnt);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_cnt, details.m_formats.data());

        ui32 present_cnt = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_cnt, nullptr);
        details.m_present_modes.resize(present_cnt);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_cnt, details.m_present_modes.data());

        return details;
    }

    static b8 checkDeviceSwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface)
    {
        auto const details = getDeviceSwapChainSupportDetails(device, surface);

        return !details.m_formats.empty() && !details.m_present_modes.empty();
    }

    static b8 isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR surface)
    {
        VkPhysicalDeviceFeatures features;
        vkGetPhysicalDeviceFeatures(device, &features);

        return findQueueFamilies(device, surface).isValid() &&
               checkDeviceExtensions(device, REQUIRED_PHYSICAL_DEVICE_EXTENSIONS) &&
               checkDeviceSwapChainSupport(device, surface) && features.samplerAnisotropy;
    }

    VkSurfaceFormatKHR chooseSwapChainSurfaceFormat(wstd::span<VkSurfaceFormatKHR const> formats)
    {
        if ((1 == formats.size()) && (VK_FORMAT_UNDEFINED == formats[0].format))
        {
            return {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
        }

        auto const format_iter =
            wstd::find_if(wstd::begin(formats), wstd::end(formats), [](VkSurfaceFormatKHR const & fmt) {
                return (fmt.format == VK_FORMAT_B8G8R8A8_UNORM) &&
                       (fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR);
            });

        if (wstd::end(formats) != format_iter)
        {
            return *format_iter;
        }

        return formats[0];
    }

    VkPresentModeKHR chooseSwapChainPresentMode(wstd::span<VkPresentModeKHR const> present_modes)
    {
        auto const present_mode_iter =
            wstd::find(wstd::begin(present_modes), wstd::end(present_modes), VK_PRESENT_MODE_MAILBOX_KHR);

        if (present_mode_iter != wstd::end(present_modes))
        {
            return VK_PRESENT_MODE_MAILBOX_KHR;
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapChainImageExtent(VkSurfaceCapabilitiesKHR const & caps, GLFWwindow * wnd_hdl)
    {
        if (caps.currentExtent.width != wstd::numeric_limits<ui32>::max())
        {
            return caps.currentExtent;
        }
        else
        {
            si32 width, height;
            glfwGetFramebufferSize(wnd_hdl, &width, &height);

            return {
                wstd::max(caps.minImageExtent.width, wstd::min(caps.maxImageExtent.width, numericCast<ui32>(width))),
                wstd::max(caps.minImageExtent.height,
                          wstd::min(caps.maxImageExtent.height, numericCast<ui32>(height)))};
        }
    }

    static MemoryArena readFile(char const * file_path, AllocatorView alloc)
    {
        File file_hdl = FS::openFileRead(file_path);

        if (!file_hdl.isNull())
        {
            auto const data_size = numericCast<usize>(file_hdl.getLength());

            if (data_size != 0)
            {
                ui8 * const data = static_cast<ui8 *>(alloc.allocate(data_size));
                auto const byte_cnt = file_hdl.read({data, numericCast<sptrdiff>(data_size)});

                sbWarn(data_size == numericCast<usize>(byte_cnt));

                return {data, data_size};
            }
        }

        return {};
    }

    VkShaderModule createShaderModule(MemoryArena byte_code)
    {
        VkShaderModuleCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.pCode = reinterpret_cast<ui32 const *>(byte_code.m_ptr);
        create_info.codeSize = numericCast<usize>(byte_code.m_size);

        VkShaderModule shader_module;
        if (VK_SUCCESS != vkCreateShaderModule(m_device, &create_info, nullptr, &shader_module))
        {
            return VK_NULL_HANDLE;
        }

        return shader_module;
    }

    void destroyShaderModule(VkShaderModule shader_module)
    {
        vkDestroyShaderModule(m_device, shader_module, nullptr);
    }

    b8 createSyncObjects()
    {
        VkSemaphoreCreateInfo sem_info = {};
        sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fence_info = {};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (ui32 idx = 0; idx != MAX_FRAMES_IN_FLIGHT; ++idx)
        {
            if ((VK_SUCCESS != vkCreateSemaphore(m_device, &sem_info, nullptr, &m_img_available_sems[idx])) ||
                (VK_SUCCESS != vkCreateSemaphore(m_device, &sem_info, nullptr, &m_render_finished_sems[idx])) ||
                (VK_SUCCESS != vkCreateFence(m_device, &fence_info, nullptr, &m_in_flight_fences[idx])))
            {
                sbLogE("vkCreateSemaphore failed");

                return false;
            }
        }

        return true;
    }

    void destroySyncObjects()
    {
        for (ui32 idx = 0; idx != MAX_FRAMES_IN_FLIGHT; ++idx)
        {
            vkDestroySemaphore(m_device, m_img_available_sems[idx], nullptr);
            vkDestroySemaphore(m_device, m_render_finished_sems[idx], nullptr);
            vkDestroyFence(m_device, m_in_flight_fences[idx], nullptr);
        }

        zeroStructArray(m_img_available_sems);
        zeroStructArray(m_render_finished_sems);
        zeroStructArray(m_in_flight_fences);
    }

    ui32 findVkDeviceMemoryTypeIndex(ui32 mem_types_mask, VkMemoryPropertyFlags props)
    {
        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(m_phys_device, &mem_props);

        for (ui32 i = 0; i != mem_props.memoryTypeCount; ++i)
        {
            if ((mem_types_mask & (1 << i)) && ((mem_props.memoryTypes[i].propertyFlags & props) == props))
            {
                return i;
            }
        }

        return wstd::numeric_limits<ui32>::max();
    }

    b8 createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_pros, VkBuffer & buffer,
                    VkDeviceMemory & buffer_mem)
    {
        VkBufferCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = size;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // This is the queue family ownership ... this buffer will only be
                                                      // used with the graphics queue

        if (VK_SUCCESS != vkCreateBuffer(m_device, &info, nullptr, &buffer))
        {
            sbLogE("vkCreateBuffer failed");
            return false;
        }

        VkMemoryRequirements mem_req;
        vkGetBufferMemoryRequirements(m_device, buffer, &mem_req);

        ui32 const mem_type_idx = findVkDeviceMemoryTypeIndex(mem_req.memoryTypeBits, mem_pros);
        if (wstd::numeric_limits<ui32>::max() == mem_type_idx)
        {
            sbLogE("failed to find memory type index for the buffer");
            return false;
        }

        VkMemoryAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_req.size;
        alloc_info.memoryTypeIndex = mem_type_idx;

        if (VK_SUCCESS != vkAllocateMemory(m_device, &alloc_info, nullptr, &buffer_mem))
        {
            sbLogE("Buffer device memory allocation failed");
            return false;
        }

        // Associate buffer object and its memory
        vkBindBufferMemory(m_device, buffer, buffer_mem, 0);

        return true;
    }

    void destroyBuffer(VkBuffer buffer, VkDeviceMemory buffer_mem)
    {
        vkFreeMemory(m_device, buffer_mem, nullptr);
        vkDestroyBuffer(m_device, buffer, nullptr);
    }

    void copyBuffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size)
    {
        VkCommandBuffer copy_cmd_buffer = beginOnTimeCommandBuffer();

        VkBufferCopy copy_region = {};
        copy_region.srcOffset = 0;
        copy_region.dstOffset = 0;
        copy_region.size = size;
        vkCmdCopyBuffer(copy_cmd_buffer, src_buffer, dst_buffer, 1, &copy_region);

        endOneTimeCommandBuffer(copy_cmd_buffer);
    }

    b8 createDescriptorSets()
    {
        m_desc_sets.resize(m_swap_chain_images.size());
        Vector<VkDescriptorSetLayout> layouts(m_swap_chain_images.size(), m_descriptor_set_layout);

        VkDescriptorSetAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorSetCount = (ui32)m_swap_chain_images.size();
        alloc_info.pSetLayouts = layouts.data();
        alloc_info.descriptorPool = m_desc_pool;

        if (VK_SUCCESS != vkAllocateDescriptorSets(m_device, &alloc_info, m_desc_sets.data()))
        {
            sbLogE("vkAllocateDescriptorSets failed");
            return false;
        }

        for (usize i = 0; i != m_swap_chain_images.size(); ++i)
        {
            VkDescriptorBufferInfo buffer_info = {};
            buffer_info.buffer = m_uniform_buffers[i];
            buffer_info.offset = 0;
            buffer_info.range = sizeof(UniforBufferObject);

            VkDescriptorImageInfo img_info = {};
            img_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            img_info.imageView = m_texture_img_view;
            img_info.sampler = m_texture_sampler;

            VkWriteDescriptorSet write_set[2] = {};

            write_set[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_set[0].dstSet = m_desc_sets[i];
            write_set[0].dstBinding = 0;
            write_set[0].dstArrayElement = 0;
            write_set[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            write_set[0].descriptorCount = 1;
            write_set[0].pBufferInfo = &buffer_info;

            write_set[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_set[1].dstSet = m_desc_sets[i];
            write_set[1].dstBinding = 1;
            write_set[1].dstArrayElement = 0;
            write_set[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_set[1].descriptorCount = 1;
            write_set[1].pImageInfo = &img_info;

            vkUpdateDescriptorSets(m_device, (ui32)wstd::size(write_set), write_set, 0, nullptr);
        }

        return true;
    }

    void destroyDescriptorSets()
    {
        // no need to free descriptor sets because they are cleaned up automatically by
        // the associdated pool
        m_desc_sets.clear();
    }

    b8 createDescriptorPool()
    {
        VkDescriptorPoolSize desc_sizes[2] = {};

        desc_sizes[0].descriptorCount = (ui32)m_swap_chain_image_views.size();
        desc_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

        desc_sizes[1].descriptorCount = (ui32)m_swap_chain_image_views.size();
        desc_sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

        VkDescriptorPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.poolSizeCount = (ui32)std::size(desc_sizes);
        pool_info.pPoolSizes = desc_sizes;
        pool_info.maxSets = (ui32)m_swap_chain_image_views.size();

        if (VK_SUCCESS != vkCreateDescriptorPool(m_device, &pool_info, nullptr, &m_desc_pool))
        {
            sbLogE("vkCreateDescriptorPool failed");
            return false;
        }

        return true;
    }

    void destroyDescriptorPool()
    {
        vkDestroyDescriptorPool(m_device, m_desc_pool, nullptr);
        m_desc_pool = VK_NULL_HANDLE;
    }

    b8 createUniformBuffers()
    {
        m_uniform_buffers.resize(m_swap_chain_frame_buffers.size());
        m_uniform_buffers_mem.resize(m_swap_chain_frame_buffers.size());

        for (usize i = 0; i != m_uniform_buffers.size(); ++i)
        {
            createBuffer(sizeof(UniforBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                         m_uniform_buffers[i], m_uniform_buffers_mem[i]);
        }

        return true;
    }

    void destroyUniformBuffers()
    {
        for (usize i = 0; i != m_uniform_buffers.size(); ++i)
        {
            destroyBuffer(m_uniform_buffers[i], m_uniform_buffers_mem[i]);
        }

        m_uniform_buffers.clear();
        m_uniform_buffers_mem.clear();
    }

    void updateUniformBuffer(ui32 img_idx)
    {
        static auto const start_time = std::chrono::high_resolution_clock::now();

        auto const curr_time = std::chrono::high_resolution_clock::now();
        float const delta_time =
            std::chrono::duration<float, std::chrono::seconds::period>(curr_time - start_time).count();

        UniforBufferObject ubo = {};
        ubo.m_model = glm::rotate(glm::mat4(1.0f), delta_time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.m_view = glm::lookAt(glm::vec3(m_camera_x, m_camera_y, m_camera_z), glm::vec3(0.0f, 0.0f, 0.0f),
                                 glm::vec3(0.0f, 1.0f, 0.0f));
        ubo.m_proj = glm::perspective(glm::radians(45.0f),
                                      m_swap_chain_img_ext.width / (float)m_swap_chain_img_ext.height, 0.1f, 200.0f);
        ubo.m_proj[1][1] *=
            -1; // GLM was originally designed for OpenGL, where the Y coordinate of the clip coordinates is inverted

        void * data = nullptr;
        vkMapMemory(m_device, m_uniform_buffers_mem[img_idx], 0, sizeof(UniforBufferObject), 0, &data);
        memcpy(data, &ubo, sizeof(UniforBufferObject));
        vkUnmapMemory(m_device, m_uniform_buffers_mem[img_idx]);
    }

    b8 createIndexBuffer(Vector<ui32> & indices)
    {
        VkBuffer staging_buffer;
        VkDeviceMemory staging_buffer_mem;
        void * buffer_data = nullptr;

        auto const indexBufferMemSize = getVectorMemorySize(indices);

        if (!createBuffer(indexBufferMemSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer,
                          staging_buffer_mem))
        {
            sbLogE("Failed to create test mesh staging index buffer");
            return false;
        }

        // Make the memory available on CPU for copy i.e. get a CPU visible pointer to this memory
        vkMapMemory(m_device, staging_buffer_mem, 0, indexBufferMemSize, 0, &buffer_data);
        memcpy(buffer_data, indices.data(), indexBufferMemSize);
        vkUnmapMemory(m_device, staging_buffer_mem);

        if (!createBuffer(indexBufferMemSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_index_buffer, m_index_buffer_mem))
        {
            sbLogE("Failed to create test mesh index buffer");
            return false;
        }

        copyBuffer(staging_buffer, m_index_buffer, indexBufferMemSize);

        destroyBuffer(staging_buffer, staging_buffer_mem);

        m_model_indices_cnt = indices.size();

        return true;
    }

    void destroyIndexBuffer()
    {
        vkDestroyBuffer(m_device, m_index_buffer, nullptr);
        m_index_buffer = VK_NULL_HANDLE;

        vkFreeMemory(m_device, m_index_buffer_mem, nullptr);
        m_index_buffer_mem = VK_NULL_HANDLE;
    }

    b8 createImage(ui32 tex_width, ui32 tex_height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
                   VkMemoryPropertyFlags properties, VkImage & img, VkDeviceMemory & img_mem)
    {
        VkImageCreateInfo img_info = {};
        img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        img_info.imageType = VK_IMAGE_TYPE_2D;
        img_info.extent.width = tex_width;
        img_info.extent.height = tex_height;
        img_info.extent.depth = 1;
        img_info.mipLevels = 1;
        img_info.arrayLayers = 1;
        img_info.format = format;
        img_info.tiling = tiling;
        img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        img_info.usage = usage;
        img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        img_info.samples = VK_SAMPLE_COUNT_1_BIT;
        img_info.flags = 0;

        if (VK_SUCCESS != vkCreateImage(m_device, &img_info, nullptr, &img))
        {
            sbLogE("Failed to create image");
            return false;
        }

        VkMemoryRequirements img_mem_req;
        vkGetImageMemoryRequirements(m_device, img, &img_mem_req);

        VkMemoryAllocateInfo img_alloc_info = {};
        img_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        img_alloc_info.allocationSize = img_mem_req.size;
        img_alloc_info.memoryTypeIndex = findVkDeviceMemoryTypeIndex(img_mem_req.memoryTypeBits, properties);

        if (VK_SUCCESS != vkAllocateMemory(m_device, &img_alloc_info, nullptr, &img_mem))
        {
            sbLogE("Failed to allocate memory for the image");
            return false;
        }

        vkBindImageMemory(m_device, img, img_mem, 0);

        return true;
    }

    VkCommandBuffer beginOnTimeCommandBuffer()
    {
        VkCommandBufferAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandBufferCount = 1;
        alloc_info.commandPool = m_cmd_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

        VkCommandBuffer cmd_buffer;

        vkAllocateCommandBuffers(m_device, &alloc_info, &cmd_buffer);

        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(cmd_buffer, &begin_info);

        return cmd_buffer;
    }

    void endOneTimeCommandBuffer(VkCommandBuffer cmd_buffer)
    {
        vkEndCommandBuffer(cmd_buffer);

        VkSubmitInfo sub_info = {};
        sub_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        sub_info.commandBufferCount = 1;
        sub_info.pCommandBuffers = &cmd_buffer;

        vkQueueSubmit(m_graphics_queue, 1, &sub_info, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_graphics_queue);

        vkFreeCommandBuffers(m_device, m_cmd_pool, 1, &cmd_buffer);
    }

    VkImageView createImageView(VkImage img, VkFormat fmt)
    {
        VkImageView img_view = VK_NULL_HANDLE;

        VkImageViewCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        info.image = img;
        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        info.format = fmt;
        info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        info.subresourceRange.baseMipLevel = 0;
        info.subresourceRange.levelCount = 1;
        info.subresourceRange.baseArrayLayer = 0;
        info.subresourceRange.layerCount = 1;

        if (VK_SUCCESS != vkCreateImageView(m_device, &info, nullptr, &img_view))
        {
            sbLogE("Failed to create image view");
        }

        return img_view;
    }

    void destroyTextureSampler()
    {
        vkDestroySampler(m_device, m_texture_sampler, nullptr);
        m_texture_sampler = VK_NULL_HANDLE;
    }

    b8 createTextureSampler()
    {
        VkSamplerCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.magFilter = VK_FILTER_LINEAR;
        info.minFilter = VK_FILTER_LINEAR;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.anisotropyEnable = VK_TRUE;
        info.maxAnisotropy = 16;
        info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        info.unnormalizedCoordinates = VK_FALSE;
        info.compareEnable = VK_FALSE;
        info.compareOp = VK_COMPARE_OP_ALWAYS;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        info.mipLodBias = 0.f;
        info.minLod = 0.f;
        info.maxLod = 0.f;

        if (VK_SUCCESS != vkCreateSampler(m_device, &info, nullptr, &m_texture_sampler))
        {
            sbLogE("failed to create texture sampler");
            return false;
        }

        return true;
    }

    b8 createTextureImageVew()
    {
        m_texture_img_view = createImageView(m_texture_img, VK_FORMAT_R8G8B8A8_UNORM);

        if (VK_NULL_HANDLE == m_texture_img_view)
        {
            sbLogE("Failed to create texture image view");
            return false;
        }

        return true;
    }

    void destroyTextureImageView()
    {
        vkDestroyImageView(m_device, m_texture_img_view, nullptr);

        m_texture_img_view = VK_NULL_HANDLE;
    }

    VkFormat findSupportedFormat(wstd::span<VkFormat const> formats, VkImageTiling tiling,
                                 VkFormatFeatureFlags features)
    {
        for (auto format : formats)
        {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(m_phys_device, format, &props);

            if ((tiling == VK_IMAGE_TILING_LINEAR) && ((features & props.linearTilingFeatures) == features))
            {
                return format;
            }
            else if ((tiling == VK_IMAGE_TILING_OPTIMAL) && ((features & props.optimalTilingFeatures) == features))
            {
                return format;
            }
        }

        return VK_FORMAT_UNDEFINED;
    }

    VkFormat findDepthBufferFormat()
    {
        VkFormat const depth_formats[] = {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
                                          VK_FORMAT_D24_UNORM_S8_UINT};

        return findSupportedFormat(depth_formats, VK_IMAGE_TILING_OPTIMAL,
                                   VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }

    b8 hasStencilComponent(VkFormat format)
    {
        return (VK_FORMAT_D32_SFLOAT_S8_UINT == format) || (VK_FORMAT_D24_UNORM_S8_UINT == format);
    }

    void destroyDepthBuffer()
    {
        if (m_depth_img_view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(m_device, m_depth_img_view, nullptr);
        }

        if (m_depth_img != VK_NULL_HANDLE)
        {
            vkFreeMemory(m_device, m_depth_img_mem, nullptr);
            vkDestroyImage(m_device, m_depth_img, nullptr);
        }

        m_depth_img = VK_NULL_HANDLE;
        m_depth_img_mem = VK_NULL_HANDLE;
        m_depth_img_view = VK_NULL_HANDLE;
    }

    b8 createDepthBuffer()
    {
        VkFormat const fmt = findDepthBufferFormat();
        sbAssert(fmt != VK_FORMAT_UNDEFINED);

        if (!createImage(m_swap_chain_img_ext.width, m_swap_chain_img_ext.height, fmt, VK_IMAGE_TILING_OPTIMAL,
                         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_depth_img,
                         m_depth_img_mem))
        {
            sbLogE("failed to create depth buffer image");
            return false;
        }

        m_depth_img_view = createImageView(m_depth_img, fmt);
        if (VK_NULL_HANDLE == m_depth_img_view)
        {
            sbLogE("failed to create depth buffer image view");
            return false;
        }

        return true;
    }

    void destroyTextureImage()
    {
        vkFreeMemory(m_device, m_texture_mem, nullptr);
        vkDestroyImage(m_device, m_texture_img, nullptr);

        m_texture_mem = VK_NULL_HANDLE;
        m_texture_img = VK_NULL_HANDLE;
    }

    b8 createTextureImage()
    {
        si32 tex_width = 0;
        si32 tex_height = 0;
        si32 tex_channels = 0;

        char abs_file_path[PPath::MAX_LEN];
        strCpyT(abs_file_path, FS::getLayerPhysicalPath(HashStr{"data"}));
        PPath::concat(abs_file_path, "flash_light.png");

        stbi_uc * pixels = stbi_load(abs_file_path, &tex_width, &tex_height, &tex_channels, STBI_rgb_alpha);

        if (nullptr == pixels)
        {
            sbLogE("failed to load texture image");
            return false;
        }

        VkDeviceSize const tex_size = numericCast<usize>(tex_width * tex_height * 4);

        VkBuffer staging_buffer;
        VkDeviceMemory staging_buffer_mem;

        if (!createBuffer(tex_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer,
                          staging_buffer_mem))
        {
            sbLogE("Failed to create texture image staging buffer");
            return false;
        }

        void * data;
        vkMapMemory(m_device, staging_buffer_mem, 0, tex_size, 0, &data);
        memcpy(data, pixels, tex_size);
        vkUnmapMemory(m_device, staging_buffer_mem);

        stbi_image_free(pixels);

        if (!createImage(numericCast<ui32>(tex_width), numericCast<ui32>(tex_height), VK_FORMAT_R8G8B8A8_UNORM,
                         VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_texture_img, m_texture_mem))
        {
            sbLogE("Failed to create texture image");
            return false;
        }

        transitionImageLayout(m_texture_img, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        copyBufferToImage(staging_buffer, m_texture_img, numericCast<ui32>(tex_width), numericCast<ui32>(tex_height));

        transitionImageLayout(m_texture_img, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                              VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

        destroyBuffer(staging_buffer, staging_buffer_mem);

        return true;
    }

    void copyBufferToImage(VkBuffer src_buffer, VkImage dst_image, ui32 width, ui32 height)
    {
        VkCommandBuffer cmd_buffer = beginOnTimeCommandBuffer();

        VkBufferImageCopy copy_info = {};
        copy_info.bufferOffset = 0;
        copy_info.bufferRowLength = 0;
        copy_info.bufferImageHeight = 0;
        copy_info.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_info.imageSubresource.baseArrayLayer = 0;
        copy_info.imageSubresource.layerCount = 1;
        copy_info.imageSubresource.mipLevel = 0;
        copy_info.imageOffset = {};
        copy_info.imageExtent = {width, height, 1};

        vkCmdCopyBufferToImage(cmd_buffer, src_buffer, dst_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_info);

        endOneTimeCommandBuffer(cmd_buffer);
    }

    void transitionImageLayout(VkImage img, VkFormat fmt, VkImageLayout src_layout, VkImageLayout dst_layout,
                               VkAccessFlags src_access, VkAccessFlags dst_access, VkPipelineStageFlags src_stage,
                               VkPipelineStageFlags dst_stage)
    {
        VkCommandBuffer cmd_buffer = beginOnTimeCommandBuffer();

        VkImageMemoryBarrier img_barrier = {};
        img_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        img_barrier.oldLayout = src_layout;
        img_barrier.newLayout = dst_layout;
        img_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        img_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        img_barrier.image = img;
        img_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        img_barrier.subresourceRange.layerCount = 1;
        img_barrier.subresourceRange.baseArrayLayer = 0;
        img_barrier.subresourceRange.levelCount = 1;
        img_barrier.subresourceRange.baseMipLevel = 0;
        img_barrier.srcAccessMask = src_access;
        img_barrier.dstAccessMask = dst_access;

        vkCmdPipelineBarrier(cmd_buffer, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &img_barrier);

        endOneTimeCommandBuffer(cmd_buffer);
    }

    b8 loadModel(Vector<Vertex> & vertices, Vector<ui32> & indices)
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string err, warn;

        char model_path[PPath::MAX_LEN];
        strCpyT(model_path, FS::getLayerPhysicalPath(HashStr{"data"}));
        PPath::concat(model_path, "flash_light.obj");

        sbLogI(model_path);

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, model_path))
        {
            sbLogE("Failed to load test model from disc");
            return false;
        }

        for (auto const & shape : shapes)
        {
            for (auto const & index : shape.mesh.indices)
            {
                Vertex vert;

                vert.m_pos = {attrib.vertices[3 * index.vertex_index + 0], attrib.vertices[3 * index.vertex_index + 1],
                              attrib.vertices[3 * index.vertex_index + 2]};

                vert.m_text = {attrib.texcoords[2 * index.texcoord_index + 0],
                               attrib.texcoords[2 * index.texcoord_index + 1]};

                vert.m_color = {1.f, 1.f, 1.f};

                vertices.push_back(vert);
                indices.push_back(numericCast<ui32>(indices.size()));
            }
        }

        return true;
    }

    template <typename TVector>
    static usize getVectorMemorySize(TVector const & vect)
    {
        return sizeof(typename TVector::value_type) * vect.size();
    }

    b8 createVertexBuffer(Vector<Vertex> const & vertices)
    {
        VkBuffer staging_buffer;
        VkDeviceMemory staging_buffer_mem;
        void * buffer_data = nullptr;

        auto const vertBufferMemSize = getVectorMemorySize(vertices);

        if (!createBuffer(vertBufferMemSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer,
                          staging_buffer_mem))
        {
            sbLogE("Failed to create test mesh staging vertex buffer");
            return false;
        }

        // Make the memory available on CPU for copy i.e. get a CPU visible pointer to this memory
        vkMapMemory(m_device, staging_buffer_mem, 0, vertBufferMemSize, 0, &buffer_data);
        memcpy(buffer_data, vertices.data(), vertBufferMemSize);
        vkUnmapMemory(m_device, staging_buffer_mem);

        if (!createBuffer(vertBufferMemSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_vertex_buffer, m_vertex_buffer_mem))
        {
            sbLogE("Failed to create test mesh vertex buffer");
            return false;
        }

        copyBuffer(staging_buffer, m_vertex_buffer, vertBufferMemSize);

        destroyBuffer(staging_buffer, staging_buffer_mem);

        return true;
    }

    void destroyVertexBuffer()
    {
        vkDestroyBuffer(m_device, m_vertex_buffer, nullptr);
        m_vertex_buffer = VK_NULL_HANDLE;

        vkFreeMemory(m_device, m_vertex_buffer_mem, nullptr);
        m_vertex_buffer_mem = VK_NULL_HANDLE;
    }

    b8 createCommandBuffers()
    {
        m_cmd_buffers.resize(m_swap_chain_frame_buffers.size());

        VkCommandBufferAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = m_cmd_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = numericCast<ui32>(m_cmd_buffers.size());

        if (VK_SUCCESS != vkAllocateCommandBuffers(m_device, &alloc_info, m_cmd_buffers.data()))
        {
            sbLogE("vkAllocateCommandBuffers failed");

            return false;
        }

        VkClearValue clear_val;
        clear_val.color = {{0.5f, 0.5f, 0.5f, 1.f}};

        for (usize i = 0; i < m_cmd_buffers.size(); ++i)
        {
            VkCommandBufferBeginInfo begin_info = {};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
            begin_info.pInheritanceInfo = nullptr;

            if (VK_SUCCESS != vkBeginCommandBuffer(m_cmd_buffers[i], &begin_info))
            {
                sbLogE("vkBeginCommandBuffer failed");
                return false;
            }

            VkRenderPassBeginInfo rndr_pass_begin_info = {};
            rndr_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rndr_pass_begin_info.renderPass = m_render_pass;
            rndr_pass_begin_info.framebuffer = m_swap_chain_frame_buffers[i];
            rndr_pass_begin_info.renderArea.offset = {0, 0};
            rndr_pass_begin_info.renderArea.extent = m_swap_chain_img_ext;
            rndr_pass_begin_info.clearValueCount = 1;
            rndr_pass_begin_info.pClearValues = &clear_val;

            vkCmdBeginRenderPass(m_cmd_buffers[i], &rndr_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

            vkCmdBindPipeline(m_cmd_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphics_pipeline);

            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(m_cmd_buffers[i], 0, 1, &m_vertex_buffer, &offset);
            vkCmdBindIndexBuffer(m_cmd_buffers[i], m_index_buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdBindDescriptorSets(m_cmd_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline_layout, 0, 1,
                                    &m_desc_sets[i], 0, nullptr);

            // vkCmdDraw(m_cmd_buffers[i], wstd::size(TEST_MESH_VERTS), 1, 0, 0);
            vkCmdDrawIndexed(m_cmd_buffers[i], (ui32)m_model_indices_cnt, 1, 0, 0, 0);

            vkCmdEndRenderPass(m_cmd_buffers[i]);

            if (VK_SUCCESS != vkEndCommandBuffer(m_cmd_buffers[i]))
            {
                sbLogE("vkEndCommandBuffer failed");
                return false;
            }
        }

        return true;
    }

    void destroyCommandBuffers()
    {
        // no need to delete them because it is handled
        // by the associated Command Pool
        m_cmd_buffers.clear();
    }

    b8 createCommandPool()
    {
        auto const queue_family = findQueueFamilies(m_phys_device, m_surface);

        // A command pool can allocate command buffers for only 1 queue
        VkCommandPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = numericCast<ui32>(queue_family.m_graphics_idx);
        pool_info.flags = 0;

        if (VK_SUCCESS != vkCreateCommandPool(m_device, &pool_info, nullptr, &m_cmd_pool))
        {
            sbLogE("vkCreateCommandPool failed");
            return false;
        }

        return true;
    }

    void destroyCommandPool()
    {
        vkDestroyCommandPool(m_device, m_cmd_pool, nullptr);
        m_cmd_pool = VK_NULL_HANDLE;
    }

    b8 createFrameBuffers()
    {
        m_swap_chain_frame_buffers.reserve(m_swap_chain_image_views.size());

        for (auto & image_view : m_swap_chain_image_views)
        {
            VkFramebufferCreateInfo frame_buffer_info = {};
            frame_buffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            frame_buffer_info.renderPass = m_render_pass;
            frame_buffer_info.attachmentCount = 1;
            frame_buffer_info.pAttachments = &image_view;
            frame_buffer_info.width = m_swap_chain_img_ext.width;
            frame_buffer_info.height = m_swap_chain_img_ext.height;
            frame_buffer_info.layers = 1;

            VkFramebuffer frame_buffer;
            if (VK_SUCCESS != vkCreateFramebuffer(m_device, &frame_buffer_info, nullptr, &frame_buffer))
            {
                sbLogE("vkCreateFramebuffer failed");
                return false;
            }

            m_swap_chain_frame_buffers.push_back(frame_buffer);
        }

        return true;
    }

    void destroyFrameBuffers()
    {
        for (auto const frame_buffer : m_swap_chain_frame_buffers)
        {
            vkDestroyFramebuffer(m_device, frame_buffer, nullptr);
        }

        m_swap_chain_frame_buffers.clear();
    }

    b8 createRenderPass()
    {
        VkAttachmentDescription color_attach = {};
        color_attach.format = m_swap_chain_img_fmt;
        color_attach.samples = VK_SAMPLE_COUNT_1_BIT;
        color_attach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // clear the attachment (to black) before rendering
        color_attach.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // We would like to write to the frame buffer
        color_attach.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_attach.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color_attach.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color_attach.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference color_attach_ref = {};
        color_attach_ref.attachment = 0;
        color_attach_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription sub_pass_desc = {};
        sub_pass_desc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub_pass_desc.colorAttachmentCount = 1;
        sub_pass_desc.pColorAttachments = &color_attach_ref;

        VkRenderPassCreateInfo render_pass_info = {};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        render_pass_info.attachmentCount = 1;
        render_pass_info.pAttachments = &color_attach;
        render_pass_info.subpassCount = 1;
        render_pass_info.pSubpasses = &sub_pass_desc;

        VkSubpassDependency dep = {};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = 0;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        render_pass_info.dependencyCount = 1;
        render_pass_info.pDependencies = &dep;

        if (VK_SUCCESS != vkCreateRenderPass(m_device, &render_pass_info, nullptr, &m_render_pass))
        {
            sbLogE("vkCreateRenderPass failed");
            return false;
        }

        return true;
    }

    void destroyRenderPass()
    {
        vkDestroyRenderPass(m_device, m_render_pass, nullptr);
        m_render_pass = VK_NULL_HANDLE;
    }

    b8 createDescriptorSetLayout()
    {
        VkDescriptorSetLayoutBinding bindings[2] = {};

        // model x view x projection matrix
        bindings[0].binding = 0;
        bindings[0].descriptorCount = 1; // if > 1, it will be an array of descriptor e.g. list of bones
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        // texture
        bindings[1].binding = 1;
        bindings[1].descriptorCount = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layout_info = {};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = numericCast<ui32>(wstd::size(bindings));
        layout_info.pBindings = bindings;

        if (VK_SUCCESS != vkCreateDescriptorSetLayout(m_device, &layout_info, nullptr, &m_descriptor_set_layout))
        {
            sbLogE("vkCreateDescriptorSetLayout failed");
            return false;
        }

        return true;
    }

    void destroyDescriptorSetLayout()
    {
        vkDestroyDescriptorSetLayout(m_device, m_descriptor_set_layout, nullptr);
        m_descriptor_set_layout = VK_NULL_HANDLE;
    }

    b8 createGraphicsPipeline()
    {
        auto global_heap = getGlobalHeapView();

        // Shader creation
        auto mem_arena = readFile("/data/vert.spv", global_heap);

        if (mem_arena.memarena_isEmpty())
        {
            sbLogE("Failed to read '/data/vert.spv'");
            return false;
        }

        VkShaderModule vert_module = createShaderModule(mem_arena);
        global_heap.deallocate(mem_arena.m_ptr);

        if (VK_NULL_HANDLE == vert_module)
        {
            return false;
        }

        mem_arena = readFile("/data/frag.spv", global_heap);

        if (mem_arena.memarena_isEmpty())
        {
            destroyShaderModule(vert_module);
            sbLogE("Failed to read '/data/frag.spv'");
            return false;
        }

        VkShaderModule frag_module = createShaderModule(mem_arena);
        global_heap.deallocate(mem_arena.m_ptr);

        if (VK_NULL_HANDLE == frag_module)
        {
            destroyShaderModule(vert_module);
            return false;
        }

        // Input Assembly Stage
        VkPipelineVertexInputStateCreateInfo vert_input_stage_info = {};

        // Declare a vertex buffer being bound and the space between 2 elements
        VkVertexInputBindingDescription vert_input_binding = {};
        vert_input_binding.binding = 0;
        vert_input_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        vert_input_binding.stride = sizeof(Vertex);

        // Declares individual vertex buffer components to map them to shader input
        auto vert_attr_desc = Vertex::getInputAttrDesc();

        vert_input_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vert_input_stage_info.vertexBindingDescriptionCount = 1;
        vert_input_stage_info.pVertexBindingDescriptions = &vert_input_binding;
        vert_input_stage_info.vertexAttributeDescriptionCount = numericCast<ui32>(vert_attr_desc.size());
        vert_input_stage_info.pVertexAttributeDescriptions = vert_attr_desc.data();

        VkPipelineInputAssemblyStateCreateInfo input_assembly_stage_info = {};
        input_assembly_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly_stage_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        input_assembly_stage_info.primitiveRestartEnable = VK_FALSE;

        VkPipelineShaderStageCreateInfo shader_stages_info[2] = {};

        // Vertex Shader Stage
        shader_stages_info[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shader_stages_info[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        shader_stages_info[0].module = vert_module;
        shader_stages_info[0].pName = "main"; // function name to invoke

        // Rasterizer Stage
        VkViewport view_port = {};
        view_port.x = 0.f;
        view_port.y = 0.f;
        view_port.width = (float)m_swap_chain_img_ext.width;
        view_port.height = (float)m_swap_chain_img_ext.height;
        view_port.minDepth = 0.f;
        view_port.maxDepth = 1.f;

        VkRect2D scissor = {};
        scissor.extent = m_swap_chain_img_ext;
        scissor.offset = {0, 0};

        VkPipelineViewportStateCreateInfo view_port_stage_info = {};
        view_port_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        view_port_stage_info.viewportCount = 1;
        view_port_stage_info.pViewports = &view_port;
        view_port_stage_info.scissorCount = 1;
        view_port_stage_info.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer_stage_info = {};
        rasterizer_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer_stage_info.depthBiasClamp = VK_FALSE;
        rasterizer_stage_info.rasterizerDiscardEnable = VK_FALSE;
        rasterizer_stage_info.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer_stage_info.lineWidth = 1.f;
        rasterizer_stage_info.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer_stage_info.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer_stage_info.depthBiasEnable = VK_FALSE;

        // Pixel Shader Stage
        shader_stages_info[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shader_stages_info[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shader_stages_info[1].module = frag_module;
        shader_stages_info[1].pName = "main";

        // Color blending
        VkPipelineColorBlendAttachmentState color_blend_attach = {};
        color_blend_attach.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        color_blend_attach.blendEnable = VK_FALSE;
        color_blend_attach.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        color_blend_attach.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        color_blend_attach.colorBlendOp = VK_BLEND_OP_ADD;
        color_blend_attach.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        color_blend_attach.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        color_blend_attach.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo color_blend_stage_info = {};
        color_blend_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blend_stage_info.logicOpEnable = VK_FALSE;
        color_blend_stage_info.logicOp = VK_LOGIC_OP_COPY;
        color_blend_stage_info.attachmentCount = 1;
        color_blend_stage_info.pAttachments = &color_blend_attach;
        color_blend_stage_info.blendConstants[0] = 0.f;
        color_blend_stage_info.blendConstants[1] = 0.f;
        color_blend_stage_info.blendConstants[2] = 0.f;
        color_blend_stage_info.blendConstants[3] = 0.f;

        // Anti Aliasing (happeing after the shaders have been shaded)
        VkPipelineMultisampleStateCreateInfo multisamp_stage_info = {};
        multisamp_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisamp_stage_info.sampleShadingEnable = VK_FALSE;
        multisamp_stage_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisamp_stage_info.minSampleShading = 1.f;
        multisamp_stage_info.pSampleMask = nullptr;
        multisamp_stage_info.alphaToCoverageEnable = VK_FALSE;
        multisamp_stage_info.alphaToOneEnable = VK_FALSE;

        // Dynamic states
        VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_LINE_WIDTH};

        VkPipelineDynamicStateCreateInfo dyn_state_info = {};
        dyn_state_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dyn_state_info.dynamicStateCount = 2;
        dyn_state_info.pDynamicStates = dyn_states;

        // Shaders uniform variables
        VkPipelineLayoutCreateInfo pipe_layout_info = {};
        pipe_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipe_layout_info.setLayoutCount = 1;
        pipe_layout_info.pSetLayouts = &m_descriptor_set_layout;
        pipe_layout_info.pushConstantRangeCount = 0;
        pipe_layout_info.pPushConstantRanges = nullptr;

        if (VK_SUCCESS != vkCreatePipelineLayout(m_device, &pipe_layout_info, nullptr, &m_pipeline_layout))
        {
            sbLogE("vkCreatePipelineLayout failed");

            destroyShaderModule(vert_module);
            destroyShaderModule(frag_module);

            return false;
        }

        VkGraphicsPipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_info.stageCount = 2;
        pipeline_info.pStages = shader_stages_info;
        pipeline_info.pVertexInputState = &vert_input_stage_info;
        pipeline_info.pInputAssemblyState = &input_assembly_stage_info;
        pipeline_info.pViewportState = &view_port_stage_info;
        pipeline_info.pRasterizationState = &rasterizer_stage_info;
        pipeline_info.pMultisampleState = &multisamp_stage_info;
        pipeline_info.pDepthStencilState = nullptr;
        pipeline_info.pColorBlendState = &color_blend_stage_info;
        pipeline_info.pDynamicState = nullptr;
        pipeline_info.layout = m_pipeline_layout;
        pipeline_info.renderPass = m_render_pass;
        pipeline_info.subpass = 0;
        pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
        pipeline_info.basePipelineIndex = -1;

        if (VK_SUCCESS !=
            vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &m_graphics_pipeline))
        {
            sbLogE("vkCreateGraphicsPipelines failed");
            return false;
        }

        // They are not needed after pipeline creation i.e. the ownership is passed to the pipeline
        destroyShaderModule(vert_module);
        destroyShaderModule(frag_module);

        return true;
    }

    void destroyGraphicsPipeline()
    {
        vkDestroyPipeline(m_device, m_graphics_pipeline, nullptr);
        m_graphics_pipeline = VK_NULL_HANDLE;

        vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
        m_pipeline_layout = VK_NULL_HANDLE;
    }

    b8 createSwapChainImageViews()
    {
        m_swap_chain_image_views.reserve(m_swap_chain_images.size());

        for (auto const & img : m_swap_chain_images)
        {
            VkImageView tmp_view = createImageView(img, m_swap_chain_img_fmt);

            if (VK_NULL_HANDLE == tmp_view)
            {
                sbLogE("vkCreateImageView failed");
                return false;
            }

            m_swap_chain_image_views.push_back(tmp_view);
        }

        return true;
    }

    void destroySwapChainImageViews()
    {
        for (auto const & view : m_swap_chain_image_views)
        {
            vkDestroyImageView(m_device, view, nullptr);
        }

        m_swap_chain_image_views.clear();
    }

    b8 createSwapChain(GLFWwindow * wnd_hdl)
    {
        auto const swap_info = getDeviceSwapChainSupportDetails(m_phys_device, m_surface);
        auto const queue_family = findQueueFamilies(m_phys_device, m_surface);

        ui32 const queue_indices[] = {numericCast<ui32>(queue_family.m_graphics_idx),
                                      numericCast<ui32>(queue_family.m_present_idx)};

        VkSurfaceFormatKHR const fmt = chooseSwapChainSurfaceFormat(swap_info.m_formats);

        VkSwapchainCreateInfoKHR create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        create_info.surface = m_surface;
        create_info.minImageCount = wstd::min(3U, swap_info.m_caps.maxImageCount);
        create_info.imageFormat = fmt.format;
        create_info.imageColorSpace = fmt.colorSpace;
        create_info.imageExtent = chooseSwapChainImageExtent(swap_info.m_caps, wnd_hdl);
        create_info.imageArrayLayers = 1; // Always 1 except if we are doing stereoscopic rendering
        create_info.imageUsage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // We will render directly into the swap chain image

        // Images from the swap chain are accessed by only one queue i.e. access is exclusive by definition
        if (queue_family.m_graphics_idx == queue_family.m_present_idx)
        {
            create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }
        // Multiple queues are accesing the swap chain image and we don't want to add code to handle this ... let Vulkan
        // doing it Alternatively, we could set it to VK_SHARING_MODE_EXCLUSIVE and explicitely tell Vulkan when we are
        // switching queues
        else
        {
            create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            create_info.queueFamilyIndexCount = numericCast<ui32>(wstd::size(queue_indices));
            create_info.pQueueFamilyIndices = queue_indices;
        }

        // Transfor to apply to the image (e.g. 90 deg rotation) before presentation
        create_info.preTransform = swap_info.m_caps.currentTransform;

        // Don't blend pixels with other windows
        create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

        create_info.presentMode = chooseSwapChainPresentMode(swap_info.m_present_modes);

        // We don't care if the pixels occluded by other operating system elements (windo on top of ours) are
        // rendered/presented or not Because we don't want to read them back
        create_info.clipped = VK_TRUE;

        create_info.oldSwapchain = VK_NULL_HANDLE;

        auto const err = vkCreateSwapchainKHR(m_device, &create_info, nullptr, &m_swap_chain);
        if (VK_SUCCESS != err)
        {
            sbLogE("vkCreateSwapchainKHR failed with error {}", getEnumValue(err));
            return false;
        }

        ui32 img_cnt = 0;
        vkGetSwapchainImagesKHR(m_device, m_swap_chain, &img_cnt, nullptr);
        m_swap_chain_images.resize(img_cnt);
        vkGetSwapchainImagesKHR(m_device, m_swap_chain, &img_cnt, m_swap_chain_images.data());

        m_swap_chain_img_fmt = create_info.imageFormat;
        m_swap_chain_img_ext = create_info.imageExtent;

        return true;
    }

    void destroySwapChain()
    {
        m_swap_chain_images.clear();

        vkDestroySwapchainKHR(m_device, m_swap_chain, nullptr);
        m_swap_chain = VK_NULL_HANDLE;
    }

    b8 createWindowSurface(GLFWwindow * wnd_hdl)
    {
        // TODO: pass the window surface as parameter
        VkResult const err = glfwCreateWindowSurface(m_instance, wnd_hdl, nullptr, &m_surface);

        if (VK_SUCCESS != err)
        {
            sbLogE("Window surface creation failed with error {}", getEnumValue(err));
            return false;
        }

        return true;
    }

    void destroyWindowSurface()
    {
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        m_surface = VK_NULL_HANDLE;
    }

    b8 createLogicalDevice()
    {
        QueueFamiliesDesc queues_desc = findQueueFamilies(m_phys_device, m_surface);

        // If the queues are from different families we will need to have separated VkDeviceQueueCreateInfo
        // The best approach would be to findout unique queue family IDs to create as less queues as possible e.g. using
        // a set
        sbAssert(queues_desc.m_graphics_idx == queues_desc.m_present_idx);

        f32 const queue_priorities = 1.f;

        VkDeviceQueueCreateInfo queue_info = {};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueCount = 1;
        queue_info.queueFamilyIndex = numericCast<ui32>(queues_desc.m_graphics_idx);
        queue_info.pQueuePriorities = &queue_priorities;

        VkPhysicalDeviceFeatures device_features = {};
        device_features.samplerAnisotropy = VK_TRUE;

        VkDeviceCreateInfo device_info = {};
        device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.pQueueCreateInfos = &queue_info;
        device_info.queueCreateInfoCount = 1;
        device_info.pEnabledFeatures = &device_features;
        device_info.enabledExtensionCount = numericCast<ui32>(wstd::size(REQUIRED_PHYSICAL_DEVICE_EXTENSIONS));
        device_info.ppEnabledExtensionNames = REQUIRED_PHYSICAL_DEVICE_EXTENSIONS;

        if (m_validation_enabled)
        {
            device_info.enabledLayerCount = numericCast<ui32>(wstd::size(DEFAULT_VALIDATION_LAYERS));
            device_info.ppEnabledLayerNames = DEFAULT_VALIDATION_LAYERS;
        }

        auto const err = vkCreateDevice(m_phys_device, &device_info, nullptr, &m_device);

        if (VK_SUCCESS != err)
        {
            sbLogE("vkCreateDevice failed with error {}", getEnumValue(err));
            return false;
        }

        vkGetDeviceQueue(m_device, numericCast<ui32>(queues_desc.m_graphics_idx), 0, &m_graphics_queue);

        if (VK_NULL_HANDLE == m_graphics_queue)
        {
            sbLogE("Failed to get graphics queue handle from the Vulkan Device");
            return false;
        }

        vkGetDeviceQueue(m_device, numericCast<ui32>(queues_desc.m_present_idx), 0, &m_present_queue);

        if (VK_NULL_HANDLE == m_present_queue)
        {
            sbLogE("Failed to get present queue handle from the Vulkan Device");
            return false;
        }

        return true;
    }

    void destroyLogicalDevice()
    {
        vkDestroyDevice(m_device, nullptr);
        m_device = VK_NULL_HANDLE;
    }

    b8 selectPhysicalDevice()
    {
        ui32 phys_device_cnt = 0;
        vkEnumeratePhysicalDevices(m_instance, &phys_device_cnt, nullptr);

        if (0 == phys_device_cnt)
        {
            sbLogE("No physical device detected");
            return false;
        }

        SmallVector<VkPhysicalDevice, 5> phys_devices(phys_device_cnt);
        vkEnumeratePhysicalDevices(m_instance, &phys_device_cnt, phys_devices.data());

        for (auto const & phys_device : phys_devices)
        {
            if (isDeviceSuitable(phys_device, m_surface))
            {
                m_phys_device = phys_device;
                break;
            }
        }

        if (VK_NULL_HANDLE == m_phys_device)
        {
            sbLogE("None of the available physical device is suitable");
            return false;
        }

        return true;
    }

    void setupDebugCallback()
    {
        auto create_fn =
            (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT");

        if (nullptr == create_fn)
        {
            sbLogI("Cannot find 'vkCreateDebugUtilsMessengerEXT' extension function");
            return;
        }

        VkDebugUtilsMessengerCreateInfoEXT create_info = {};

        create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        create_info.pfnUserCallback = &debugCallback;

        VkResult const err = (*create_fn)(m_instance, &create_info, nullptr, &m_dbg_cb);

        if (VK_SUCCESS != err)
        {
            sbLogW("vkCreateDebugUtilsMessengerEXT failed with error {}", getEnumValue(err));
        }
    }

    void removeDebugCallback()
    {
        if (VK_NULL_HANDLE != m_dbg_cb)
        {
            auto const destroy_fn = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
                m_instance, "vkDestroyDebugUtilsMessengerEXT");

            if (nullptr == destroy_fn)
            {
                sbLogI("Cannot find 'vkDestroyDebugUtilsMessengerEXT' extension function");
                return;
            }

            (*destroy_fn)(m_instance, m_dbg_cb, nullptr);
            m_dbg_cb = VK_NULL_HANDLE;
        }
    }

    b8 createInstance()
    {
        VkApplicationInfo app_info = {};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "Vulkan 101";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "Sunburst Engine";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo inst_info = {};
        inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        inst_info.pApplicationInfo = &app_info;

        auto const required_vk_exts = getRequiredExtensions(m_validation_enabled);

        ui32 layer_cnt = 0;

        if (m_validation_enabled)
        {
            ui32 vk_layer_cnt = 0;
            vkEnumerateInstanceLayerProperties(&vk_layer_cnt, nullptr);
            ui32 validation_layer_cnt = numericCast<ui32>(wstd::size(DEFAULT_VALIDATION_LAYERS));

            if ((0 != vk_layer_cnt) && (0 != validation_layer_cnt))
            {
                SmallVector<VkLayerProperties, 5> vk_layers(vk_layer_cnt);

                vkEnumerateInstanceLayerProperties(&vk_layer_cnt, vk_layers.data());

                if (m_validation_enabled)
                {
                    sbLogI("Available Vulkan layers:");
                    for (auto const & layer_props : vk_layers)
                    {
                        sbLogI("    - {}", layer_props.layerName);
                    }
                }

                for (auto const & props : vk_layers)
                {
                    auto const layer_name_iter = wstd::find_if(
                        wstd::begin(DEFAULT_VALIDATION_LAYERS), wstd::end(DEFAULT_VALIDATION_LAYERS),
                        [&props](char const * layer_name) { return strcmp(layer_name, props.layerName) == 0; });

                    if (wstd::end(DEFAULT_VALIDATION_LAYERS) != layer_name_iter)
                    {
                        --validation_layer_cnt;

                        if (0 == validation_layer_cnt)
                        {
                            layer_cnt = numericCast<ui32>(wstd::size(DEFAULT_VALIDATION_LAYERS));
                            break;
                        }
                    }
                }
            }
        }

        if (m_verbose)
        {
            if (0 != layer_cnt)
            {
                sbLogI("Vulkan validation layers enabled:");
                for (auto layer_name : DEFAULT_VALIDATION_LAYERS)
                {
                    sbLogI("    - {}", layer_name);
                }
            }

            if (!required_vk_exts.empty())
            {
                sbLogI("Required Vulkan extensions:");
                for (auto ext_name : required_vk_exts)
                {
                    sbLogI("    - {}", ext_name);
                }
            }
        }

        inst_info.enabledExtensionCount = numericCast<ui32>(required_vk_exts.size());
        inst_info.ppEnabledExtensionNames = required_vk_exts.data();
        inst_info.enabledLayerCount = layer_cnt;
        inst_info.ppEnabledLayerNames = DEFAULT_VALIDATION_LAYERS;
        inst_info.enabledLayerCount = 0;

        auto const vk_err = vkCreateInstance(&inst_info, nullptr, &m_instance);

        if (VK_SUCCESS != vk_err)
        {
            sbLogE("vkCreateInstance failed with code {}", getEnumValue(vk_err));

            return false;
        }

        if (m_verbose)
        {
            ui32 vk_ext_cnt = 0;
            vkEnumerateInstanceExtensionProperties(nullptr, &vk_ext_cnt, nullptr);

            if (0 != vk_ext_cnt)
            {
                Vector<VkExtensionProperties> vk_exts(vk_ext_cnt);

                vkEnumerateInstanceExtensionProperties(nullptr, &vk_ext_cnt, vk_exts.data());

                sbLogI("Available Vulkan Instance extensions:");

                for (auto const & ext_props : vk_exts)
                {
                    sbLogI("    - {}", ext_props.extensionName);
                }
            }
        }

        return true;
    }

    void destroyInstance()
    {
        vkDestroyInstance(m_instance, nullptr);
        m_instance = VK_NULL_HANDLE;
    }
};

static void framebufferResizeCallback(GLFWwindow * window, int width, int height)
{
    auto app = reinterpret_cast<VkTestApp *>(glfwGetWindowUserPointer(window));
    app->surfaceResized();
}

int main()
{
    char working_dir[PPath::MAX_LEN];

    FS::InitParams fs_params;
    FS::LayerDesc layer_descs[] = {
        {"/data/", FS::createLocalFileSystemLayer(getWorkingDirectory(working_dir, wstd::size(working_dir))),
         HashStr{"data"}}};

    fs_params.m_layers = layer_descs;

    b8 sb_succeded = FS::initialize(fs_params);
    sbAssert(sb_succeded);

    VkTestApp app{};

    int glfw_err = glfwInit();
    sbAssert(GLFW_TRUE == glfw_err);

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    GLFWwindow * const wnd_hdl = glfwCreateWindow(800, 600, "Vulkan 101", nullptr, nullptr);
    sbAssert(nullptr != wnd_hdl);

    glfwSetWindowUserPointer(wnd_hdl, &app);
    glfwSetFramebufferSizeCallback(wnd_hdl, &framebufferResizeCallback);

    b8 const app_err = app.initialize(wnd_hdl, true, true);
    sbAssert(app_err);

    // Timer::Ctx timer_ctx = Timer::GetCtx();

    while (!glfwWindowShouldClose(wnd_hdl))
    {
        app.render(wnd_hdl);

        glfwPollEvents();
    }

    app.terminate();

    glfwDestroyWindow(wnd_hdl);

    glfwTerminate();

    FS::terminate();

    return 0;
}
