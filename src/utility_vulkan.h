#pragma once

#include <vulkan/vulkan.h>

#include <sb_core/enum.h>
#include <sb_core/core.h>
#include <sb_core/container/dynamic_array.h>

#include <sb_std/span>

namespace sb {

enum class VkQueueFamilyFeature : u32
{
    GRAPHICS,
    COMPUTE,
    PRESENT
};

using VkQueueFamilyIndex = u32;

struct VkQueueFamilyIndices
{
    EnumMask<VkQueueFamilyFeature> families;

    VkQueueFamilyIndex graphics;
    VkQueueFamilyIndex present;
};

struct VkSurfaceSwapChainProperties
{
    VkSurfaceCapabilitiesKHR caps;
    DArray<VkSurfaceFormatKHR> formats;
    DArray<VkPresentModeKHR> present_modes;
};

struct VkBufferMem
{
    VkBuffer buffer;
    VkDeviceMemory memory;
};

struct VkImageMem
{
    VkImage image;
    VkDeviceMemory memory;
};

VkResult createVkDebugUtilsMessenger(VkInstance instance, VkDebugUtilsMessengerCreateInfoEXT const * create_info,
                                     VkAllocationCallbacks const * alloc_cb, VkDebugUtilsMessengerEXT * dbg_messenger);

VkResult createVkShaderModule(VkDevice device, sbstd::span<u8 const> byte_code, VkShaderModule * shader_module);

void destroyVkDebugUtilsMessenger(VkInstance instance, VkDebugUtilsMessengerEXT dbg_messenger,
                                  VkAllocationCallbacks const * alloc_cb);

VkQueueFamilyIndices getVkQueueFamilyIndicies(VkPhysicalDevice device, VkSurfaceKHR const * surface);

b32 checkDeviceExtensionsSupport(VkPhysicalDevice phys_device, sbstd::span<char const * const> extensions);

VkSurfaceSwapChainProperties getVkSurfaceSwapChainProperties(VkPhysicalDevice phys_device, VkSurfaceKHR surface);

u32 findVkDeviceMemoryTypeIndex(VkPhysicalDevice device, u32 possible_types, VkMemoryPropertyFlags property_flags);

VkResult createVkBuffer(VkPhysicalDevice phys_device, VkDevice device, VkDeviceSize size,
                        VkBufferUsageFlags usage_flags, VkMemoryPropertyFlags mem_prop_flags, VkBufferMem * buffer_mem);
void destroyVkBuffer(VkDevice device, VkBufferMem buffer_mem);

VkResult createVkImage(VkPhysicalDevice phys_device, VkDevice device, uint32_t width, uint32_t height, u32 mip_count, VkSampleCountFlagBits sample_cnt, VkFormat format,VkImageTiling tiling, VkImageUsageFlags usage,VkMemoryPropertyFlags properties, VkImageMem * image);
void destroyVkImage(VkDevice device, VkImageMem image_mem);

void copyVkBuffer(VkDevice device, VkCommandPool cmd_pool, VkQueue cmd_quue, VkBuffer src_buffer, VkBuffer dst_buffer,
                  VkDeviceSize buffer_size);

void copyVkBufferToImage(VkDevice device, VkCommandPool cmd_pool, VkQueue cmd_quue, VkBuffer src_buffer, VkImage dst_image, VkExtent3D img_extents);


VkResult uploadVkBufferDataToDevice(VkPhysicalDevice phys_device, VkDevice device, void * data,
                                    VkDeviceSize buffer_size, VkCommandPool cmd_pool, VkQueue cmd_quue,
                                    VkBuffer dst_buffer);

VkCommandBuffer beginVkSingleTimeCommandBuffer(VkDevice device, VkCommandPool cmd_pool);
void endVkSingleTimeCommandBuffer(VkDevice device, VkCommandPool cmd_pool, VkQueue queue, VkCommandBuffer cmd_buffer);

VkResult transitionVkImageLayout(VkDevice device, VkQueue cmd_queue, VkCommandPool cmd_pool, VkImage image, VkFormat fmt, VkImageLayout old_layout, VkImageLayout new_layout, u32 mip_count);

VkFormat findVkSupportedImageFormat(VkPhysicalDevice phys_device, sbstd::span<VkFormat const> formats, VkImageTiling tiling_mode, VkFormatFeatureFlags features);

VkFormat findVkDepthImageFormat(VkPhysicalDevice phys_device);

sb::b8 hasVkSencilComponent(VkFormat fmt);

void generateMipmaps(VkPhysicalDevice phys_device, VkDevice device, VkCommandPool cmd_pool, VkQueue queue, int width, int height, int mip_count, VkImage img, VkFormat fmt);
} // namespace sb
