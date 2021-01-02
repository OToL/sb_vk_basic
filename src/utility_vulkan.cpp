#include "utility_vulkan.h"

#include <sb_core/string/string_format.h>
#include <sb_core/core.h>
#include <sb_core/log.h>
#include <sb_core/enum.h>
#include <sb_core/container/small_array.h>
#include <sb_core/error/error.h>

VkResult sb::createVkDebugUtilsMessenger(VkInstance instance, VkDebugUtilsMessengerCreateInfoEXT const * create_info,
                                         VkAllocationCallbacks const * alloc_cb,
                                         VkDebugUtilsMessengerEXT * dbg_messenger)
{
    auto const msg_create_func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (nullptr == msg_create_func)
    {
        sbLogE("Failed to locate 'vkCreateDebugUtilsMessengerEXT'");
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
    else
    {
        VkResult const res = msg_create_func(instance, create_info, alloc_cb, dbg_messenger);
        if (VK_SUCCESS != res)
        {
            sbLogE("Failed to create Vulkan Debug Utils Messeger (error = '{}')", getEnumValue(res));
            return res;
        }
    }

    return VK_SUCCESS;
}

void sb::destroyVkDebugUtilsMessenger(VkInstance instance, VkDebugUtilsMessengerEXT dbg_messenger,
                                      VkAllocationCallbacks const * alloc_cb)
{
    auto const msg_destroy_func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (nullptr == msg_destroy_func)
    {
        sbLogE("Failed to locate 'vkDestroyDebugUtilsMessengerEXT'");
    }
    else
    {
        msg_destroy_func(instance, dbg_messenger, alloc_cb);
    }
}

sb::VkQueueFamilyIndices sb::getVkQueueFamilyIndicies(VkPhysicalDevice device, VkSurfaceKHR const * surface)
{
    u32 family_cnt = 0;
    SArray<VkQueueFamilyProperties, 10> families;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &family_cnt, nullptr);

    families.resize(family_cnt);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &family_cnt, families.data());

    VkQueueFamilyIndices queue_indices = {};

    VkQueueFamilyIndex idx = 0;
    for (auto const & family : families)
    {
        if (family.queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            queue_indices.families.value |= makeEnumMaskValue(VkQueueFamilyFeature::GRAPHICS);
            queue_indices.graphics = idx;
        }

        if (VK_NULL_HANDLE != surface)
        {
            VkBool32 present_support = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, idx, *surface, &present_support);

            if (present_support)
            {
                queue_indices.families.value |= makeEnumMaskValue(VkQueueFamilyFeature::PRESENT);
                queue_indices.present = idx;
            }
        }

        ++idx;
    }

    return queue_indices;
}

sb::b32 sb::checkDeviceExtensionsSupport(VkPhysicalDevice phys_device, sbstd::span<char const * const> extensions)
{
    if (extensions.empty())
    {
        return true;
    }

    u32 device_ext_cnt = 0;
    SArray<VkExtensionProperties, 10> device_exts;
    vkEnumerateDeviceExtensionProperties(phys_device, nullptr, &device_ext_cnt, nullptr);
    sbAssert(device_ext_cnt != 0);
    device_exts.resize(device_ext_cnt);
    vkEnumerateDeviceExtensionProperties(phys_device, nullptr, &device_ext_cnt, device_exts.data());

    usize ext_found_cnt = 0;
    for (auto req_ext : extensions)
    {
        auto const ext_idx = sbstd::find_if(begin(device_exts), end(device_exts), [req_ext](auto const & ext) {
            return 0 == strcmpi(req_ext, ext.extensionName);
        });

        if (ext_idx != end(device_exts))
        {
            ++ext_found_cnt;
        }
    }

    return extensions.size() == ext_found_cnt;
}

sb::VkSurfaceSwapChainProperties sb::getVkSurfaceSwapChainProperties(VkPhysicalDevice phys_device, VkSurfaceKHR surface)
{
    VkSurfaceSwapChainProperties props;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys_device, surface, &props.caps);

    u32 fmt_cnt = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys_device, surface, &fmt_cnt, nullptr);
    if (fmt_cnt != 0)
    {
        props.formats.resize(fmt_cnt);
        vkGetPhysicalDeviceSurfaceFormatsKHR(phys_device, surface, &fmt_cnt, props.formats.data());
    }

    u32 present_mode_cnt = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(phys_device, surface, &present_mode_cnt, nullptr);
    if (present_mode_cnt != 0)
    {
        props.present_modes.resize(present_mode_cnt);
        vkGetPhysicalDeviceSurfacePresentModesKHR(phys_device, surface, &present_mode_cnt, props.present_modes.data());
    }

    return props;
}

VkResult sb::createVkShaderModule(VkDevice device, sbstd::span<u8 const> byte_code, VkShaderModule * shader_module)
{
    sbAssert(nullptr != shader_module);

    VkShaderModuleCreateInfo module_info = {};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = byte_code.size();
    module_info.pCode = reinterpret_cast<u32 const *>(byte_code.data());

    VkResult const res = vkCreateShaderModule(device, &module_info, nullptr, shader_module);

    return res;
}

sb::u32 sb::findVkDeviceMemoryTypeIndex(VkPhysicalDevice device, u32 possible_types,
                                        VkMemoryPropertyFlags property_flags)
{
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(device, &mem_props);

    for (u32 idx = 0; idx != mem_props.memoryTypeCount; ++idx)
    {
        if (((1 << idx) & possible_types) &&
            ((mem_props.memoryTypes[idx].propertyFlags & property_flags) == property_flags))
        {
            return idx;
        }
    }

    return UINT32_MAX;
}

void sb::destroyVkImage(VkDevice device, VkImageMem image_mem)
{
    if (VK_NULL_HANDLE != image_mem.memory)
    {
        vkFreeMemory(device, image_mem.memory, nullptr);
    }

    if (VK_NULL_HANDLE != image_mem.image)
    {
        vkDestroyImage(device, image_mem.image, nullptr);
    }
}

VkResult sb::createVkImage(VkPhysicalDevice phys_device, VkDevice device, uint32_t width, uint32_t height,
                           u32 mip_count, VkSampleCountFlagBits sample_cnt, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
                           VkMemoryPropertyFlags properties, VkImageMem * image_mem)
{
    sbAssert(nullptr != image_mem);

    VkImageCreateInfo img_info = {};
    img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_info.imageType = VK_IMAGE_TYPE_2D;
    img_info.extent = VkExtent3D{.width = numericConv<u32>(width), .height = numericConv<u32>(height), .depth = 1};
    img_info.mipLevels = mip_count;
    img_info.arrayLayers = 1;
    img_info.format = format;
    img_info.tiling = tiling;
    img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    img_info.usage = usage;
    img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    img_info.samples = sample_cnt;
    img_info.flags = 0;

    auto vk_res = vkCreateImage(device, &img_info, nullptr, &image_mem->image);
    if (VK_SUCCESS != vk_res)
    {
        return vk_res;
    }

    VkMemoryRequirements img_mem_req = {};
    vkGetImageMemoryRequirements(device, image_mem->image, &img_mem_req);

    VkMemoryAllocateInfo img_alloc_info = {};
    img_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    img_alloc_info.allocationSize = img_mem_req.size;
    img_alloc_info.memoryTypeIndex = findVkDeviceMemoryTypeIndex(phys_device, img_mem_req.memoryTypeBits, properties);

    vk_res = vkAllocateMemory(device, &img_alloc_info, nullptr, &image_mem->memory);
    if (VK_SUCCESS != vk_res)
    {
        return vk_res;
    }

    vkBindImageMemory(device, image_mem->image, image_mem->memory, 0);

    return VK_SUCCESS;
}

VkResult sb::createVkBuffer(VkPhysicalDevice phys_device, VkDevice device, VkDeviceSize size,
                            VkBufferUsageFlags usage_flags, VkMemoryPropertyFlags mem_prop_flags,
                            VkBufferMem * buffer_mem)
{
    sbAssert(nullptr != buffer_mem);

    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage_flags;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // will only be used with graphics queue

    VkResult vk_res = vkCreateBuffer(device, &buffer_info, nullptr, &buffer_mem->buffer);
    if (VK_SUCCESS != vk_res)
    {
        return vk_res;
    }

    VkMemoryRequirements mem_req = {};
    vkGetBufferMemoryRequirements(device, buffer_mem->buffer, &mem_req);

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_req.size;
    alloc_info.memoryTypeIndex = findVkDeviceMemoryTypeIndex(phys_device, mem_req.memoryTypeBits, mem_prop_flags);

    vk_res = vkAllocateMemory(device, &alloc_info, nullptr, &buffer_mem->memory);
    if (VK_SUCCESS != vk_res)
    {
        return vk_res;
    }

    vkBindBufferMemory(device, buffer_mem->buffer, buffer_mem->memory, 0);

    return VK_SUCCESS;
}

void sb::copyVkBufferToImage(VkDevice device, VkCommandPool cmd_pool, VkQueue cmd_queue, VkBuffer src_buffer,
                             VkImage dst_image, VkExtent3D img_extents)
{
    VkCommandBuffer cmd_buffer = beginVkSingleTimeCommandBuffer(device, cmd_pool);

    VkBufferImageCopy copy_info = {};
    copy_info.bufferOffset = 0;
    copy_info.bufferRowLength = 0;
    copy_info.bufferImageHeight = 0;

    copy_info.imageOffset = {0, 0, 0};
    copy_info.imageExtent = img_extents;
    copy_info.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_info.imageSubresource.mipLevel = 0;
    copy_info.imageSubresource.baseArrayLayer = 0;
    copy_info.imageSubresource.layerCount = 1;

    vkCmdCopyBufferToImage(cmd_buffer, src_buffer, dst_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_info);

    endVkSingleTimeCommandBuffer(device, cmd_pool, cmd_queue, cmd_buffer);
}

void sb::copyVkBuffer(VkDevice device, VkCommandPool cmd_pool, VkQueue cmd_queue, VkBuffer src_buffer,
                      VkBuffer dst_buffer, VkDeviceSize buffer_size)
{
    VkCommandBuffer cmd_buffer = beginVkSingleTimeCommandBuffer(device, cmd_pool);

    VkBufferCopy copy_region = {};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = buffer_size;

    vkCmdCopyBuffer(cmd_buffer, src_buffer, dst_buffer, 1, &copy_region);

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buffer;

    endVkSingleTimeCommandBuffer(device, cmd_pool, cmd_queue, cmd_buffer);
}

void sb::destroyVkBuffer(VkDevice device, VkBufferMem buffer_mem)
{
    if (buffer_mem.buffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(device, buffer_mem.buffer, nullptr);
    }
    if (buffer_mem.memory != VK_NULL_HANDLE)
    {
        vkFreeMemory(device, buffer_mem.memory, nullptr);
    }
}

VkResult sb::uploadVkBufferDataToDevice(VkPhysicalDevice phys_device, VkDevice device, void * data,
                                        VkDeviceSize buffer_size, VkCommandPool cmd_pool, VkQueue cmd_queue,
                                        VkBuffer dst_buffer)
{
    VkBufferMem staging_mem;
    auto vk_res =
        createVkBuffer(phys_device, device, buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &staging_mem);

    if (VK_SUCCESS != vk_res)
    {
        return vk_res;
    }

    void * buffer_data;
    vkMapMemory(device, staging_mem.memory, 0, buffer_size, 0, &buffer_data);
    memcpy(buffer_data, data, buffer_size);
    vkUnmapMemory(device, staging_mem.memory);

    copyVkBuffer(device, cmd_pool, cmd_queue, staging_mem.buffer, dst_buffer, buffer_size);

    vkFreeMemory(device, staging_mem.memory, nullptr);
    vkDestroyBuffer(device, staging_mem.buffer, nullptr);

    return VK_SUCCESS;
}

VkCommandBuffer sb::beginVkSingleTimeCommandBuffer(VkDevice device, VkCommandPool cmd_pool)
{
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandBufferCount = 1;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = cmd_pool;

    VkCommandBuffer cmd_buffer;
    vkAllocateCommandBuffers(device, &alloc_info, &cmd_buffer);

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(cmd_buffer, &begin_info);

    return cmd_buffer;
}

void sb::endVkSingleTimeCommandBuffer(VkDevice device, VkCommandPool cmd_pool, VkQueue queue,
                                      VkCommandBuffer cmd_buffer)
{
    vkEndCommandBuffer(cmd_buffer);

    VkSubmitInfo sub_info = {};
    sub_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    sub_info.commandBufferCount = 1;
    sub_info.pCommandBuffers = &cmd_buffer;

    vkQueueSubmit(queue, 1, &sub_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkFreeCommandBuffers(device, cmd_pool, 1, &cmd_buffer);
}

VkResult sb::transitionVkImageLayout(VkDevice device, VkQueue cmd_queue, VkCommandPool cmd_pool, VkImage image,
                                     [[maybe_unused]] VkFormat fmt, VkImageLayout old_layout, VkImageLayout new_layout,
                                     u32 mip_count)
{
    auto cmd_buffer = beginVkSingleTimeCommandBuffer(device, cmd_pool);

    VkPipelineStageFlags src_stage = 0;
    VkPipelineStageFlags dst_stage = 0;

    VkImageMemoryBarrier img_barrier = {};

    if ((old_layout == VK_IMAGE_LAYOUT_UNDEFINED) && (new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL))
    {
        src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        img_barrier.srcAccessMask = 0;
        img_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    }
    else if ((old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) &&
             (new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL))
    {
        src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

        img_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        img_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    }
    else
    {
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

    img_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    img_barrier.oldLayout = old_layout;
    img_barrier.newLayout = new_layout;
    img_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    img_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    img_barrier.image = image;
    img_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    img_barrier.subresourceRange.baseMipLevel = 0;
    img_barrier.subresourceRange.levelCount = mip_count;
    img_barrier.subresourceRange.baseArrayLayer = 0;
    img_barrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(cmd_buffer, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &img_barrier);

    endVkSingleTimeCommandBuffer(device, cmd_pool, cmd_queue, cmd_buffer);

    return VK_SUCCESS;
}

VkFormat sb::findVkSupportedImageFormat(VkPhysicalDevice phys_device, sbstd::span<VkFormat const> formats,
                                        VkImageTiling tiling_mode, VkFormatFeatureFlags features)
{
    for (auto fmt : formats)
    {
        VkFormatProperties fmt_props;
        vkGetPhysicalDeviceFormatProperties(phys_device, fmt, &fmt_props);

        if ((tiling_mode == VK_IMAGE_TILING_LINEAR) && ((fmt_props.linearTilingFeatures & features) == features))
        {
            return fmt;
        }
        else if ((tiling_mode == VK_IMAGE_TILING_OPTIMAL) && ((fmt_props.optimalTilingFeatures & features) == features))
        {
            return fmt;
        }
    }

    return VK_FORMAT_UNDEFINED;
}

VkFormat sb::findVkDepthImageFormat(VkPhysicalDevice phys_device)
{
    VkFormat const DEPTH_FORMATS[] = {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};

    return findVkSupportedImageFormat(phys_device, DEPTH_FORMATS, VK_IMAGE_TILING_OPTIMAL,
                                      VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

sb::b8 sb::hasVkSencilComponent(VkFormat fmt)
{
    return (fmt == VK_FORMAT_D32_SFLOAT_S8_UINT) || (fmt == VK_FORMAT_D24_UNORM_S8_UINT);
}

void sb::generateMipmaps(VkPhysicalDevice phys_device,VkDevice device, VkCommandPool cmd_pool, VkQueue queue, int width, int height, int mip_count,
                         VkImage img, VkFormat fmt)
{
    VkFormatProperties fmt_props = {};
    vkGetPhysicalDeviceFormatProperties(phys_device, fmt, &fmt_props);
    sbAssert(fmt_props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT);

    auto cmd_buffer = beginVkSingleTimeCommandBuffer(device, cmd_pool);

    VkImageMemoryBarrier barrier_info = {};
    barrier_info.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier_info.image = img;
    barrier_info.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_info.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier_info.subresourceRange.baseArrayLayer = 0;
    barrier_info.subresourceRange.layerCount = 1;
    barrier_info.subresourceRange.levelCount = 1;

    s32 curr_width = width;
    s32 curr_height = height;

    for (s32 i = 1; i < mip_count; ++i)
    {
        s32 next_width = (curr_width > 1) ? (s32)curr_width / 2 : 1;
        s32 next_height = (curr_height > 1) ? (s32)curr_height / 2 : 1;

        barrier_info.subresourceRange.baseMipLevel = i - 1;
        barrier_info.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier_info.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier_info.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier_info.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(cmd_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                             0, nullptr, 1, &barrier_info);

        VkImageBlit cmd_blit = {};
        cmd_blit.srcOffsets[0] = {0, 0, 0};
        cmd_blit.srcOffsets[1] = {curr_width, curr_height, 1};
        cmd_blit.srcSubresource.mipLevel = i - 1;
        cmd_blit.srcSubresource.baseArrayLayer = 0;
        cmd_blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        cmd_blit.dstSubresource.layerCount = 1;
        cmd_blit.dstOffsets[0] = {0, 0, 0};
        cmd_blit.dstOffsets[1] = {next_width, next_height, 1};
        cmd_blit.dstSubresource.mipLevel = i;
        cmd_blit.dstSubresource.baseArrayLayer = 0;
        cmd_blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        cmd_blit.srcSubresource.layerCount = 1;

        vkCmdBlitImage(cmd_buffer, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &cmd_blit, VK_FILTER_LINEAR);

        barrier_info.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier_info.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier_info.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier_info.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &barrier_info);

        curr_width = next_width;
        curr_height = next_height;
    }

    barrier_info.subresourceRange.baseMipLevel = mip_count - 1;
    barrier_info.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier_info.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier_info.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier_info.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &barrier_info);

    endVkSingleTimeCommandBuffer(device, cmd_pool, queue, cmd_buffer);
}