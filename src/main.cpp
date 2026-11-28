//--------------------------------------------------------------------------------------//
// PROJECT2: PARTICLE ARTIST
//--------------------------------------------------------------------------------------//
#define VK_ENABLE_BETA_EXTENSIONS
#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h> //if we want to print out enum names
#define GLFW_INCLUDE_NONE //"no, do not include OpenGL"
#include <GLFW/glfw3.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <array>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

//macro function to check the return value of Vulkan API calls. Print an error message and aborts the program if it fails
#define VK_CHECK(call)                                                  \
    do {                                                                \
        VkResult vk_check_result = (call);                              \
        if (vk_check_result != VK_SUCCESS) {                            \
            std::fprintf(stderr, "%s:%d: %s failed with VkResult %d\n", \
                        __FILE__, __LINE__, #call, vk_check_result);    \
            std::abort();                                               \
        }                                                               \
    } while (0)

struct Vertex {
    glm::vec3 pos;
    glm::vec2 uv;
    glm::vec3 normal;
    uint32_t material;
};

//-------------------------------------------------------------------------------------//
// PART A: THE PARTICLE
//-------------------------------------------------------------------------------------//
// The Monday lecture effectively went over all parts of the required features. The first
// step is a correct particle struct. The Particle struct, similar to the Vertex struct,
// must be synchronized between the host for initialization, compute shader for
// simulation, and vertex shader for rendering. Minimally, it must have position,
// velocity, plus some lifetime system, but you will likely find yourself adding more
// fields as your features ask for them. On the GPU, the particle buffer will be read
// with std430 layout, just like Material, PointLight, and LightHeader. The GPU expects
// each instance to be sized in some multiple of its largest member's alignment. Since
// vec3's (and vec4's) alignment is 16 bytes, its size should align to some multiple of
// that. You MUST add a static_assert like the other structs to prove your particle
// struct fits this formatting. 
//
// Later, you will define the following two structs, ComputePushConstants and
// ParticlePushConstants.

struct Particle {
    // minimal struct referenced on Slide 20 of Chapter 16 slides
    glm::vec3 position;
    float current_life;
    glm::vec3 velocity;

    // additional variables
    float phase;
    float age;
    float lifespan;
    uint32_t pad[2]; 
};

struct ComputePushConstants {
    float dt;
    float time;
    uint32_t particle_count;
    float floor_y;
    glm::vec4 wind;
    glm::vec4 spawn_min;
    glm::vec4 spawn_max;
};
// Edits for Per Particle Phase Variation
// Based on tutorial linked in my references
struct ParticlePushConstants {
    glm::mat4 vp;
    float point_scale;
    float time;
    float radiusX;
    float radiusZ;
};

struct Material {
    glm::vec4 diffuse;  //rgb = Kd 
    glm::vec4 specular; //rgb = Ks, a = shininess
    uint32_t has_diffuse_texture; //0 = no
    uint32_t pad[3]; //since we have a vec4 in the struct it needs to be in multiple of 16 bytes
};

struct PointLight {
    glm::vec4 position; //xyz = world position
    glm::vec4 color;    //rgb = color, a = intensity
};

struct LightHeader {
    glm::vec4 camera_pos;
    uint32_t light_count;
    float ambient;
    // uint32_t pad[2];
    // I added these constants for Major feature "Point Light Shadows"
    uint32_t which_shadow; 
    float shadow_far_plane; 
    glm::mat4 sun_VP;
    glm::vec4 sun_direction;
    glm::vec4 sun_color;
};

struct GraphicsPushConstants {
    glm::mat4 VP;
};

struct LightPointPushConstants {
    glm::mat4 VP;
    float point_scale;
};

static_assert(sizeof(Material) == 48, "Material must match std430 layout");
static_assert(sizeof(PointLight) == 32, "PointLight must match std430 layout");
static_assert(sizeof(LightHeader) == 128, "LightHeader must match std430 layout");
//(P2) add Particle static_assert here
// Part A asserts
static_assert(sizeof(Particle) == 48, "Particle must match std430 layout");
static_assert(sizeof(Particle) % 16 == 0, "Particle size must be a multiple of 16 (vec3 alignment)");
static_assert(sizeof(ComputePushConstants) == 64, "ComputePushConstants must match layout");
static_assert(sizeof(ParticlePushConstants) == 80, "ParticlePushConstants must match layout");

struct VertexKey {
    int position;
    int texcoord;
    int normal;
    uint32_t material;

    bool operator==(const VertexKey& other) const {
        return position == other.position && texcoord == other.texcoord && normal == other.normal && material == other.material;
    }
};

struct VertexKeyHash {
    size_t operator()(const VertexKey& key) const {
        size_t h = std::hash<int>()(key.position);
        h ^= std::hash<int>()(key.texcoord) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(key.normal) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>()(key.material) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

enum KeyState {
    KEY_STATE_UP,
    KEY_STATE_PRESSED,
    KEY_STATE_DOWN,
    KEY_STATE_RELEASED
};

//loaded once, shared by any number of objects
struct Mesh {
    std::string name;
    uint32_t first_index;
    uint32_t index_count;
    int32_t vertex_offset;
};

enum AnimationType {
    ANIMATION_NONE = 0,
};

struct Object {
    uint32_t mesh;
    glm::vec3 position;
    glm::vec3 rotation;
    glm::vec3 scale;
    AnimationType animation;
};

//1 = vertex/index data lives in DEVICE_LOCAL memory (cpu writes to RAM then copies to VRAM)
//0 = lives in HOST_VISIBLE | HOST_COHERENT memory (data stays in RAM and is fetched every draw call)
#define USE_DEVICE_LOCAL_MESHES 1
//1 = per-frame model matrices (and lights) prefer DEVICE_LOCAL | HOST_VISIBLE/COHERENT (cpu writes straight into VRAM)
//0 = plain HOST_VISIBLE | COHERENT (data stays in RAM)
#define USE_BAR_MODEL_MATRICES 1

static const char* APPLICATION_TITLE =                      "PARTICLE ARTIST";
static const uint32_t WINDOW_WIDTH =                        800;
static const uint32_t WINDOW_HEIGHT =                       800;
static const uint32_t FRAME_WIDTH =                         WINDOW_WIDTH;
static const uint32_t FRAME_HEIGHT =                        WINDOW_HEIGHT;
static const uint32_t FRAMES_IN_FLIGHT =                    3;
static const uint32_t MAX_OBJECTS =                         50;
static const char* VERTEX_SHADER_PATH =                     SHADER_DIR "/myvertshader.vert.spv";
static const char* FRAGMENT_SHADER_PATH =                   SHADER_DIR "/myfragshader.frag.spv";
static const char* LIGHT_POINT_VERTEX_SHADER_PATH =         SHADER_DIR "/lightpoint.vert.spv";
static const char* LIGHT_POINT_FRAGMENT_SHADER_PATH =       SHADER_DIR "/lightpoint.frag.spv";
static const float LIGHT_POINT_SIZE =                       0.25f; //light point width in world space
static const char* TEXTURE_DIR =                            DATA_DIR "/textures/";
static const VkFormat COLOR_FORMAT =                        VK_FORMAT_R8G8B8A8_SRGB;
static const VkFormat DEPTH_FORMAT =                        VK_FORMAT_D32_SFLOAT;
static const VkFormat TEXTURE_FORMAT =                      VK_FORMAT_R8G8B8A8_SRGB;

// (Project2) NEW_CONSTS: adjust all of these (except the shader paths) to your liking
static const VkPresentModeKHR PREFERRED_PRESENT_MODE =      VK_PRESENT_MODE_MAILBOX_KHR;
static const VkPhysicalDeviceType PREFERRED_DEVICE_TYPE =   VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
static const bool PREFER_SECOND_MONITOR =                   false;
static const uint32_t MAX_LIGHTS =                          25;
static const glm::vec3 SUN_DIRECTION =                      glm::vec3(-0.4f, -1.f, -0.3f);
// Turned off the sun so that you can see the point light shadows of the lanterns!
static const glm::vec4 SUN_COLOR =                          glm::vec4(1.f, 1.f, 1.f, 0.0f);
static const uint32_t SHADOW_MAP_SIZE =                     4096;
static const float SHADOW_EXTENT =                          10.f;
static const float SHADOW_DISTANCE =                        10.f;
static const uint32_t PARTICLE_COUNT = 8000; // I modified this to make the snow look good
static const float PARTICLE_SIZE =                          0.025f; //particle width in world space
static const char* COMPUTE_SHADER_PATH =                    SHADER_DIR "/particle.comp.spv";
static const char* PARTICLE_VERTEX_SHADER_PATH =            SHADER_DIR "/particle.vert.spv";
static const char* PARTICLE_FRAGMENT_SHADER_PATH =          SHADER_DIR "/particle.frag.spv";
// Added consts not included in the original scaffold
static const float FLOOR_Y = 0.f;
static const glm::vec4 SPAWN_MIN = glm::vec4(-8.f, 6.f, -8.f, 0.f);
static const glm::vec4 SPAWN_MAX = glm::vec4( 8.f, 6.5f, 8.f, 0.f);
// Needed to meet minor feature "basic forces"
static const glm::vec4 WIND = glm::vec4(1.4f, 0.f, 0.2f, 2.f);
// Needed to meet major feature "point light shadows"
static const uint32_t POINT_SHADOW_SIZE = 1024;
static const float POINT_SHADOW_NEAR = 0.05f;
static const float POINT_SHADOW_FAR = 20.f;
static const uint32_t POINT_SHADOW_LIGHT = 3;

//validation layers pass debug messages to this function
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity, VkDebugUtilsMessageTypeFlagsEXT message_types, const VkDebugUtilsMessengerCallbackDataEXT* callback_data, void* user_data);
//set up our validation.txt file
static std::ofstream& ValidationLog();
//check if our physical device supports a certain device extension
static bool DeviceSupportsExtension(VkPhysicalDevice physical_device, const char* extension_name);
//check if our Vulkan instance supports a certain instance extension
static bool InstanceSupportsExtension(const char* extension_name);
//check if a certain bit index is set in a mask
static bool BitIsSet(uint32_t mask, uint32_t bit_index);
//find first memory type supporting type_bits and property_flags, returns UINT32_MAX otherwise
static uint32_t FindMemoryType(const VkPhysicalDeviceMemoryProperties& memory_properties, uint32_t type_bits, VkMemoryPropertyFlags property_flags);
//helper function to interpret shader code as a uint32 vec
static std::vector<uint32_t> ReadSpirv(const char* path);
//glfw only reports the scroll wheel through a callback so notches pile up here until UpdateInput drains them
static double scroll_accumulator = 0.0;
//glfw calls this during glfwPollEvents for every scroll event
static void scroll_callback(GLFWwindow* window, double x_offset, double y_offset);

int WindowCreation(const char* window_title, GLFWwindow*& window);
int InstanceCreation(const char* window_title, VkInstance& instance, VkDebugUtilsMessengerEXT& debug_messenger);
int SurfaceCreation(const VkInstance& instance, GLFWwindow*& window, VkSurfaceKHR& surface);
int PhysicalDeviceSelection(const VkInstance& instance, const VkSurfaceKHR& surface, VkPhysicalDevice& physical_device, uint32_t& queue_family_index);
int DeviceAndQueueCreation(const uint32_t& queue_family_index, const VkPhysicalDevice& physical_device, VkDevice& device, VkQueue& queue);
int SwapchainCreation(const VkPhysicalDevice& physical_device, const VkSurfaceKHR& surface, GLFWwindow*& window, const VkDevice& device, VkSurfaceFormatKHR& surface_format, VkExtent2D& swapchain_extent, VkSwapchainKHR& swapchain, uint32_t& swapchain_image_count, std::vector<VkImage>& swapchain_images);
int CommandBufferCreation(const uint32_t& queue_family_index, const VkDevice& device, VkCommandPool& command_pool, std::vector<VkCommandBuffer>& command_buffers);
int FrameCreation(const VkDevice& device, const VkPhysicalDevice& physical_device, std::vector<VkImage>& frame, VkPhysicalDeviceMemoryProperties& memory_properties, VkDeviceMemory& frame_memory, std::vector<VkImageView>& frame_view, std::vector<VkImage>& depth, VkDeviceMemory& depth_memory, std::vector<VkImageView>& depth_view);
int GraphicsDescriptorSetCreation(const VkDevice& device, const VkPhysicalDeviceMemoryProperties& memory_properties, std::vector<VkBuffer>& model_buffers, std::vector<VkDeviceMemory>& model_buffer_memory, std::vector<glm::mat4*>& model_matrices, std::vector<VkBuffer>& light_buffers, std::vector<VkDeviceMemory>& light_buffer_memory, std::vector<void*>& light_data, VkDescriptorPool& graphics_descriptor_pool, VkDescriptorSetLayout& graphics_descriptor_set_layout, std::vector<VkDescriptorSet>& graphics_descriptor_set, VkDescriptorSetLayout& material_descriptor_set_layout, VkDescriptorSet& material_descriptor_set);
int GraphicsPipelineCreation(const VkDevice& device, const VkDescriptorSetLayout& graphics_descriptor_set_layout, const VkDescriptorSetLayout& material_descriptor_set_layout, VkPipelineLayout& graphics_pipeline_layout, VkPipeline& graphics_pipeline);
int LightPointPipelineCreation(const VkDevice& device, const VkDescriptorSetLayout& graphics_descriptor_set_layout, VkPipelineLayout& light_point_pipeline_layout, VkPipeline& light_point_pipeline);
int ShadowMapCreation(const VkDevice& device, const VkPhysicalDevice& physical_device, const VkPhysicalDeviceMemoryProperties& memory_properties, const std::vector<VkDescriptorSet>& graphics_descriptor_set, std::vector<VkImage>& shadow_map, VkDeviceMemory& shadow_map_memory, std::vector<VkImageView>& shadow_map_view, VkSampler& shadow_sampler);
int ShadowPipelineCreation(const VkDevice& device, const VkPipelineLayout& graphics_pipeline_layout, VkPipeline& shadow_pipeline);
int SynchronizationObjectCreation(const VkDevice& device, const uint32_t& swapchain_image_count, std::vector<VkSemaphore>& image_available_semaphores, std::vector<VkSemaphore>& render_finished_semaphores, std::vector<VkFence>& in_flight_fences);

// (Project2) NEW FUNCTIONS
int ComputeDescriptorSetCreation(const VkDevice& device, const VkBuffer& particle_buffer, VkDescriptorPool& descriptor_pool, VkDescriptorSetLayout& descriptor_set_layout, VkDescriptorSet& descriptor_set);
int ComputePipelineCreation(const VkDevice& device, const VkDescriptorSetLayout& descriptor_set_layout, VkPipelineLayout& compute_pipeline_layout, VkPipeline& compute_pipeline);
int ParticlePipelineCreation(const VkDevice& device, const VkDescriptorSetLayout& graphics_descriptor_set_layout, VkPipelineLayout& particle_pipeline_layout, VkPipeline& particle_pipeline);

int TransitionDepthToAttachment(const VkQueue& queue, const std::vector<VkCommandBuffer>& command_buffers, const std::vector<VkImage>& depth, const std::vector<VkFence>& in_flight_fences);
int UpdateInput(GLFWwindow*& window, const std::vector<int>& keys_to_track, std::array<KeyState, GLFW_KEY_LAST + 1>& key_states, double& cursor_x, double& cursor_y, double& cursor_dx, double& cursor_dy, double& scroll_dy);
int CreateHostVisibleBuffer(const VkDevice& device, const VkPhysicalDeviceMemoryProperties& memory_properties, VkDeviceSize size, VkBufferUsageFlags usage, bool prefer_device_local, VkBuffer& buffer, VkDeviceMemory& memory, void*& mapped);
int CreateDeviceLocalBuffer(const VkDevice& device, const VkPhysicalDeviceMemoryProperties& memory_properties, const VkQueue& queue, const VkCommandPool& command_pool, const void* data, VkDeviceSize size, VkBufferUsageFlags usage, VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access, VkBuffer& buffer, VkDeviceMemory& memory);
int CreateTextureImage(const VkDevice& device, const VkPhysicalDeviceMemoryProperties& memory_properties, const VkQueue& queue, const VkCommandPool& command_pool, const std::string& texture_filename, VkImage& image, VkDeviceMemory& memory, VkImageView& view);
int CreateTextureSampler(const VkDevice& device, VkSampler& sampler);
int LoadMtl(const std::string& mtl_filename, std::unordered_set<std::string>& loaded_mtl_files, std::unordered_map<std::string, uint32_t>& material_lookup, std::vector<Material>& materials, std::string& diffuse_texture_filename);
int LoadObj(const std::string& filename, std::unordered_set<std::string>& loaded_mtl_files, std::unordered_map<std::string, uint32_t>& material_lookup, std::vector<Material>& materials, std::string& diffuse_texture_filename, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices);
int MeshImport(std::vector<Mesh>& meshes, std::unordered_map<std::string, uint32_t>& mesh_lookup, std::vector<Material>& materials, std::string& diffuse_texture_filename, std::vector<Vertex>& all_vertices, std::vector<uint32_t>& all_indices);
int MeshUpload(const VkDevice& device, const VkPhysicalDeviceMemoryProperties& memory_properties, const VkQueue& queue, const VkCommandPool& command_pool, const std::vector<Vertex>& all_vertices, const std::vector<uint32_t>& all_indices, VkBuffer& model_vertex_buffer, VkDeviceMemory& model_vertex_buffer_memory, VkBuffer& model_index_buffer, VkDeviceMemory& model_index_buffer_memory);
int MaterialUpload(const VkDevice& device, const VkPhysicalDeviceMemoryProperties& memory_properties, const VkQueue& queue, const VkCommandPool& command_pool, const std::vector<Material>& materials, const VkDescriptorSet& material_descriptor_set, VkBuffer& material_buffer, VkDeviceMemory& material_buffer_memory);
int TextureUpload(const VkDevice& device, const VkPhysicalDeviceMemoryProperties& memory_properties, const VkQueue& queue, const VkCommandPool& command_pool, const std::string& diffuse_texture_filename, const VkDescriptorSet& material_descriptor_set, VkImage& diffuse_texture, VkDeviceMemory& diffuse_texture_memory, VkImageView& diffuse_texture_view, VkSampler& diffuse_sampler);

// Functions I added to implement features
// Function for Point Light Shadows
int CreatePointShadow(const VkDevice& device, const VkPhysicalDeviceMemoryProperties& properties, const std::vector<VkDescriptorSet>& set, const VkSampler& sampler, std::vector<VkImage>& cube, VkDeviceMemory& cube_memory, std::vector<VkImageView>& cube_view, std::vector<VkImageView>& cube_face);


int main()
{
    //--------------------------------------------------------------------------------------//
    // WINDOW CREATION
    //--------------------------------------------------------------------------------------//

    const char* window_title = APPLICATION_TITLE;
    GLFWwindow* window = nullptr;
    int wc = WindowCreation(window_title, window);
    if (wc != 0) return wc;

    //--------------------------------------------------------------------------------------//
    // INSTANCE CREATION
    //--------------------------------------------------------------------------------------//

    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
    int ic = InstanceCreation(window_title, instance, debug_messenger);
    if (ic != 0) return ic;

    //--------------------------------------------------------------------------------------//
    // SURFACE CREATION
    //--------------------------------------------------------------------------------------//

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    int sc = SurfaceCreation(instance, window, surface);
    if (sc != 0) return sc;

    //--------------------------------------------------------------------------------------//
    // PHYSICAL DEVICE SELECTION
    //--------------------------------------------------------------------------------------//

    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    uint32_t queue_family_index = UINT32_MAX;
    int pds = PhysicalDeviceSelection(instance, surface, physical_device, queue_family_index);
    if (pds != 0) return pds;

    //--------------------------------------------------------------------------------------//
    // LOGICAL DEVICE & QUEUE CREATION
    //--------------------------------------------------------------------------------------//

    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    int daqc = DeviceAndQueueCreation(queue_family_index, physical_device, device, queue);
    if (daqc != 0) return daqc;

    //--------------------------------------------------------------------------------------//
    // SWAPCHAIN CREATION
    //--------------------------------------------------------------------------------------//

    VkSurfaceFormatKHR surface_format;
    VkExtent2D swapchain_extent;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    uint32_t swapchain_image_count = 0;
    std::vector<VkImage> swapchain_images;
    int swc = SwapchainCreation(physical_device, surface, window, device, surface_format, swapchain_extent, swapchain, swapchain_image_count, swapchain_images);
    if (swc != 0) return swc;

    //--------------------------------------------------------------------------------------//
    // COMMAND POOL CREATION & COMMAND BUFFER ALLOCATION 
    //--------------------------------------------------------------------------------------//

    VkCommandPool command_pool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> command_buffers(FRAMES_IN_FLIGHT);
    int cbc = CommandBufferCreation(queue_family_index, device, command_pool, command_buffers);
    if (cbc != 0) return cbc;

    //--------------------------------------------------------------------------------------//
    // IMAGE ALLOCATION
    //--------------------------------------------------------------------------------------//

    std::vector<VkImage> frame(FRAMES_IN_FLIGHT);
    VkPhysicalDeviceMemoryProperties memory_properties = {};
    VkDeviceMemory frame_memory = VK_NULL_HANDLE;
    std::vector<VkImageView> frame_view(FRAMES_IN_FLIGHT);

    std::vector<VkImage> depth(FRAMES_IN_FLIGHT);
    VkDeviceMemory depth_memory = VK_NULL_HANDLE;
    std::vector<VkImageView> depth_view(FRAMES_IN_FLIGHT);

    int fc = FrameCreation(device, physical_device, frame, memory_properties, frame_memory, frame_view, depth, depth_memory, depth_view);
    if (fc != 0) return fc;

    //--------------------------------------------------------------------------------------//
    // GRAPHICS DESCRIPTOR SET CREATION
    //--------------------------------------------------------------------------------------//

    std::vector<VkBuffer> model_buffers(FRAMES_IN_FLIGHT);
    std::vector<VkDeviceMemory> model_buffer_memory(FRAMES_IN_FLIGHT);
    std::vector<glm::mat4*> model_matrices(FRAMES_IN_FLIGHT);
    std::vector<VkBuffer> light_buffers(FRAMES_IN_FLIGHT);
    std::vector<VkDeviceMemory> light_buffer_memory(FRAMES_IN_FLIGHT);
    std::vector<void*> light_data(FRAMES_IN_FLIGHT);
    VkDescriptorPool graphics_descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout graphics_descriptor_set_layout = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> graphics_descriptor_set(FRAMES_IN_FLIGHT);
    VkDescriptorSetLayout material_descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorSet material_descriptor_set = VK_NULL_HANDLE;
    int gdsc = GraphicsDescriptorSetCreation(device, memory_properties, model_buffers, model_buffer_memory, model_matrices, light_buffers, light_buffer_memory, light_data, graphics_descriptor_pool, graphics_descriptor_set_layout, graphics_descriptor_set, material_descriptor_set_layout, material_descriptor_set);
    if (gdsc != 0) return gdsc;

    //--------------------------------------------------------------------------------------//
    // SHADOW MAP CREATION
    //--------------------------------------------------------------------------------------//

    std::vector<VkImage> shadow_map(FRAMES_IN_FLIGHT);
    VkDeviceMemory shadow_map_memory = VK_NULL_HANDLE;
    std::vector<VkImageView> shadow_map_view(FRAMES_IN_FLIGHT);
    VkSampler shadow_sampler = VK_NULL_HANDLE;
    int smc = ShadowMapCreation(device, physical_device, memory_properties, graphics_descriptor_set, shadow_map, shadow_map_memory, shadow_map_view, shadow_sampler);
    if (smc != 0) return smc;

    // Point Light Shadows - Create Point Shadow

    std::vector<VkImage> cube_image(FRAMES_IN_FLIGHT);
    VkDeviceMemory cube_memory = VK_NULL_HANDLE;
    std::vector<VkImageView> cube_view(FRAMES_IN_FLIGHT);
    std::vector<VkImageView> cube_face_view(FRAMES_IN_FLIGHT * 6);
    int psc = CreatePointShadow(device, memory_properties, graphics_descriptor_set, shadow_sampler, cube_image, cube_memory, cube_view, cube_face_view);
    if (psc != 0) return psc;

    //--------------------------------------------------------------------------------------//
    // GRAPHICS PIPELINE CREATION
    //--------------------------------------------------------------------------------------//

    VkPipelineLayout graphics_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline graphics_pipeline = VK_NULL_HANDLE;
    int gpc = GraphicsPipelineCreation(device, graphics_descriptor_set_layout, material_descriptor_set_layout, graphics_pipeline_layout, graphics_pipeline);
    if (gpc != 0) return gpc;

    //second pipeline for drawing large points over light positions
    VkPipelineLayout light_point_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline light_point_pipeline = VK_NULL_HANDLE;
    int lppc = LightPointPipelineCreation(device, graphics_descriptor_set_layout, light_point_pipeline_layout, light_point_pipeline);
    if (lppc != 0) return lppc;

    VkPipeline shadow_pipeline = VK_NULL_HANDLE;
    int spc = ShadowPipelineCreation(device, graphics_pipeline_layout, shadow_pipeline);
    if (spc != 0) return spc;

    //--------------------------------------------------------------------------------------//
    // MESH IMPORT
    //--------------------------------------------------------------------------------------//

    std::vector<Mesh> meshes;
    std::unordered_map<std::string, uint32_t> mesh_lookup;
    std::vector<Material> materials;
    std::string diffuse_texture_filename;
    std::vector<Vertex> all_vertices;
    std::vector<uint32_t> all_indices;
    int mi = MeshImport(meshes, mesh_lookup, materials, diffuse_texture_filename, all_vertices, all_indices);
    if (mi != 0) return mi;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    VkBuffer model_vertex_buffer = VK_NULL_HANDLE;
    VkDeviceMemory model_vertex_buffer_memory = VK_NULL_HANDLE;
    VkBuffer model_index_buffer = VK_NULL_HANDLE;
    VkDeviceMemory model_index_buffer_memory = VK_NULL_HANDLE;
    int mu = MeshUpload(device, memory_properties, queue, command_pool, all_vertices, all_indices, model_vertex_buffer, model_vertex_buffer_memory, model_index_buffer, model_index_buffer_memory);
    if (mu != 0) return mu;

    //--------------------------------------------------------------------------------------//
    // MATERIAL UPLOAD
    //--------------------------------------------------------------------------------------//

    VkBuffer material_buffer = VK_NULL_HANDLE;
    VkDeviceMemory material_buffer_memory = VK_NULL_HANDLE;
    int mtu = MaterialUpload(device, memory_properties, queue, command_pool, materials, material_descriptor_set, material_buffer, material_buffer_memory);
    if (mtu != 0) return mtu;

    //--------------------------------------------------------------------------------------//
    // TEXTURE UPLOAD
    //--------------------------------------------------------------------------------------//

    VkImage diffuse_texture = VK_NULL_HANDLE;
    VkDeviceMemory diffuse_texture_memory = VK_NULL_HANDLE;
    VkImageView diffuse_texture_view = VK_NULL_HANDLE;
    VkSampler diffuse_sampler = VK_NULL_HANDLE;
    int tu = TextureUpload(device, memory_properties, queue, command_pool, diffuse_texture_filename, material_descriptor_set, diffuse_texture, diffuse_texture_memory, diffuse_texture_view, diffuse_sampler);
    if (tu != 0) return tu;

    //--------------------------------------------------------------------------------------//
    // SCENE SETUP
    //--------------------------------------------------------------------------------------//

    std::vector<Object> objects = {
        {.mesh = mesh_lookup.at("Floor.obj"),        .position = {0.f, 0.f, 0.f},     .rotation = {0.f, 0.f, 0.f},    .scale = {0.5f, 0.5f, 0.5f},    .animation = ANIMATION_NONE},
        {.mesh = mesh_lookup.at("Lamp.obj"),         .position = {0.3f, 0.f, -3.8f}, .rotation = {0.f, -90.f, 0.f}, .scale = {0.5f, 0.5f, 0.5f},    .animation = ANIMATION_NONE},
        {.mesh = mesh_lookup.at("Lamp.obj"),         .position = {3.8f, 0.f, -2.3f}, .rotation = {0.f, -90.f, 0.f}, .scale = {0.5f, 0.5f, 0.5f},    .animation = ANIMATION_NONE},
        {.mesh = mesh_lookup.at("Lamp.obj"),         .position = {-4.2f, 0.f, 2.3f}, .rotation = {0.f, -90.f, 0.f}, .scale = {0.5f, 0.5f, 0.5f},    .animation = ANIMATION_NONE},
        {.mesh = mesh_lookup.at("Lamp.obj"),         .position = {1.8f, 0.f, 1.7f},  .rotation = {0.f, -90.f, 0.f}, .scale = {0.5f, 0.5f, 0.5f},    .animation = ANIMATION_NONE},
        {.mesh = mesh_lookup.at("Lamp.obj"),         .position = {-2.3f, 0.f, 3.8f}, .rotation = {0.f, -90.f, 0.f}, .scale = {0.5f, 0.5f, 0.5f},    .animation = ANIMATION_NONE},
        {.mesh = mesh_lookup.at("Lamp.obj"),         .position = {3.7f, 0.f, 4.2f},  .rotation = {0.f, -90.f, 0.f}, .scale = {0.5f, 0.5f, 0.5f},    .animation = ANIMATION_NONE},
        {.mesh = mesh_lookup.at("logcottage.obj"), .position = {-3.f, 0.f, 0.f}, .rotation = {0.f, 90.f, 0.f}, .scale = {0.15f, 0.15f, 0.15f}, .animation = ANIMATION_NONE},
        {.mesh = mesh_lookup.at("Fir-tree_obj.obj"), .position = {-3.8f, 0.f, -4.2f}, .rotation = {0.f, 0.f, 0.f}, .scale = {0.3f, 0.3f, 0.3f},    .animation = ANIMATION_NONE},
        {.mesh = mesh_lookup.at("Fir-tree_obj.obj"), .position = {-1.7f, 0.f, -3.7f}, .rotation = {0.f, 70.f, 0.f}, .scale = {0.3f, 0.3f, 0.3f},    .animation = ANIMATION_NONE},
        {.mesh = mesh_lookup.at("Fir-tree_obj.obj"), .position = {2.3f, 0.f, -4.2f}, .rotation = {0.f, 140.f, 0.f}, .scale = {0.3f, 0.3f, 0.3f},    .animation = ANIMATION_NONE},
        {.mesh = mesh_lookup.at("Fir-tree_obj.obj"), .position = {4.2f, 0.f, -3.7f}, .rotation = {0.f, 215.f, 0.f}, .scale = {0.3f, 0.3f, 0.3f},    .animation = ANIMATION_NONE},
        {.mesh = mesh_lookup.at("Fir-tree_obj.obj"), .position = {-4.3f, 0.f, -1.8f}, .rotation = {0.f, 30.f, 0.f}, .scale = {0.3f, 0.3f, 0.3f},    .animation = ANIMATION_NONE},
        {.mesh = mesh_lookup.at("Fir-tree_obj.obj"), .position = {1.7f, 0.f, -2.2f}, .rotation = {0.f, 105.f, 0.f}, .scale = {0.3f, 0.3f, 0.3f},    .animation = ANIMATION_NONE},
        {.mesh = mesh_lookup.at("Fir-tree_obj.obj"), .position = {2.3f, 0.f, 0.3f}, .rotation = {0.f, 180.f, 0.f}, .scale = {0.3f, 0.3f, 0.3f},    .animation = ANIMATION_NONE},
        {.mesh = mesh_lookup.at("Fir-tree_obj.obj"), .position = {-0.3f, 0.f, 2.2f}, .rotation = {0.f, 250.f, 0.f}, .scale = {0.3f, 0.3f, 0.3f},    .animation = ANIMATION_NONE},
        {.mesh = mesh_lookup.at("Fir-tree_obj.obj"), .position = {4.2f, 0.f, 1.8f}, .rotation = {0.f, 55.f, 0.f},  .scale = {0.3f, 0.3f, 0.3f},    .animation = ANIMATION_NONE},
        {.mesh = mesh_lookup.at("Fir-tree_obj.obj"), .position = {-3.7f, 0.f, 4.3f}, .rotation = {0.f, 125.f, 0.f}, .scale = {0.3f, 0.3f, 0.3f},    .animation = ANIMATION_NONE},
        {.mesh = mesh_lookup.at("Fir-tree_obj.obj"), .position = {2.2f, 0.f, 3.7f}, .rotation = {0.f, 300.f, 0.f}, .scale = {0.3f, 0.3f, 0.3f},    .animation = ANIMATION_NONE},
    };

    if (objects.size() > MAX_OBJECTS) {
        std::fprintf(stderr, "Too many objects: %zu (MAX_OBJECTS = %u)\n", objects.size(), MAX_OBJECTS);
        return EXIT_FAILURE;
    }

    float ambient_strength = 0.05f;
    // Edited for major feature "point light shadows." The commented out one is the original test light from the scaffold.
    // The goal for this feature was to use the lanterns to light the scene, so each light is placed where the lantern would
    // be in the scene. I had to play with the sizing a bunch to get it right.
    std::vector<PointLight> lights = {
        // Original light I made into the moon by making it bigger and moving it up and to the corner
        // {.position = {5.f, 3.f, -5.f, 1.f},  .color = {1.f, 1.f, 1.f, 3.f}},

        // Lanterns
        {.position = {0.3f, 1.25f, -3.8f, 1.f}, .color = {1.f, 0.7f, 0.35f, 2.5f}},
        {.position = {3.8f, 1.25f, -2.3f, 1.f}, .color = {1.f, 0.78f, 0.5f, 2.0f}},
        {.position = {-4.2f, 1.25f, 2.3f, 1.f}, .color = {1.f, 0.78f, 0.5f, 2.0f}},
        {.position = {1.8f, 1.25f, 1.7f, 1.f}, .color = {1.f, 0.78f, 0.5f, 2.0f}},
        {.position = {-2.3f, 1.25f, 3.8f, 1.f}, .color = {1.f, 0.78f, 0.5f, 2.0f}},
        {.position = {3.7f, 1.25f, 4.2f, 1.f}, .color = {1.f, 0.7f, 0.35f, 2.5f}},
    };

    float camera_light_intensity = 0.f;
    const float LIGHT_STEP = 1.25f;
    const float LIGHT_MIN = 1.f;
    const float LIGHT_MAX = 10000.f;

    auto step_intensity = [&](float& intensity, double steps) {
        if (intensity == 0.f) {
            if (steps > 0.0) intensity = LIGHT_MIN;
        }
        else {
            intensity *= std::pow(LIGHT_STEP, (float)steps);
            if (intensity < LIGHT_MIN) intensity = 0.f;
            intensity = glm::min(intensity, LIGHT_MAX);
        }
    };

    if (lights.size() + 1 > MAX_LIGHTS) {
        std::fprintf(stderr, "Too many lights: %zu + camera light (MAX_LIGHTS = %u)\n", lights.size(), MAX_LIGHTS);
        return EXIT_FAILURE;
    }

    //--------------------------------------------------------------------------------------//
    // PARTICLE BUFFER CONSTRUCTION
    //--------------------------------------------------------------------------------------//

    //-------------------------------------------------------------------------------------//
    // PART B: INITIAL PARTICLE DATA
    //-------------------------------------------------------------------------------------//
    // Once PARTICLE_COUNT is set to be > 0, you can populate the particles vector and create
    // the device local buffer. This is the only time the host will touch or see particle
    // data. Once this code block passes, it is only handled on the GPU.
    //
    // The following 3 main() code blocks (compute descriptor set creation, compute pipeline
    // creation, and particle pipeline creation) do not need to be touched. You simply need
    // to define the functions they call. Do so before worrying about the render loop.
    //
    // Hint:    For a continuous and smooth simulation, should every particle start with the
    //          same age? What happens if they all die at the same time?

    std::vector<Particle> particles;
    VkDeviceSize particle_bytes = sizeof(Particle) * PARTICLE_COUNT;
    VkBuffer particle_buffer = VK_NULL_HANDLE;
    VkDeviceMemory particle_memory = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < PARTICLE_COUNT; ++i) {
        // Particles currently live between 5-15 seconds
        float lifespan = 5.f + (10.f * dis(gen));
        particles.push_back({
            .position = {-8.f + (16.f * dis(gen)), 6.f * dis(gen), -8.f + (16.f * dis(gen))},
            .current_life = lifespan * dis(gen),
            .velocity = {0.f, -0.5f, 0.f},
            .phase = 6.2831853f * dis(gen),
            .lifespan = lifespan,
        });
    }
    //works once particle_bytes > 0 (uncommented existing code!)
    int pb = CreateDeviceLocalBuffer(device, memory_properties, queue, command_pool, particles.data(), particle_bytes, 
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT, particle_buffer, particle_memory);
    if (pb != 0) return pb;

    //--------------------------------------------------------------------------------------//
    // COMPUTE DESCRIPTOR SET CREATION
    //--------------------------------------------------------------------------------------//

    VkDescriptorPool compute_descriptor_pool = VK_NULL_HANDLE; 
    VkDescriptorSetLayout compute_descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorSet compute_descriptor_set = VK_NULL_HANDLE;
    int cdsc = ComputeDescriptorSetCreation(device, particle_buffer, compute_descriptor_pool, compute_descriptor_set_layout, compute_descriptor_set);
    if (cdsc != 0) return cdsc;

    //--------------------------------------------------------------------------------------//
    // COMPUTE PIPELINE CREATION
    //--------------------------------------------------------------------------------------//

    VkPipelineLayout compute_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline compute_pipeline = VK_NULL_HANDLE;
    int cpc = ComputePipelineCreation(device, compute_descriptor_set_layout, compute_pipeline_layout, compute_pipeline);
    if (cpc != 0) return cpc;

    //--------------------------------------------------------------------------------------//
    // PARTICLE PIPELINE CREATION
    //--------------------------------------------------------------------------------------//

    VkPipelineLayout particle_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline particle_pipeline = VK_NULL_HANDLE;
    int pppc = ParticlePipelineCreation(device, graphics_descriptor_set_layout, particle_pipeline_layout, particle_pipeline);
    if (pppc != 0) return pppc;

    //--------------------------------------------------------------------------------------//
    // SYNCHRONIZATION OBJECT CREATION
    //--------------------------------------------------------------------------------------//

    std::vector<VkSemaphore> image_available_semaphores(FRAMES_IN_FLIGHT);
    std::vector<VkSemaphore> render_finished_semaphores(swapchain_image_count);
    std::vector<VkFence> in_flight_fences(FRAMES_IN_FLIGHT);
    int soc = SynchronizationObjectCreation(device, swapchain_image_count, image_available_semaphores, render_finished_semaphores, in_flight_fences);
    if (soc != 0) return soc;

    //--------------------------------------------------------------------------------------//
    // FRAME LOOP VARIABLES & CONSTANTS
    //--------------------------------------------------------------------------------------//

    TransitionDepthToAttachment(queue, command_buffers, depth, in_flight_fences);

    uint32_t image_index = 0;
    VkCommandBufferBeginInfo command_buffer_begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = NULL,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VkImageSubresourceRange subresource_range = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
    VkImageSubresourceRange shadow_subresource_range = {
        .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
    VkViewport shadow_viewport = {
        .x = 0.f,
        .y = 0.f,
        .width = float(SHADOW_MAP_SIZE),
        .height = float(SHADOW_MAP_SIZE),
        .minDepth = 0.f,
        .maxDepth = 1.f,
    };
    VkRect2D shadow_scissor = {
        .offset = {0, 0},
        .extent = VkExtent2D{SHADOW_MAP_SIZE, SHADOW_MAP_SIZE},
    };
    glm::vec3 sun_direction = glm::normalize(SUN_DIRECTION);
    glm::vec3 sun_up = glm::abs(sun_direction.y) > 0.99f ? glm::vec3(0.f, 0.f, 1.f) : glm::vec3(0.f, 1.f, 0.f);
    glm::mat4 sun_view = glm::lookAtRH(-sun_direction * SHADOW_DISTANCE, glm::vec3(0.f), sun_up);
    glm::mat4 sun_proj = glm::orthoRH_ZO(-SHADOW_EXTENT, SHADOW_EXTENT, -SHADOW_EXTENT, SHADOW_EXTENT, 0.f, 2.f * SHADOW_DISTANCE);
    glm::mat4 sun_VP = sun_proj * sun_view;
    VkImageBlit image_blit = {
        .srcSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .srcOffsets = { {0, 0, 0}, {FRAME_WIDTH, FRAME_HEIGHT, 1} },
        .dstSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .dstOffsets = { {0, 0, 0}, {(int32_t)swapchain_extent.width, (int32_t)swapchain_extent.height, 1} },
    };
    VkViewport viewport = {
        .x = 0.f,
        .y = 0,
        .width = float(FRAME_WIDTH),
        .height = float(FRAME_HEIGHT),
        .minDepth = 0.f,
        .maxDepth = 1.f,
    };
    VkRect2D scissor = {
        .offset = {0, 0},
        .extent = VkExtent2D{FRAME_WIDTH, FRAME_HEIGHT},
    };
    VkDeviceSize vert_offset = 0;
    glm::mat4 translation;
    glm::mat4 rotation;
    glm::mat4 scale;
    glm::mat4 rotation_x;
    glm::mat4 rotation_y;
    glm::mat4 rotation_z;
    bool animate = true;

    float aspect_ratio = float(FRAME_WIDTH)/float(FRAME_HEIGHT);
    float h_fov = 50.f;
    float v_fov = glm::degrees(2.f * glm::atan(glm::tan(glm::radians(h_fov) / 2.f) / aspect_ratio));
    float near_plane = 0.1f;
    float far_plane = 100.f;
    float sensitivity = 1.f;
    float y_sens_multiplier = 1.f;
    float deg_per_count = 0.022f;

    float movement_speed = 1.f;
    bool fast_movement = false;
    float fast_multiplier = 2.f;
    glm::vec3 move_input = {0.f, 0.f, 0.f};
    glm::mat4 view;
    glm::mat4 camera;
    glm::mat4 cam_trs_m;
    glm::mat4 cam_rot_m;
    glm::mat4 cam_rot_x;
    glm::mat4 cam_rot_y;
    glm::mat4 cam_rot_z;
    glm::vec3 cam_pos = {0.f, 1.f, 4.5f};
    glm::vec3 cam_rot = {0.f, 0.f, 0.f};

    glm::mat4 proj;
    glm::mat4 VP;

    uint32_t selected_light = 0;
    float light_move_speed = 3.f;
    glm::vec3 light_move_input = {0.f, 0.f, 0.f};
    bool alt_held = false;
    bool print_fps = true;
    bool render_light_points = true;

    std::array<KeyState, GLFW_KEY_LAST + 1> key_states{};
    std::vector<int> keys_to_track = {GLFW_KEY_SPACE, GLFW_KEY_LEFT_CONTROL, GLFW_KEY_W, GLFW_KEY_A, GLFW_KEY_S, GLFW_KEY_D, GLFW_KEY_ESCAPE, GLFW_KEY_1, GLFW_KEY_3, GLFW_KEY_T, GLFW_KEY_UP, GLFW_KEY_DOWN, GLFW_KEY_LEFT, GLFW_KEY_RIGHT, GLFW_KEY_TAB, GLFW_KEY_LEFT_SHIFT, GLFW_KEY_LEFT_ALT, GLFW_KEY_GRAVE_ACCENT};
    double cursor_x = 0, cursor_y = 0, cursor_dx = 0, cursor_dy = 0;
    double scroll_dy = 0;
    glfwGetCursorPos(window, &cursor_x, &cursor_y);

    VkCommandBuffer command_buffer;
    VkSemaphore image_available_semaphore;
    VkSemaphore render_finished_semaphore;
    VkFence in_flight_fence;

    uint32_t current_frame = 0;
    uint64_t total_frames = 0;
    uint64_t recent_frames = 0;
    using clock_type = std::chrono::steady_clock;
    clock_type::time_point start_time = clock_type::now();
    clock_type::time_point end_time;
    clock_type::time_point curr_time = start_time;
    clock_type::time_point last_frame_time = start_time;
    clock_type::time_point last_cycle_time = start_time;
    float elapsed = 0.f;
    float last_elapsed = 0.f;
    float dt = 0.f;
    double fps = 0.0;
    float frame_cycle = 2.f;

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    //--------------------------------------------------------------------------------------//
    // FRAME LOOP
    //--------------------------------------------------------------------------------------//

    while (!glfwWindowShouldClose(window)) {

        //--------------------------------------------------------------------------------------//
        // WAIT AND ACQUIRE SWAPCHAIN IMAGE
        //--------------------------------------------------------------------------------------//

        command_buffer = command_buffers[current_frame];
        image_available_semaphore = image_available_semaphores[current_frame];
        in_flight_fence = in_flight_fences[current_frame];

        vkWaitForFences(device, 1, &in_flight_fence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &in_flight_fence);

        image_index = 0;
        VkResult acquire_result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, image_available_semaphore, VK_NULL_HANDLE, &image_index);

        if (acquire_result != VK_SUCCESS && acquire_result != VK_SUBOPTIMAL_KHR) {
            std::fprintf(stderr, "vkAcquireNextImageKHR failed with VkResult %d\n", acquire_result);
            return EXIT_FAILURE;
        }

        render_finished_semaphore = render_finished_semaphores[image_index];

        //--------------------------------------------------------------------------------------//
        // INPUT DETECTION
        //--------------------------------------------------------------------------------------//

        glfwPollEvents();

        move_input = {0.f, 0.f, 0.f};
        light_move_input = {0.f, 0.f, 0.f};

        UpdateInput(window, keys_to_track, key_states, cursor_x, cursor_y, cursor_dx, cursor_dy, scroll_dy);

        alt_held = key_states[GLFW_KEY_LEFT_ALT] == KEY_STATE_PRESSED || key_states[GLFW_KEY_LEFT_ALT] == KEY_STATE_DOWN;

        for (uint32_t i = 0; i < keys_to_track.size(); ++i) {
            int key_id = keys_to_track[i];
            switch (key_id) {
                case GLFW_KEY_SPACE: {
                    if (key_states[key_id] == KEY_STATE_PRESSED || key_states[key_id] == KEY_STATE_DOWN) {
                        move_input.y += 1.f;
                    }
                    break;
                }
                case GLFW_KEY_LEFT_CONTROL: {
                    if (key_states[key_id] == KEY_STATE_PRESSED || key_states[key_id] == KEY_STATE_DOWN) {
                        move_input.y -= 1.f;
                    }
                    break;
                }
                case GLFW_KEY_W: {
                    if (key_states[key_id] == KEY_STATE_PRESSED || key_states[key_id] == KEY_STATE_DOWN) {
                        move_input.z += 1.f;
                    }
                    break;
                }
                case GLFW_KEY_S: {
                    if (key_states[key_id] == KEY_STATE_PRESSED || key_states[key_id] == KEY_STATE_DOWN) {
                        move_input.z -= 1.f;
                    }
                    break;
                }
                case GLFW_KEY_A: {
                    if (key_states[key_id] == KEY_STATE_PRESSED || key_states[key_id] == KEY_STATE_DOWN) {
                        move_input.x -= 1.f;
                    }
                    break;
                }
                case GLFW_KEY_D: {
                    if (key_states[key_id] == KEY_STATE_PRESSED || key_states[key_id] == KEY_STATE_DOWN) {
                        move_input.x += 1.f;
                    }
                    break;
                }
                case GLFW_KEY_ESCAPE: {
                    if (key_states[key_id] == KEY_STATE_PRESSED) {
                        if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED) glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                        else if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_NORMAL) glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                    }
                    break;
                }
                case GLFW_KEY_1: {
                    if (key_states[key_id] == KEY_STATE_PRESSED) {
                        animate = !animate;
                    }
                    break;
                }
                case GLFW_KEY_3: {
                    if (key_states[key_id] == KEY_STATE_PRESSED) {
                        render_light_points = !render_light_points;
                    }
                    break;
                }
                case GLFW_KEY_T: {
                    if (key_states[key_id] == KEY_STATE_PRESSED) {
                        printf("Camera:\t%s\nLight[%u]:\t%s\n", glm::to_string(cam_pos).c_str(), selected_light, glm::to_string(glm::vec3(lights[selected_light].position)).c_str());
                    }
                    break;
                }
                case GLFW_KEY_UP: {
                    if (key_states[key_id] == KEY_STATE_PRESSED || key_states[key_id] == KEY_STATE_DOWN) {
                        if (alt_held) light_move_input.z -= 1.f;
                        else light_move_input.y += 1.f;
                    }
                    break;
                }
                case GLFW_KEY_DOWN: {
                    if (key_states[key_id] == KEY_STATE_PRESSED || key_states[key_id] == KEY_STATE_DOWN) {
                        if (alt_held) light_move_input.z += 1.f;
                        else light_move_input.y -= 1.f;
                    }
                    break;
                }
                case GLFW_KEY_LEFT: {
                    if (key_states[key_id] == KEY_STATE_PRESSED || key_states[key_id] == KEY_STATE_DOWN) {
                        light_move_input.x -= 1.f;
                    }
                    break;
                }
                case GLFW_KEY_RIGHT: {
                    if (key_states[key_id] == KEY_STATE_PRESSED || key_states[key_id] == KEY_STATE_DOWN) {
                        light_move_input.x += 1.f;
                    }
                    break;
                }
                case GLFW_KEY_TAB: {
                    if (key_states[key_id] == KEY_STATE_PRESSED) {
                        selected_light = (selected_light + 1) % (uint32_t)lights.size();
                        printf("Selected light: %u\n", selected_light);
                    }
                    break;
                }
                case GLFW_KEY_GRAVE_ACCENT: {
                    if (key_states[key_id] == KEY_STATE_PRESSED) {
                        print_fps = !print_fps;
                    }
                    break;
                }
                case GLFW_KEY_LEFT_SHIFT: {
                    if (key_states[key_id] == KEY_STATE_PRESSED || key_states[key_id] == KEY_STATE_DOWN) {
                        fast_movement = true;
                    }
                    else {
                        fast_movement = false;
                    }
                    break;
                }
            }
        }

        if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED && (cursor_dx != 0.f || cursor_dy != 0.f)) {
            cam_rot.y -= sensitivity * cursor_dx * deg_per_count;
            cam_rot.x -= sensitivity * cursor_dy * deg_per_count * y_sens_multiplier;
        }

        cam_rot.x = glm::clamp(cam_rot.x, -89.f, 89.f);

        if (glm::length(light_move_input) > 0.f) {
            lights[selected_light].position += glm::vec4(glm::normalize(light_move_input) * light_move_speed * (fast_movement ? fast_multiplier : 1.f) * dt, 0.f);
        }

        if (scroll_dy != 0.0) {
            if (alt_held) {
                step_intensity(lights[selected_light].color.a, scroll_dy);
                std::printf("Light[%u] intensity: %.2f\n", selected_light, lights[selected_light].color.a);
            }
            else {
                step_intensity(camera_light_intensity, scroll_dy);
                std::printf("Camera light intensity: %.2f\n", camera_light_intensity);
            }
        }

        //--------------------------------------------------------------------------------------//
        // BEGIN COMMAND BUFFER RECORDING
        //--------------------------------------------------------------------------------------//

        VK_CHECK(vkResetCommandBuffer(command_buffer, 0));

        VK_CHECK(vkBeginCommandBuffer(command_buffer, &command_buffer_begin_info));

        //--------------------------------------------------------------------------------------//
        // TRANSITION FOR RENDER PASS
        //--------------------------------------------------------------------------------------//

        VkImageMemoryBarrier2 barrier_to_color_attachment_output = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext = NULL,

            .srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT,
            .srcAccessMask = VK_ACCESS_2_NONE,

            .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,

            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,

            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,

            .image = frame[current_frame],
            .subresourceRange = subresource_range,
        };

        VkDependencyInfo dependency_info_to_cao = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = NULL,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier_to_color_attachment_output,
        };

        vkCmdPipelineBarrier2(command_buffer, &dependency_info_to_cao);

        //--------------------------------------------------------------------------------------//
        // SETUP DRAW CALL
        //--------------------------------------------------------------------------------------//

        VkRenderingAttachmentInfo color_attachment = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = NULL,
            .imageView = frame_view[current_frame],
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = {.color = {.float32 = {0.f, 0.f, 0.f, 1.f}}},
        };

        VkRenderingAttachmentInfo depth_attachment = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = NULL,
            .imageView = depth_view[current_frame],
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .clearValue = {.depthStencil = {.depth = 1.0f}},
        };

        VkRenderingInfo rendering_info = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = NULL,
            .flags = 0,
            .renderArea = { {0, 0}, {FRAME_WIDTH, FRAME_HEIGHT}},
            .layerCount = 1,
            .viewMask = 0,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color_attachment,
            .pDepthAttachment = &depth_attachment,
        };

        //--------------------------------------------------------------------------------------//
        // MATRIX TRANSFORMATIONS
        //--------------------------------------------------------------------------------------//

        cam_rot_x = {
            1.f, 0.f, 0.f, 0.f,
            0.f, glm::cos(glm::radians(cam_rot.x)), glm::sin(glm::radians(cam_rot.x)), 0.f,
            0.f, -glm::sin(glm::radians(cam_rot.x)), glm::cos(glm::radians(cam_rot.x)), 0.f,
            0.f, 0.f, 0.f, 1.f
        };
        cam_rot_y = {
            glm::cos(glm::radians(cam_rot.y)), 0.f, -glm::sin(glm::radians(cam_rot.y)), 0.f,
            0.f, 1.f, 0.f, 0.f,
            glm::sin(glm::radians(cam_rot.y)), 0.f, glm::cos(glm::radians(cam_rot.y)), 0.f,
            0.f, 0.f, 0.f, 1.f
        };
        cam_rot_z = {
            glm::cos(glm::radians(cam_rot.z)), glm::sin(glm::radians(cam_rot.z)), 0.f, 0.f,
            -glm::sin(glm::radians(cam_rot.z)), glm::cos(glm::radians(cam_rot.z)), 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f,
            0.f, 0.f, 0.f, 1.f
        };

        cam_rot_m = cam_rot_y * cam_rot_x * cam_rot_z;

        glm::vec3 cam_right =  glm::vec3(cam_rot_m[0]);
        glm::vec3 cam_forward = -glm::vec3(cam_rot_m[2]);
        glm::vec3 world_up =  glm::vec3(0.f, 1.f, 0.f);

        if (glm::length(move_input) > 0.f) {
            move_input = glm::normalize(move_input);
        }

        cam_pos += (cam_right * move_input.x + world_up * move_input.y + cam_forward * move_input.z) * movement_speed * (fast_movement ? fast_multiplier : 1.f) * dt;

        cam_trs_m = {
            1.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f,
            cam_pos.x, cam_pos.y, cam_pos.z, 1.f
        };

        camera = cam_trs_m * cam_rot_m;

        view = glm::inverse(camera);

        proj = {
            1.f / (aspect_ratio * glm::tan(glm::radians(v_fov)/2.f)), 0.f, 0.f, 0.f,
            0.f, 1.f / (glm::tan(glm::radians(v_fov)/2.f)), 0.f, 0.f,
            0.f, 0.f, far_plane/(near_plane - far_plane), -1.f,
            0.f, 0.f, (far_plane * near_plane)/(near_plane - far_plane), 0.f
        };

        proj[1][1] *= -1.f;

        VP = proj * view;

        glm::mat4* frame_models = model_matrices[current_frame];

        for (uint32_t i = 0; i < objects.size(); ++i) {
            Object& object = objects[i];

            if (animate) {
                switch (object.animation) {
                    case ANIMATION_NONE:
                    default: {
                        break;
                    }
                }
            }

            glm::vec3 p = object.position;
            glm::vec3 r = object.rotation;
            glm::vec3 s = object.scale;

            translation = {
                1.f, 0.f, 0.f, 0.f,
                0.f, 1.f, 0.f, 0.f,
                0.f, 0.f, 1.f, 0.f,
                p.x, p.y, p.z, 1.f
            };
            rotation_x = {
                1.f, 0.f, 0.f, 0.f,
                0.f, glm::cos(glm::radians(r.x)), glm::sin(glm::radians(r.x)), 0.f,
                0.f, -glm::sin(glm::radians(r.x)), glm::cos(glm::radians(r.x)), 0.f,
                0.f, 0.f, 0.f, 1.f
            };
            rotation_y = {
                glm::cos(glm::radians(r.y)), 0.f, -glm::sin(glm::radians(r.y)), 0.f,
                0.f, 1.f, 0.f, 0.f,
                glm::sin(glm::radians(r.y)), 0.f, glm::cos(glm::radians(r.y)), 0.f,
                0.f, 0.f, 0.f, 1.f
            };
            rotation_z = {
                glm::cos(glm::radians(r.z)), glm::sin(glm::radians(r.z)), 0.f, 0.f,
                -glm::sin(glm::radians(r.z)), glm::cos(glm::radians(r.z)), 0.f, 0.f,
                0.f, 0.f, 1.f, 0.f,
                0.f, 0.f, 0.f, 1.f
            };
            rotation = rotation_y * rotation_x * rotation_z;
            scale = {
                s.x, 0.f, 0.f, 0.f,
                0.f, s.y, 0.f, 0.f,
                0.f, 0.f, s.z, 0.f,
                0.f, 0.f, 0.f, 1.f
            };

            frame_models[i] = translation * rotation * scale;
        }

        //--------------------------------------------------------------------------------------//
        // LIGHT DATA
        //--------------------------------------------------------------------------------------//

        //layout: [LightHeader (128 bytes)][PointLight 0][PointLight 1]...[camera light]
        uint8_t* frame_lights = (uint8_t*)light_data[current_frame];

        //uses the camera position so the light sits exactly at the eye
        PointLight camera_light = {
            .position = glm::vec4(cam_pos, 1.f),
            .color = {1.f, 1.f, 1.f, camera_light_intensity},
        };

        LightHeader light_header = {
            .camera_pos = glm::vec4(cam_pos, 1.f),
            .light_count = (uint32_t)lights.size() + 1, //camera light is not part of lights array
            .ambient = ambient_strength,
            // edited for "point-light shadows"
            .which_shadow = POINT_SHADOW_LIGHT,
            .shadow_far_plane = POINT_SHADOW_FAR,
            .sun_VP = sun_VP,
            .sun_direction = glm::vec4(sun_direction, 0.f),
            .sun_color = SUN_COLOR,
        };

        std::memcpy(frame_lights, &light_header, sizeof(LightHeader));
        std::memcpy(frame_lights + sizeof(LightHeader), lights.data(), sizeof(PointLight) * lights.size());
        std::memcpy(frame_lights + sizeof(LightHeader) + sizeof(PointLight) * lights.size(), &camera_light, sizeof(PointLight));

        //--------------------------------------------------------------------------------------//
        // SHADOW PASS
        //--------------------------------------------------------------------------------------//

        VkImageMemoryBarrier2 barrier_to_shadow_attachment = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext = NULL,

            .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_NONE,

            .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,

            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,

            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,

            .image = shadow_map[current_frame],
            .subresourceRange = shadow_subresource_range,
        };

        VkDependencyInfo dependency_info_to_shadow_attachment = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = NULL,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier_to_shadow_attachment,
        };

        vkCmdPipelineBarrier2(command_buffer, &dependency_info_to_shadow_attachment);

        VkRenderingAttachmentInfo shadow_attachment = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = NULL,
            .imageView = shadow_map_view[current_frame],
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = {.depthStencil = {.depth = 1.0f}},
        };

        VkRenderingInfo shadow_rendering_info = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = NULL,
            .flags = 0,
            .renderArea = { {0, 0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}},
            .layerCount = 1,
            .viewMask = 0,
            .colorAttachmentCount = 0,
            .pColorAttachments = nullptr,
            .pDepthAttachment = &shadow_attachment,
        };

        vkCmdBeginRendering(command_buffer, &shadow_rendering_info);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);
        vkCmdSetViewport(command_buffer, 0, 1, &shadow_viewport);
        vkCmdSetScissor(command_buffer, 0, 1, &shadow_scissor);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_layout, 0, 1, &graphics_descriptor_set[current_frame], 0, nullptr);

        GraphicsPushConstants shadow_push_constants = {.VP = sun_VP};

        vkCmdPushConstants(command_buffer, graphics_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GraphicsPushConstants), &shadow_push_constants);
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &model_vertex_buffer, &vert_offset);
        vkCmdBindIndexBuffer(command_buffer, model_index_buffer, 0, VK_INDEX_TYPE_UINT32);

        for (uint32_t i = 0; i < objects.size(); ++i) {
            const Mesh& mesh = meshes[objects[i].mesh];
            vkCmdDrawIndexed(command_buffer, mesh.index_count, 1, mesh.first_index, mesh.vertex_offset, i);
        }

        vkCmdEndRendering(command_buffer);

        VkImageMemoryBarrier2 barrier_to_shadow_read = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext = NULL,

            .srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,

            .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,

            .oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,

            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,

            .image = shadow_map[current_frame],
            .subresourceRange = shadow_subresource_range,
        };

        // Added for Point Light Shadow
        const glm::vec3 face_dir[6] = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}};
        const glm::vec3 face_up[6] = {{0,-1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}, {0,-1,0}, {0,-1,0}};
        glm::vec3 shadow_pos = glm::vec3(lights[POINT_SHADOW_LIGHT].position);
        glm::mat4 cube_proj = glm::perspectiveRH_ZO(glm::radians(90.f), 1.f, POINT_SHADOW_NEAR, POINT_SHADOW_FAR);   //note: NO y flip

        VkImageSubresourceRange cube_range = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 6};

        VkImageMemoryBarrier2 barrier_to_cube_attachment = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_NONE,
            .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = cube_image[current_frame],
            .subresourceRange = cube_range,
        };
        VkDependencyInfo cube_dep_in = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier_to_cube_attachment,
        };
        vkCmdPipelineBarrier2(command_buffer, &cube_dep_in);

        VkViewport cube_viewport = {0.f, 0.f, float(POINT_SHADOW_SIZE), float(POINT_SHADOW_SIZE), 0.f, 1.f};
        VkRect2D cube_scissor = {{0, 0}, {POINT_SHADOW_SIZE, POINT_SHADOW_SIZE}};

        for (uint32_t f = 0; f < 6; ++f) {
            VkRenderingAttachmentInfo face_attachment = {
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .imageView = cube_face_view[current_frame * 6 + f],
                .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue = {.depthStencil = {.depth = 1.0f}},
            };
            VkRenderingInfo face_rendering = {
                .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .renderArea = {{0, 0}, {POINT_SHADOW_SIZE, POINT_SHADOW_SIZE}},
                .layerCount = 1,
                .colorAttachmentCount = 0,
                .pDepthAttachment = &face_attachment,
            };

            vkCmdBeginRendering(command_buffer, &face_rendering);
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);
            vkCmdSetViewport(command_buffer, 0, 1, &cube_viewport);
            vkCmdSetScissor(command_buffer, 0, 1, &cube_scissor);
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_layout, 0, 1, &graphics_descriptor_set[current_frame], 0, nullptr);

            GraphicsPushConstants face_push = {.VP = cube_proj * glm::lookAtRH(shadow_pos, shadow_pos + face_dir[f], face_up[f])};
            vkCmdPushConstants(command_buffer, graphics_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GraphicsPushConstants), &face_push);
            vkCmdBindVertexBuffers(command_buffer, 0, 1, &model_vertex_buffer, &vert_offset);
            vkCmdBindIndexBuffer(command_buffer, model_index_buffer, 0, VK_INDEX_TYPE_UINT32);

            for (uint32_t i = 0; i < objects.size(); ++i) {
                if (i == 1 + POINT_SHADOW_LIGHT) continue;   //the light sits inside its own lamp mesh, which would block everything
                const Mesh& mesh = meshes[objects[i].mesh];
                vkCmdDrawIndexed(command_buffer, mesh.index_count, 1, mesh.first_index, mesh.vertex_offset, i);
            }
            vkCmdEndRendering(command_buffer);
        }

        VkImageMemoryBarrier2 barrier_to_cube_read = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = cube_image[current_frame],
            .subresourceRange = cube_range,
        };
        VkDependencyInfo cube_dep_out = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier_to_cube_read,
        };
        vkCmdPipelineBarrier2(command_buffer, &cube_dep_out);

        VkDependencyInfo dependency_info_to_shadow_read = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = NULL,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier_to_shadow_read,
        };

        vkCmdPipelineBarrier2(command_buffer, &dependency_info_to_shadow_read);

        //-------------------------------------------------------------------------------------//
        // PART F: COMPUTE DISPATCH
        //-------------------------------------------------------------------------------------//
        // Your particle compute shader must be dispatched before you render the particles. It
        // also must FINISH before rendering. After a synchronization barrier to block for the
        // previous frame's particle rendering, you have to bind the correct pipeline, descriptor
        // sets, and push constants before finally being able to dispatch. For reasons I have
        // discussed in past lectures, I recommend you keep your dispatch dimensions
        // one-dimensional, with your workgroup size (not dispatch count) being 64 particles.
        // Finally, you have to make sure they finish updating before going on to render them.
        //
        // Unlike other barriers which were VkImageMemoryBarrier2s, you will be utilizing
        // VkBufferMemoryBarrier2s since you are writing to buffers, not images. For both
        // barriers, you will be asked to justify your choice of srcStage/AccessMask and 
        // dstStage/AccessMask in the write-up. If your choice is wrong (or suboptimal), or your
        // explanation is insufficient, a point will be taken off.



        //-------------------------------------------------------------------------------------//
        // PART F: COMPUTE DISPATCH
        //-------------------------------------------------------------------------------------//

        VkBufferMemoryBarrier2 barrier_before_compute = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .pNext = NULL,
            .srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT,
            .srcAccessMask = VK_ACCESS_2_NONE,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = particle_buffer,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        };

        VkDependencyInfo dependency_info_before_compute = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = NULL,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &barrier_before_compute,
            .imageMemoryBarrierCount = 0,
            .pImageMemoryBarriers = nullptr,
        };

        vkCmdPipelineBarrier2(command_buffer, &dependency_info_before_compute);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_layout, 0, 1, &compute_descriptor_set, 0, nullptr);

        ComputePushConstants compute_push_constants = {
            .dt = dt,
            .time = elapsed,
            .particle_count = PARTICLE_COUNT,
            .floor_y = FLOOR_Y,
            .wind = WIND,
            .spawn_min = SPAWN_MIN,
            .spawn_max = SPAWN_MAX,
        };

        vkCmdPushConstants(command_buffer, compute_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstants), &compute_push_constants);
        vkCmdDispatch(command_buffer, (PARTICLE_COUNT + 63) / 64, 1, 1);

        VkBufferMemoryBarrier2 barrier_after_compute = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .pNext = NULL,
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT,
            .dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = particle_buffer,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        };

        VkDependencyInfo dependency_info_after_compute = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = NULL,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &barrier_after_compute,
            .imageMemoryBarrierCount = 0,
            .pImageMemoryBarriers = nullptr,
        };

        vkCmdPipelineBarrier2(command_buffer, &dependency_info_after_compute);
        
        //--------------------------------------------------------------------------------------//
        // BEGIN RENDER PASS
        //--------------------------------------------------------------------------------------//

        vkCmdBeginRendering(command_buffer, &rendering_info);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);
        vkCmdSetViewport(command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);

        //set 0 = models + lights, set 1 = materials
        VkDescriptorSet bound_descriptor_sets[2] = {graphics_descriptor_set[current_frame], material_descriptor_set};
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_layout, 0, 2, bound_descriptor_sets, 0, nullptr);
        GraphicsPushConstants graphics_push_constants = {.VP = VP};

        vkCmdPushConstants(command_buffer, graphics_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GraphicsPushConstants), &graphics_push_constants);
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &model_vertex_buffer, &vert_offset);
        vkCmdBindIndexBuffer(command_buffer, model_index_buffer, 0, VK_INDEX_TYPE_UINT32);

        //firstInstance = object index -> gl_InstanceIndex picks this object's matrix from the SSBO
        for (uint32_t i = 0; i < objects.size(); ++i) {
            const Mesh& mesh = meshes[objects[i].mesh];
            vkCmdDrawIndexed(command_buffer, mesh.index_count, 1, mesh.first_index, mesh.vertex_offset, i);
        }

        //draw vertices at light positions for worldspace visualization
        if (render_light_points) {
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, light_point_pipeline);

            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, light_point_pipeline_layout, 0, 1, &graphics_descriptor_set[current_frame], 0, nullptr);

            float light_point_scale = viewport.height * glm::abs(proj[1][1]) * 0.5f * LIGHT_POINT_SIZE;

            LightPointPushConstants light_point_push_constants = {.VP = VP, .point_scale = light_point_scale};

            vkCmdPushConstants(command_buffer, light_point_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(LightPointPushConstants), &light_point_push_constants);

            uint32_t light_point_count = (uint32_t)lights.size();

            //depth test for all lights
            vkCmdSetDepthTestEnable(command_buffer, VK_TRUE);
            vkCmdDraw(command_buffer, light_point_count, 1, 0, 0);

            //re-render selected light without depth test
            vkCmdSetDepthTestEnable(command_buffer, VK_FALSE);
            vkCmdDraw(command_buffer, 1, 1, selected_light, 0);
        }

        //-------------------------------------------------------------------------------------//
        // PART G: PARTICLE DRAW
        //-------------------------------------------------------------------------------------//
        // The process of drawing particles is very similar to that of other geometry. You must
        // bind descriptor sets, dynamic state, push constants, and your vertex buffer (no index
        // buffer exists), before drawing.
        //
        // Hint: Take a look at how the light point rendering gives those points consistent
        //       world-space sizing (in host code, light_point_scale), and reuse the logic for
        //       your own particles. Just because you may be reusing the graphics descriptor sets
        //       doesn't mean you should be reusing its pipeline layout.


        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, particle_pipeline);
        vkCmdSetViewport(command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, particle_pipeline_layout, 0, 1, &graphics_descriptor_set[current_frame], 0, nullptr);

        float particle_point_scale = viewport.height * glm::abs(proj[1][1]) * 0.5f * PARTICLE_SIZE;
        ParticlePushConstants particle_push_constants = {
            .vp = VP,
            .point_scale = particle_point_scale,
            .time = elapsed,
            .radiusX = 2.0f,
            .radiusZ = 2.0f,
        };

        vkCmdPushConstants(command_buffer, particle_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ParticlePushConstants), &particle_push_constants);
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &particle_buffer, &vert_offset);
        vkCmdDraw(command_buffer, PARTICLE_COUNT, 1, 0, 0);

        vkCmdEndRendering(command_buffer);

        //--------------------------------------------------------------------------------------//
        // TRANSITION TO BLIT
        //--------------------------------------------------------------------------------------//

        VkImageMemoryBarrier2 barrier_to_transfer_src = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext = NULL,

            .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,

            .dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,

            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,

            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,

            .image = frame[current_frame],
            .subresourceRange = subresource_range,
        };

        VkImageMemoryBarrier2 barrier_to_transfer_dst = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext = NULL,

            .srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT,
            .srcAccessMask = VK_ACCESS_2_NONE,

            .dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,

            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            
            .image = swapchain_images[image_index],
            .subresourceRange = subresource_range,
        };

        VkImageMemoryBarrier2 barriers_before_blit[2] = {barrier_to_transfer_src, barrier_to_transfer_dst};

        VkDependencyInfo dependency_info_before_blit = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = NULL,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = 2,
            .pImageMemoryBarriers = barriers_before_blit,
        };

        vkCmdPipelineBarrier2(command_buffer, &dependency_info_before_blit);

        //--------------------------------------------------------------------------------------//
        // BLIT TO SWAPCHAIN
        //--------------------------------------------------------------------------------------//

        vkCmdBlitImage(command_buffer, frame[current_frame], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchain_images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &image_blit, VK_FILTER_LINEAR);

        //--------------------------------------------------------------------------------------//
        // TRANSITION TO PRESENT
        //--------------------------------------------------------------------------------------//

        VkImageMemoryBarrier2 barrier_to_present = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext = NULL,

            .srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,

            .dstStageMask = VK_PIPELINE_STAGE_2_NONE,
            .dstAccessMask = VK_ACCESS_2_NONE,

            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,

            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,

            .image = swapchain_images[image_index],
            .subresourceRange = subresource_range,
        };

        VkDependencyInfo dependency_info_to_present = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = NULL,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier_to_present,
        };

        vkCmdPipelineBarrier2(command_buffer, &dependency_info_to_present);

        VK_CHECK(vkEndCommandBuffer(command_buffer));

        //--------------------------------------------------------------------------------------//
        // SUBMIT COMMAND BUFFER
        //--------------------------------------------------------------------------------------//

        VkSemaphoreSubmitInfo wait_semaphore_info = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext = NULL,
            .semaphore = image_available_semaphore,
            .value = 0,
            .stageMask = VK_PIPELINE_STAGE_2_BLIT_BIT,
            .deviceIndex = 0,
        };

        VkCommandBufferSubmitInfo command_buffer_submit_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .pNext = NULL,
            .commandBuffer = command_buffer,
            .deviceMask = 0,
        };

        VkSemaphoreSubmitInfo signal_semaphore_info = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext = NULL,
            .semaphore = render_finished_semaphore,
            .value = 0,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .deviceIndex = 0,
        };

        VkSubmitInfo2 submit_info = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .pNext = NULL,
            .flags = 0,
            .waitSemaphoreInfoCount = 1,
            .pWaitSemaphoreInfos = &wait_semaphore_info,
            .commandBufferInfoCount = 1,
            .pCommandBufferInfos = &command_buffer_submit_info,
            .signalSemaphoreInfoCount = 1,
            .pSignalSemaphoreInfos = &signal_semaphore_info,
        };

        VK_CHECK(vkQueueSubmit2(queue, 1, &submit_info, in_flight_fence));

        //--------------------------------------------------------------------------------------//
        // PRESENT
        //--------------------------------------------------------------------------------------//

        VkPresentInfoKHR present_info = {
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = NULL,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &render_finished_semaphore,
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = &image_index,
            .pResults = nullptr,
        };

        VkResult present_result = vkQueuePresentKHR(queue, &present_info);
        if (present_result != VK_SUCCESS && present_result != VK_SUBOPTIMAL_KHR) {
            std::fprintf(stderr, "vkQueuePresentKHR failed with VkResult %d\n", present_result);
            return EXIT_FAILURE;
        }

        //--------------------------------------------------------------------------------------//
        // CLOCK BOOKKEEPING
        //--------------------------------------------------------------------------------------//

        curr_time = clock_type::now();
        dt = glm::min(std::chrono::duration<float>(curr_time - last_frame_time).count(), 0.1f);
        elapsed = std::chrono::duration<float>(curr_time - start_time).count();
        if (fmod(elapsed, frame_cycle) < fmod(last_elapsed, frame_cycle)) {
            fps = (double)recent_frames / std::chrono::duration<float>(curr_time - last_cycle_time).count();
            if (print_fps) std::printf("Avg FPS: %i\n", int(fps));
            last_cycle_time = curr_time;
            recent_frames = 0;
        } else {
            recent_frames++;
        }
        last_frame_time = curr_time;
        last_elapsed = elapsed;
        current_frame = (current_frame + 1) % FRAMES_IN_FLIGHT;
        total_frames++;

    }

    //let's wait for everything to finish on the host side before cleaning up or finalizing timers
    VK_CHECK(vkDeviceWaitIdle(device));

    end_time = clock_type::now();
    double total_seconds = std::chrono::duration<double>(end_time - start_time).count();

    fps = (double)total_frames / total_seconds;
    std::printf("Avg FPS: %i\n", int(fps));

    //-------------------------------------------------------------------------------------//
    // PART H: CLEANUP
    //-------------------------------------------------------------------------------------//
    // Everything you created in Parts B through E needs destroying. Validation errors will
    // tell you what you missed when the window closes.

    //destruction generally goes in reverse order of creation

    vkDestroySampler(device, diffuse_sampler, nullptr);
    vkDestroyImageView(device, diffuse_texture_view, nullptr);
    vkDestroyImage(device, diffuse_texture, nullptr);
    vkFreeMemory(device, diffuse_texture_memory, nullptr);

    vkDestroyBuffer(device, material_buffer, nullptr);
    vkFreeMemory(device, material_buffer_memory, nullptr);

    // Part B buffer destruction
    vkDestroyBuffer(device, particle_buffer, nullptr);
    vkFreeMemory(device, particle_memory, nullptr);

    vkDestroyBuffer(device, model_index_buffer, nullptr);
    vkFreeMemory(device, model_index_buffer_memory, nullptr);
    vkDestroyBuffer(device, model_vertex_buffer, nullptr);
    vkFreeMemory(device, model_vertex_buffer_memory, nullptr);

    // Part C pool destruction
    vkDestroyDescriptorPool(device, compute_descriptor_pool, nullptr);
    vkDestroyDescriptorSetLayout(device, compute_descriptor_set_layout, nullptr);

    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        vkDestroyBuffer(device, light_buffers[i], nullptr);
        vkFreeMemory(device, light_buffer_memory[i], nullptr);
        vkDestroyBuffer(device, model_buffers[i], nullptr);
        vkFreeMemory(device, model_buffer_memory[i], nullptr);
    }
    vkDestroyDescriptorPool(device, graphics_descriptor_pool, nullptr);
    vkDestroyDescriptorSetLayout(device, material_descriptor_set_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, graphics_descriptor_set_layout, nullptr);

    for (uint32_t i = 0; i < swapchain_image_count; ++i) {
        vkDestroySemaphore(device, render_finished_semaphores[i], nullptr);
    }

    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        vkDestroySemaphore(device, image_available_semaphores[i], nullptr);
        vkDestroyFence(device, in_flight_fences[i], nullptr);
    }
    // Cleanup for Point Light Shadow
    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        vkDestroyImageView(device, cube_view[i], nullptr);
        vkDestroyImage(device, cube_image[i], nullptr);
    }
    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT * 6; ++i) {
        vkDestroyImageView(device, cube_face_view[i], nullptr);
    }
    vkFreeMemory(device, cube_memory, nullptr);

    vkDestroyPipeline(device, shadow_pipeline, nullptr);
    vkDestroyPipeline(device, light_point_pipeline, nullptr);
    vkDestroyPipeline(device, graphics_pipeline, nullptr);
    vkDestroyPipelineLayout(device, graphics_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, light_point_pipeline_layout, nullptr);
    vkDestroyPipeline(device, particle_pipeline, nullptr);
    vkDestroyPipeline(device, compute_pipeline, nullptr);
    vkDestroyPipelineLayout(device, particle_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, compute_pipeline_layout, nullptr);

    vkDestroyCommandPool(device, command_pool, nullptr);

    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        vkDestroyImageView(device, frame_view[i], nullptr);
        vkDestroyImage(device, frame[i], nullptr);
        vkDestroyImageView(device, depth_view[i], nullptr);
        vkDestroyImage(device, depth[i], nullptr);
    }

    vkFreeMemory(device, frame_memory, nullptr);
    vkFreeMemory(device, depth_memory, nullptr);

    vkDestroySampler(device, shadow_sampler, nullptr);
    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        vkDestroyImageView(device, shadow_map_view[i], nullptr);
        vkDestroyImage(device, shadow_map[i], nullptr);
    }
    vkFreeMemory(device, shadow_map_memory, nullptr);

    vkDestroySwapchainKHR(device, swapchain, nullptr);

    vkDestroyDevice(device, nullptr);

    vkDestroySurfaceKHR(instance, surface, nullptr);

    PFN_vkDestroyDebugUtilsMessengerEXT pfn_destroy_debug_utils_messengerEXT =
        (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkDestroyDebugUtilsMessengerEXT");
    pfn_destroy_debug_utils_messengerEXT(instance, debug_messenger, nullptr);

    vkDestroyInstance(instance, nullptr);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                                           VkDebugUtilsMessageTypeFlagsEXT message_types,
                                           const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
                                           void* user_data)
{
    (void)message_severity;
    (void)message_types;
    (void)user_data;

    std::fprintf(stderr, "[validation] %s\n", callback_data->pMessage);

    std::ofstream& log_file = ValidationLog();

    if (log_file.is_open()) {
        log_file << "[validation] " << callback_data->pMessage << "\n";
        log_file << std::flush;
    }

    return VK_FALSE;
}

static std::ofstream& ValidationLog()
{
    static std::ofstream log_file("validation.txt", std::ios::out | std::ios::trunc);
    return log_file;
}

static bool DeviceSupportsExtension(VkPhysicalDevice physical_device, const char* extension_name)
{
    uint32_t extension_count = 0;
    VK_CHECK(vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr));

    std::vector<VkExtensionProperties> extensions(extension_count);
    VK_CHECK(vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, extensions.data()));

    for (uint32_t i = 0; i < extension_count; ++i) {
        if (std::strcmp(extensions[i].extensionName, extension_name) == 0) {
            return true;
        }
    }

    return false;
}

static bool InstanceSupportsExtension(const char* extension_name)
{
    uint32_t extension_count = 0;
    VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr));

    std::vector<VkExtensionProperties> extensions(extension_count);
    VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extensions.data()));

    for (uint32_t i = 0; i < extension_count; ++i) {
        if (std::strcmp(extensions[i].extensionName, extension_name) == 0) {
            return true;
        }
    }

    return false;
}

static bool BitIsSet(uint32_t mask, uint32_t bit_index)
{
    return (mask & (1u << bit_index)) != 0;
}

static uint32_t FindMemoryType(const VkPhysicalDeviceMemoryProperties& memory_properties, uint32_t type_bits, VkMemoryPropertyFlags property_flags)
{
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
        bool type_allowed = BitIsSet(type_bits, i);
        bool has_properties = (memory_properties.memoryTypes[i].propertyFlags & property_flags) == property_flags;

        if (type_allowed && has_properties) {
            return i;
        }
    }

    return UINT32_MAX;
}

static std::vector<uint32_t> ReadSpirv(const char* path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);

    if (!file.is_open()) {
        std::fprintf(stderr, "failed to open shader file: %s\n", std::filesystem::absolute(path).string().c_str());
        std::abort();
    }

    size_t size_in_bytes = (size_t)file.tellg();
    std::vector<uint32_t> code(size_in_bytes / sizeof(uint32_t));

    file.seekg(0);
    file.read((char*)code.data(), size_in_bytes);

    return code;
}

int WindowCreation(const char* window_title, GLFWwindow*& window)
{
    //initialize glfw
    if (!glfwInit()) {
        std::fprintf(stderr, "glfwInit failed\n");
        return EXIT_FAILURE; //EXIT_FAILURE resolves to 1
    }

    //disables OpenGL context creation
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, window_title, nullptr, nullptr);

    //if there's a second monitor, center the window on it. with one monitor this does nothing
    int monitor_count = 0;
    GLFWmonitor** monitors = glfwGetMonitors(&monitor_count);
    if (monitor_count > 1 && PREFER_SECOND_MONITOR) {
        int work_x = 0, work_y = 0, work_width = 0, work_height = 0;
        glfwGetMonitorWorkarea(monitors[1], &work_x, &work_y, &work_width, &work_height);
        glfwSetWindowPos(window, work_x + (work_width - (int)WINDOW_WIDTH) / 2, work_y + (work_height - (int)WINDOW_HEIGHT) / 2);
    }

    if (!window) {
        std::fprintf(stderr, "glfwCreateWindow failed\n");
        return EXIT_FAILURE;
    }

    //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (glfwRawMouseMotionSupported()) glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);

    glfwSetScrollCallback(window, scroll_callback);

    ValidationLog() << "=== VALIDATION LOG: " << window_title << " ===\n" << std::flush;
    std::fprintf(stderr, "[info] validation log location: %s\n", std::filesystem::absolute("validation.txt").string().c_str());

    return 0;
}

int InstanceCreation(const char* window_title, VkInstance& instance, VkDebugUtilsMessengerEXT& debug_messenger)
{
    //glfw needs certain vulkan instance extensions to be enabled to perform core functions like creating a window surface. we query for them here
    uint32_t glfw_extension_count = 0;
    const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
    if (!glfw_extensions) {
        std::fprintf(stderr, "glfwGetRequiredInstanceExtensions failed (Vulkan not supported by GLFW on this platform)\n");
        return EXIT_FAILURE;
    }

    std::vector<const char*> instance_extensions(glfw_extensions, glfw_extensions + glfw_extension_count);

    //let's add the debug extension so that our debug messanger works
    instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    //enable validation layers that create debug messages
    const char* enabled_layers[] = { "VK_LAYER_KHRONOS_validation" };

    //describe our debug messanger
    VkDebugUtilsMessengerCreateInfoEXT debug_messenger_create_info = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = NULL,
        .flags = 0,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debug_callback,
        .pUserData = nullptr,
    };

    //let's enable synchronization validation to make sure our stage mask targets are correct
    instance_extensions.push_back(VK_EXT_LAYER_SETTINGS_EXTENSION_NAME);

    VkBool32 validate_sync_enabled = VK_TRUE;

    VkLayerSettingEXT layer_settings[] = {
        {
            .pLayerName = "VK_LAYER_KHRONOS_validation",
            .pSettingName = "validate_sync",
            .type = VK_LAYER_SETTING_TYPE_BOOL32_EXT,
            .valueCount = 1,
            .pValues = &validate_sync_enabled,
        },
    };

    VkLayerSettingsCreateInfoEXT layer_settings_create_info = {
        .sType = VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT,
        .pNext = &debug_messenger_create_info,
        .settingCount = sizeof(layer_settings) / sizeof(layer_settings[0]),
        .pSettings = layer_settings,
    };

    VkInstanceCreateFlags instance_create_flags = 0;
    //mac support
    if (InstanceSupportsExtension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        instance_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        instance_create_flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }

    //describe our application
    VkApplicationInfo application_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = NULL,
        .pApplicationName = window_title,
        .applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
        .pEngineName = "none",
        .engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),

        .apiVersion = VK_API_VERSION_1_3,
    };

    //describe our Vulkan instance
    VkInstanceCreateInfo instance_create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = &layer_settings_create_info,
        .flags = instance_create_flags,
        .pApplicationInfo = &application_info,
        .enabledLayerCount = sizeof(enabled_layers) / sizeof(enabled_layers[0]),
        .ppEnabledLayerNames = enabled_layers,
        .enabledExtensionCount = (uint32_t)instance_extensions.size(),
        .ppEnabledExtensionNames = instance_extensions.data(),
    };

    //create our Vulkan instance
    VkResult instance_creation_result = vkCreateInstance(&instance_create_info, nullptr, &instance);
    VK_CHECK(instance_creation_result);

    //since extension functions are not loaded in by default (after all, they are extensions), we have to query the function as a pointer in order to use it
    PFN_vkCreateDebugUtilsMessengerEXT pfn_create_debug_utils_messengerEXT =
        (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkCreateDebugUtilsMessengerEXT");

    //establish the debug messanger callback
    VK_CHECK(pfn_create_debug_utils_messengerEXT(instance, &debug_messenger_create_info, nullptr, &debug_messenger));

    return 0;
}

int SurfaceCreation(const VkInstance& instance, GLFWwindow*& window, VkSurfaceKHR& surface)
{
    VK_CHECK(glfwCreateWindowSurface(instance, window, nullptr, &surface));

    return 0;
}

int PhysicalDeviceSelection(const VkInstance& instance, const VkSurfaceKHR& surface, VkPhysicalDevice& physical_device, uint32_t& queue_family_index)
{
    //simply check the number of available physical devices (GPUs), don't get their data yet
    uint32_t physical_device_count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &physical_device_count, nullptr));
    if (physical_device_count == 0) {
        std::fprintf(stderr, "vkEnumeratePhysicalDevices failed (no Vulkan-capable devices found)\n");
        return EXIT_FAILURE;
    }

    //properly get the physical device data, look through them and find the best* one
    //"best" meaning preferably a dedicated GPU (as opposed to an integrated one on the CPU), and minimally must support version 1.3 of Vulkan 
    std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &physical_device_count, physical_devices.data()));

    for (uint32_t i = 0; i < physical_device_count; ++i) {
        VkPhysicalDeviceProperties physical_device_properties = {};
        vkGetPhysicalDeviceProperties(physical_devices[i], &physical_device_properties);

        if (physical_device_properties.apiVersion < VK_API_VERSION_1_3) {
            continue;
        }

        //devices have one or more queue families that support different pipelines
        //to keep it simple, let's grab a family that supports both the graphics pipeline and glfw surfaces 
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[i], &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[i], &queue_family_count, queue_families.data());

        uint32_t device_queue_family_index = UINT32_MAX;

        for (uint32_t j = 0; j < queue_family_count; ++j) {
            if (!(queue_families[j].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                continue;
            }

            VkBool32 present_surface_support = VK_FALSE;
            VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(physical_devices[i], j, surface, &present_surface_support));

            if (present_surface_support == VK_TRUE) {
                device_queue_family_index = j;
                break;
            }
        }

        if (device_queue_family_index == UINT32_MAX) {
            continue;
        }

        if (physical_device != VK_NULL_HANDLE && physical_device_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU && PREFERRED_DEVICE_TYPE != VK_PHYSICAL_DEVICE_TYPE_CPU) {
            continue;
        }

        physical_device = physical_devices[i];
        queue_family_index = device_queue_family_index;

        if (physical_device_properties.deviceType == PREFERRED_DEVICE_TYPE) {
            break;
        }
    }

    if (physical_device == VK_NULL_HANDLE) {
        std::fprintf(stderr, "No device supporting Vulkan 1.3 + present\n");
        return EXIT_FAILURE;
    }

    VkPhysicalDeviceProperties physical_device_properties = {};
    vkGetPhysicalDeviceProperties(physical_device, &physical_device_properties);

    if (physical_device_properties.deviceType != PREFERRED_DEVICE_TYPE) {
        std::printf("[info] WARNING: failed to get preferred device type (%s), took %s instead.\n", string_VkPhysicalDeviceType(PREFERRED_DEVICE_TYPE), string_VkPhysicalDeviceType(physical_device_properties.deviceType));
    }

    return 0;
}

int DeviceAndQueueCreation(const uint32_t& queue_family_index, const VkPhysicalDevice& physical_device, VkDevice& device, VkQueue& queue)
{
    //queue priority is useful for > 1 queue but still necessary for == 1, ranges from 0.0-1.0
    //(it's used for prioritizing queues when scheduling)
    float queue_priority = 1.f;

    //describe our queue
    VkDeviceQueueCreateInfo device_queue_create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .queueFamilyIndex = queue_family_index,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };

    VkPhysicalDeviceFeatures supported{};
    vkGetPhysicalDeviceFeatures(physical_device, &supported);
    if (!supported.fillModeNonSolid) {
        std::fprintf(stderr, "fillModeNonSolid not supported.\n");
        return EXIT_FAILURE;
    }
    if (!supported.largePoints) {
        std::fprintf(stderr, "largePoints not supported.\n");
        return EXIT_FAILURE;
    }

    //note that simply being on vulkan 1.3 doesn't automatically turn 1.3 features on.
    //they default to VK_FALSE, so we enable them here and pass it to the logical device creation info
    VkPhysicalDeviceVulkan13Features vulkan_1_3_features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = NULL,
        .shaderDemoteToHelperInvocation = VK_TRUE,
        .synchronization2 = VK_TRUE,
        .dynamicRendering = VK_TRUE,
    };

    VkPhysicalDeviceFeatures2 features_2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &vulkan_1_3_features,
        .features = {
            .fillModeNonSolid = VK_TRUE,
            .largePoints = VK_TRUE,
        },
    };

    //presenting a surface (and by extension, using a swapchain) is not core Vulkan but rather a device extension
    std::vector<const char*> device_extensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    //mac support
    if (DeviceSupportsExtension(physical_device, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
        device_extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    }

    //describe our logical device that serves as the interface between our Vulkan instance and our physical device (GPU)
    VkDeviceCreateInfo device_create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &features_2,
        .flags = 0,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &device_queue_create_info,
        .enabledExtensionCount = (uint32_t)device_extensions.size(),
        .ppEnabledExtensionNames = device_extensions.data(),
    };

    //create our logical device and its queues 
    VK_CHECK(vkCreateDevice(physical_device, &device_create_info, nullptr, &device));

    //retrieve the handle of the queue generated with the logical device
    uint32_t queue_index = 0;
    vkGetDeviceQueue(device, queue_family_index, queue_index, &queue);

    return 0;
}

int SwapchainCreation(const VkPhysicalDevice& physical_device, const VkSurfaceKHR& surface, GLFWwindow*& window, const VkDevice& device, VkSurfaceFormatKHR& surface_format, VkExtent2D& swapchain_extent, VkSwapchainKHR& swapchain, uint32_t& swapchain_image_count, std::vector<VkImage>& swapchain_images)
{
    //let's check if our physical device supports the transfer op we're going to use later
    //(don't worry, it practically always is)
    VkSurfaceCapabilitiesKHR surface_capabilities = {};
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &surface_capabilities));

    //VK_IMAGE_USAGE_TRANSFER_DST_BIT is the flag that signifies the image can be used as the destination for a transfer op
    if (!(surface_capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        std::fprintf(stderr, "Surface does not support VK_IMAGE_USAGE_TRANSFER_DST_BIT\n");
        return EXIT_FAILURE;
    }

    //query the number of supported surface (image) formats first before fetching them
    uint32_t surface_format_count = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &surface_format_count, nullptr));

    //...then fetch them
    std::vector<VkSurfaceFormatKHR> surface_formats(surface_format_count);
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &surface_format_count, surface_formats.data()));

    //for our purposes, this is the format and color space we want. in case it isn't there, we grab the first one
    //at least one format is guaranteed to exist
    const VkFormat preferred_swapchain_formats[] = {VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_R8G8B8A8_SRGB};
    bool found_srgb_format = false;

    surface_format = surface_formats[0];
    for (VkFormat preferred : preferred_swapchain_formats) {
        for (uint32_t i = 0; i < surface_format_count; ++i) {
            if (surface_formats[i].format == preferred &&
                surface_formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                surface_format = surface_formats[i];
                found_srgb_format = true;
                break;
            }
        }
        if (found_srgb_format) break;
    }

    if (!found_srgb_format) {
        std::fprintf(stderr, "Warning: no _SRGB swapchain format offered, falling back to %s (output will look too dark)\n", string_VkFormat(surface_format.format));
    }
    std::printf("[info] swapchain format: %s\n", string_VkFormat(surface_format.format));

    //device will support up to a specific surface size, or any size (UINT32_MAX).
    //since we presumably get to choose, let's just use the framebuffer dimensions
    swapchain_extent = surface_capabilities.currentExtent;
    if (swapchain_extent.width == UINT32_MAX) {
        int framebuffer_width = 0;
        int framebuffer_height = 0;
        glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
        swapchain_extent.width = (uint32_t)framebuffer_width;
        swapchain_extent.height = (uint32_t)framebuffer_height;
    }

    //minImageCount can be as low as 1...
    //let's make sure there are at least two images to use in our swapchain to "swap" betweeen
    uint32_t swapchain_min_image_count = surface_capabilities.minImageCount + 1;
    if (surface_capabilities.maxImageCount > 0 && swapchain_min_image_count > surface_capabilities.maxImageCount) {
        swapchain_min_image_count = surface_capabilities.maxImageCount;
    }

    uint32_t present_mode_count = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_mode_count, nullptr));

    //...then fetch them
    std::vector<VkPresentModeKHR> present_modes(present_mode_count);
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_mode_count, present_modes.data()));
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    for (uint32_t i = 0; i < present_mode_count; ++i) {
        if (present_modes[i] == PREFERRED_PRESENT_MODE) {
            present_mode = present_modes[i];
            break;
        }
    }
    if (present_mode != PREFERRED_PRESENT_MODE) {
        std::printf("[info] WARNING: preferred present mode (%s) not supported, defaulting to VK_PRESENT_MODE_FIFO_KHR.\n", string_VkPresentModeKHR(PREFERRED_PRESENT_MODE));
    }

    //describe our swapchain
    VkSwapchainCreateInfoKHR swapchain_create_info = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .surface = surface,
        .minImageCount = swapchain_min_image_count,
        .imageFormat = surface_format.format,
        .imageColorSpace = surface_format.colorSpace,
        .imageExtent = swapchain_extent,
        //more than 1 only used for stereoscopic rendering
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .preTransform = surface_capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        //FIFO is guaranteed to be supported
        .presentMode = present_mode, 
        .clipped = VK_TRUE,
        .oldSwapchain = VK_NULL_HANDLE,
    };

    //create our swapchain
    VK_CHECK(vkCreateSwapchainKHR(device, &swapchain_create_info, nullptr, &swapchain));

    //query number of images in the swapchain we just made
    VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain, &swapchain_image_count, nullptr));
    swapchain_images.resize(swapchain_image_count);
    
    //grab the images the swapchain made - we do not generate them ourselves
    VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain, &swapchain_image_count, swapchain_images.data()));

    return 0;
}

int CommandBufferCreation(const uint32_t& queue_family_index, const VkDevice& device, VkCommandPool& command_pool, std::vector<VkCommandBuffer>& command_buffers)
{
    //describe our command pool
    VkCommandPoolCreateInfo command_pool_create_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = NULL,
        //this flag enables vkResetCommandBuffer which we need because we are re-recording the command buffers on each frame
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        //a command pool is bound to (& effectively owned by) one queue family.
        //its buffers can only be submitted to queues in that family
        .queueFamilyIndex = queue_family_index,
    };

    //create our command pool
    VK_CHECK(vkCreateCommandPool(device, &command_pool_create_info, nullptr, &command_pool));

    //describe the buffer(s) we are going to create
    VkCommandBufferAllocateInfo command_buffer_allocate_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = NULL,
        .commandPool = command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = FRAMES_IN_FLIGHT,
    };

    //allocate (create) command buffers for the pool we've made
    VK_CHECK(vkAllocateCommandBuffers(device, &command_buffer_allocate_info, command_buffers.data()));

    return 0;
}

int FrameCreation(const VkDevice& device, const VkPhysicalDevice& physical_device, std::vector<VkImage>& frame, VkPhysicalDeviceMemoryProperties& memory_properties, VkDeviceMemory& frame_memory, std::vector<VkImageView>& frame_view, std::vector<VkImage>& depth, VkDeviceMemory& depth_memory, std::vector<VkImageView>& depth_view)
{
    VkImageCreateInfo image_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = COLOR_FORMAT,
        .extent = {
            .width = FRAME_WIDTH,
            .height = FRAME_HEIGHT,
            .depth = 1,
        },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, //SRC -> blit, COLOR_ATTACHMENT -> render target
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    
    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        VK_CHECK(vkCreateImage(device, &image_info, nullptr, &frame[i])); //creates the VkImage handle (equivalent to VkBuffer)
    }

    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

    VkMemoryRequirements image_memory_requirements = {};
    vkGetImageMemoryRequirements(device, frame[0], &image_memory_requirements);
    VkMemoryPropertyFlags property_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    uint32_t memory_type_index = FindMemoryType(memory_properties, image_memory_requirements.memoryTypeBits, property_flags);

    if (memory_type_index == UINT32_MAX) {
        std::fprintf(stderr, "No device-local memory type available\n");
        return EXIT_FAILURE;
    }

    VkDeviceSize stride = ((image_memory_requirements.size + image_memory_requirements.alignment - 1) / image_memory_requirements.alignment) * image_memory_requirements.alignment;

    VkDeviceSize frame_size = stride * FRAMES_IN_FLIGHT;

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = NULL,
        .allocationSize = frame_size,
        .memoryTypeIndex = memory_type_index,
    };

    VK_CHECK(vkAllocateMemory(device, &alloc_info, nullptr, &frame_memory));

    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        VkDeviceSize bind_offset = i * stride;
        VK_CHECK(vkBindImageMemory(device, frame[i], frame_memory, bind_offset)); //binds VkImage handle to VkMemory block

        //to create the image view...
        VkImageViewCreateInfo view_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = frame[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = COLOR_FORMAT,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        VK_CHECK(vkCreateImageView(device, &view_info, nullptr, &frame_view[i])); //creates the VkImageView handle
    }

    VkImageCreateInfo depth_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = DEPTH_FORMAT,
        .extent = {
            .width = FRAME_WIDTH,
            .height = FRAME_HEIGHT,
            .depth = 1,
        },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    
    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        VK_CHECK(vkCreateImage(device, &depth_info, nullptr, &depth[i]));
    }

    image_memory_requirements = {};
    vkGetImageMemoryRequirements(device, depth[0], &image_memory_requirements);
    property_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    memory_type_index = FindMemoryType(memory_properties, image_memory_requirements.memoryTypeBits, property_flags);

    if (memory_type_index == UINT32_MAX) {
        std::fprintf(stderr, "No device-local memory type available\n");
        return EXIT_FAILURE;
    }

    stride = ((image_memory_requirements.size + image_memory_requirements.alignment - 1) / image_memory_requirements.alignment) * image_memory_requirements.alignment;

    VkDeviceSize depth_size = stride * FRAMES_IN_FLIGHT;

    alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = NULL,
        .allocationSize = depth_size,
        .memoryTypeIndex = memory_type_index,
    };

    VK_CHECK(vkAllocateMemory(device, &alloc_info, nullptr, &depth_memory));

    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        VkDeviceSize bind_offset = i * stride;
        VK_CHECK(vkBindImageMemory(device, depth[i], depth_memory, bind_offset)); //binds VkImage handle to VkMemory block

        //to create the image view...
        VkImageViewCreateInfo view_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = depth[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = DEPTH_FORMAT,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        VK_CHECK(vkCreateImageView(device, &view_info, nullptr, &depth_view[i])); //creates the VkImageView handle
    }

    return 0;
}

int GraphicsDescriptorSetCreation(const VkDevice& device, const VkPhysicalDeviceMemoryProperties& memory_properties, std::vector<VkBuffer>& model_buffers, std::vector<VkDeviceMemory>& model_buffer_memory, std::vector<glm::mat4*>& model_matrices, std::vector<VkBuffer>& light_buffers, std::vector<VkDeviceMemory>& light_buffer_memory, std::vector<void*>& light_data, VkDescriptorPool& graphics_descriptor_pool, VkDescriptorSetLayout& graphics_descriptor_set_layout, std::vector<VkDescriptorSet>& graphics_descriptor_set, VkDescriptorSetLayout& material_descriptor_set_layout, VkDescriptorSet& material_descriptor_set)
{
    VkDeviceSize model_buffer_size = sizeof(glm::mat4) * MAX_OBJECTS;
    VkDeviceSize light_buffer_size = sizeof(LightHeader) + sizeof(PointLight) * MAX_LIGHTS;

    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        void* mapped = nullptr;
        int hvb = CreateHostVisibleBuffer(device, memory_properties, model_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, USE_BAR_MODEL_MATRICES, model_buffers[i], model_buffer_memory[i], mapped);
        if (hvb != 0) return hvb;
        model_matrices[i] = (glm::mat4*)mapped;

        //one per FIF since updated every frame
        int lhvb = CreateHostVisibleBuffer(device, memory_properties, light_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, USE_BAR_MODEL_MATRICES, light_buffers[i], light_buffer_memory[i], light_data[i]);
        if (lhvb != 0) return lhvb;
    }

    VkDescriptorPoolSize descriptor_pool_sizes[2] = {
        {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = FRAMES_IN_FLIGHT * 2 + 1, //set 0 needs FIF * 2, set 1 needs 1
        },
        {
            // .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            // .descriptorCount = FRAMES_IN_FLIGHT + 1,
            // Updated for point light shadows
            .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = FRAMES_IN_FLIGHT * 2 + 1
        },
    };

    VkDescriptorPoolCreateInfo descriptor_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .maxSets = FRAMES_IN_FLIGHT + 1,
        .poolSizeCount = 2,
        .pPoolSizes = descriptor_pool_sizes,
    };

    VK_CHECK(vkCreateDescriptorPool(device, &descriptor_pool_info, nullptr, &graphics_descriptor_pool));

    //set 0: per-frame data (model matrices & lights)
    VkDescriptorSetLayoutBinding descriptor_set_layout_bindings[4] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        // Added for point light shadows
        {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
    };

    VkDescriptorSetLayoutCreateInfo descriptor_set_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .bindingCount = 4,
        .pBindings = descriptor_set_layout_bindings,
    };

    VK_CHECK(vkCreateDescriptorSetLayout(device, &descriptor_set_layout_info, nullptr, &graphics_descriptor_set_layout));

    //set 1: static data (materials)
    VkDescriptorSetLayoutBinding material_set_layout_bindings[2] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
    };

    VkDescriptorSetLayoutCreateInfo material_set_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .bindingCount = 2,
        .pBindings = material_set_layout_bindings,
    };

    VK_CHECK(vkCreateDescriptorSetLayout(device, &material_set_layout_info, nullptr, &material_descriptor_set_layout));

    VkDescriptorSetAllocateInfo descriptor_set_allocate_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = NULL,
        .descriptorPool = graphics_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &graphics_descriptor_set_layout,
    };

    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        VK_CHECK(vkAllocateDescriptorSets(device, &descriptor_set_allocate_info, &graphics_descriptor_set[i]));

        //the set points AT the buffer; we write the set once, then change the buffer's contents every frame
        VkDescriptorBufferInfo model_buffer_info = {
            .buffer = model_buffers[i],
            .offset = 0,
            .range = VK_WHOLE_SIZE,
        };

        VkDescriptorBufferInfo light_buffer_info = {
            .buffer = light_buffers[i],
            .offset = 0,
            .range = VK_WHOLE_SIZE,
        };

        VkWriteDescriptorSet write_descriptor_sets[2] = {
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = NULL,
                .dstSet = graphics_descriptor_set[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &model_buffer_info,
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = NULL,
                .dstSet = graphics_descriptor_set[i],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &light_buffer_info,
            },
        };

        vkUpdateDescriptorSets(device, 2, write_descriptor_sets, 0, nullptr);
    }

    //allocated now so the pipeline layout can be built, but written after mesh import once the material buffer exists
    VkDescriptorSetAllocateInfo material_set_allocate_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = NULL,
        .descriptorPool = graphics_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &material_descriptor_set_layout,
    };

    VK_CHECK(vkAllocateDescriptorSets(device, &material_set_allocate_info, &material_descriptor_set));

    return 0;
}


int ComputeDescriptorSetCreation(const VkDevice& device, const VkBuffer& particle_buffer, VkDescriptorPool& descriptor_pool, VkDescriptorSetLayout& descriptor_set_layout, VkDescriptorSet& descriptor_set)
{
    //-------------------------------------------------------------------------------------//
    // PART C: COMPUTE DESCRIPTOR SET
    //-------------------------------------------------------------------------------------//
    // At least to start, you should only be worried about passing in one piece of data into
    // your compute shader via descriptors, that being the particle buffer itself. You should
    // be familiar with this process by now. Whatever binding number you choose for this
    // descriptor set, your particle comp shader has to match it. Since our descriptors are
    // just pointers to (in this case) VkBuffers, you don't have to worry about constantly
    // updating them to refresh the particle data. You only need to link the buffer to the
    // descriptor once at set creation. Also, since these functions lie outside main, the
    // function parameters include references to non-const variables to write to. This goes
    // for the other functions you will write, as well. Pay attention to what exits the
    // functions.

    // layout info
    VkDescriptorSetLayoutBinding binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    };

    VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .bindingCount = 1,
        .pBindings = &binding,
    };

    VK_CHECK(vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptor_set_layout));

    // pool info
    VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
    };

    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };

    VK_CHECK(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool));

    // allocate from pool
    VkDescriptorSetAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = NULL,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptor_set_layout,
    };

    VK_CHECK(vkAllocateDescriptorSets(device, &allocate_info, &descriptor_set));

    // point to particle
    VkDescriptorBufferInfo buffer_info = {
        .buffer = particle_buffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };

    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = NULL,
        .dstSet = descriptor_set,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buffer_info,
    };

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);



    return 0;
}

int ComputePipelineCreation(const VkDevice& device, const VkDescriptorSetLayout& descriptor_set_layout, VkPipelineLayout& compute_pipeline_layout, VkPipeline& compute_pipeline)
{
    //-------------------------------------------------------------------------------------//
    // PART D: COMPUTE PIPELINE
    //-------------------------------------------------------------------------------------//
    // Just like Assignment3, you have to create a compute pipeline in order to dispatch the
    // work. Your push constants should go in the new ComputePushConstants struct, and you
    // should use sizeof() the struct for the push constant range. At minimum, you should be
    // passing in delta time to keep your simulation frame-rate-independent.

    std::vector<uint32_t> compute_code = ReadSpirv(COMPUTE_SHADER_PATH);

    VkShaderModuleCreateInfo shader_module_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .codeSize = compute_code.size() * sizeof(uint32_t),
        .pCode = compute_code.data(),
    };

    VkShaderModule compute_shader_module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &shader_module_info, nullptr, &compute_shader_module));

    // push constants
    VkPushConstantRange push_constant_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(ComputePushConstants),
    };

    // layout
    VkPipelineLayoutCreateInfo pipeline_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant_range,
    };

    VK_CHECK(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &compute_pipeline_layout));

    //pipeline
    VkPipelineShaderStageCreateInfo stage_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = compute_shader_module,
        .pName = "main",
        .pSpecializationInfo = NULL,
    };

    VkComputePipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .stage = stage_info,
        .layout = compute_pipeline_layout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };

    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &compute_pipeline));

    vkDestroyShaderModule(device, compute_shader_module, nullptr);

    return 0;
}

int ParticlePipelineCreation(const VkDevice& device, const VkDescriptorSetLayout& graphics_descriptor_set_layout, VkPipelineLayout& particle_pipeline_layout, VkPipeline& particle_pipeline)
{
    //-------------------------------------------------------------------------------------//
    // PART E: PARTICLE PIPELINE
    //-------------------------------------------------------------------------------------//
    // To start, particles are drawn as points straight from the particle buffer, with no
    // indexing. LightPointPipelineCreation is similar in this regard, and since particles
    // are input as vertices, GraphicsPipelineCreation can be used as reference. You need
    // pieces of both, so copy with intent. You may find yourself returning to this function
    // later on to add support for features like blending.
    //
    // Hint:    Since particles need to see lights, it may be smart to simply reuse
    //          graphics_descriptor_set_layout in your pipeline layout, and only use the
    //          bindings relevant for this purpose.

    VkPipelineRenderingCreateInfo pipeline_rendering_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .pNext = NULL,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &COLOR_FORMAT,
        .depthAttachmentFormat = DEPTH_FORMAT,
    };

    // shaders
    std::vector<uint32_t> fragment_code = ReadSpirv(PARTICLE_FRAGMENT_SHADER_PATH);
    VkShaderModuleCreateInfo fragment_shader_module_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .codeSize = fragment_code.size() * sizeof(uint32_t),
        .pCode = fragment_code.data(),
    };
    VkShaderModule fragment_shader_module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &fragment_shader_module_info, nullptr, &fragment_shader_module));

    std::vector<uint32_t> vertex_code = ReadSpirv(PARTICLE_VERTEX_SHADER_PATH);
    VkShaderModuleCreateInfo vertex_shader_module_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .codeSize = vertex_code.size() * sizeof(uint32_t),
        .pCode = vertex_code.data(),
    };
    VkShaderModule vertex_shader_module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &vertex_shader_module_info, nullptr, &vertex_shader_module));

    VkPipelineShaderStageCreateInfo shader_stage_infos[2] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertex_shader_module,
            .pName = "main",
            .pSpecializationInfo = NULL,
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragment_shader_module,
            .pName = "main",
            .pSpecializationInfo = NULL,
        },
    };

    // the particle buffer, one per vertex
    VkVertexInputBindingDescription vertex_input_binding = {
        .binding = 0,
        .stride = sizeof(Particle),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };

    VkVertexInputAttributeDescription vertex_input_attributes[1] = {
        {
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Particle, position),
        },
    };

    VkPipelineVertexInputStateCreateInfo vertex_input_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &vertex_input_binding,
        .vertexAttributeDescriptionCount = 1,
        .pVertexAttributeDescriptions = vertex_input_attributes,
    };

    VkPipelineInputAssemblyStateCreateInfo input_assembly_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
    };

    VkPipelineViewportStateCreateInfo viewport_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .viewportCount = 1,
        .pViewports = VK_NULL_HANDLE,
        .scissorCount = 1,
        .pScissors = VK_NULL_HANDLE,
    };

    VkPipelineRasterizationStateCreateInfo rasterization_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.f,
    };

    VkPipelineMultisampleStateCreateInfo multisample_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    // based on slide 22
    VkPipelineDepthStencilStateCreateInfo depth_stencil_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = NULL,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_FALSE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };

    VkPipelineColorBlendAttachmentState color_blend_attachment_state_info = {
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    VkPipelineColorBlendStateCreateInfo color_blend_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment_state_info,
    };

    VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

    VkPipelineDynamicStateCreateInfo pipeline_dynamic_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .dynamicStateCount = sizeof(dynamic_states)/sizeof(dynamic_states[0]),
        .pDynamicStates = dynamic_states,
    };

    VkPushConstantRange particle_push_constant_range = {
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(ParticlePushConstants),
    };

    VkPipelineLayoutCreateInfo particle_pipeline_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &graphics_descriptor_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &particle_push_constant_range,
    };

    VK_CHECK(vkCreatePipelineLayout(device, &particle_pipeline_layout_info, nullptr, &particle_pipeline_layout));

    VkGraphicsPipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &pipeline_rendering_info,
        .flags = 0,
        .stageCount = 2,
        .pStages = shader_stage_infos,
        .pVertexInputState = &vertex_input_state_info,
        .pInputAssemblyState = &input_assembly_state_info,
        .pTessellationState = VK_NULL_HANDLE,
        .pViewportState = &viewport_state_info,
        .pRasterizationState = &rasterization_state_info,
        .pMultisampleState = &multisample_state_info,
        .pDepthStencilState = &depth_stencil_state_info,
        .pColorBlendState = &color_blend_state_info,
        .pDynamicState = &pipeline_dynamic_state_info,
        .layout = particle_pipeline_layout,
        .renderPass = VK_NULL_HANDLE,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };

    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &particle_pipeline));

    vkDestroyShaderModule(device, fragment_shader_module, nullptr);
    vkDestroyShaderModule(device, vertex_shader_module, nullptr);

    return 0;
}

int GraphicsPipelineCreation(const VkDevice& device, const VkDescriptorSetLayout& graphics_descriptor_set_layout, const VkDescriptorSetLayout& material_descriptor_set_layout, VkPipelineLayout& graphics_pipeline_layout, VkPipeline& graphics_pipeline)
{
    VkPipelineRenderingCreateInfo pipeline_rendering_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .pNext = NULL,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &COLOR_FORMAT,
        .depthAttachmentFormat = DEPTH_FORMAT,
    };

    std::vector<uint32_t> fragment_code = ReadSpirv(FRAGMENT_SHADER_PATH);

    VkShaderModuleCreateInfo fragment_shader_module_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .codeSize = fragment_code.size() * sizeof(uint32_t),
        .pCode = fragment_code.data(),
    };

    VkShaderModule fragment_shader_module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &fragment_shader_module_info, nullptr, &fragment_shader_module));

    VkPipelineShaderStageCreateInfo graphics_pipeline_fragment_shader_stage_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = fragment_shader_module,
        .pName = "main",
        .pSpecializationInfo = NULL,
    };

    std::vector<uint32_t> vertex_code = ReadSpirv(VERTEX_SHADER_PATH);

    VkShaderModuleCreateInfo vertex_shader_module_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .codeSize = vertex_code.size() * sizeof(uint32_t),
        .pCode = vertex_code.data(),
    };

    VkShaderModule vertex_shader_module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &vertex_shader_module_info, nullptr, &vertex_shader_module));

    VkPipelineShaderStageCreateInfo graphics_pipeline_vertex_shader_stage_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vertex_shader_module,
        .pName = "main",
        .pSpecializationInfo = NULL,
    };

    VkPipelineShaderStageCreateInfo graphics_pipeline_shader_stage_infos[2] = {graphics_pipeline_vertex_shader_stage_info, graphics_pipeline_fragment_shader_stage_info};

    VkVertexInputBindingDescription vertex_input_binding = {
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };

    VkVertexInputAttributeDescription vertex_pos_input_attribute = {
        .location = 0,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(Vertex, pos),
    };

    VkVertexInputAttributeDescription vertex_uv_input_attribute = {
        .location = 1,
        .binding = 0,
        .format = VK_FORMAT_R32G32_SFLOAT,
        .offset = offsetof(Vertex, uv),
    };

    VkVertexInputAttributeDescription vertex_normal_input_attribute = {
        .location = 2,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(Vertex, normal),
    };

    VkVertexInputAttributeDescription vertex_material_input_attribute = {
        .location = 3,
        .binding = 0,
        .format = VK_FORMAT_R32_UINT,
        .offset = offsetof(Vertex, material),
    };

    VkVertexInputAttributeDescription vertex_input_attributes[] = {vertex_pos_input_attribute, vertex_uv_input_attribute, vertex_normal_input_attribute, vertex_material_input_attribute};

    VkPipelineVertexInputStateCreateInfo vertex_input_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &vertex_input_binding,
        .vertexAttributeDescriptionCount = sizeof(vertex_input_attributes)/sizeof(vertex_input_attributes[0]), //4
        .pVertexAttributeDescriptions = vertex_input_attributes,
    };

    VkPipelineInputAssemblyStateCreateInfo input_assembly_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    VkPipelineViewportStateCreateInfo viewport_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .viewportCount = 1,
        .pViewports = VK_NULL_HANDLE,
        .scissorCount = 1,
        .pScissors = VK_NULL_HANDLE,
    };

    VkPipelineRasterizationStateCreateInfo rasterization_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.f,
    };

    VkPipelineMultisampleStateCreateInfo multisample_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    VkPipelineDepthStencilStateCreateInfo depth_stencil_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = NULL,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };

    VkPipelineColorBlendAttachmentState color_blend_attachment_state_info = {
        .blendEnable = false,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    VkPipelineColorBlendStateCreateInfo color_blend_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment_state_info,
    };

    VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

    VkPipelineDynamicStateCreateInfo pipeline_dynamic_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .dynamicStateCount = sizeof(dynamic_states)/sizeof(dynamic_states[0]),
        .pDynamicStates = dynamic_states,
    };

    VkPushConstantRange graphics_push_constant_range = {
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(GraphicsPushConstants),
    };

    //array index = set number within the shader (how we know set = 1 is the one we want it to be)
    VkDescriptorSetLayout set_layouts[2] = {graphics_descriptor_set_layout, material_descriptor_set_layout};

    VkPipelineLayoutCreateInfo graphics_pipeline_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .setLayoutCount = 2,
        .pSetLayouts = set_layouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &graphics_push_constant_range,
    };

    VK_CHECK(vkCreatePipelineLayout(device, &graphics_pipeline_layout_info, nullptr, &graphics_pipeline_layout));

    VkGraphicsPipelineCreateInfo graphics_pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &pipeline_rendering_info,
        .flags = 0,
        .stageCount = sizeof(graphics_pipeline_shader_stage_infos)/sizeof(graphics_pipeline_shader_stage_infos[0]), //2
        .pStages = graphics_pipeline_shader_stage_infos,
        .pVertexInputState = &vertex_input_state_info,
        .pInputAssemblyState = &input_assembly_state_info,
        .pTessellationState = VK_NULL_HANDLE,
        .pViewportState = &viewport_state_info,
        .pRasterizationState = &rasterization_state_info,
        .pMultisampleState = &multisample_state_info,
        .pDepthStencilState = &depth_stencil_state_info,
        .pColorBlendState = &color_blend_state_info,
        .pDynamicState = &pipeline_dynamic_state_info,
        .layout = graphics_pipeline_layout,
        .renderPass = VK_NULL_HANDLE,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };

    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &graphics_pipeline_info, nullptr, &graphics_pipeline));

    vkDestroyShaderModule(device, fragment_shader_module, nullptr);
    vkDestroyShaderModule(device, vertex_shader_module, nullptr);

    return 0;
}

int LightPointPipelineCreation(const VkDevice& device, const VkDescriptorSetLayout& graphics_descriptor_set_layout, VkPipelineLayout& light_point_pipeline_layout, VkPipeline& light_point_pipeline)
{
    VkPipelineRenderingCreateInfo pipeline_rendering_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .pNext = NULL,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &COLOR_FORMAT,
        .depthAttachmentFormat = DEPTH_FORMAT,
    };

    std::vector<uint32_t> fragment_code = ReadSpirv(LIGHT_POINT_FRAGMENT_SHADER_PATH);

    VkShaderModuleCreateInfo fragment_shader_module_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .codeSize = fragment_code.size() * sizeof(uint32_t),
        .pCode = fragment_code.data(),
    };

    VkShaderModule fragment_shader_module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &fragment_shader_module_info, nullptr, &fragment_shader_module));

    std::vector<uint32_t> vertex_code = ReadSpirv(LIGHT_POINT_VERTEX_SHADER_PATH);

    VkShaderModuleCreateInfo vertex_shader_module_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .codeSize = vertex_code.size() * sizeof(uint32_t),
        .pCode = vertex_code.data(),
    };

    VkShaderModule vertex_shader_module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &vertex_shader_module_info, nullptr, &vertex_shader_module));

    VkPipelineShaderStageCreateInfo shader_stage_infos[2] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertex_shader_module,
            .pName = "main",
            .pSpecializationInfo = NULL,
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragment_shader_module,
            .pName = "main",
            .pSpecializationInfo = NULL,
        },
    };

    VkPipelineVertexInputStateCreateInfo vertex_input_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .vertexBindingDescriptionCount = 0,
        .pVertexBindingDescriptions = nullptr,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions = nullptr,
    };

    VkPipelineInputAssemblyStateCreateInfo input_assembly_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
    };

    VkPipelineViewportStateCreateInfo viewport_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .viewportCount = 1,
        .pViewports = VK_NULL_HANDLE,
        .scissorCount = 1,
        .pScissors = VK_NULL_HANDLE,
    };

    VkPipelineRasterizationStateCreateInfo rasterization_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.f,
    };

    VkPipelineMultisampleStateCreateInfo multisample_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    VkPipelineDepthStencilStateCreateInfo depth_stencil_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = NULL,
        .depthTestEnable = VK_TRUE, //overridden because marked as dynamic
        .depthWriteEnable = VK_FALSE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };

    VkPipelineColorBlendAttachmentState color_blend_attachment_state_info = {
        .blendEnable = false,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    VkPipelineColorBlendStateCreateInfo color_blend_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment_state_info,
    };

    VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE};

    VkPipelineDynamicStateCreateInfo pipeline_dynamic_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .dynamicStateCount = sizeof(dynamic_states)/sizeof(dynamic_states[0]), //3
        .pDynamicStates = dynamic_states,
    };

    VkPushConstantRange light_point_push_constant_range = {
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(LightPointPushConstants),
    };

    VkPipelineLayoutCreateInfo light_point_pipeline_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &graphics_descriptor_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &light_point_push_constant_range,
    };

    VK_CHECK(vkCreatePipelineLayout(device, &light_point_pipeline_layout_info, nullptr, &light_point_pipeline_layout));

    VkGraphicsPipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &pipeline_rendering_info,
        .flags = 0,
        .stageCount = 2,
        .pStages = shader_stage_infos,
        .pVertexInputState = &vertex_input_state_info,
        .pInputAssemblyState = &input_assembly_state_info,
        .pTessellationState = VK_NULL_HANDLE,
        .pViewportState = &viewport_state_info,
        .pRasterizationState = &rasterization_state_info,
        .pMultisampleState = &multisample_state_info,
        .pDepthStencilState = &depth_stencil_state_info,
        .pColorBlendState = &color_blend_state_info,
        .pDynamicState = &pipeline_dynamic_state_info,
        .layout = light_point_pipeline_layout,
        .renderPass = VK_NULL_HANDLE,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };

    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &light_point_pipeline));

    vkDestroyShaderModule(device, fragment_shader_module, nullptr);
    vkDestroyShaderModule(device, vertex_shader_module, nullptr);

    return 0;
}

int ShadowMapCreation(const VkDevice& device, const VkPhysicalDevice& physical_device, const VkPhysicalDeviceMemoryProperties& memory_properties, const std::vector<VkDescriptorSet>& graphics_descriptor_set, std::vector<VkImage>& shadow_map, VkDeviceMemory& shadow_map_memory, std::vector<VkImageView>& shadow_map_view, VkSampler& shadow_sampler)
{
    VkImageCreateInfo shadow_map_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = DEPTH_FORMAT,
        .extent = {
            .width = SHADOW_MAP_SIZE,
            .height = SHADOW_MAP_SIZE,
            .depth = 1,
        },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        VK_CHECK(vkCreateImage(device, &shadow_map_info, nullptr, &shadow_map[i]));
    }

    VkMemoryRequirements image_memory_requirements = {};
    vkGetImageMemoryRequirements(device, shadow_map[0], &image_memory_requirements);

    uint32_t memory_type_index = FindMemoryType(memory_properties, image_memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (memory_type_index == UINT32_MAX) {
        std::fprintf(stderr, "No device-local memory type available\n");
        return EXIT_FAILURE;
    }

    VkDeviceSize stride = ((image_memory_requirements.size + image_memory_requirements.alignment - 1) / image_memory_requirements.alignment) * image_memory_requirements.alignment;

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = NULL,
        .allocationSize = stride * FRAMES_IN_FLIGHT,
        .memoryTypeIndex = memory_type_index,
    };

    VK_CHECK(vkAllocateMemory(device, &alloc_info, nullptr, &shadow_map_memory));

    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        VK_CHECK(vkBindImageMemory(device, shadow_map[i], shadow_map_memory, i * stride));

        VkImageViewCreateInfo view_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = shadow_map[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = DEPTH_FORMAT,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        VK_CHECK(vkCreateImageView(device, &view_info, nullptr, &shadow_map_view[i]));
    }

    VkFormatProperties format_properties = {};
    vkGetPhysicalDeviceFormatProperties(physical_device, DEPTH_FORMAT, &format_properties);
    VkFilter shadow_filter = (format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;

    VkSamplerCreateInfo sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .magFilter = shadow_filter,
        .minFilter = shadow_filter,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .mipLodBias = 0.f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.f,
        .compareEnable = VK_TRUE,
        .compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
        .minLod = 0.f,
        .maxLod = 0.f,
        .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
        .unnormalizedCoordinates = VK_FALSE,
    };

    VK_CHECK(vkCreateSampler(device, &sampler_info, nullptr, &shadow_sampler));

    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        VkDescriptorImageInfo shadow_map_image_info = {
            .sampler = shadow_sampler,
            .imageView = shadow_map_view[i],
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };

        VkWriteDescriptorSet write_descriptor_set = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = NULL,
            .dstSet = graphics_descriptor_set[i],
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &shadow_map_image_info,
        };

        vkUpdateDescriptorSets(device, 1, &write_descriptor_set, 0, nullptr);
    }

    return 0;
}

int ShadowPipelineCreation(const VkDevice& device, const VkPipelineLayout& graphics_pipeline_layout, VkPipeline& shadow_pipeline)
{
    VkPipelineRenderingCreateInfo pipeline_rendering_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .pNext = NULL,
        .colorAttachmentCount = 0,
        .pColorAttachmentFormats = nullptr,
        .depthAttachmentFormat = DEPTH_FORMAT,
    };

    std::vector<uint32_t> vertex_code = ReadSpirv(VERTEX_SHADER_PATH);

    VkShaderModuleCreateInfo vertex_shader_module_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .codeSize = vertex_code.size() * sizeof(uint32_t),
        .pCode = vertex_code.data(),
    };

    VkShaderModule vertex_shader_module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &vertex_shader_module_info, nullptr, &vertex_shader_module));

    VkPipelineShaderStageCreateInfo vertex_shader_stage_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vertex_shader_module,
        .pName = "main",
        .pSpecializationInfo = NULL,
    };

    VkVertexInputBindingDescription vertex_input_binding = {
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };

    VkVertexInputAttributeDescription vertex_input_attributes[4] = {
        {
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, pos),
        },
        {
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R32G32_SFLOAT,
            .offset = offsetof(Vertex, uv),
        },
        {
            .location = 2,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, normal),
        },
        {
            .location = 3,
            .binding = 0,
            .format = VK_FORMAT_R32_UINT,
            .offset = offsetof(Vertex, material),
        },
    };

    VkPipelineVertexInputStateCreateInfo vertex_input_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &vertex_input_binding,
        .vertexAttributeDescriptionCount = 4,
        .pVertexAttributeDescriptions = vertex_input_attributes,
    };

    VkPipelineInputAssemblyStateCreateInfo input_assembly_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    VkPipelineViewportStateCreateInfo viewport_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .viewportCount = 1,
        .pViewports = VK_NULL_HANDLE,
        .scissorCount = 1,
        .pScissors = VK_NULL_HANDLE,
    };

    VkPipelineRasterizationStateCreateInfo rasterization_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_TRUE,
        .depthBiasConstantFactor = 1.25f,
        .depthBiasClamp = 0.f,
        .depthBiasSlopeFactor = 1.75f,
        .lineWidth = 1.f,
    };

    VkPipelineMultisampleStateCreateInfo multisample_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    VkPipelineDepthStencilStateCreateInfo depth_stencil_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = NULL,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };

    VkPipelineColorBlendStateCreateInfo color_blend_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .attachmentCount = 0,
        .pAttachments = nullptr,
    };

    VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

    VkPipelineDynamicStateCreateInfo pipeline_dynamic_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .dynamicStateCount = sizeof(dynamic_states)/sizeof(dynamic_states[0]),
        .pDynamicStates = dynamic_states,
    };

    VkGraphicsPipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &pipeline_rendering_info,
        .flags = 0,
        .stageCount = 1,
        .pStages = &vertex_shader_stage_info,
        .pVertexInputState = &vertex_input_state_info,
        .pInputAssemblyState = &input_assembly_state_info,
        .pTessellationState = VK_NULL_HANDLE,
        .pViewportState = &viewport_state_info,
        .pRasterizationState = &rasterization_state_info,
        .pMultisampleState = &multisample_state_info,
        .pDepthStencilState = &depth_stencil_state_info,
        .pColorBlendState = &color_blend_state_info,
        .pDynamicState = &pipeline_dynamic_state_info,
        .layout = graphics_pipeline_layout,
        .renderPass = VK_NULL_HANDLE,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };

    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &shadow_pipeline));

    vkDestroyShaderModule(device, vertex_shader_module, nullptr);

    return 0;
}

int CreateHostVisibleBuffer(const VkDevice& device, const VkPhysicalDeviceMemoryProperties& memory_properties, VkDeviceSize size, VkBufferUsageFlags usage, bool prefer_device_local, VkBuffer& buffer, VkDeviceMemory& memory, void*& mapped)
{
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };

    VK_CHECK(vkCreateBuffer(device, &buffer_info, nullptr, &buffer));

    VkMemoryRequirements memory_requirements = {};
    vkGetBufferMemoryRequirements(device, buffer, &memory_requirements);

    //host visible so we can map it, coherent so writes are visible to the GPU without vkFlushMappedMemoryRanges
    VkMemoryPropertyFlags property_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t memory_type_index = UINT32_MAX;

    if (prefer_device_local) {
        memory_type_index = FindMemoryType(memory_properties, memory_requirements.memoryTypeBits, property_flags | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    if (memory_type_index == UINT32_MAX) {
        memory_type_index = FindMemoryType(memory_properties, memory_requirements.memoryTypeBits, property_flags);
    }

    if (memory_type_index == UINT32_MAX) {
        std::fprintf(stderr, "No host-visible coherent memory type available\n");
        return EXIT_FAILURE;
    }

    if (prefer_device_local) {
        //std::printf("[info] prefer-VRAM buffer got memory type %u: %s\n", memory_type_index, string_VkMemoryPropertyFlags(memory_properties.memoryTypes[memory_type_index].propertyFlags).c_str());
    }

    VkMemoryAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = nullptr,
        .allocationSize = memory_requirements.size,
        .memoryTypeIndex = memory_type_index,
    };

    VK_CHECK(vkAllocateMemory(device, &allocate_info, nullptr, &memory));
    VK_CHECK(vkBindBufferMemory(device, buffer, memory, 0));

    //caller decides whether to unmap (static data) or keep it mapped (per-frame data)
    VK_CHECK(vkMapMemory(device, memory, 0, size, 0, &mapped));

    return 0;
}

//static data path: cpu -> host-visible staging buffer -> (gpu copy) -> device-local buffer, done once
int CreateDeviceLocalBuffer(const VkDevice& device, const VkPhysicalDeviceMemoryProperties& memory_properties, const VkQueue& queue, const VkCommandPool& command_pool, const void* data, VkDeviceSize size, VkBufferUsageFlags usage, VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access, VkBuffer& buffer, VkDeviceMemory& memory)
{
    //staging buffer: the only part the cpu ever touches
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    int hvb = CreateHostVisibleBuffer(device, memory_properties, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false, staging_buffer, staging_memory, mapped);
    if (hvb != 0) return hvb;
    std::memcpy(mapped, data, size);
    vkUnmapMemory(device, staging_memory);

    //destination buffer: TRANSFER_DST so we can copy into it
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .size = size,
        .usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };

    VK_CHECK(vkCreateBuffer(device, &buffer_info, nullptr, &buffer));

    VkMemoryRequirements memory_requirements = {};
    vkGetBufferMemoryRequirements(device, buffer, &memory_requirements);

    uint32_t memory_type_index = FindMemoryType(memory_properties, memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (memory_type_index == UINT32_MAX) {
        std::fprintf(stderr, "No device-local memory type available\n");
        return EXIT_FAILURE;
    }

    VkMemoryAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = nullptr,
        .allocationSize = memory_requirements.size,
        .memoryTypeIndex = memory_type_index,
    };

    VK_CHECK(vkAllocateMemory(device, &allocate_info, nullptr, &memory));
    VK_CHECK(vkBindBufferMemory(device, buffer, memory, 0));

    //one-shot command buffer for the copy
    VkCommandBufferAllocateInfo command_buffer_allocate_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = NULL,
        .commandPool = command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    VkCommandBuffer upload_command_buffer = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device, &command_buffer_allocate_info, &upload_command_buffer));

    VkCommandBufferBeginInfo command_buffer_begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = NULL,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    VK_CHECK(vkBeginCommandBuffer(upload_command_buffer, &command_buffer_begin_info));

    VkBufferCopy copy_region = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = size,
    };

    vkCmdCopyBuffer(upload_command_buffer, staging_buffer, buffer, 1, &copy_region);

    VkBufferMemoryBarrier2 barrier_to_read = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .pNext = NULL,

        .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,

        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,

        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,

        .buffer = buffer,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };

    VkDependencyInfo dependency_info_to_read = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext = NULL,
        .dependencyFlags = 0,
        .memoryBarrierCount = 0,
        .pMemoryBarriers = nullptr,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &barrier_to_read,
        .imageMemoryBarrierCount = 0,
        .pImageMemoryBarriers = nullptr,
    };

    vkCmdPipelineBarrier2(upload_command_buffer, &dependency_info_to_read);

    VK_CHECK(vkEndCommandBuffer(upload_command_buffer));

    VkCommandBufferSubmitInfo command_buffer_submit_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext = NULL,
        .commandBuffer = upload_command_buffer,
        .deviceMask = 0,
    };

    VkSubmitInfo2 submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = NULL,
        .flags = 0,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &command_buffer_submit_info,
    };

    VK_CHECK(vkQueueSubmit2(queue, 1, &submit_info, VK_NULL_HANDLE));

    //startup only, so just block. staging buffer can't be freed until the copy has actually executed
    VK_CHECK(vkQueueWaitIdle(queue));

    vkFreeCommandBuffers(device, command_pool, 1, &upload_command_buffer);
    vkDestroyBuffer(device, staging_buffer, nullptr);
    vkFreeMemory(device, staging_memory, nullptr);

    return 0;
}

int CreateTextureImage(const VkDevice& device, const VkPhysicalDeviceMemoryProperties& memory_properties, const VkQueue& queue, const VkCommandPool& command_pool, const std::string& texture_filename, VkImage& image, VkDeviceMemory& memory, VkImageView& view)
{
    int width = 1;
    int height = 1;
    int channels = 4;
    const uint8_t white_texel[4] = {255, 255, 255, 255};
    stbi_uc* pixels = nullptr;

    if (!texture_filename.empty()) {
        std::string texture_filepath = TEXTURE_DIR + texture_filename;

        pixels = stbi_load(texture_filepath.c_str(), &width, &height, &channels, STBI_rgb_alpha);

        if (!pixels) {
            std::fprintf(stderr, "Failed to load texture %s: %s\n", texture_filepath.c_str(), stbi_failure_reason());
            return EXIT_FAILURE;
        }

        std::printf("[info] loaded texture %s: %i x %i (%i channels in file, 4 uploaded)\n", texture_filename.c_str(), width, height, channels);
    }

    VkDeviceSize image_bytes = (VkDeviceSize)width * (VkDeviceSize)height * 4;

    //staging buffer: identical to CreateDeviceLocalBuffer
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    int hvb = CreateHostVisibleBuffer(device, memory_properties, image_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false, staging_buffer, staging_memory, mapped);
    if (hvb != 0) return hvb;
    std::memcpy(mapped, pixels ? pixels : white_texel, image_bytes);
    vkUnmapMemory(device, staging_memory);

    if (pixels) stbi_image_free(pixels);

    VkImageCreateInfo image_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = TEXTURE_FORMAT,
        .extent = {
            .width = (uint32_t)width,
            .height = (uint32_t)height,
            .depth = 1,
        },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, //DST -> copy into it, SAMPLED -> fragment shader read
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    VK_CHECK(vkCreateImage(device, &image_info, nullptr, &image));

    //memory: same pattern as FrameCreation
    VkMemoryRequirements memory_requirements = {};
    vkGetImageMemoryRequirements(device, image, &memory_requirements);

    uint32_t memory_type_index = FindMemoryType(memory_properties, memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (memory_type_index == UINT32_MAX) {
        std::fprintf(stderr, "No device-local memory type available\n");
        return EXIT_FAILURE;
    }

    VkMemoryAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = nullptr,
        .allocationSize = memory_requirements.size,
        .memoryTypeIndex = memory_type_index,
    };

    VK_CHECK(vkAllocateMemory(device, &allocate_info, nullptr, &memory));
    VK_CHECK(vkBindImageMemory(device, image, memory, 0));

    VkCommandBufferAllocateInfo command_buffer_allocate_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = NULL,
        .commandPool = command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    VkCommandBuffer upload_command_buffer = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device, &command_buffer_allocate_info, &upload_command_buffer));

    VkCommandBufferBeginInfo command_buffer_begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = NULL,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    VK_CHECK(vkBeginCommandBuffer(upload_command_buffer, &command_buffer_begin_info));

    VkImageSubresourceRange subresource_range = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };

    VkImageMemoryBarrier2 barrier_to_transfer_dst = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = NULL,

        .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
        .srcAccessMask = VK_ACCESS_2_NONE,

        .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
        .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,

        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,

        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,

        .image = image,
        .subresourceRange = subresource_range,
    };

    VkDependencyInfo dependency_info_to_transfer_dst = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext = NULL,
        .dependencyFlags = 0,
        .memoryBarrierCount = 0,
        .pMemoryBarriers = nullptr,
        .bufferMemoryBarrierCount = 0,
        .pBufferMemoryBarriers = nullptr,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier_to_transfer_dst,
    };

    vkCmdPipelineBarrier2(upload_command_buffer, &dependency_info_to_transfer_dst);

    VkBufferImageCopy copy_region = {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .imageOffset = {0, 0, 0},
        .imageExtent = {(uint32_t)width, (uint32_t)height, 1},
    };

    vkCmdCopyBufferToImage(upload_command_buffer, staging_buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

    VkImageMemoryBarrier2 barrier_to_shader_read = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = NULL,

        .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,

        .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,

        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,

        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,

        .image = image,
        .subresourceRange = subresource_range,
    };

    VkDependencyInfo dependency_info_to_shader_read = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext = NULL,
        .dependencyFlags = 0,
        .memoryBarrierCount = 0,
        .pMemoryBarriers = nullptr,
        .bufferMemoryBarrierCount = 0,
        .pBufferMemoryBarriers = nullptr,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier_to_shader_read,
    };

    vkCmdPipelineBarrier2(upload_command_buffer, &dependency_info_to_shader_read);

    VK_CHECK(vkEndCommandBuffer(upload_command_buffer));

    VkCommandBufferSubmitInfo command_buffer_submit_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext = NULL,
        .commandBuffer = upload_command_buffer,
        .deviceMask = 0,
    };

    VkSubmitInfo2 submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = NULL,
        .flags = 0,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &command_buffer_submit_info,
    };

    VK_CHECK(vkQueueSubmit2(queue, 1, &submit_info, VK_NULL_HANDLE));

    VK_CHECK(vkQueueWaitIdle(queue));

    vkFreeCommandBuffers(device, command_pool, 1, &upload_command_buffer);
    vkDestroyBuffer(device, staging_buffer, nullptr);
    vkFreeMemory(device, staging_memory, nullptr);

    VkImageViewCreateInfo view_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = TEXTURE_FORMAT,
        .subresourceRange = subresource_range,
    };

    VK_CHECK(vkCreateImageView(device, &view_info, nullptr, &view));

    return 0;
}

int CreateTextureSampler(const VkDevice& device, VkSampler& sampler)
{
    VkSamplerCreateInfo sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .mipLodBias = 0.f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .minLod = 0.f,
        .maxLod = 0.f,
        .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };

    VK_CHECK(vkCreateSampler(device, &sampler_info, nullptr, &sampler));

    return 0;
}

int LoadMtl(const std::string& mtl_filename, std::unordered_set<std::string>& loaded_mtl_files, std::unordered_map<std::string, uint32_t>& material_lookup, std::vector<Material>& materials, std::string& diffuse_texture_filename)
{
    if (loaded_mtl_files.find(mtl_filename) != loaded_mtl_files.end()) {
        return 0;
    }

    std::string mtl_filepath = DATA_DIR "/materials/" + mtl_filename;
    std::ifstream mtl_file(mtl_filepath);

    if (!mtl_file.is_open()) {
        std::fprintf(stderr, "Warning: material file not found: %s\n", mtl_filepath.c_str());
        return 0;
    }

    loaded_mtl_files.insert(mtl_filename);

    std::string line;
    int32_t current = -1;

    while (std::getline(mtl_file, line)) {
        std::istringstream iss(line);
        std::string keyword;
        iss >> keyword;

        if (keyword == "newmtl") {
            std::string mtl_name;
            iss >> mtl_name;
            current = (int32_t)materials.size();
            material_lookup[mtl_filename + "." + mtl_name] = (uint32_t)current;
            materials.push_back({.diffuse = {1.f, 0.f, 1.0f, 1.f}, .specular = {0.f, 0.f, 0.f, 32.f}}); //default mat for missing data
        }
        else if (current < 0) {
            continue;
        }
        else if (keyword == "Kd") { //diffuse color
            glm::vec3 kd;
            iss >> kd.x >> kd.y >> kd.z;
            materials[current].diffuse = glm::vec4(kd, materials[current].diffuse.a);
        }
        else if (keyword == "d") {
            iss >> materials[current].diffuse.a;
        }
        else if (keyword == "Tr") {
            float tr = 0.f;
            iss >> tr;
            materials[current].diffuse.a = 1.f - tr;
        }
        else if (keyword == "Ks") { //specular color
            glm::vec3 ks;
            iss >> ks.x >> ks.y >> ks.z;
            materials[current].specular = glm::vec4(ks, materials[current].specular.w);
        }
        else if (keyword == "Ns") { //shininess (specular exponent)
            iss >> materials[current].specular.w;
        }
        else if (keyword == "map_Kd") {
            std::string token, texture_name;
            while (iss >> token) texture_name = token;

            if (texture_name.empty()) continue;

            if (!diffuse_texture_filename.empty() && diffuse_texture_filename != texture_name) {
                std::fprintf(stderr, "Warning: only one diffuse texture supported; %s ignored, %s still sampled for %s\n", texture_name.c_str(), diffuse_texture_filename.c_str(), mtl_filename.c_str());
            }
            else {
                diffuse_texture_filename = texture_name;
            }

            materials[current].has_diffuse_texture = 1;
        }
    }

    return 0;
}

int MeshImport(std::vector<Mesh>& meshes, std::unordered_map<std::string, uint32_t>& mesh_lookup, std::vector<Material>& materials, std::string& diffuse_texture_filename, std::vector<Vertex>& all_vertices, std::vector<uint32_t>& all_indices)
{
    //every .obj in data/objects becomes a mesh. objects reference them by filename through mesh_lookup
    std::vector<std::string> mesh_files;
    std::error_code fs_error;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(DATA_DIR "/objects", fs_error)) {
        if (entry.is_regular_file() && entry.path().extension() == ".obj") {
            mesh_files.push_back(entry.path().filename().string());
        }
    }

    if (fs_error) {
        std::fprintf(stderr, "Failed to read %s: %s\n", DATA_DIR "/objects", fs_error.message().c_str());
        return EXIT_FAILURE;
    }
    std::sort(mesh_files.begin(), mesh_files.end());

    meshes.resize(mesh_files.size());

    materials = {
        {.diffuse = {1.f, 0.f, 1.f, 1.f}, .specular = {0.f, 0.f, 0.f, 1.f}},
    };
    std::unordered_map<std::string, uint32_t> material_lookup;
    std::unordered_set<std::string> loaded_mtl_files; //prevents repeat .mtl file reads

    for (uint32_t i = 0; i < mesh_files.size(); ++i) {
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        int lo = LoadObj(mesh_files[i], loaded_mtl_files, material_lookup, materials, diffuse_texture_filename, vertices, indices);
        if (lo != 0) return lo;

        meshes[i] = {
            .name = mesh_files[i],
            .first_index = (uint32_t)all_indices.size(),
            .index_count = (uint32_t)indices.size(),
            .vertex_offset = (int32_t)all_vertices.size(),
        };
        mesh_lookup[mesh_files[i]] = i;

        all_vertices.insert(all_vertices.end(), vertices.begin(), vertices.end());
        all_indices.insert(all_indices.end(), indices.begin(), indices.end());

        //std::printf("%s: %zi vertices, %zi indices, %zi tris!\n", mesh_files[i].c_str(), vertices.size(), indices.size(), indices.size()/3);
    }

    //std::printf("%zu materials loaded (including default)\n", materials.size());

    return 0;
}

int MeshUpload(const VkDevice& device, const VkPhysicalDeviceMemoryProperties& memory_properties, const VkQueue& queue, const VkCommandPool& command_pool, const std::vector<Vertex>& all_vertices, const std::vector<uint32_t>& all_indices, VkBuffer& model_vertex_buffer, VkDeviceMemory& model_vertex_buffer_memory, VkBuffer& model_index_buffer, VkDeviceMemory& model_index_buffer_memory)
{
    VkDeviceSize model_vertex_bytes = sizeof(Vertex) * all_vertices.size();
    VkDeviceSize model_index_bytes = sizeof(uint32_t) * all_indices.size();

#if USE_DEVICE_LOCAL_MESHES
    //std::printf("[info] mesh buffers: DEVICE_LOCAL (staged)\n");

    int mvb = CreateDeviceLocalBuffer(device, memory_properties, queue, command_pool, all_vertices.data(), model_vertex_bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                      VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT, VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT, model_vertex_buffer, model_vertex_buffer_memory);
    if (mvb != 0) return mvb;

    int mib = CreateDeviceLocalBuffer(device, memory_properties, queue, command_pool, all_indices.data(), model_index_bytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                      VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT, VK_ACCESS_2_INDEX_READ_BIT, model_index_buffer, model_index_buffer_memory);
    if (mib != 0) return mib;
#else
    //std::printf("[info] mesh buffers: HOST_VISIBLE\n");

    void* mapped = nullptr;

    int mvb = CreateHostVisibleBuffer(device, memory_properties, model_vertex_bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, false, model_vertex_buffer, model_vertex_buffer_memory, mapped);
    if (mvb != 0) return mvb;
    std::memcpy(mapped, all_vertices.data(), model_vertex_bytes);
    vkUnmapMemory(device, model_vertex_buffer_memory);

    int mib = CreateHostVisibleBuffer(device, memory_properties, model_index_bytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, false, model_index_buffer, model_index_buffer_memory, mapped);
    if (mib != 0) return mib;
    std::memcpy(mapped, all_indices.data(), model_index_bytes);
    vkUnmapMemory(device, model_index_buffer_memory);
#endif

    return 0;
}

int MaterialUpload(const VkDevice& device, const VkPhysicalDeviceMemoryProperties& memory_properties, const VkQueue& queue, const VkCommandPool& command_pool, const std::vector<Material>& materials, const VkDescriptorSet& material_descriptor_set, VkBuffer& material_buffer, VkDeviceMemory& material_buffer_memory)
{
    VkDeviceSize material_bytes = sizeof(Material) * materials.size();

    int mtb = CreateDeviceLocalBuffer(device, memory_properties, queue, command_pool, materials.data(), material_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT, material_buffer, material_buffer_memory);
    if (mtb != 0) return mtb;

    VkDescriptorBufferInfo material_buffer_info = {
        .buffer = material_buffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };

    VkWriteDescriptorSet material_write_descriptor_set = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = NULL,
        .dstSet = material_descriptor_set,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &material_buffer_info,
    };

    vkUpdateDescriptorSets(device, 1, &material_write_descriptor_set, 0, nullptr);

    return 0;
}

int TextureUpload(const VkDevice& device, const VkPhysicalDeviceMemoryProperties& memory_properties, const VkQueue& queue, const VkCommandPool& command_pool, const std::string& diffuse_texture_filename, const VkDescriptorSet& material_descriptor_set, VkImage& diffuse_texture, VkDeviceMemory& diffuse_texture_memory, VkImageView& diffuse_texture_view, VkSampler& diffuse_sampler)
{
    int cti = CreateTextureImage(device, memory_properties, queue, command_pool, diffuse_texture_filename, diffuse_texture, diffuse_texture_memory, diffuse_texture_view);
    if (cti != 0) return cti;

    int cts = CreateTextureSampler(device, diffuse_sampler);
    if (cts != 0) return cts;

    VkDescriptorImageInfo diffuse_texture_info = {
        .sampler = diffuse_sampler,
        .imageView = diffuse_texture_view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    VkWriteDescriptorSet texture_write_descriptor_set = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = NULL,
        .dstSet = material_descriptor_set,
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &diffuse_texture_info,
    };

    vkUpdateDescriptorSets(device, 1, &texture_write_descriptor_set, 0, nullptr);

    return 0;
}

//parses one .obj (+ its .mtl) into local vertex/index lists. indices are relative to this mesh
int LoadObj(const std::string& filename, std::unordered_set<std::string>& loaded_mtl_files, std::unordered_map<std::string, uint32_t>& material_lookup, std::vector<Material>& materials, std::string& diffuse_texture_filename, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices)
{
    std::string object_filepath = DATA_DIR "/objects/" + filename;
    std::ifstream object_file(object_filepath);

    if (!object_file.is_open()) {
        std::fprintf(stderr, "Invalid file name: %s\n", object_filepath.c_str());
        return EXIT_FAILURE;
    }

    std::string line;
    std::string mtl_filename;
    uint32_t current_material = 0;

    //pos and nrm have different indexes, both 1-indexed
    std::vector<glm::vec3> positions;
    std::vector<glm::vec2> texcoords;
    std::vector<glm::vec3> normals;
    std::unordered_map<VertexKey, uint32_t, VertexKeyHash> unique_vertices;
    uint32_t skipped_transparent_faces = 0;

    while (std::getline(object_file, line)) {
        std::istringstream iss(line);
        std::string keyword;
        iss >> keyword;

        if (keyword == "v") {
            glm::vec3 position;
            iss >> position.x >> position.y >> position.z;
            positions.push_back(position);
        }
        else if (keyword == "vt") {
            glm::vec2 texcoord;
            iss >> texcoord.x >> texcoord.y;
            texcoord.y = 1.f - texcoord.y;
            texcoords.push_back(texcoord);
        }
        else if (keyword == "vn") {
            glm::vec3 normal;
            iss >> normal.x >> normal.y >> normal.z;
            normals.push_back(normal);
        }
        else if (keyword == "f") {
            if (current_material < materials.size() && materials[current_material].diffuse.a <= 0.f) {
                ++skipped_transparent_faces;
                continue;
            }

            std::vector<uint32_t> face_indices;
            std::string vertex_data;

            while (iss >> vertex_data) { //face format per vertex can either be "p" "p/t" "p/t/n" or "p//n"
                std::istringstream viss(vertex_data);
                std::string pos_idx, tex_idx, norm_idx;

                std::getline(viss, pos_idx, '/');
                std::getline(viss, tex_idx, '/');
                std::getline(viss, norm_idx, '/');

                int pidx = std::stoi(pos_idx) - 1;
                int tidx = tex_idx.empty() ? -1 : std::stoi(tex_idx) - 1;
                int nidx = norm_idx.empty() ? -1 : std::stoi(norm_idx) - 1;

                if (pidx < 0 || pidx >= (int)positions.size()) {
                    std::fprintf(stderr, "Invalid position index for %s: %i\n", filename.c_str(), pidx);
                    return EXIT_FAILURE;
                }
                if (tidx != -1 && (tidx < 0 || tidx >= (int)texcoords.size())) {
                    std::fprintf(stderr, "Invalid texcoord index for %s: %i\n", filename.c_str(), tidx);
                    return EXIT_FAILURE;
                }
                if (nidx != -1 && (nidx < 0 || nidx >= (int)normals.size())) {
                    std::fprintf(stderr, "Invalid normal index for %s: %i\n", filename.c_str(), nidx);
                    return EXIT_FAILURE;
                }

                VertexKey key = {pidx, tidx, nidx, current_material};
                std::unordered_map<VertexKey, uint32_t, VertexKeyHash>::iterator it = unique_vertices.find(key); //reuse if vertex has been described

                if (it == unique_vertices.end()) {
                    uint32_t new_index = (uint32_t)vertices.size();
                    vertices.push_back({
                        .pos = positions[pidx],
                        .uv = (tidx != -1) ? texcoords[tidx] : glm::vec2(0.f),
                        .normal = (nidx != -1) ? normals[nidx] : glm::vec3(0.f), //fill empty normals
                        .material = current_material,
                    });
                    unique_vertices[key] = new_index;
                    face_indices.push_back(new_index);
                }
                else {
                    face_indices.push_back(it->second);
                }
            }

            //fan triangulation
            for (uint32_t j = 1; j + 1 < face_indices.size(); ++j) {
                indices.push_back(face_indices[0]);
                indices.push_back(face_indices[j]);
                indices.push_back(face_indices[j + 1]);
            }
        }
        else if (keyword == "mtllib") {
            iss >> mtl_filename;
            int lm = LoadMtl(mtl_filename, loaded_mtl_files, material_lookup, materials, diffuse_texture_filename);
            if (lm != 0) return lm;
        }
        else if (keyword == "usemtl") {
            std::string mtl_name;
            iss >> mtl_name;
            std::string mtl_full_name = mtl_filename + "." + mtl_name;

            std::unordered_map<std::string, uint32_t>::iterator it = material_lookup.find(mtl_full_name);
            if (it == material_lookup.end()) {
                std::fprintf(stderr, "Warning: unknown material %s in %s, using default\n", mtl_full_name.c_str(), filename.c_str());
                current_material = 0;
            }
            else {
                current_material = it->second;
            }
        }
    }
    object_file.close();

    if (skipped_transparent_faces > 0) {
        std::printf("[info] %s: skipped %u faces with fully transparent materials (d = 0)\n", filename.c_str(), skipped_transparent_faces);
    }

    //construct smooth normals by averaging each vertex's faces' normals
    if (normals.empty()) {
        std::printf("Generating normals for %s\n", filename.c_str());

        for (size_t i = 0; i < indices.size(); i += 3) {
            uint32_t i0 = indices[i];
            uint32_t i1 = indices[i + 1];
            uint32_t i2 = indices[i + 2];

            glm::vec3 edge1 = vertices[i1].pos - vertices[i0].pos;
            glm::vec3 edge2 = vertices[i2].pos - vertices[i0].pos;
            glm::vec3 face_normal = glm::cross(edge1, edge2); //bigger faces get more weight (unnormalized)

            vertices[i0].normal += face_normal;
            vertices[i1].normal += face_normal;
            vertices[i2].normal += face_normal;
        }

        for (Vertex& vertex : vertices) {
            if (glm::length(vertex.normal) > 0.f) {
                vertex.normal = glm::normalize(vertex.normal);
            }
        }
    }

    return 0;
}

int SynchronizationObjectCreation(const VkDevice& device, const uint32_t& swapchain_image_count, std::vector<VkSemaphore>& image_available_semaphores, std::vector<VkSemaphore>& render_finished_semaphores, std::vector<VkFence>& in_flight_fences)
{
    VkSemaphoreCreateInfo semaphore_create_info = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
    };

    VkFenceCreateInfo command_buffer_fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = 0,
    };

    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        VK_CHECK(vkCreateSemaphore(device, &semaphore_create_info, nullptr, &image_available_semaphores[i]));
        VK_CHECK(vkCreateFence(device, &command_buffer_fence_info, nullptr, &in_flight_fences[i]));
    }

    for (uint32_t i = 0; i < swapchain_image_count; ++i) {
        VK_CHECK(vkCreateSemaphore(device, &semaphore_create_info, nullptr, &render_finished_semaphores[i]));
    }

    return 0;
}

int TransitionDepthToAttachment(const VkQueue& queue, const std::vector<VkCommandBuffer>& command_buffers, const std::vector<VkImage>& depth, const std::vector<VkFence>& in_flight_fences)
{
    for (int i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        VK_CHECK(vkResetCommandBuffer(command_buffers[i], 0));

        VkCommandBufferBeginInfo command_buffer_begin_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = NULL,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };

        VK_CHECK(vkBeginCommandBuffer(command_buffers[i], &command_buffer_begin_info));

        VkImageSubresourceRange depth_subresource_range = {
            .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };

        VkImageMemoryBarrier2 barrier_to_dao = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext = NULL,

            .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
            .srcAccessMask = VK_ACCESS_2_NONE,

            .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,

            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,

            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,

            .image = depth[i],
            .subresourceRange = depth_subresource_range,
        };

        VkDependencyInfo dependency_info_to_loop = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = NULL,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier_to_dao,
        };

        vkCmdPipelineBarrier2(command_buffers[i], &dependency_info_to_loop);
        
        VK_CHECK(vkEndCommandBuffer(command_buffers[i]));

        VkCommandBufferSubmitInfo command_buffer_submit_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .pNext = NULL,
            .commandBuffer = command_buffers[i],
            .deviceMask = 0,
        };

        VkSubmitInfo2 submit_info = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .pNext = NULL,
            .flags = 0,
            .commandBufferInfoCount = 1,
            .pCommandBufferInfos = &command_buffer_submit_info,
        };

        VK_CHECK(vkQueueSubmit2(queue, 1, &submit_info, in_flight_fences[i]));
    }

    return 0;
}

int UpdateInput(GLFWwindow*& window, const std::vector<int>& keys_to_track, std::array<KeyState, GLFW_KEY_LAST + 1>& key_states, double& cursor_x, double& cursor_y, double& cursor_dx, double& cursor_dy, double& scroll_dy)
{
    for (uint32_t i = 0; i < keys_to_track.size(); ++i) {
        int key_id = keys_to_track[i];
        int key_state = glfwGetKey(window, key_id);
        switch (key_states[key_id]) {
            case KEY_STATE_UP: {
                if (key_state == GLFW_PRESS) {
                    key_states[key_id] = KEY_STATE_PRESSED;
                } else if (key_state == GLFW_RELEASE) {
                    //do nothing
                }
                break;
            }
            case KEY_STATE_PRESSED: {
                if (key_state == GLFW_PRESS) {
                    key_states[key_id] = KEY_STATE_DOWN;
                } else if (key_state == GLFW_RELEASE) {
                    key_states[key_id] = KEY_STATE_RELEASED;
                }
                break;
            }
            case KEY_STATE_DOWN: {
                if (key_state == GLFW_PRESS) {
                    //do nothing
                } else if (key_state == GLFW_RELEASE) {
                    key_states[key_id] = KEY_STATE_RELEASED;
                }
                break;
            }
            case KEY_STATE_RELEASED: {
                if (key_state == GLFW_PRESS) {
                    key_states[key_id] = KEY_STATE_PRESSED;
                } else if (key_state == GLFW_RELEASE) {
                    key_states[key_id] = KEY_STATE_UP;
                }
                break;
            }
        }
    }

    double cursor_nx, cursor_ny;
    glfwGetCursorPos(window, &cursor_nx, &cursor_ny);
    cursor_dx = cursor_nx - cursor_x;
    cursor_dy = cursor_ny - cursor_y;
    cursor_x = cursor_nx;
    cursor_y = cursor_ny;

    scroll_dy = scroll_accumulator;
    scroll_accumulator = 0.0;

    return 0;
}

static void scroll_callback(GLFWwindow* window, double x_offset, double y_offset)
{
    (void)window;
    (void)x_offset;

    scroll_accumulator += y_offset;
}

// Functions that I added for the major and minor features are below!
// Function based on ShadowMapCreation with six layers and cube compatability
int CreatePointShadow(const VkDevice& device, const VkPhysicalDeviceMemoryProperties& memory_properties, const std::vector<VkDescriptorSet>& graphics_descriptor_set, const VkSampler& shadow_sampler, std::vector<VkImage>& cube_image, VkDeviceMemory& cube_memory, std::vector<VkImageView>& cube_view, std::vector<VkImageView>& cube_face_view)
{
    VkImageCreateInfo image_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = DEPTH_FORMAT,
        .extent = {POINT_SHADOW_SIZE, POINT_SHADOW_SIZE, 1},
        .mipLevels = 1,
        .arrayLayers = 6,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        VK_CHECK(vkCreateImage(device, &image_info, nullptr, &cube_image[i]));
    }

    VkMemoryRequirements reqs = {};
    vkGetImageMemoryRequirements(device, cube_image[0], &reqs);
    uint32_t type_index = FindMemoryType(memory_properties, reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type_index == UINT32_MAX) {
        std::fprintf(stderr, "No device-local memory type available\n");
        return EXIT_FAILURE;
    }
    VkDeviceSize stride = ((reqs.size + reqs.alignment - 1) / reqs.alignment) * reqs.alignment;

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = stride * FRAMES_IN_FLIGHT,
        .memoryTypeIndex = type_index,
    };
    VK_CHECK(vkAllocateMemory(device, &alloc_info, nullptr, &cube_memory));

    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        VK_CHECK(vkBindImageMemory(device, cube_image[i], cube_memory, i * stride));

        VkImageViewCreateInfo cube_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = cube_image[i],
            .viewType = VK_IMAGE_VIEW_TYPE_CUBE,
            .format = DEPTH_FORMAT,
            .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 6},
        };
        VK_CHECK(vkCreateImageView(device, &cube_info, nullptr, &cube_view[i]));

        for (uint32_t f = 0; f < 6; ++f) {
            VkImageViewCreateInfo face_info = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = cube_image[i],
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = DEPTH_FORMAT,
                .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, f, 1},
            };
            VK_CHECK(vkCreateImageView(device, &face_info, nullptr, &cube_face_view[i * 6 + f]));
        }

        VkDescriptorImageInfo image_desc = {
            .sampler = shadow_sampler,
            .imageView = cube_view[i],
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        VkWriteDescriptorSet write = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = graphics_descriptor_set[i],
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &image_desc,
        };
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
    return 0;
}