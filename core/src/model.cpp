/*
 * Omni Core - Model Loading and Management
 *
 * Loads .omni format models with mmap for zero-copy access
 */

#include "../include/omni.h"
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Magic and constants
#define OMNI_MAGIC_VALUE 0x494E4D4F // "OMNI" in little-endian
#define OMNI_VERSION 1
#define OMNI_ALIGNMENT 64 // Apple Silicon cache line

// Simple JSON parser for metadata
namespace {
std::unordered_map<std::string, std::string> parse_json_flat(const char *json,
                                                             size_t len) {
  std::unordered_map<std::string, std::string> result;
  std::string key, value;
  bool in_key = false, in_value = false;
  bool in_string = false;

  for (size_t i = 0; i < len; i++) {
    char c = json[i];

    if (c == '"') {
      if (!in_string) {
        in_string = true;
        if (!in_key && !in_value) {
          in_key = true;
          key.clear();
        }
      } else {
        in_string = false;
        if (in_key) {
          in_key = false;
        } else if (in_value) {
          result[key] = value;
          in_value = false;
        }
      }
    } else if (c == ':' && !in_string) {
      in_value = true;
      value.clear();
    } else if (c == ',' && !in_string) {
      if (!value.empty() && !key.empty()) {
        result[key] = value;
        value.clear();
      }
    } else if (in_string) {
      if (in_key)
        key += c;
      else if (in_value)
        value += c;
    } else if (in_value && (isdigit(c) || c == '.' || c == '-')) {
      value += c;
    }
  }

  if (!key.empty() && !value.empty()) {
    result[key] = value;
  }

  return result;
}

int parse_int(const std::string &s, int def = 0) {
  try {
    return std::stoi(s);
  } catch (...) {
    return def;
  }
}

float parse_float(const std::string &s, float def = 0.0f) {
  try {
    return std::stof(s);
  } catch (...) {
    return def;
  }
}
} // namespace

// Load model from file
omni_model *omni_load_model(const char *path) {
  // Open file
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    std::cerr << "Failed to open file: " << path << std::endl;
    return nullptr;
  }

  // Get file size
  struct stat st;
  if (fstat(fd, &st) < 0) {
    close(fd);
    std::cerr << "Failed to stat file" << std::endl;
    return nullptr;
  }
  size_t file_size = st.st_size;

  // Memory map the file
  void *mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped == MAP_FAILED) {
    close(fd);
    std::cerr << "Failed to mmap file" << std::endl;
    return nullptr;
  }

  // Create model structure
  omni_model *model = new omni_model();
  model->fd = fd;
  model->mapped_data = mapped;
  model->mapped_size = file_size;

  // Parse header
  uint8_t *ptr = (uint8_t *)mapped;

  // Magic (4 bytes)
  uint32_t magic;
  memcpy(&magic, ptr, 4);
  ptr += 4;
  if (magic != OMNI_MAGIC_VALUE) {
    std::cerr << "Invalid magic number" << std::endl;
    omni_free_model(model);
    return nullptr;
  }

  // Version (4 bytes)
  memcpy(&model->version, ptr, 4);
  ptr += 4;

  // Architecture (4 bytes)
  uint32_t arch;
  memcpy(&arch, ptr, 4);
  ptr += 4;
  model->architecture = arch;

  // Alignment (4 bytes)
  memcpy(&model->alignment, ptr, 4);
  ptr += 4;

  // Number of tensors (8 bytes)
  memcpy(&model->n_tensors, ptr, 8);
  ptr += 8;

  // Number of KV pairs (8 bytes)
  memcpy(&model->n_kv, ptr, 8);
  ptr += 8;

  // Metadata JSON length (8 bytes)
  uint64_t json_len;
  memcpy(&json_len, ptr, 8);
  ptr += 8;

  // Parse metadata JSON
  model->metadata = parse_json_flat((const char *)ptr, json_len);
  ptr += json_len;

  // Align to boundary
  size_t current_pos = ptr - (uint8_t *)mapped;
  size_t aligned_pos =
      (current_pos + model->alignment - 1) & ~(model->alignment - 1);
  ptr = (uint8_t *)mapped + aligned_pos;

  // Parse tensor info
  for (uint64_t i = 0; i < model->n_tensors; i++) {
    omni_tensor tensor;

    // Name length (2 bytes)
    uint16_t name_len;
    memcpy(&name_len, ptr, 2);
    ptr += 2;

    // Name
    tensor.name = std::string((const char *)ptr, name_len);
    ptr += name_len;

    // Number of dimensions (1 byte)
    uint8_t ndim;
    memcpy(&ndim, ptr, 1);
    ptr += 1;

    // Dimensions
    tensor.shape.resize(ndim);
    for (uint8_t j = 0; j < ndim; j++) {
      memcpy(&tensor.shape[j], ptr, 8);
      ptr += 8;
    }

    // Data type (1 byte)
    uint8_t dtype;
    memcpy(&dtype, ptr, 1);
    ptr += 1;
    tensor.dtype = static_cast<omni_dtype>(dtype);

    // Offset (8 bytes)
    memcpy(&tensor.offset, ptr, 8);
    ptr += 8;

    // Size (8 bytes)
    memcpy(&tensor.size, ptr, 8);
    ptr += 8;

    model->tensors[tensor.name] = tensor;
  }

  // Calculate data section offset
  size_t header_size = ptr - (uint8_t *)mapped;
  model->data_offset =
      (header_size + model->alignment - 1) & ~(model->alignment - 1);

  // Set tensor data pointers
  for (auto &[name, tensor] : model->tensors) {
    tensor.data = (uint8_t *)mapped + model->data_offset + tensor.offset;
  }

  // Extract common config
  model->hidden_size = parse_int(model->metadata["hidden_size"], 2048);
  model->num_layers = parse_int(model->metadata["num_layers"], 16);
  model->num_heads = parse_int(model->metadata["num_heads"], 32);
  model->num_kv_heads = parse_int(model->metadata["num_kv_heads"], 8);
  model->vocab_size = parse_int(model->metadata["vocab_size"], 65536);
  model->max_seq_len = parse_int(model->metadata["max_seq_len"], 128000);
  model->rope_theta = parse_float(model->metadata["rope_theta"], 1000000.0f);
  model->norm_eps = 1e-5f;

  return model;
}

// Free model
void omni_free_model(omni_model *model) {
  if (!model)
    return;

  if (model->mapped_data) {
    munmap(model->mapped_data, model->mapped_size);
  }
  if (model->fd >= 0) {
    close(model->fd);
  }

  delete model;
}

// Get tensor by name
const omni_tensor *omni_get_tensor(omni_model *model, const char *name) {
  auto it = model->tensors.find(name);
  if (it != model->tensors.end()) {
    return &it->second;
  }
  return nullptr;
}

// Get raw tensor data pointer (no dequantization)
const void *omni_get_tensor_data(omni_model *model, const char *name) {
  const omni_tensor *tensor = omni_get_tensor(model, name);
  return tensor ? tensor->data : nullptr;
}

// Print model info
void omni_print_info(omni_model *model) {
  std::cout << "\n========================================\n";
  std::cout << "Omni Model\n";
  std::cout << "========================================\n";
  std::cout << "Version:      " << model->version << "\n";
  std::cout << "Architecture: " << model->architecture << "\n";
  std::cout << "Tensors:      " << model->n_tensors << "\n";
  std::cout << "Alignment:    " << model->alignment << " bytes\n";
  std::cout << "\nConfig:\n";
  std::cout << "  hidden_size:  " << model->hidden_size << "\n";
  std::cout << "  num_layers:   " << model->num_layers << "\n";
  std::cout << "  num_heads:    " << model->num_heads << "\n";
  std::cout << "  num_kv_heads: " << model->num_kv_heads << "\n";
  std::cout << "  vocab_size:   " << model->vocab_size << "\n";
  std::cout << "  rope_theta:   " << model->rope_theta << "\n";
  std::cout << "========================================\n\n";
}

// Get metadata value
const char *omni_get_metadata(omni_model *model, const char *key) {
  auto it = model->metadata.find(key);
  if (it != model->metadata.end()) {
    return it->second.c_str();
  }
  return nullptr;
}
