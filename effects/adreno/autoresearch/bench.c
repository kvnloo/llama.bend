/* Spliced after Bend 2.0.37's runtime, using its existing foreign-effect ABI.
 * Only argument marshalling/packing touches Env. File I/O, parsing, OpenCL and
 * evidence emission run on io_work's helper; there is no CPU GEMV fallback.
 */
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define ARB_ROWS 576u
#define ARB_COLS 576u
#define ARB_TENSOR_BYTES ((size_t)ARB_ROWS * ARB_COLS * sizeof(float))
#define ARB_SOURCE_MAX 65536u
#define ARB_LOG_MAX 1048576u
#define ARB_SAMPLES_MAX 101u
#define ARB_BATCH_MAX 64u
#define ARB_ID_MAX 1024u
/* Stable protocol: malformed kernels/unsupported workgroups are candidate errors,
 * while unavailable profiling/device/runtime instrumentation remains fatal. */
#define ARB_CANDIDATE_REJECT 65
#define ARB_DRIVER "/vendor/lib64/libOpenCL.so"
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "The Android benchmark requires a little-endian host."
#endif
_Static_assert(sizeof(float) == 4 && sizeof(uint32_t) == 4, "F32 word size");

struct arb_cl_api {
#define API(name) __typeof__(&name) name
  API(clGetPlatformIDs); API(clGetPlatformInfo); API(clGetDeviceIDs);
  API(clGetDeviceInfo); API(clCreateContext); API(clCreateCommandQueue);
  API(clGetCommandQueueInfo); API(clCreateBuffer); API(clCreateProgramWithSource);
  API(clBuildProgram); API(clGetProgramBuildInfo); API(clCreateKernel);
  API(clGetKernelWorkGroupInfo); API(clSetKernelArg); API(clEnqueueWriteBuffer);
  API(clEnqueueNDRangeKernel); API(clEnqueueReadBuffer); API(clGetEventInfo);
  API(clGetEventProfilingInfo); API(clFinish); API(clReleaseEvent);
  API(clReleaseKernel); API(clReleaseProgram); API(clReleaseMemObject);
  API(clReleaseCommandQueue); API(clReleaseContext);
#undef API
};

struct arb_job {
  char *path, *source_path, *params;
  uint64_t path_length, source_path_length, params_length;
  uint32_t pattern, global, local, samples, warmups, batch;
  char error[512];
};

struct arb_sample {
  uint64_t duration_sum, wall_ns;
};

struct arb_identity {
  char text[7][ARB_ID_MAX];
  size_t length[7];
};

static int arb_error(struct arb_job *job, int code, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vsnprintf(job->error, sizeof(job->error), fmt, args);
  va_end(args);
  fprintf(stderr, "%s\n", job->error);
  return code > 0 ? code : EIO;
}

static int arb_parse(struct arb_job *job) {
  if (!job->path || !job->source_path || !job->params)
    return arb_error(job, ENOMEM, "Bench.run: copying arguments");
  if (!job->path_length || !job->source_path_length ||
      job->path_length > 4095 || job->source_path_length > 4095 ||
      memchr(job->path, '\0', (size_t)job->path_length) ||
      memchr(job->source_path, '\0', (size_t)job->source_path_length))
    return arb_error(job, EINVAL, "Bench.run: paths must be nonempty, NUL-free and at most 4095 bytes");
  if (!job->params_length || job->params_length > 65 ||
      memchr(job->params, '\0', (size_t)job->params_length))
    return arb_error(job, EINVAL, "Bench.run: expected six unsigned decimal fields, at most 65 bytes");
  uint32_t fields[6] = {0};
  const unsigned char *cursor = (const unsigned char *)job->params;
  for (unsigned field = 0; field < 6; ++field) {
    if (*cursor < '0' || *cursor > '9')
      return arb_error(job, EINVAL, "Bench.run: field %u requires unsigned decimal digits", field);
    do {
      unsigned digit = *cursor - '0';
      if (fields[field] > (UINT32_MAX - digit) / 10u)
        return arb_error(job, ERANGE, "Bench.run: parameter field %u overflows U32", field);
      fields[field] = fields[field] * 10u + digit;
      ++cursor;
    } while (*cursor >= '0' && *cursor <= '9');
    if (*cursor != (field == 5 ? '\0' : ':'))
      return arb_error(job, EINVAL, "Bench.run: expected pattern:global:local:samples:warmups:batch");
    if (field != 5) ++cursor;
  }
  job->pattern = fields[0]; job->global = fields[1]; job->local = fields[2];
  job->samples = fields[3]; job->warmups = fields[4]; job->batch = fields[5];
  if (job->pattern > 4 || !job->global || job->global > 147456 ||
      (job->local && (job->local > 256 || (job->local & (job->local - 1)) ||
                      job->global % job->local)) ||
      job->samples < 3 || job->samples > ARB_SAMPLES_MAX ||
      !job->warmups || job->warmups > 128 || !job->batch || job->batch > ARB_BATCH_MAX)
    return arb_error(job, ERANGE,
      "Bench.run: bounds: pattern0..4 global1..147456 local0|power-of-two<=256 dividing global samples3..101 warmups1..128 batch1..64");
  return 0;
}

/* Validate the opened descriptor, not a racy prior path check. Nonblocking open
 * also prevents an untrusted FIFO from hanging before fstat can reject it. */
static int arb_read_file(struct arb_job *job, const char *path, int source,
                         unsigned char **data, size_t *length) {
  int fd = open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
  const char *label = source ? "source" : "tensor";
  if (fd < 0) return arb_error(job, errno, "Bench.run: open %s: %s", label, strerror(errno));
  int result = 0;
  struct stat info;
  if (fstat(fd, &info) != 0) {
    result = arb_error(job, errno, "Bench.run: stat %s: %s", label, strerror(errno));
    goto close_file;
  }
  if (!S_ISREG(info.st_mode) || info.st_size < 0 ||
      (source ? (info.st_size == 0 || info.st_size > ARB_SOURCE_MAX)
              : info.st_size != (off_t)ARB_TENSOR_BYTES)) {
    result = arb_error(job, EINVAL, "Bench.run: %s must be regular, %s; size=%jd",
      label, source ? "1..65536 bytes" : "exactly 1327104 bytes", (intmax_t)info.st_size);
    goto close_file;
  }
  *length = (size_t)info.st_size;
  *data = malloc(*length + (source ? 1u : 0u));
  if (!*data) {
    result = arb_error(job, ENOMEM, "Bench.run: allocating %s storage", label);
    goto close_file;
  }
  size_t offset = 0;
  while (offset < *length) {
    ssize_t count = read(fd, *data + offset, *length - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      result = count < 0
        ? arb_error(job, errno, "Bench.run: read %s: %s", label, strerror(errno))
        : arb_error(job, EINVAL, "Bench.run: truncated %s at byte=%zu", label, offset);
      goto close_file;
    }
    offset += (size_t)count;
  }
  unsigned char extra;
  ssize_t count;
  do { count = read(fd, &extra, 1); } while (count < 0 && errno == EINTR);
  if (count != 0) {
    result = count < 0
      ? arb_error(job, errno, "Bench.run: checking %s end: %s", label, strerror(errno))
      : arb_error(job, EINVAL, "Bench.run: %s has trailing bytes", label);
    goto close_file;
  }
  if (source) {
    if (memchr(*data, '\0', *length))
      result = arb_error(job, EINVAL, "Bench.run: kernel source contains an embedded NUL");
    (*data)[*length] = '\0';
  } else {
    const float *weights = (const float *)*data;
    for (size_t i = 0; i < ARB_ROWS * ARB_COLS; ++i) {
      if (!isfinite(weights[i])) {
        result = arb_error(job, EDOM, "Bench.run: nonfinite tensor element=%zu", i);
        break;
      }
    }
  }
close_file:
  if (close(fd) != 0)
    result = arb_error(job, errno, "Bench.run: close %s: %s", label, strerror(errno));
  return result;
}

/* Preserve UTF-8 identities as ASCII JSON escapes. Invalid log bytes are escaped
 * individually; no driver-provided quote/control byte can inject evidence. */
static void arb_json(FILE *out, const char *text, size_t length) {
  fputc('"', out);
  for (size_t i = 0; i < length; ++i) {
    unsigned char byte = (unsigned char)text[i];
    if (byte == '"' || byte == '\\') { fputc('\\', out); fputc(byte, out); }
    else if (byte < 32 || byte == 127) fprintf(out, "\\u%04x", (unsigned)byte);
    else if (byte < 128) fputc(byte, out);
    else {
      unsigned count = byte >= 0xc2 && byte <= 0xdf ? 2
        : byte >= 0xe0 && byte <= 0xef ? 3 : byte >= 0xf0 && byte <= 0xf4 ? 4 : 0;
      uint32_t cp = count ? byte & (0x7fu >> count) : 0;
      int valid = count && count <= length - i;
      for (unsigned j = 1; valid && j < count; ++j) {
        unsigned char next = (unsigned char)text[i + j];
        if ((next & 0xc0u) != 0x80u) valid = 0;
        else cp = (cp << 6) | (next & 0x3fu);
      }
      if (!valid || cp < (count == 2 ? 0x80u : count == 3 ? 0x800u : 0x10000u) ||
          cp > 0x10ffffu || (cp >= 0xd800u && cp <= 0xdfffu)) {
        fprintf(out, "\\u%04x", (unsigned)byte);
      } else {
        if (cp <= 0xffffu) fprintf(out, "\\u%04x", (unsigned)cp);
        else {
          cp -= 0x10000u;
          fprintf(out, "\\u%04x\\u%04x", (unsigned)(0xd800u + (cp >> 10)),
                  (unsigned)(0xdc00u + (cp & 0x3ffu)));
        }
        i += count - 1;
      }
    }
  }
  fputc('"', out);
}

static int arb_identity_read(struct arb_job *job, struct arb_cl_api *api,
                             cl_platform_id platform, cl_device_id device,
                             struct arb_identity *identity) {
  const cl_uint fields[7] = {CL_PLATFORM_NAME, CL_PLATFORM_VENDOR, CL_PLATFORM_VERSION,
    CL_DEVICE_VENDOR, CL_DEVICE_NAME, CL_DEVICE_VERSION, CL_DRIVER_VERSION};
  for (unsigned i = 0; i < 7; ++i) {
    size_t length = 0, actual = 0;
    cl_int status = i < 3
      ? api->clGetPlatformInfo(platform, fields[i], 0, NULL, &length)
      : api->clGetDeviceInfo(device, fields[i], 0, NULL, &length);
    if (status != CL_SUCCESS || !length || length > ARB_ID_MAX)
      return arb_error(job, EIO, "Bench.run: identity field=%u size=%zu status=%d", i, length, status);
    status = i < 3
      ? api->clGetPlatformInfo(platform, fields[i], length, identity->text[i], &actual)
      : api->clGetDeviceInfo(device, fields[i], length, identity->text[i], &actual);
    if (status != CL_SUCCESS || actual != length || identity->text[i][length - 1] != '\0' ||
        memchr(identity->text[i], '\0', length - 1))
      return arb_error(job, EIO, "Bench.run: invalid identity field=%u size=%zu actual=%zu status=%d",
                       i, length, actual, status);
    identity->length[i] = length - 1;
  }
  return 0;
}

static int arb_build_record(struct arb_job *job, struct arb_cl_api *api,
                            cl_program program, cl_device_id device, cl_int built) {
  cl_build_status state = CL_BUILD_NONE;
  cl_int status = api->clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS,
                                            sizeof(state), &state, NULL);
  fprintf(stderr, "AUTORESEARCH_BUILD status=%d build_status=%d build_status_query=%d options=-cl-std=CL1.2\n",
          built, state, status);
  if (status != CL_SUCCESS)
    return arb_error(job, EIO, "Bench.run: querying build status=%d", status);
  size_t length = 0, actual = 0;
  status = api->clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &length);
  if (status != CL_SUCCESS || length > ARB_LOG_MAX)
    return arb_error(job, EIO, "Bench.run: build log query status=%d bytes=%zu limit=%u",
                     status, length, ARB_LOG_MAX);
  char *log = malloc(length ? length : 1);
  if (!log) return arb_error(job, ENOMEM, "Bench.run: allocating build log");
  if (length) {
    status = api->clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, length, log, &actual);
    if (status != CL_SUCCESS || actual != length || log[length - 1] != '\0') {
      free(log);
      return arb_error(job, EIO, "Bench.run: reading build log status=%d bytes=%zu actual=%zu",
                       status, length, actual);
    }
  }
  fputs("AUTORESEARCH_BUILD_LOG ", stderr);
  arb_json(stderr, log, length ? length - 1 : 0);
  fputc('\n', stderr);
  free(log);
  if (fflush(stderr) != 0 || ferror(stderr))
    return arb_error(job, EIO, "Bench.run: recording build evidence failed");
  if (built != CL_SUCCESS || state != CL_BUILD_SUCCESS)
    return arb_error(job, (built == CL_BUILD_PROGRAM_FAILURE || state == CL_BUILD_ERROR)
      ? ARB_CANDIDATE_REJECT : EIO, "Bench.run: kernel build API status=%d build_status=%d", built, state);
  return 0;
}

static int arb_clock(struct arb_job *job, uint64_t *value) {
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
    return arb_error(job, errno, "Bench.run: monotonic wall clock: %s", strerror(errno));
  if (now.tv_sec < 0 || now.tv_nsec < 0 || now.tv_nsec >= 1000000000L ||
      (uint64_t)now.tv_sec > (UINT64_MAX - (uint64_t)now.tv_nsec) / UINT64_C(1000000000))
    return arb_error(job, ERANGE, "Bench.run: invalid monotonic wall timestamp");
  *value = (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
  return 0;
}

static int arb_event(struct arb_job *job, struct arb_cl_api *api, cl_event event,
                      uint64_t *duration) {
  cl_int execution = -1;
  cl_int status = api->clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                     sizeof(execution), &execution, NULL);
  if (status != CL_SUCCESS || execution != CL_COMPLETE)
    return arb_error(job, EIO, "Bench.run: event status query=%d execution=%d", status, execution);
  cl_ulong start = 0, end = 0;
  status = api->clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
  if (status != CL_SUCCESS)
    return arb_error(job, EIO, "Bench.run: event START profiling status=%d", status);
  status = api->clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
  if (status != CL_SUCCESS)
    return arb_error(job, EIO, "Bench.run: event END profiling status=%d", status);
  if (!start || !end || end <= start)
    return arb_error(job, EIO, "Bench.run: invalid GPU timestamps START=%" PRIu64 " END=%" PRIu64,
                     (uint64_t)start, (uint64_t)end);
  *duration = (uint64_t)(end - start);
  return 0;
}

static int arb_outputs(struct arb_job *job, const uint32_t *values, unsigned sample) {
  for (uint32_t row = 0; row < ARB_ROWS; ++row) {
    if ((values[row] & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000))
      return arb_error(job, EDOM, "Bench.run: nonfinite or unwritten output sample=%u row=%" PRIu32,
                       sample, row);
  }
  return 0;
}

static int arb_bench(struct arb_job *job) {
  struct arb_cl_api api = {0};
  struct arb_identity identity = {0};
  struct arb_sample samples[ARB_SAMPLES_MAX] = {{0}};
  cl_platform_id platforms[16];
  cl_device_id device = NULL;
  cl_platform_id platform = NULL;
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_mem weight_buffer = NULL, input_buffer = NULL, output_buffer = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_event events[ARB_BATCH_MAX] = {0};
  unsigned char *weights = NULL, *source = NULL;
  size_t weight_length = 0, source_length = 0;
  float input[ARB_COLS];
  uint32_t sentinel[ARB_ROWS], values[ARB_ROWS], baseline[ARB_ROWS];
  void *library = NULL;
  cl_int status = CL_SUCCESS;
  int result = arb_parse(job);
  if (result != 0) return result;
  result = arb_read_file(job, job->path, 0, &weights, &weight_length);
  if (result != 0) goto cleanup;
  result = arb_read_file(job, job->source_path, 1, &source, &source_length);
  if (result != 0) goto cleanup;
  for (uint32_t i = 0; i < ARB_COLS; ++i) {
    switch (job->pattern) {
      case 0: input[i] = 1.0f; break;
      case 1: input[i] = i % 2u ? -1.0f : 1.0f; break;
      case 2: input[i] = i == ARB_COLS - 1 ? 1.0f : 0.0f; break;
      case 3: input[i] = (float)((int32_t)(i % 29u) - 14) * 0.0625f; break;
      default: input[i] = (float)((int32_t)((i * 73u + 19u) % 257u) - 128) * 0.0078125f; break;
    }
    sentinel[i] = UINT32_C(0x7fc00000);
  }
  library = dlopen(ARB_DRIVER, RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    result = arb_error(job, ENOTSUP, "Bench.run: vendor OpenCL unavailable: %s", dlerror());
    goto cleanup;
  }
#define LOAD(name) do { \
    void *symbol = dlsym(library, #name); \
    if (!symbol) { \
      result = arb_error(job, ENOTSUP, "Bench.run: missing OpenCL entry %s", #name); \
      goto cleanup; \
    } \
    _Static_assert(sizeof(api.name) == sizeof(symbol), "Android LP64 function pointer ABI"); \
    memcpy(&api.name, &symbol, sizeof(symbol)); \
  } while (0)
  LOAD(clGetPlatformIDs); LOAD(clGetPlatformInfo); LOAD(clGetDeviceIDs);
  LOAD(clGetDeviceInfo); LOAD(clCreateContext); LOAD(clCreateCommandQueue);
  LOAD(clGetCommandQueueInfo); LOAD(clCreateBuffer); LOAD(clCreateProgramWithSource);
  LOAD(clBuildProgram); LOAD(clGetProgramBuildInfo); LOAD(clCreateKernel);
  LOAD(clGetKernelWorkGroupInfo); LOAD(clSetKernelArg); LOAD(clEnqueueWriteBuffer);
  LOAD(clEnqueueNDRangeKernel); LOAD(clEnqueueReadBuffer); LOAD(clGetEventInfo);
  LOAD(clGetEventProfilingInfo); LOAD(clFinish); LOAD(clReleaseEvent);
  LOAD(clReleaseKernel); LOAD(clReleaseProgram); LOAD(clReleaseMemObject);
  LOAD(clReleaseCommandQueue); LOAD(clReleaseContext);
#undef LOAD
#define CHECK(call) do { \
    status = (call); \
    if (status != CL_SUCCESS) { \
      result = arb_error(job, EIO, "Bench.run: %s status=%d", #call, status); \
      goto cleanup; \
    } \
  } while (0)
#define REQUIRE(object) do { \
    if (!(object)) { \
      result = arb_error(job, EIO, "Bench.run: %s returned NULL", #object); \
      goto cleanup; \
    } \
  } while (0)
  cl_uint platform_count = 0;
  CHECK(api.clGetPlatformIDs(0, NULL, &platform_count));
  if (!platform_count || platform_count > 16) {
    result = arb_error(job, ENOTSUP, "Bench.run: unsupported platform count=%u", platform_count);
    goto cleanup;
  }
  CHECK(api.clGetPlatformIDs(platform_count, platforms, NULL));
  for (cl_uint p = 0; p < platform_count && !device; ++p) {
    cl_uint count = 0;
    status = api.clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, NULL, &count);
    if (status == CL_DEVICE_NOT_FOUND) continue;
    if (status != CL_SUCCESS || count > 16) {
      result = arb_error(job, ENOTSUP, "Bench.run: GPU enumeration status=%d count=%u", status, count);
      goto cleanup;
    }
    if (count) {
      cl_device_id devices[16];
      CHECK(api.clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, count, devices, NULL));
      device = devices[0]; platform = platforms[p];
    }
  }
  if (!device) {
    result = arb_error(job, ENOTSUP, "Bench.run: no OpenCL GPU; CPU fallback forbidden");
    goto cleanup;
  }
  cl_device_type type = 0;
  cl_bool available = CL_FALSE, compiler = CL_FALSE, little_endian = CL_FALSE;
  CHECK(api.clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL));
  CHECK(api.clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(available), &available, NULL));
  CHECK(api.clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE, sizeof(compiler), &compiler, NULL));
  CHECK(api.clGetDeviceInfo(device, CL_DEVICE_ENDIAN_LITTLE, sizeof(little_endian), &little_endian, NULL));
  if (type != CL_DEVICE_TYPE_GPU || !available || !compiler || !little_endian) {
    result = arb_error(job, ENOTSUP,
      "Bench.run: unsupported GPU type=%" PRIu64 " available=%u compiler=%u little_endian=%u",
      (uint64_t)type, available, compiler, little_endian);
    goto cleanup;
  }
  result = arb_identity_read(job, &api, platform, device, &identity);
  if (result != 0) goto cleanup;
  context = api.clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  CHECK(status); REQUIRE(context);
  queue = api.clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  CHECK(status); REQUIRE(queue);
  cl_command_queue_properties properties = 0;
  CHECK(api.clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES, sizeof(properties), &properties, NULL));
  if (!(properties & CL_QUEUE_PROFILING_ENABLE) || (properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)) {
    result = arb_error(job, ENOTSUP, "Bench.run: an in-order profiling queue is required; properties=%" PRIu64,
                       (uint64_t)properties);
    goto cleanup;
  }
  weight_buffer = api.clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     weight_length, weights, &status);
  CHECK(status); REQUIRE(weight_buffer);
  input_buffer = api.clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(input), input, &status);
  CHECK(status); REQUIRE(input_buffer);
  output_buffer = api.clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     sizeof(sentinel), sentinel, &status);
  CHECK(status); REQUIRE(output_buffer);
  const char *source_ptr = (const char *)source;
  program = api.clCreateProgramWithSource(context, 1, &source_ptr, &source_length, &status);
  CHECK(status); REQUIRE(program);
  status = api.clBuildProgram(program, 1, &device, "-cl-std=CL1.2", NULL, NULL);
  result = arb_build_record(job, &api, program, device, status);
  if (result != 0) goto cleanup;
  kernel = api.clCreateKernel(program, "gemv", &status);
  CHECK(status); REQUIRE(kernel);
  CHECK(api.clSetKernelArg(kernel, 0, sizeof(weight_buffer), &weight_buffer));
  CHECK(api.clSetKernelArg(kernel, 1, sizeof(input_buffer), &input_buffer));
  CHECK(api.clSetKernelArg(kernel, 2, sizeof(output_buffer), &output_buffer));
  size_t kernel_limit = 0, device_limit = 0, required[3] = {0};
  cl_ulong kernel_memory = 0, device_memory = 0;
  CHECK(api.clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                    sizeof(kernel_limit), &kernel_limit, NULL));
  CHECK(api.clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
                                    sizeof(required), required, NULL));
  CHECK(api.clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_LOCAL_MEM_SIZE,
                                    sizeof(kernel_memory), &kernel_memory, NULL));
  CHECK(api.clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(device_limit), &device_limit, NULL));
  CHECK(api.clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(device_memory), &device_memory, NULL));
  if ((job->local && (job->local > kernel_limit || job->local > device_limit)) ||
      kernel_memory > device_memory ||
      (required[0] && (!job->local || required[0] != job->local || required[1] != 1 || required[2] != 1))) {
    result = arb_error(job, ARB_CANDIDATE_REJECT, "Bench.run: unsupported local size or kernel local-memory requirement");
    goto cleanup;
  }
  size_t global = job->global, local = job->local;
  const size_t *local_ptr = local ? &local : NULL;
  /* All host storage, buffers and argument state already exist. OpenCL creates
   * event objects, but their bounded handle array is reused, never allocated in
   * a launch loop. A blocking reset is deliberately outside each wall interval. */
  for (uint32_t iteration = 0; iteration < job->warmups + job->samples; ++iteration) {
    int timed = iteration >= job->warmups;
    uint32_t index = timed ? iteration - job->warmups : iteration;
    uint32_t count = timed ? job->batch : 1;
    CHECK(api.clEnqueueWriteBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(sentinel),
                                   sentinel, 0, NULL, NULL));
    uint64_t start = 0, end = 0, sum = 0;
    if (timed) {
      result = arb_clock(job, &start);
      if (result != 0) goto cleanup;
    }
    for (uint32_t event = 0; event < count; ++event) {
      CHECK(api.clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, local_ptr,
                                      0, NULL, &events[event]));
      REQUIRE(events[event]);
    }
    CHECK(api.clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(values),
                                  values, count, events, NULL));
    if (timed) {
      result = arb_clock(job, &end);
      if (result != 0) goto cleanup;
      if (end <= start) {
        result = arb_error(job, EIO, "Bench.run: nonpositive wall duration sample=%u", index);
        goto cleanup;
      }
    }
    for (uint32_t event = 0; event < count; ++event) {
      uint64_t duration = 0;
      result = arb_event(job, &api, events[event], &duration);
      if (result != 0) goto cleanup;
      if (sum > UINT64_MAX - duration) {
        result = arb_error(job, ERANGE, "Bench.run: GPU duration sum overflow sample=%u", index);
        goto cleanup;
      }
      sum += duration;
      status = api.clReleaseEvent(events[event]);
      events[event] = NULL;
      if (status != CL_SUCCESS) {
        result = arb_error(job, EIO, "Bench.run: cleanup clReleaseEvent sample=%u event=%u status=%d",
                           index, event, status);
        goto cleanup;
      }
    }
    result = arb_outputs(job, values, index);
    if (result != 0) goto cleanup;
    if (timed) {
      if (index == 0) memcpy(baseline, values, sizeof(baseline));
      else if (memcmp(baseline, values, sizeof(baseline)) != 0) {
        result = arb_error(job, EDOM, "Bench.run: nondeterministic GPU output sample=%u", index);
        goto cleanup;
      }
      samples[index].duration_sum = sum;
      samples[index].wall_ns = end - start;
    }
  }
  result = 0;
cleanup:
  /* Complete queued work before any associated host storage is freed, even on
   * error. Cleanup failures are evidence and always make the IO result nonzero. */
  if (queue) {
    cl_int finished = api.clFinish(queue);
    if (finished != CL_SUCCESS)
      result = arb_error(job, EIO, "Bench.run: cleanup clFinish status=%d", finished);
  }
#define RELEASE(object, method) do { \
    if (object) { \
      cl_int released = api.method(object); \
      if (released != CL_SUCCESS) \
        result = arb_error(job, EIO, "Bench.run: cleanup %s status=%d", #method, released); \
    } \
  } while (0)
  for (uint32_t i = 0; i < ARB_BATCH_MAX; ++i) RELEASE(events[i], clReleaseEvent);
  RELEASE(kernel, clReleaseKernel); RELEASE(program, clReleaseProgram);
  RELEASE(output_buffer, clReleaseMemObject); RELEASE(input_buffer, clReleaseMemObject);
  RELEASE(weight_buffer, clReleaseMemObject); RELEASE(queue, clReleaseCommandQueue);
  RELEASE(context, clReleaseContext);
#undef RELEASE
#undef REQUIRE
#undef CHECK
  free(source); free(weights);
  if (library && dlclose(library) != 0)
    result = arb_error(job, EIO, "Bench.run: cleanup dlclose: %s", dlerror());
  /* Emit nothing that could look successful until all samples AND cleanup pass.
   * These words are raw GPU evidence, never numerical acceptance or a PASS. */
  if (result == 0) {
    printf("AUTORESEARCH_META {\"pattern\":%" PRIu32 ",\"rows\":576,\"cols\":576,"
           "\"device_type\":4,\"gpu_required\":true,\"cpu_fallback\":false,\"profiling\":true,"
           "\"global_size\":%" PRIu32 ",\"local_size\":%" PRIu32 ",\"samples\":%" PRIu32 ","
           "\"warmups\":%" PRIu32 ",\"batch\":%" PRIu32,
           job->pattern, job->global, job->local, job->samples, job->warmups, job->batch);
    const char *labels[7] = {"platform", "platform_vendor", "platform_version",
      "vendor", "device", "device_version", "driver_version"};
    for (unsigned i = 0; i < 7; ++i) {
      printf(",\"%s\":", labels[i]);
      arb_json(stdout, identity.text[i], identity.length[i]);
    }
    fputs("}\n", stdout);
    for (uint32_t i = 0; i < job->samples; ++i)
      printf("AUTORESEARCH_SAMPLE {\"index\":%" PRIu32 ",\"gpu_ns\":%.9f,\"wall_ns\":%" PRIu64
             ",\"events\":%" PRIu32 ",\"completed\":true}\n", i,
             (double)samples[i].duration_sum / (double)job->batch, samples[i].wall_ns, job->batch);
    printf("AUTORESEARCH_BITS pattern=%" PRIu32 " values=", job->pattern);
    for (uint32_t row = 0; row < ARB_ROWS; ++row)
      printf("%s%" PRIu32, row == 0 ? "" : " ", baseline[row]);
    fputc('\n', stdout);
    if (fflush(stdout) != 0 || ferror(stdout))
      result = arb_error(job, EIO, "Bench.run: writing profiling evidence failed");
  }
  return result;
}

static void arb_call(IoWork *work) {
  work->code = arb_bench((struct arb_job *)work->data);
}

static Term arb_pack(Env env, IoWork *work) {
  struct arb_job *job = (struct arb_job *)work->data;
  Term result = work->code == 0
    ? io_done(env, term_pak(CID(Unit), 0))
    : io_fail(env, (u32)work->code, job->error);
  /* io_cstr owns copies, and io_fail copies its message before we free state.
   * Runtime owns work/args. Borrowed platform/device IDs are never released. */
  free(job->params); free(job->source_path); free(job->path); free(job);
  work->data = NULL;
  return result;
}

static Term arb_run(Env env, Term *args, IoWork *work) {
  struct arb_job *job = calloc(1, sizeof(*job));
  if (!job) return io_fail(env, ENOMEM, "Bench.run: allocating helper state");
  job->path = io_cstr(env, args[0], &job->path_length);
  job->source_path = io_cstr(env, args[1], &job->source_path_length);
  job->params = io_cstr(env, args[2], &job->params_length);
  work->data = (char *)job;
  return io_work(work, arb_call, arb_pack);
}

static void __attribute__((constructor)) arb_use(void) {
  io_eff(CID(Bench.run), arb_run);
}
