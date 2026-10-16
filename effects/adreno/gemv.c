/* Spliced after Bend 2.0.37's C runtime; uses its documented foreign-effect ABI.
 * OpenCL declarations come from external Khronos headers, never vendor binaries.
 * All GPU/file work runs on io_work's helper, with no Env or Bend heap access.
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
#include <unistd.h>

#define ADRENO_ROWS 576u
#define ADRENO_COLS 576u
#define ADRENO_BYTES ((size_t)ADRENO_ROWS * ADRENO_COLS * sizeof(float))
#define ADRENO_DRIVER "/vendor/lib64/libOpenCL.so"
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "The fixed Android example requires a little-endian host."
#endif
_Static_assert(sizeof(float) == 4 && sizeof(uint32_t) == 4, "F32 word size");

struct adreno_cl_api {
#define API(name) __typeof__(&name) name
  API(clGetPlatformIDs); API(clGetPlatformInfo); API(clGetDeviceIDs);
  API(clGetDeviceInfo); API(clCreateContext); API(clCreateCommandQueue);
  API(clCreateBuffer); API(clCreateProgramWithSource); API(clBuildProgram);
  API(clGetProgramBuildInfo); API(clCreateKernel); API(clGetKernelWorkGroupInfo); API(clSetKernelArg);
  API(clEnqueueNDRangeKernel); API(clEnqueueReadBuffer); API(clGetEventInfo);
  API(clFinish); API(clReleaseEvent); API(clReleaseKernel); API(clReleaseProgram);
  API(clReleaseMemObject); API(clReleaseCommandQueue); API(clReleaseContext);
#undef API
};

struct adreno_gemv_work {
  char *path;
  uint32_t pattern;
  char error[512];
};

static int adreno_error(struct adreno_gemv_work *job, int code, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vsnprintf(job->error, sizeof(job->error), fmt, args);
  va_end(args);
  fprintf(stderr, "%s\n", job->error);
  return code;
}

static int adreno_read_weights(struct adreno_gemv_work *job, float *weights) {
  /* O_NONBLOCK prevents a supplied FIFO from blocking even before fstat rejects it.
   * The open descriptor, not a path checked earlier, determines size and contents. */
  int fd = open(job->path, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
  if (fd < 0) return adreno_error(job, errno, "Adreno.gemv: open tensor: %s", strerror(errno));
  int result = 0;
  struct stat info;
  if (fstat(fd, &info) != 0) {
    result = adreno_error(job, errno, "Adreno.gemv: stat tensor: %s", strerror(errno));
    goto close_file;
  }
  if (!S_ISREG(info.st_mode) || info.st_size != (off_t)ADRENO_BYTES) {
    result = adreno_error(job, EINVAL,
      "Adreno.gemv: expected regular 576x576 F32 tensor of %zu bytes; got size=%jd",
      ADRENO_BYTES, (intmax_t)info.st_size);
    goto close_file;
  }
  size_t offset = 0;
  while (offset < ADRENO_BYTES) {
    ssize_t count = read(fd, (unsigned char *)weights + offset, ADRENO_BYTES - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      result = count < 0
        ? adreno_error(job, errno, "Adreno.gemv: read tensor: %s", strerror(errno))
        : adreno_error(job, EINVAL, "Adreno.gemv: truncated tensor at byte=%zu", offset);
      goto close_file;
    }
    offset += (size_t)count;
  }
  unsigned char extra;
  ssize_t count;
  do { count = read(fd, &extra, 1); } while (count < 0 && errno == EINTR);
  if (count != 0) {
    result = count < 0
      ? adreno_error(job, errno, "Adreno.gemv: checking tensor end: %s", strerror(errno))
      : adreno_error(job, EINVAL, "Adreno.gemv: tensor has trailing bytes");
    goto close_file;
  }
  for (size_t i = 0; i < ADRENO_ROWS * ADRENO_COLS; ++i) {
    if (!isfinite(weights[i])) {
      result = adreno_error(job, EDOM, "Adreno.gemv: nonfinite tensor element=%zu", i);
      goto close_file;
    }
  }
close_file:
  if (close(fd) != 0 && result == 0)
    result = adreno_error(job, errno, "Adreno.gemv: close tensor: %s", strerror(errno));
  return result;
}

static int adreno_gemv(struct adreno_gemv_work *job) {
  /* Exact trusted kernels.generate cooperative(lanes=32, local_size=256, unroll=8).
   * Source SHA256: 5ec98c557d421017dffe0e02f06c18d709d0b5a81145dc4bddbde9f5c13fcb0e.
   * 72 groups each cover eight rows: the fixed launch must not be rounded up.
   * Reduction order differs from TA.dotr; acceptance remains the independent
   * all-row Bend oracle comparison, not a claim of bitwise CPU equivalence. */
  static const char source[] =
    "#pragma OPENCL FP_CONTRACT OFF\n"
    "__attribute__((reqd_work_group_size(256, 1, 1)))\n"
    "__kernel void gemv(__global const float *w, __global const float *x,\n"
    "                   __global float *y) {\n"
    "  uint lid = (uint)get_local_id(0);\n"
    "  uint lane = lid % 32u;\n"
    "  uint row = (uint)get_group_id(0) * 8u + lid / 32u;\n"
    "  uint base = row * 576u;\n"
    "  __local float partial[256];\n"
    "  float sum = 0.0f;\n"
    "  for (uint k = lane; k < 512u; k += 256u) {\n"
    "    sum = sum + x[k + 0u] * w[base + k + 0u];\n"
    "    sum = sum + x[k + 32u] * w[base + k + 32u];\n"
    "    sum = sum + x[k + 64u] * w[base + k + 64u];\n"
    "    sum = sum + x[k + 96u] * w[base + k + 96u];\n"
    "    sum = sum + x[k + 128u] * w[base + k + 128u];\n"
    "    sum = sum + x[k + 160u] * w[base + k + 160u];\n"
    "    sum = sum + x[k + 192u] * w[base + k + 192u];\n"
    "    sum = sum + x[k + 224u] * w[base + k + 224u];\n"
    "  }\n"
    "  if (lane + 512u < 576u) {\n"
    "    sum = sum + x[lane + 512u] * w[base + lane + 512u];\n"
    "  }\n"
    "  if (lane + 544u < 576u) {\n"
    "    sum = sum + x[lane + 544u] * w[base + lane + 544u];\n"
    "  }\n"
    "  partial[lid] = sum;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 16u) {\n"
    "    partial[lid] = partial[lid] + partial[lid + 16u];\n"
    "  }\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 8u) {\n"
    "    partial[lid] = partial[lid] + partial[lid + 8u];\n"
    "  }\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 4u) {\n"
    "    partial[lid] = partial[lid] + partial[lid + 4u];\n"
    "  }\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 2u) {\n"
    "    partial[lid] = partial[lid] + partial[lid + 2u];\n"
    "  }\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane == 0u) y[row] = partial[lid] + partial[lid + 1u];\n"
    "}\n";
  struct adreno_cl_api api = {0};
  cl_platform_id platforms[16], platform = NULL;
  cl_device_id device = NULL;
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_mem weight_buffer = NULL, input_buffer = NULL, output_buffer = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_event event = NULL;
  void *library = NULL;
  float *weights = NULL;
  float input[ADRENO_COLS];
  uint32_t values[ADRENO_ROWS];
  cl_int status = CL_SUCCESS;
  int result = EIO;

  if (job->pattern > 2)
    return adreno_error(job, EINVAL, "Adreno.gemv: pattern must be 0, 1 or 2");
  weights = malloc(ADRENO_BYTES);
  if (!weights) return adreno_error(job, ENOMEM, "Adreno.gemv: allocating tensor");
  result = adreno_read_weights(job, weights);
  if (result != 0) goto cleanup;
  result = EIO;
  for (uint32_t i = 0; i < ADRENO_COLS; ++i) {
    input[i] = job->pattern == 0 ? 1.0f : job->pattern == 1
      ? (i % 2 == 0 ? 1.0f : -1.0f) : (i == ADRENO_COLS - 1 ? 1.0f : 0.0f);
  }
  /* NaN sentinels make an unwritten row fail finite-output validation. */
  for (uint32_t i = 0; i < ADRENO_ROWS; ++i) values[i] = UINT32_C(0x7fc00000);
  printf("tensor_bytes=%zu rows=%u cols=%u pattern=%" PRIu32 "\n",
         ADRENO_BYTES, ADRENO_ROWS, ADRENO_COLS, job->pattern);
  library = dlopen(ADRENO_DRIVER, RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    result = adreno_error(job, ENOTSUP, "Adreno.gemv: vendor OpenCL unavailable: %s", dlerror());
    goto cleanup;
  }
#define LOAD(name) do { \
    void *symbol = dlsym(library, #name); \
    if (!symbol) { \
      result = adreno_error(job, ENOTSUP, "Adreno.gemv: missing OpenCL entry %s", #name); \
      goto cleanup; \
    } \
    _Static_assert(sizeof(api.name) == sizeof(symbol), "Android LP64 function pointer ABI"); \
    memcpy(&api.name, &symbol, sizeof(symbol)); \
  } while (0)
  LOAD(clGetPlatformIDs); LOAD(clGetPlatformInfo); LOAD(clGetDeviceIDs);
  LOAD(clGetDeviceInfo); LOAD(clCreateContext); LOAD(clCreateCommandQueue);
  LOAD(clCreateBuffer); LOAD(clCreateProgramWithSource); LOAD(clBuildProgram);
  LOAD(clGetProgramBuildInfo); LOAD(clCreateKernel); LOAD(clGetKernelWorkGroupInfo); LOAD(clSetKernelArg);
  LOAD(clEnqueueNDRangeKernel); LOAD(clEnqueueReadBuffer); LOAD(clGetEventInfo);
  LOAD(clFinish); LOAD(clReleaseEvent); LOAD(clReleaseKernel); LOAD(clReleaseProgram);
  LOAD(clReleaseMemObject); LOAD(clReleaseCommandQueue); LOAD(clReleaseContext);
#undef LOAD
#define CHECK(call) do { \
    status = (call); \
    if (status != CL_SUCCESS) { \
      result = adreno_error(job, EIO, "Adreno.gemv: %s status=%d", #call, status); \
      goto cleanup; \
    } \
  } while (0)
#define REQUIRE(object) do { \
    if (!(object)) { \
      result = adreno_error(job, EIO, "Adreno.gemv: %s returned NULL", #object); \
      goto cleanup; \
    } \
  } while (0)
  cl_uint platform_count = 0;
  CHECK(api.clGetPlatformIDs(0, NULL, &platform_count));
  if (!platform_count || platform_count > 16) {
    result = adreno_error(job, ENOTSUP, "Adreno.gemv: unsupported platform count=%u", platform_count);
    goto cleanup;
  }
  CHECK(api.clGetPlatformIDs(platform_count, platforms, NULL));
  for (cl_uint p = 0; p < platform_count && !device; ++p) {
    cl_uint gpu_count = 0;
    status = api.clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, NULL, &gpu_count);
    if (status == CL_DEVICE_NOT_FOUND) continue;
    if (status != CL_SUCCESS || gpu_count > 16) {
      result = adreno_error(job, ENOTSUP, "Adreno.gemv: GPU enumeration status=%d count=%u", status, gpu_count);
      goto cleanup;
    }
    if (gpu_count) {
      cl_device_id devices[16];
      CHECK(api.clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, gpu_count, devices, NULL));
      device = devices[0];
      platform = platforms[p];
    }
  }
  if (!device) {
    result = adreno_error(job, ENOTSUP, "Adreno.gemv: no OpenCL GPU; CPU fallback forbidden");
    goto cleanup;
  }
  cl_device_type type = 0;
  cl_bool available = CL_FALSE, compiler = CL_FALSE, little_endian = CL_FALSE;
  CHECK(api.clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL));
  CHECK(api.clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(available), &available, NULL));
  CHECK(api.clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE, sizeof(compiler), &compiler, NULL));
  CHECK(api.clGetDeviceInfo(device, CL_DEVICE_ENDIAN_LITTLE, sizeof(little_endian), &little_endian, NULL));
  if (!(type & CL_DEVICE_TYPE_GPU) || (type & CL_DEVICE_TYPE_CPU) || !available || !compiler || !little_endian) {
    result = adreno_error(job, ENOTSUP,
      "Adreno.gemv: unsupported device type=%" PRIu64 " available=%u compiler=%u little_endian=%u",
      (uint64_t)type, available, compiler, little_endian);
    goto cleanup;
  }
  char identity[1024];
  const cl_platform_info platform_fields[] = {CL_PLATFORM_NAME, CL_PLATFORM_VENDOR, CL_PLATFORM_VERSION};
  const char *platform_labels[] = {"platform", "platform_vendor", "platform_version"};
  for (size_t i = 0; i < sizeof(platform_fields)/sizeof(platform_fields[0]); ++i) {
    memset(identity, 0, sizeof(identity));
    CHECK(api.clGetPlatformInfo(platform, platform_fields[i], sizeof(identity), identity, NULL));
    identity[sizeof(identity)-1] = '\0';
    printf("%s=%s\n", platform_labels[i], identity);
  }
  const cl_device_info fields[] = {CL_DEVICE_VENDOR, CL_DEVICE_NAME, CL_DEVICE_VERSION, CL_DRIVER_VERSION};
  const char *labels[] = {"vendor", "device", "device_version", "driver_version"};
  for (size_t i = 0; i < sizeof(fields)/sizeof(fields[0]); ++i) {
    memset(identity, 0, sizeof(identity));
    CHECK(api.clGetDeviceInfo(device, fields[i], sizeof(identity), identity, NULL));
    identity[sizeof(identity)-1] = '\0';
    printf("%s=%s\n", labels[i], identity);
  }
  printf("device_type=%" PRIu64 " gpu_required=true cpu_fallback=false\n", (uint64_t)type);
  fflush(stdout);
  context = api.clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  CHECK(status); REQUIRE(context);
  queue = api.clCreateCommandQueue(context, device, 0, &status);
  CHECK(status); REQUIRE(queue);
  weight_buffer = api.clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    ADRENO_BYTES, weights, &status);
  CHECK(status); REQUIRE(weight_buffer);
  input_buffer = api.clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(input), input, &status);
  CHECK(status); REQUIRE(input_buffer);
  output_buffer = api.clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    sizeof(values), values, &status);
  CHECK(status); REQUIRE(output_buffer);
  const char *source_ptr = source;
  program = api.clCreateProgramWithSource(context, 1, &source_ptr, NULL, &status);
  CHECK(status); REQUIRE(program);
  status = api.clBuildProgram(program, 1, &device, "-cl-std=CL1.2", NULL, NULL);
  printf("kernel_build_status=%d kernel_build_options=-cl-std=CL1.2 fp_contract=off\n", status);
  if (status != CL_SUCCESS) {
    char build_log[8192] = {0};
    (void)api.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                  sizeof(build_log)-1, build_log, NULL);
    fprintf(stderr, "OpenCL build log: %s\n", build_log);
    result = adreno_error(job, EIO, "Adreno.gemv: kernel build status=%d", status);
    goto cleanup;
  }
  kernel = api.clCreateKernel(program, "gemv", &status);
  CHECK(status); REQUIRE(kernel);
  CHECK(api.clSetKernelArg(kernel, 0, sizeof(weight_buffer), &weight_buffer));
  CHECK(api.clSetKernelArg(kernel, 1, sizeof(input_buffer), &input_buffer));
  CHECK(api.clSetKernelArg(kernel, 2, sizeof(output_buffer), &output_buffer));
  const size_t global = 18432, local = 256;
  size_t kernel_limit = 0, device_limit = 0, required[3] = {0};
  cl_ulong kernel_memory = 0, device_memory = 0;
  CHECK(api.clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                    sizeof(kernel_limit), &kernel_limit, NULL));
  CHECK(api.clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
                                    sizeof(required), required, NULL));
  CHECK(api.clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_LOCAL_MEM_SIZE,
                                    sizeof(kernel_memory), &kernel_memory, NULL));
  CHECK(api.clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                           sizeof(device_limit), &device_limit, NULL));
  CHECK(api.clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE,
                           sizeof(device_memory), &device_memory, NULL));
  /* OpenCL reports one size_t per supported dimension. A bounded stack buffer
   * covers the normal three dimensions without a per-call heap allocation. */
  cl_uint dimensions = 0;
  size_t item_limits[16] = {0};
  CHECK(api.clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                           sizeof(dimensions), &dimensions, NULL));
  if (!dimensions || dimensions > sizeof(item_limits) / sizeof(item_limits[0])) {
    result = adreno_error(job, ENOTSUP, "Adreno.gemv: unsupported work-item dimensions=%u", dimensions);
    goto cleanup;
  }
  CHECK(api.clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                           dimensions * sizeof(item_limits[0]), item_limits, NULL));
  if (local > kernel_limit || local > device_limit || local > item_limits[0] ||
      kernel_memory > device_memory ||
      required[0] != local || required[1] != 1 || required[2] != 1) {
    result = adreno_error(job, ENOTSUP,
      "Adreno.gemv: unsupported local size or kernel local-memory requirement");
    goto cleanup;
  }
  status = api.clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, &event);
  printf("enqueue_ndrange_status=%d global=%zu local=%zu\n", status, global, local);
  fflush(stdout);
  CHECK(status); REQUIRE(event);
  CHECK(api.clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(values),
                               values, 1, &event, NULL));
  cl_int execution = -1;
  CHECK(api.clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(execution), &execution, NULL));
  printf("kernel_event_execution_status=%d blocking_readback=success\n", execution);
  if (execution != CL_COMPLETE) {
    result = adreno_error(job, EIO, "Adreno.gemv: kernel event not complete status=%d", execution);
    goto cleanup;
  }
  for (uint32_t row = 0; row < ADRENO_ROWS; ++row) {
    if ((values[row] & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
      result = adreno_error(job, EDOM, "Adreno.gemv: nonfinite GPU output row=%" PRIu32, row);
      goto cleanup;
    }
  }
  result = 0;
cleanup:
  /* clFinish before freeing host storage on an error after a queued operation.
   * OpenCL objects retain dependencies; release in reverse order. Device/platform
   * IDs are borrowed opaque API values: never dereference or release them. */
  if (result != 0 && queue) {
    cl_int finished = api.clFinish(queue);
    if (finished != CL_SUCCESS)
      result = adreno_error(job, EIO, "Adreno.gemv: cleanup clFinish status=%d", finished);
  }
#define RELEASE(object, method) do { \
    if (object) { \
      cl_int released = api.method(object); \
      if (released != CL_SUCCESS) \
        result = adreno_error(job, EIO, "Adreno.gemv: cleanup %s status=%d", #method, released); \
    } \
  } while (0)
  RELEASE(event, clReleaseEvent); RELEASE(kernel, clReleaseKernel);
  RELEASE(program, clReleaseProgram); RELEASE(output_buffer, clReleaseMemObject);
  RELEASE(input_buffer, clReleaseMemObject); RELEASE(weight_buffer, clReleaseMemObject);
  RELEASE(queue, clReleaseCommandQueue); RELEASE(context, clReleaseContext);
#undef RELEASE
#undef REQUIRE
#undef CHECK
  free(weights);
  if (library && dlclose(library) != 0)
    result = adreno_error(job, EIO, "Adreno.gemv: dlclose: %s", dlerror());
  /* This is raw GPU evidence, not numerical acceptance. Only the independent
   * Bend M.mv oracle comparison may accept a result; never print a GPU PASS. */
  if (result == 0) {
    printf("ADRENO_GEMV_BITS pattern=%" PRIu32 " values=", job->pattern);
    for (uint32_t row = 0; row < ADRENO_ROWS; ++row)
      printf("%s%" PRIu32, row == 0 ? "" : " ", values[row]);
    putchar('\n');
  }
  if (fflush(stdout) != 0 || ferror(stdout))
    result = adreno_error(job, EIO, "Adreno.gemv: writing GPU evidence failed");
  return result;
}

static void adreno_gemv_call(IoWork *work) {
  work->code = adreno_gemv((struct adreno_gemv_work *)work->data);
}

static Term adreno_gemv_pack(Env env, IoWork *work) {
  struct adreno_gemv_work *job = (struct adreno_gemv_work *)work->data;
  Term result = work->code == 0
    ? io_done(env, term_pak(CID(Unit), 0))
    : io_fail(env, (u32)work->code, job->error);
  /* io_cstr returns an owned copy. Runtime owns work/args; this effect owns
   * work->data and its path. Pack runs on the loop after the helper has finished;
   * io_fail copies the message before either allocation is freed. No custom handle. */
  free(job->path);
  free(job);
  work->data = NULL;
  return result;
}

static Term adreno_gemv_run(Env env, Term *args, IoWork *work) {
  struct adreno_gemv_work *job = calloc(1, sizeof(*job));
  if (!job) return io_fail(env, ENOMEM, "Adreno.gemv: allocating helper state");
  uint64_t length = 0;
  job->path = io_cstr(env, args[0], &length);
  job->pattern = (u32)args[1];
  work->data = (char *)job;
  if (length == 0 || io_nul(job->path, length) || job->pattern > 2) {
    work->code = adreno_error(job, EINVAL, "Adreno.gemv: nonempty NUL-free path and pattern 0..2 required");
    return adreno_gemv_pack(env, work);
  }
  return io_work(work, adreno_gemv_call, adreno_gemv_pack);
}

static void __attribute__((constructor)) adreno_gemv_use(void) {
  io_eff(CID(Adreno.gemv), adreno_gemv_run);
}
