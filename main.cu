#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rans_byte_cuda.h"

// 定义 CUDA 可调用的宏
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

// 将 rans_byte.h 中的所有函数前添加 CUDA_CALLABLE，以便在设备上调用
// 请确保在 rans_byte.h 中所有函数前添加 CUDA_CALLABLE
// 读取文件的函数
static uint8_t* read_file(char const* filename, size_t* out_size)
{
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "File not found: %s\n", filename);
        exit(1);
    }

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t* buf = new uint8_t[size];
    if (fread(buf, size, 1, f) != 1) {
        fprintf(stderr, "Read failed\n");
        exit(1);
    }

    fclose(f);
    if (out_size)
        *out_size = size;

    return buf;
}

// 写入文件的函数
static void write_file(char const* filename, uint8_t* data, size_t size)
{
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        exit(1);
    }

    if (fwrite(data, size, 1, f) != 1) {
        fprintf(stderr, "Write failed\n");
        exit(1);
    }

    fclose(f);
}

// CUDA 编码内核函数
__global__ void RansEncodeKernel(uint8_t* in_bytes_d, size_t* part_offsets_d, size_t* part_sizes_d,
                                 RansEncSymbol* esyms_d, uint8_t* out_buf_d, size_t* out_sizes_d,
                                 size_t num_parts, size_t out_max_size_per_part) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parts)
        return;

    size_t in_offset = part_offsets_d[idx];
    size_t in_size = part_sizes_d[idx];
    uint8_t* in = in_bytes_d + in_offset;

    uint8_t* out_buf = out_buf_d + idx * out_max_size_per_part;

    // 初始化编码器状态
    RansState rans;
    RansEncInit(&rans);

    uint8_t* ptr = out_buf + out_max_size_per_part; // 输出缓冲区的末尾

    // 进行编码（反向处理）
    for (size_t i = in_size; i > 0; i--) {
        int s = in[i - 1];
        RansEncPutSymbol(&rans, &ptr, &esyms_d[s]);
    }
    RansEncFlush(&rans, &ptr);

    // 计算输出大小
    size_t out_size = out_buf + out_max_size_per_part - ptr;

    // 将编码数据复制到输出缓冲区的开头
    for (size_t i = 0; i < out_size; i++) {
        out_buf[i] = ptr[i];
    }

    // 存储输出大小
    out_sizes_d[idx] = out_size;
}

// CUDA 解码内核函数
__global__ void RansDecodeKernel(uint8_t* out_buf_d, size_t* out_sizes_d, size_t* part_sizes_d,
                                 uint8_t* dec_bytes_d, RansDecSymbol* dsyms_d, uint8_t* cum2sym_d,
                                 size_t num_parts, uint32_t prob_bits, size_t out_max_size_per_part) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parts)
        return;
    uint8_t* out_buf = out_buf_d + idx * out_max_size_per_part;
    size_t out_size = out_sizes_d[idx];
    size_t dec_size = part_sizes_d[idx];
    // 打印dec_size
    // printf("Thread %lu, dec_size: %lu\n", idx, dec_size);
    uint8_t* dec_bytes = dec_bytes_d + idx * part_sizes_d[0];

    uint8_t* ptr = out_buf;

    // 初始化解码器状态
    RansState rans;
    RansDecInit(&rans, &ptr);

    for (size_t i = 0; i < dec_size; i++) {
        uint32_t s = cum2sym_d[RansDecGet(&rans, prob_bits)];
        dec_bytes[i] = (uint8_t)s;
        RansDecAdvanceSymbol(&rans, &ptr, &dsyms_d[s], prob_bits);
    }
}


// ---- Stats

struct SymbolStats
{
    uint32_t freqs[256];
    uint32_t cum_freqs[257];

    void count_freqs(uint8_t const* in, size_t nbytes);
    void calc_cum_freqs();
    void normalize_freqs(uint32_t target_total);
};

void SymbolStats::count_freqs(uint8_t const* in, size_t nbytes)
{
    for (int i=0; i < 256; i++)
        freqs[i] = 0;

    for (size_t i=0; i < nbytes; i++)
        freqs[in[i]]++;
}

void SymbolStats::calc_cum_freqs()
{
    cum_freqs[0] = 0;
    for (int i=0; i < 256; i++)
        cum_freqs[i+1] = cum_freqs[i] + freqs[i];
}

void SymbolStats::normalize_freqs(uint32_t target_total)
{
    RansAssert(target_total >= 256);
    
    calc_cum_freqs();
    uint32_t cur_total = cum_freqs[256];
    
    // resample distribution based on cumulative freqs
    for (int i = 1; i <= 256; i++)
        cum_freqs[i] = ((uint64_t)target_total * cum_freqs[i])/cur_total;

    // if we nuked any non-0 frequency symbol to 0, we need to steal
    // the range to make the frequency nonzero from elsewhere.
    //
    // this is not at all optimal, i'm just doing the first thing that comes to mind.
    for (int i=0; i < 256; i++) {
        if (freqs[i] && cum_freqs[i+1] == cum_freqs[i]) {
            // symbol i was set to zero freq

            // find best symbol to steal frequency from (try to steal from low-freq ones)
            uint32_t best_freq = ~0u;
            int best_steal = -1;
            for (int j=0; j < 256; j++) {
                uint32_t freq = cum_freqs[j+1] - cum_freqs[j];
                if (freq > 1 && freq < best_freq) {
                    best_freq = freq;
                    best_steal = j;
                }
            }
            RansAssert(best_steal != -1);

            // and steal from it!
            if (best_steal < i) {
                for (int j = best_steal + 1; j <= i; j++)
                    cum_freqs[j]--;
            } else {
                RansAssert(best_steal > i);
                for (int j = i + 1; j <= best_steal; j++)
                    cum_freqs[j]++;
            }
        }
    }

    // calculate updated freqs and make sure we didn't screw anything up
    RansAssert(cum_freqs[0] == 0 && cum_freqs[256] == target_total);
    for (int i=0; i < 256; i++) {
        if (freqs[i] == 0)
            RansAssert(cum_freqs[i+1] == cum_freqs[i]);
        else
            RansAssert(cum_freqs[i+1] > cum_freqs[i]);

        // calc updated freq
        freqs[i] = cum_freqs[i+1] - cum_freqs[i];
    }
}

float calculate_accuracy(const unsigned char* in_bytes, const unsigned char* dec_bytes, size_t in_size) {
    int match_count = 0;
    for (size_t i = 0; i < in_size; i++) {
        if (in_bytes[i] == dec_bytes[i]) {
            match_count++;
        }
    }
    // 计算正确率
    return (float)match_count / in_size * 100;
}

int main(int argc, char** argv) {
    // 检查命令行参数
    if (argc < 5) {
        printf("Usage: %s <num_parts> <block_size> <freq_file> <input_file>\n", argv[0]);
        return -1;
    }

    // 解析命令行参数
    size_t num_parts = atoi(argv[1]);
    size_t block_size = atoi(argv[2]);
    const char* freq_file = argv[3];
    const char* input_file = argv[4];

    printf("Number of parts: %zu\n", num_parts);
    printf("Block size: %zu\n", block_size);
    printf("Frequency file: %s\n", freq_file);
    printf("Input file: %s\n", input_file);

    // 获取和显示 CUDA 设备信息
    cudaDeviceProp deviceProp;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Using CUDA device: %s\n", deviceProp.name);
    printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);

    // 定义全局概率模型的参数
    static const uint32_t prob_bits = 14;
    static const uint32_t prob_scale = 1 << prob_bits;

    // 从频率文件读取数据并统计全局频率
    size_t freq_file_size;
    uint8_t* freq_file_data = read_file(freq_file, &freq_file_size);

    SymbolStats stats;
    stats.count_freqs(freq_file_data, freq_file_size);
    stats.normalize_freqs(prob_scale);

    // 构建 cum2sym 表
    uint8_t cum2sym[prob_scale];
    for (int s = 0; s < 256; s++)
        for (uint32_t i = stats.cum_freqs[s]; i < stats.cum_freqs[s + 1]; i++)
            cum2sym[i] = s;

    // 初始化编码和解码符号表
    RansEncSymbol esyms[256];
    RansDecSymbol dsyms[256];

    for (int i = 0; i < 256; i++) {
        RansEncSymbolInit(&esyms[i], stats.cum_freqs[i], stats.freqs[i], prob_bits);
        RansDecSymbolInit(&dsyms[i], stats.cum_freqs[i], stats.freqs[i]);
    }

    // 从输入文件读取数据
    size_t in_size;
    uint8_t* in_bytes = read_file(input_file, &in_size);

    // 将文件分成 num_parts 份
    size_t part_size = (in_size + num_parts - 1) / num_parts; // 向上取整
    size_t last_part_size = in_size - part_size * (num_parts - 1); // 最后一部分的大小

    // 创建偏移量和大小数组
    size_t* part_offsets = new size_t[num_parts];
    size_t* part_sizes = new size_t[num_parts];
    for (size_t i = 0; i < num_parts; i++) {
        part_offsets[i] = i * part_size;
        if (i == num_parts - 1) {
            part_sizes[i] = last_part_size;
        } else {
            part_sizes[i] = part_size;
        }
    }

    // 将输入数据复制到设备
    uint8_t* in_bytes_d;
    cudaMalloc(&in_bytes_d, in_size * sizeof(uint8_t));
    cudaMemcpy(in_bytes_d, in_bytes, in_size * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // 将偏移量和大小数组复制到设备
    size_t* part_offsets_d;
    cudaMalloc(&part_offsets_d, num_parts * sizeof(size_t));
    cudaMemcpy(part_offsets_d, part_offsets, num_parts * sizeof(size_t), cudaMemcpyHostToDevice);

    size_t* part_sizes_d;
    cudaMalloc(&part_sizes_d, num_parts * sizeof(size_t));
    cudaMemcpy(part_sizes_d, part_sizes, num_parts * sizeof(size_t), cudaMemcpyHostToDevice);

    RansEncSymbol* esyms_d;
    cudaMalloc(&esyms_d, 256 * sizeof(RansEncSymbol));
    cudaMemcpy(esyms_d, esyms, 256 * sizeof(RansEncSymbol), cudaMemcpyHostToDevice);

    // 设置输出缓冲区大小
    size_t out_max_size_per_part = part_size * 2; // 根据需要调整
    uint8_t* out_buf_d;
    cudaMalloc(&out_buf_d, num_parts * out_max_size_per_part * sizeof(uint8_t));

    size_t* out_sizes_d;
    cudaMalloc(&out_sizes_d, num_parts * sizeof(size_t));

    // 调整线程块大小
    dim3 blockSize(block_size);
    dim3 gridSize((num_parts + blockSize.x - 1) / block_size);

    // 记录编码时间
    cudaEvent_t enc_start, enc_stop;
    cudaEventCreate(&enc_start);
    cudaEventCreate(&enc_stop);

    cudaEventRecord(enc_start);

    // gridSize.x = 512
    // blockSize = 16
    RansEncodeKernel<<<gridSize, blockSize>>>(in_bytes_d, part_offsets_d, part_sizes_d, esyms_d,
                                              out_buf_d, out_sizes_d, num_parts, out_max_size_per_part);

    cudaEventRecord(enc_stop);
    cudaEventSynchronize(enc_stop);

    float enc_milliseconds = 0;
    cudaEventElapsedTime(&enc_milliseconds, enc_start, enc_stop);

    // 计算编码吞吐量 (MB/s)
    float total_data_size_mb = in_size / (1024.0 * 1024.0);  // 转换为MB
    float enc_throughput = total_data_size_mb / (enc_milliseconds / 1000.0);  // MB/s

    printf("Encoding Time: %.2f ms\n", enc_milliseconds);
    printf("Encoding Throughput: %.2f MB/s\n", enc_throughput);

    // 从设备复制编码结果
    size_t* out_sizes = new size_t[num_parts];
    cudaMemcpy(out_sizes, out_sizes_d, num_parts * sizeof(size_t), cudaMemcpyDeviceToHost);

    uint8_t* out_buf = new uint8_t[num_parts * out_max_size_per_part];
    cudaMemcpy(out_buf, out_buf_d, num_parts * out_max_size_per_part * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // 计算编码后的总大小
    size_t total_encoded_size = 0;
    for (size_t i = 0; i < num_parts; i++) {
        total_encoded_size += out_sizes[i];
    }

    // 计算压缩率
    float compression_ratio = (float)total_encoded_size / (float)in_size * 100.0f;
    printf("Original size: %zu bytes\n", in_size);
    printf("Encoded size: %zu bytes\n", total_encoded_size);
    printf("Compression ratio: %.2f%%\n", compression_ratio);

    // 保存编码后的文件
    FILE* f_encoded = fopen("encoded.bin", "wb");
    if (!f_encoded) {
        fprintf(stderr, "Failed to open encoded.bin for writing\n");
        exit(1);
    }
    // 写入部分数量和每个部分的大小，以便解码时使用
    fwrite(&num_parts, sizeof(size_t), 1, f_encoded);
    fwrite(out_sizes, sizeof(size_t), num_parts, f_encoded);
    for (size_t i = 0; i < num_parts; i++) {
        fwrite(out_buf + i * out_max_size_per_part, out_sizes[i], 1, f_encoded); // 写入编码数据
    }
    fclose(f_encoded);

    // 准备解码所需的数据
    uint8_t* cum2sym_d;
    cudaMalloc(&cum2sym_d, prob_scale * sizeof(uint8_t));
    cudaMemcpy(cum2sym_d, cum2sym, prob_scale * sizeof(uint8_t), cudaMemcpyHostToDevice);

    RansDecSymbol* dsyms_d;
    cudaMalloc(&dsyms_d, 256 * sizeof(RansDecSymbol));
    cudaMemcpy(dsyms_d, dsyms, 256 * sizeof(RansDecSymbol), cudaMemcpyHostToDevice);

    uint8_t* dec_bytes_d;
    cudaMalloc(&dec_bytes_d, in_size * sizeof(uint8_t));

    // 从编码文件中读取编码数据
    FILE* f_encoded_in = fopen("encoded.bin", "rb");
    if (!f_encoded_in) {
        fprintf(stderr, "Failed to open encoded.bin for reading\n");
        exit(1);
    }
    size_t num_parts_in;
    fread(&num_parts_in, sizeof(size_t), 1, f_encoded_in);
    if (num_parts_in != num_parts) {
        fprintf(stderr, "Number of parts mismatch!\n");
        exit(1);
    }
    fread(out_sizes, sizeof(size_t), num_parts, f_encoded_in);
    for (size_t i = 0; i < num_parts; i++) {
        fread(out_buf + i * out_max_size_per_part, out_sizes[i], 1, f_encoded_in);
    }
    fclose(f_encoded_in);

    // 将编码数据复制到设备
    cudaMemcpy(out_buf_d, out_buf, num_parts * out_max_size_per_part * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(out_sizes_d, out_sizes, num_parts * sizeof(size_t), cudaMemcpyHostToDevice);

    // 记录解码时间
    cudaEvent_t dec_start, dec_stop;
    cudaEventCreate(&dec_start);
    cudaEventCreate(&dec_stop);

    cudaEventRecord(dec_start);

    // 调用解码内核函数
    RansDecodeKernel<<<gridSize, blockSize>>>(out_buf_d, out_sizes_d, part_sizes_d, dec_bytes_d,
                                              dsyms_d, cum2sym_d, num_parts, prob_bits, out_max_size_per_part);

    cudaEventRecord(dec_stop);
    cudaEventSynchronize(dec_stop);

    float dec_milliseconds = 0;
    cudaEventElapsedTime(&dec_milliseconds, dec_start, dec_stop);

    // 计算解码吞吐量 (MB/s)
    float dec_throughput = total_data_size_mb / (dec_milliseconds / 1000.0);  // MB/s

    printf("Decoding Time: %.2f ms\n", dec_milliseconds);
    printf("Decoding Throughput: %.2f MB/s\n", dec_throughput);

    // 从设备复制解码结果
    uint8_t* dec_bytes = new uint8_t[in_size];
    cudaMemcpy(dec_bytes, dec_bytes_d, in_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // 保存解码后的文件
    write_file("decoded_output.bin", dec_bytes, in_size);

    // 检查解码结果是否正确
    if (memcmp(in_bytes, dec_bytes, in_size) == 0) {
        printf("Decoding successful! The decoded data matches the original data.\n");
    } else {
        // printf("Decoding failed! The decoded data does not match the original data.\n");
        // 计算正确率
        float accuracy = calculate_accuracy(in_bytes, dec_bytes, in_size);
        printf("Decoding accuracy: %.2f%%\n", accuracy);
    }


    // 释放资源
    delete[] freq_file_data;
    delete[] in_bytes;
    delete[] part_offsets;
    delete[] part_sizes;
    delete[] out_sizes;
    delete[] out_buf;
    delete[] dec_bytes;
    cudaFree(in_bytes_d);
    cudaFree(part_offsets_d);
    cudaFree(part_sizes_d);
    cudaFree(esyms_d);
    cudaFree(out_buf_d);
    cudaFree(out_sizes_d);
    cudaFree(cum2sym_d);
    cudaFree(dsyms_d);
    cudaFree(dec_bytes_d);

    cudaEventDestroy(enc_start);
    cudaEventDestroy(enc_stop);
    cudaEventDestroy(dec_start);
    cudaEventDestroy(dec_stop);

    return 0;
}
