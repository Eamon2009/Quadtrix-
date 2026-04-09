// This need to be tested on a real gpu
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Config
typedef struct
{
      int vocab_size;
      int block_size;
      int n_embd;
      int n_head;
      int n_layer;
} Config;

// ── Error checking
#define CUDA_CHECK(call)                                              \
      do                                                              \
      {                                                               \
            cudaError_t err = call;                                   \
            if (err != cudaSuccess)                                   \
            {                                                         \
                  fprintf(stderr, "[CUDA Error] %s at line %d: %s\n", \
                          #call, __LINE__, cudaGetErrorString(err));  \
                  exit(1);                                            \
            }                                                         \
      } while (0)

#define CUBLAS_CHECK(call)                                              \
      do                                                                \
      {                                                                 \
            cublasStatus_t err = call;                                  \
            if (err != CUBLAS_STATUS_SUCCESS)                           \
            {                                                           \
                  fprintf(stderr, "[cuBLAS Error] %s at line %d: %d\n", \
                          #call, __LINE__, err);                        \
                  exit(1);                                              \
            }                                                           \
      } while (0)

// Signal handler
static volatile int running = 1;
void handle_sigint(int s)
{
      (void)s;
      printf("\n\n[Stopped by user]\n");
      running = 0;
}

// Build x[T x C] from token + position embeddings
__global__ void embed_kernel(float *x, float *tok_emb, float *pos_emb,
                             int *tokens, int T, int C)
{
      int t = blockIdx.x;
      int c = threadIdx.x;
      if (t < T && c < C)
            x[t * C + c] = tok_emb[tokens[t] * C + c] + pos_emb[t * C + c];
}

// LayerNorm over one row of length C
__global__ void layernorm_kernel(float *out, float *x, float *w, float *b,
                                 int T, int C)
{
      int t = blockIdx.x;
      if (t >= T)
            return;

      float *xr = x + t * C;
      float *outr = out + t * C;

      float mean = 0.0f;
      for (int i = 0; i < C; i++)
            mean += xr[i];
      mean /= C;

      float var = 0.0f;
      for (int i = 0; i < C; i++)
            var += (xr[i] - mean) * (xr[i] - mean);
      var = var / C + 1e-5f;
      float inv = rsqrtf(var);

      for (int i = 0; i < C; i++)
            outr[i] = (xr[i] - mean) * inv * w[i] + b[i];
}

// Add bias in-place: x[T x N] += b[N]
__global__ void add_bias_kernel(float *x, float *b, int T, int N)
{
      int t = blockIdx.x;
      int n = threadIdx.x;
      if (t < T && n < N)
            x[t * N + n] += b[n];
}

// ReLU in-place
__global__ void relu_kernel(float *x, int n)
{
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n && x[i] < 0.0f)
            x[i] = 0.0f;
}

// Residual add: a += b
__global__ void residual_kernel(float *a, float *b, int n)
{
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n)
            a[i] += b[i];
}

// Causal attention scores + softmax for one head
__global__ void attention_kernel(float *att, float *Q, float *K,
                                 int T, int hs)
{
      int i = blockIdx.x;  // query position
      int j = threadIdx.x; // key position
      if (i >= T || j >= T)
            return;

      if (j > i)
      {
            att[i * T + j] = -1e9f;
            return;
      }
      float scale = rsqrtf((float)hs);
      float dot = 0.0f;
      for (int k = 0; k < hs; k++)
            dot += Q[i * hs + k] * K[j * hs + k];
      att[i * T + j] = dot * scale;

      // Softmax (done per row, only thread 0 of each row)
      __syncthreads();
      if (j == 0)
      {
            float *row = att + i * T;
            float mx = -1e9f;
            for (int x = 0; x <= i; x++)
                  if (row[x] > mx)
                        mx = row[x];
            float sum = 0.0f;
            for (int x = 0; x < T; x++)
            {
                  row[x] = (x <= i) ? expf(row[x] - mx) : 0.0f;
                  sum += row[x];
            }
            for (int x = 0; x < T; x++)
                  row[x] /= sum;
      }
}

// Weighted sum: hv[T x hs] = att[T x T] @ V[T x hs]
__global__ void attn_value_kernel(float *hv, float *att, float *V,
                                  int T, int hs)
{
      int i = blockIdx.x;
      int k = threadIdx.x;
      if (i >= T || k >= hs)
            return;
      float s = 0.0f;
      for (int j = 0; j <= i; j++)
            s += att[i * T + j] * V[j * hs + k];
      hv[i * hs + k] = s;
}

// Scatter head output into full [T x C] buffer at offset h*hs
__global__ void scatter_head_kernel(float *head_out, float *hv,
                                    int T, int C, int hs, int h_offset)
{
      int t = blockIdx.x;
      int k = threadIdx.x;
      if (t < T && k < hs)
            head_out[t * C + h_offset + k] = hv[t * hs + k];
}

// Softmax over logits (single row, run on CPU side for simplicity)
static void softmax_cpu(float *x, int n)
{
      float mx = x[0];
      for (int i = 1; i < n; i++)
            if (x[i] > mx)
                  mx = x[i];
      float sum = 0.0f;
      for (int i = 0; i < n; i++)
      {
            x[i] = expf(x[i] - mx);
            sum += x[i];
      }
      for (int i = 0; i < n; i++)
            x[i] /= sum;
}

static int sample(float *probs, int n)
{
      float r = (float)rand() / ((float)RAND_MAX + 1.0f);
      float cdf = 0.0f;
      for (int i = 0; i < n; i++)
      {
            cdf += probs[i];
            if (r < cdf)
                  return i;
      }
      return n - 1;
}

// Weight struct (GPU pointers,need to be tested on a GPU )
typedef struct
{
      float *tok_emb; // [vocab x C]
      float *pos_emb; // [block x C]
      float **head_k; // [n_layer x n_head][hs x C]
      float **head_q;
      float **head_v;
      float **sa_proj_w; // [n_layer][C x C]
      float **sa_proj_b; // [n_layer][C]
      float **ff_w1;     // [n_layer][4C x C]
      float **ff_b1;     // [n_layer][4C]
      float **ff_w2;     // [n_layer][C x 4C]
      float **ff_b2;     // [n_layer][C]
      float **ln1_w;     // [n_layer][C]
      float **ln1_b;
      float **ln2_w;
      float **ln2_b;
      float *ln_f_w; // [C]
      float *ln_f_b;
      float *lm_w; // [vocab x C]
      float *lm_b; // [vocab]
} Weights;

// Read tensor from file and upload to GPU
static float *read_upload(FILE *f)
{
      int ndim;
      fread(&ndim, sizeof(int), 1, f);
      int total = 1;
      for (int i = 0; i < ndim; i++)
      {
            int d;
            fread(&d, sizeof(int), 1, f);
            total *= d;
      }
      float *cpu = (float *)malloc(total * sizeof(float));
      fread(cpu, sizeof(float), total, f);
      float *gpu;
      CUDA_CHECK(cudaMalloc(&gpu, total * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(gpu, cpu, total * sizeof(float), cudaMemcpyHostToDevice));
      free(cpu);
      return gpu;
}

//  Forward pass
static void forward(float *d_logits, int *d_tokens, int T,
                    Config *cfg, Weights *W, cublasHandle_t cublas)
{
      int C = cfg->n_embd;
      int nh = cfg->n_head;
      int hs = C / nh;
      int ff = 4 * C;

      float one = 1.0f, zero = 0.0f;

      // Allocate working buffers
      float *d_x, *d_ln_out, *d_head_out, *d_attn_out;
      float *d_K, *d_Q, *d_V, *d_att, *d_hv;
      float *d_ff1, *d_ff2;

      CUDA_CHECK(cudaMalloc(&d_x, T * C * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_ln_out, T * C * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_head_out, T * C * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_attn_out, T * C * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_K, T * hs * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_Q, T * hs * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_V, T * hs * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_att, T * T * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_hv, T * hs * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_ff1, T * ff * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_ff2, T * C * sizeof(float)));

      // Embed
      embed_kernel<<<T, C>>>(d_x, W->tok_emb, W->pos_emb, d_tokens, T, C);

      for (int l = 0; l < cfg->n_layer; l++)
      {
            int base = l * nh;

            // LayerNorm 1
            layernorm_kernel<<<T, 1>>>(d_ln_out, d_x,
                                       W->ln1_w[l], W->ln1_b[l], T, C);

            // Multi-head attention
            CUDA_CHECK(cudaMemset(d_head_out, 0, T * C * sizeof(float)));

            for (int h = 0; h < nh; h++)
            {
                  CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                           hs, T, C, &one,
                                           W->head_k[base + h], C,
                                           d_ln_out, C,
                                           &zero, d_K, hs));

                  CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                           hs, T, C, &one,
                                           W->head_q[base + h], C,
                                           d_ln_out, C,
                                           &zero, d_Q, hs));

                  CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                           hs, T, C, &one,
                                           W->head_v[base + h], C,
                                           d_ln_out, C,
                                           &zero, d_V, hs));

                  // Attention scores + softmax
                  attention_kernel<<<T, T>>>(d_att, d_Q, d_K, T, hs);

                  // Weighted V
                  attn_value_kernel<<<T, hs>>>(d_hv, d_att, d_V, T, hs);

                  // Scatter into head_out
                  scatter_head_kernel<<<T, hs>>>(d_head_out, d_hv, T, C, hs, h * hs);
            }

            // SA projection
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                     C, T, C, &one,
                                     W->sa_proj_w[l], C,
                                     d_head_out, C,
                                     &zero, d_attn_out, C));
            add_bias_kernel<<<T, C>>>(d_attn_out, W->sa_proj_b[l], T, C);

            // Residual
            residual_kernel<<<(T * C + 255) / 256, 256>>>(d_x, d_attn_out, T * C);

            // LayerNorm 2
            layernorm_kernel<<<T, 1>>>(d_ln_out, d_x,
                                       W->ln2_w[l], W->ln2_b[l], T, C);

            // FF layer 1
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                     ff, T, C, &one,
                                     W->ff_w1[l], C,
                                     d_ln_out, C,
                                     &zero, d_ff1, ff));
            add_bias_kernel<<<T, ff>>>(d_ff1, W->ff_b1[l], T, ff);
            relu_kernel<<<(T * ff + 255) / 256, 256>>>(d_ff1, T * ff);

            // FF layer 2
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                     C, T, ff, &one,
                                     W->ff_w2[l], ff,
                                     d_ff1, ff,
                                     &zero, d_ff2, C));
            add_bias_kernel<<<T, C>>>(d_ff2, W->ff_b2[l], T, C);

            // Residual
            residual_kernel<<<(T * C + 255) / 256, 256>>>(d_x, d_ff2, T * C);
      }

      // Final layernorm on last token only
      float *d_last;
      CUDA_CHECK(cudaMalloc(&d_last, C * sizeof(float)));
      float *d_xf;
      CUDA_CHECK(cudaMalloc(&d_xf, C * sizeof(float)));

      CUDA_CHECK(cudaMemcpy(d_last, d_x + (T - 1) * C,
                            C * sizeof(float), cudaMemcpyDeviceToDevice));
      layernorm_kernel<<<1, 1>>>(d_xf, d_last, W->ln_f_w, W->ln_f_b, 1, C);

      // LM head: logits[vocab] = lm_w[vocab x C] @ xf[C]
      CUBLAS_CHECK(cublasSgemv(cublas, CUBLAS_OP_T,
                               C, cfg->vocab_size, &one,
                               W->lm_w, C,
                               d_xf, 1,
                               &zero, d_logits, 1));

      // Add lm bias
      float *d_lm_b_scaled;
      CUDA_CHECK(cudaMalloc(&d_lm_b_scaled, cfg->vocab_size * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(d_lm_b_scaled, W->lm_b,
                            cfg->vocab_size * sizeof(float),
                            cudaMemcpyDeviceToDevice));
      residual_kernel<<<(cfg->vocab_size + 255) / 256, 256>>>(
          d_logits, d_lm_b_scaled, cfg->vocab_size);

      // Free buffers
      cudaFree(d_x);
      cudaFree(d_ln_out);
      cudaFree(d_head_out);
      cudaFree(d_attn_out);
      cudaFree(d_K);
      cudaFree(d_Q);
      cudaFree(d_V);
      cudaFree(d_att);
      cudaFree(d_hv);
      cudaFree(d_ff1);
      cudaFree(d_ff2);
      cudaFree(d_last);
      cudaFree(d_xf);
      cudaFree(d_lm_b_scaled);
}

// Main
int main(void)
{
      signal(SIGINT, handle_sigint);
      srand((unsigned)time(NULL));

      // Check GPU
      int dev_count = 0;
      CUDA_CHECK(cudaGetDeviceCount(&dev_count));
      if (dev_count == 0)
      {
            fprintf(stderr, "[Error] No CUDA GPU found.\n");
            return 1;
      }
      cudaDeviceProp prop;
      CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
      printf("[INFO] GPU: %s\n", prop.name);

      // Load vocab
      FILE *fv = fopen("../vocab.bin", "rb");
      if (!fv)
      {
            fprintf(stderr, "[Error] Cannot open vocab.bin\n");
            return 1;
      }
      int vocab_size;
      fread(&vocab_size, sizeof(int), 1, fv);
      char *vocab = (char *)malloc(vocab_size);
      for (int i = 0; i < vocab_size; i++)
      {
            unsigned char c;
            fread(&c, 1, 1, fv);
            vocab[i] = (char)c;
      }
      fclose(fv);

      // Load weights
      FILE *fw = fopen("../weights.bin", "rb");
      if (!fw)
      {
            fprintf(stderr, "[Error] Cannot open weights.bin\n");
            return 1;
      }

      Config cfg;
      fread(&cfg.vocab_size, sizeof(int), 1, fw);
      fread(&cfg.block_size, sizeof(int), 1, fw);
      fread(&cfg.n_embd, sizeof(int), 1, fw);
      fread(&cfg.n_head, sizeof(int), 1, fw);
      fread(&cfg.n_layer, sizeof(int), 1, fw);

      int nl = cfg.n_layer, nh = cfg.n_head;

      Weights W = {0};
      W.tok_emb = read_upload(fw);
      W.pos_emb = read_upload(fw);

      W.head_k = (float **)malloc(nl * nh * sizeof(float *));
      W.head_q = (float **)malloc(nl * nh * sizeof(float *));
      W.head_v = (float **)malloc(nl * nh * sizeof(float *));
      W.sa_proj_w = (float **)malloc(nl * sizeof(float *));
      W.sa_proj_b = (float **)malloc(nl * sizeof(float *));
      W.ff_w1 = (float **)malloc(nl * sizeof(float *));
      W.ff_b1 = (float **)malloc(nl * sizeof(float *));
      W.ff_w2 = (float **)malloc(nl * sizeof(float *));
      W.ff_b2 = (float **)malloc(nl * sizeof(float *));
      W.ln1_w = (float **)malloc(nl * sizeof(float *));
      W.ln1_b = (float **)malloc(nl * sizeof(float *));
      W.ln2_w = (float **)malloc(nl * sizeof(float *));
      W.ln2_b = (float **)malloc(nl * sizeof(float *));

      for (int l = 0; l < nl; l++)
      {
            for (int h = 0; h < nh; h++)
            {
                  W.head_k[l * nh + h] = read_upload(fw);
                  W.head_q[l * nh + h] = read_upload(fw);
                  W.head_v[l * nh + h] = read_upload(fw);
            }
            W.sa_proj_w[l] = read_upload(fw);
            W.sa_proj_b[l] = read_upload(fw);
            W.ff_w1[l] = read_upload(fw);
            W.ff_b1[l] = read_upload(fw);
            W.ff_w2[l] = read_upload(fw);
            W.ff_b2[l] = read_upload(fw);
            W.ln1_w[l] = read_upload(fw);
            W.ln1_b[l] = read_upload(fw);
            W.ln2_w[l] = read_upload(fw);
            W.ln2_b[l] = read_upload(fw);
      }

      W.ln_f_w = read_upload(fw);
      W.ln_f_b = read_upload(fw);
      W.lm_w = read_upload(fw);
      W.lm_b = read_upload(fw);
      fclose(fw);

      printf("--- Model loaded ---\n");
      printf("[INFO] vocab=%d  block=%d  embd=%d  heads=%d  layers=%d\n",
             cfg.vocab_size, cfg.block_size, cfg.n_embd, cfg.n_head, cfg.n_layer);
      printf("Generating text (Ctrl+C to stop)...\n\n");
      printf("--------------------------------------------------\n");

      // cuBLAS handle
      cublasHandle_t cublas;
      CUBLAS_CHECK(cublasCreate(&cublas));

      // GPU token buffer
      int *d_tokens;
      CUDA_CHECK(cudaMalloc(&d_tokens, cfg.block_size * sizeof(int)));

      // GPU logits buffer
      float *d_logits;
      CUDA_CHECK(cudaMalloc(&d_logits, cfg.vocab_size * sizeof(float)));

      // CPU logits for sampling
      float *logits = (float *)malloc(cfg.vocab_size * sizeof(float));

      // Context window
      int *ctx = (int *)calloc(cfg.block_size, sizeof(int));
      int ctx_len = 1;
      ctx[0] = 0;

      while (running)
      {
            int T = ctx_len < cfg.block_size ? ctx_len : cfg.block_size;
            int *window = ctx + (ctx_len - T);

            // Upload tokens to GPU
            CUDA_CHECK(cudaMemcpy(d_tokens, window,
                                  T * sizeof(int), cudaMemcpyHostToDevice));

            forward(d_logits, d_tokens, T, &cfg, &W, cublas);

            // Download logits
            CUDA_CHECK(cudaMemcpy(logits, d_logits,
                                  cfg.vocab_size * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            softmax_cpu(logits, cfg.vocab_size);
            int next = sample(logits, cfg.vocab_size);

            printf("%c", vocab[next]);
            fflush(stdout);

            if (ctx_len < cfg.block_size)
                  ctx[ctx_len++] = next;
            else
            {
                  memmove(ctx, ctx + 1, (cfg.block_size - 1) * sizeof(int));
                  ctx[cfg.block_size - 1] = next;
            }
      }

      cublasDestroy(cublas);
      cudaFree(d_tokens);
      cudaFree(d_logits);
      free(logits);
      free(ctx);
      free(vocab);

      return 0;
}