#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

int v_num = 0;
int e_num = 0;
int F0 = 0, F1 = 0, F2 = 0;

vector<vector<int>> edge_index;
vector<vector<float>> edge_val;
vector<int> degree;
vector<int> raw_graph;

float *X0, *W1, *W2, *X1, *X1_inter, *X2, *X2_inter;

void readGraph(char *fname) {
  ifstream infile(fname);

  int source;
  int end;

  infile >> v_num >> e_num;

  while (!infile.eof()) {
    infile >> source >> end;
    if (infile.peek() == EOF) break;
    raw_graph.push_back(source);
    raw_graph.push_back(end);
  }
}

void raw_graph_to_AdjacencyList() {
  int src;
  int dst;

  edge_index.resize(v_num);
  edge_val.resize(v_num);
  degree.resize(v_num, 0);

  for (int i = 0; i < raw_graph.size() / 2; i++) {
    src = raw_graph[2 * i];
    dst = raw_graph[2 * i + 1];
    edge_index[dst].push_back(src);
    degree[src]++;
  }
}

void edgeNormalization() {
  for (int i = 0; i < v_num; i++) {
    for (int j = 0; j < edge_index[i].size(); j++) {
      float val = 1 / sqrt(degree[i]) / sqrt(degree[edge_index[i][j]]);
      edge_val[i].push_back(val);
    }
  }
}

void readFloat(char *fname, float *&dst, int num) {
  dst = (float *)malloc(num * sizeof(float));
  FILE *fp = fopen(fname, "rb");
  fread(dst, num * sizeof(float), 1, fp);
  fclose(fp);
}

void initFloat(float *&dst, int num) {
  dst = (float *)malloc(num * sizeof(float));
  memset(dst, 0, num * sizeof(float));
}

void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W) {
  float(*tmp_in_X)[in_dim] = (float(*)[in_dim])in_X;
  float(*tmp_out_X)[out_dim] = (float(*)[out_dim])out_X;
  float(*tmp_W)[out_dim] = (float(*)[out_dim])W;

#pragma omp parallel for
  for (int i = 0; i < v_num; i++) {
    for (int j = 0; j < out_dim; j++) {
      for (int k = 0; k < in_dim; k++) {
        tmp_out_X[i][j] += tmp_in_X[i][k] * tmp_W[k][j];
      }
    }
  }
}

void AX(int dim, float *in_X, float *out_X) {
  float(*tmp_in_X)[dim] = (float(*)[dim])in_X;
  float(*tmp_out_X)[dim] = (float(*)[dim])out_X;

#pragma omp parallel for
  for (int i = 0; i < v_num; i++) {
    vector<int> &nlist = edge_index[i];
    for (int j = 0; j < nlist.size(); j++) {
      int nbr = nlist[j];
      for (int k = 0; k < dim; k++) {
        tmp_out_X[i][k] += tmp_in_X[nbr][k] * edge_val[i][j];
      }
    }
  }
}

void ReLU(int dim, float *X) {
#pragma omp parallel for
  for (int i = 0; i < v_num * dim; i++)
    if (X[i] < 0) X[i] = 0;
}

void LogSoftmax(int dim, float *X) {
  float(*tmp_X)[dim] = (float(*)[dim])X;

#pragma omp parallel for
  for (int i = 0; i < v_num; i++) {
    float max = tmp_X[i][0];
    for (int j = 1; j < dim; j++) {
      if (tmp_X[i][j] > max) max = tmp_X[i][j];
    }

    float sum = 0;
    for (int j = 0; j < dim; j++) {
      sum += exp(tmp_X[i][j] - max);
    }
    sum = log(sum);

    for (int j = 0; j < dim; j++) {
      tmp_X[i][j] = tmp_X[i][j] - max - sum;
    }
  }
}

float MaxRowSum(float *X, int dim) {
  float(*tmp_X)[dim] = (float(*)[dim])X;
  float max = -__FLT_MAX__;

#pragma omp parallel for reduction(max : max)
  for (int i = 0; i < v_num; i++) {
    float sum = 0;
    for (int j = 0; j < dim; j++) {
      sum += tmp_X[i][j];
    }
    if (sum > max) max = sum;
  }
  return max;
}

void freeFloats() {
  free(X0);
  free(W1);
  free(W2);
  free(X1);
  free(X2);
  free(X1_inter);
  free(X2_inter);
}

void somePreprocessing() {
  raw_graph_to_AdjacencyList();
}

void residualBlock(int in_dim, int out_dim, float *in_X, float *out_X, float *W, bool apply_relu) {
  initFloat(out_X, v_num * out_dim);
  float* intermediate_X = (float*)malloc(v_num * out_dim * sizeof(float));
  memset(intermediate_X, 0, v_num * out_dim * sizeof(float));

  XW(in_dim, out_dim, in_X, intermediate_X, W);
  AX(out_dim, intermediate_X, out_X);

  if (apply_relu) {
    ReLU(out_dim, out_X);
  }

  // Add residual connection
#pragma omp parallel for
  for (int i = 0; i < v_num; i++) {
    for (int j = 0; j < out_dim; j++) {
      out_X[i * out_dim + j] += in_X[i * in_dim + j];
    }
  }

  free(intermediate_X);
}

int main(int argc, char **argv) {
  F0 = atoi(argv[1]);
  F1 = atoi(argv[2]);
  F2 = atoi(argv[3]);

  readGraph(argv[4]);
  readFloat(argv[5], X0, v_num * F0);
  readFloat(argv[6], W1, F0 * F1);
  readFloat(argv[7], W2, F1 * F2);

  initFloat(X1, v_num * F1);
  initFloat(X1_inter, v_num * F1);
  initFloat(X2, v_num * F2);
  initFloat(X2_inter, v_num * F2);

  TimePoint start = chrono::steady_clock::now();

  somePreprocessing();
  edgeNormalization();

  // Layer 1 with residual connection
  residualBlock(F0, F1, X0, X1, W1, true);

  // Layer 2 with residual connection
  residualBlock(F1, F2, X1, X2, W2, false);

  LogSoftmax(F2, X2);

  float max_sum = MaxRowSum(X2, F2);

  TimePoint end = chrono::steady_clock::now();
  chrono::duration<double> l_durationSec = end - start;
  double l_timeMs = l_durationSec.count() * 1e3;

  printf("%.8f\n", max_sum);
  printf("%.8lf\n", l_timeMs);

  freeFloats();
}
