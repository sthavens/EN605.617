__kernel void matrixMultiplyTiled(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int N,
    const int K,
    __local float4* tileA_vec,
    __local float4* tileB_vec,
    const int rowOffset,
    const int columnOffset,
    const int pitchB
) {
    int row = get_global_id(0) + rowOffset;
    int column = get_global_id(1) * 4 + columnOffset;

    int localRow = get_local_id(0);
    int localColumn = get_local_id(1);

    int tileSize = get_local_size(0); // assuming a square tile in this case
    int vecWidth = 4;

    float4 sum = (float4) 0.0f;

    for (int t = 0; t < K; t += tileSize) {

        tileA_vec[localRow * (tileSize/vecWidth) + localColumn / vecWidth] = vload4(0, &A[row * K + t + localColumn * vecWidth]);

        tileB_vec[localRow * (tileSize/vecWidth) + localColumn/vecWidth] = vload4(0, &B[(t + localRow) * pitchB + localColumn * vecWidth]);


        // Prevents advancement until all threads have finished loading
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < tileSize / vecWidth; ++k) {
            sum += tileA_vec[localRow * (tileSize/vecWidth) + k] * tileB_vec[k * (tileSize/vecWidth) + localColumn / vecWidth];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && column < N) {
        vstore4(sum, 0, &C[row * N + column]);
    }
}