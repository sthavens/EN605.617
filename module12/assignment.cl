__kernel void matrixMultiplyTiled(
    __global const float* A,          
    __global const float* B,          
    __global float* C,                
    const int K,                      
    __local float4* tileA_vec,
    __local float4* tileB_vec,
    const int N_pitch
) {
    int localRow = get_local_id(0);   
    int localColumn = get_local_id(1);   
    int vecWidth = 4;               
    int tile_k = 32;

    int tileM = get_local_size(0);
    int tileN = get_local_size(1) * vecWidth;

    float4 sum = (float4)(0.0f);

    for (int t = 0; t < K; t += tile_k) {
        // Load tiles into local memory
        tileA_vec[localRow * (tile_k/vecWidth) + localColumn] = 
            vload4(0, &A[localRow * K + t + localColumn * vecWidth]);

        tileB_vec[localRow * (tile_k/vecWidth) + localColumn] = 
            vload4(0, &B[localRow * K * t + localColumn * vecWidth]);

        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply tiles
        for (int k = 0; k < tile_k / vecWidth; ++k) {
            sum += tileA_vec[localRow * (tile_k/vecWidth) + k] *
                   tileB_vec[localColumn * (tile_k / vecWidth) + k];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result to sub-buffer (local indexing is enough)
    vstore4(sum, 0, &C[localRow * N_pitch + localColumn * vecWidth]);
}
