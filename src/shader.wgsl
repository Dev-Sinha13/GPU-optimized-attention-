// Flash Attention Compute Shader (WGSL)
// This implements a simplified Flash Attention forward pass logic:
// O = softmax(Q * K^T / sqrt(d)) * V
// We use a tile-based approach utilizing workgroup memory to minimize global memory access.

struct MatrixInfo {
    seq_len: u32,  // N (Sequence length)
    head_dim: u32, // d (Head dimension size)
    // For simplicity, we assume sequence length N is a multiple of BLOCK_SIZE
    // and head_dim d is equal to BLOCK_SIZE
}

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<uniform> info: MatrixInfo;

// Define the tile size.
// To maximize hardware occupancy and stay within workgroup memory limits,
// we use blocks of 16x16 or 32x32. Here we use 16x16 for broad compatibility.
const BLOCK_SIZE: u32 = 16u;

// Workgroup memory allocations
var<workgroup> q_tile: array<f32, 256>; // BLOCK_SIZE * BLOCK_SIZE (16x16 = 256 f32s)
var<workgroup> k_tile: array<f32, 256>;
var<workgroup> v_tile: array<f32, 256>;

// Online softmax state (stored per query row in a block)
var<workgroup> m_i: array<f32, 16>; // running max (size = BLOCK_SIZE)
var<workgroup> l_i: array<f32, 16>; // running sum (size = BLOCK_SIZE)

@compute @workgroup_size(16, 16, 1) // (BLOCK_SIZE, BLOCK_SIZE, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let tx = local_id.x; // col in tile
    let ty = local_id.y; // row in tile
    let n = info.seq_len;
    let d = info.head_dim;
    let global_row = group_id.y * BLOCK_SIZE + ty; // Assuming 1D grid of workgroups handling blocks of Q
    
    // Each workgroup computes one BLOCK_SIZE x d block of the Output matrix.
    // In our simplified setup, we assume d = BLOCK_SIZE = 16 to fit cleanly in one tile width.
    // Real-world implementations handle d > BLOCK_SIZE by wrapping/chunking.

    if (global_row >= n) { return; }

    // Initialize online softmax states and output accumulator
    if (tx == 0u) {
        m_i[ty] = -1e38; // Initial max
        l_i[ty] = 0.0;   // Initial sum
    }
    
    // Accumulator for attention scores dot products
    var o_acc: f32 = 0.0;

    // Outer loop: Iterate over K and V blocks (j direction)
    let num_blocks: u32 = n / BLOCK_SIZE; // Assumes n is divisible by BLOCK_SIZE
    
    for (var j: u32 = 0u; j < num_blocks; j++) {
        // --- 1. Load tiles into shared memory ---
        // Q tile: loaded once (technically we could load this outside the loop if workgroup mem was larger, 
        // but reloading or holding it relies on register/L1 caching)
        let q_col = tx;
        let q_row = ty;
        let q_idx = (group_id.y * BLOCK_SIZE + q_row) * d + q_col;
        q_tile[q_row * BLOCK_SIZE + q_col] = Q[q_idx];

        // K tile (Note: carefully transpose memory access conceptually later)
        let k_col = tx;
        let k_row = ty;
        let k_idx = (j * BLOCK_SIZE + k_row) * d + k_col; // Access row-major original K
        k_tile[k_row * BLOCK_SIZE + k_col] = K[k_idx];

        // V tile
        let v_col = tx;
        let v_row = ty;
        let v_idx = (j * BLOCK_SIZE + v_row) * d + v_col;
        v_tile[v_row * BLOCK_SIZE + v_col] = V[v_idx];

        workgroupBarrier();

        // --- 2. Compute Q * K^T tile ---
        // S_ij = Q_i * K_j^T
        var s_ij: f32 = 0.0;
        for (var k: u32 = 0u; k < BLOCK_SIZE; k++) {
            // q_tile is BLOCK_SIZE x BLOCK_SIZE, read row `ty`
            // k_tile is conceptually transposed, so read row `tx` of K
            let q_val = q_tile[ty * BLOCK_SIZE + k];
            let k_val = k_tile[tx * BLOCK_SIZE + k]; // Notice we use tx as row, k as col for transposed access
            s_ij += q_val * k_val;
        }
        
        // Scale by 1 / sqrt(d)
        // Note: scaling in WGSL requires sqrt(f32) cast
        s_ij = s_ij / sqrt(f32(d));

        workgroupBarrier(); // Sync before calculating local max

        // Calculate max for this block row for numerically stable softmax
        // We need a reduction across the row. Since workgroup size is small, we can 
        // accumulate in a local array and let thread 0 do it, or do an unrolled reduction.
        // For simplicity and to avoid complex shuffles, thread x=0 scans the row `s_ij` values.
        // Because `s_ij` is only held in registers currently, we'll store it in workgroup mem temporarily.
        // We reuse `q_tile` as scratch space for S_ij since Q is already used for this step!
        workgroupBarrier();
        q_tile[ty * BLOCK_SIZE + tx] = s_ij;
        workgroupBarrier();

        // Online softmax state update (done by thread 0 of the row)
        var new_m_i: f32 = 0.0;
        var m_i_old: f32 = 0.0;

        if (tx == 0u) {
            m_i_old = m_i[ty];

            // find max in chunk for numerical stability
            var m_ij: f32 = -1e38;
            for (var c: u32 = 0u; c < BLOCK_SIZE; c++) {
                let v = q_tile[ty * BLOCK_SIZE + c];
                if (v > m_ij) { m_ij = v; }
            }

            // new max combining history and this chunk
            new_m_i = max(m_i_old, m_ij);

            // calculate sum of exp for this chunk
            var l_ij: f32 = 0.0;
            for (var c: u32 = 0u; c < BLOCK_SIZE; c++) {
                let v = q_tile[ty * BLOCK_SIZE + c];
                // Exponential of chunk, offset by new max (for stability)
                let p = exp(v - new_m_i); 
                q_tile[ty * BLOCK_SIZE + c] = p; // Store P_ij back
                l_ij += p;
            }

            // Update running denominator
            let old_l_i_scaled = exp(m_i_old - new_m_i) * l_i[ty];
            let new_l_i = old_l_i_scaled + l_ij;

            // Store new states
            m_i[ty] = new_m_i;
            l_i[ty] = new_l_i;

            // Store scalar for previous accumulated output
            // We reuse k_tile[ty] as temporary storage for these scalars to broadcast to row threads
            k_tile[ty] = exp(m_i_old - new_m_i); 
        }
        
        workgroupBarrier(); // Sync so all threads across the row see the states and updated P_ij

        // --- 3. Compute P * V tile and accumulate Output ---
        // O_i = (exp(m_old - m_new) * O_old) + P_i * V_j
        
        let p_scale_old = k_tile[ty]; 

        var p_v_dot: f32 = 0.0;
        for (var k: u32 = 0u; k < BLOCK_SIZE; k++) {
            let p_val = q_tile[ty * BLOCK_SIZE + k]; // P_ij
            let v_val = v_tile[k * BLOCK_SIZE + tx]; // V_j
            p_v_dot += p_val * v_val;
        }

        // Apply recursive scaling to the accumulator
        o_acc = p_scale_old * o_acc + p_v_dot;

        workgroupBarrier(); 
    }

    // --- 4. Final Output Write ---
    // Output = O / final_l
    // Notice O uses (local_id.y * n) to write to the correct row and col
    let final_l = l_i[ty];
    let final_o = o_acc / final_l;

    let o_col = tx;
    let o_row = group_id.y * BLOCK_SIZE + ty;
    let o_idx = o_row * d + o_col;

    O[o_idx] = final_o;
}
