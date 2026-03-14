use std::borrow::Cow;
use wgpu::util::DeviceExt;
use rand::Rng;

// A struct to represent the configuration of our Q, K, V matrices.
// We must derive bytemuck traits to cast this struct safely into raw bytes for the GPU to read.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MatrixInfo {
    seq_len: u32,
    head_dim: u32,
    // Add 8 bytes of padding to align to 16 bytes, a requirement for WebGPU Uniform buffers.
    _padding: [u32; 2],
}

// Ensure the block size matches what we hardcoded in WGSL.
const BLOCK_SIZE: usize = 16;

async fn run() {
    let seq_len = 64; // N (Must be multiple of 16 for our simple shader)
    let head_dim = 16; // d (Must be exactly 16 for our simple shader)

    println!("Initializing Flash Attention WebGPU test with N={} d={}", seq_len, head_dim);

    // 1. Initialize WebGPU
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("Failed to find an appropriate adapter");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(), // Our shader is simple enough
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .expect("Failed to create device");

    println!("Using GPU Adapter: {:?}", adapter.get_info().name);

    // 2. Load the Shader
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Flash Attention Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    // 3. Generate random standard normal data for Q, K, V on the CPU
    // Size = seq_len * head_dim
    let num_elements = seq_len * head_dim;
    let mut rng = rand::thread_rng();

    let mut q_data: Vec<f32> = (0..num_elements).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let mut k_data: Vec<f32> = (0..num_elements).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let mut v_data: Vec<f32> = (0..num_elements).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // 4. Copy data to buffers on the GPU
    // Using `create_buffer_init` handles allocating GPU VRAM and transferring the initial byte data safely.
    let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Q Buffer"),
        contents: bytemuck::cast_slice(&q_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let k_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("K Buffer"),
        contents: bytemuck::cast_slice(&k_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("V Buffer"),
        contents: bytemuck::cast_slice(&v_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // We only need a buffer large enough to map back the results (Output O)
    let o_size = (num_elements * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
    
    // Create an output storage buffer that the shader can write to it
    let o_buf_storage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Storage Buffer"),
        size: o_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Create a staging buffer so the CPU can map memory from the GPU back safely.
    // It must have MAP_READ and COPY_DST usage flags.
    let o_buf_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Staging Buffer"),
        size: o_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Uniform buffer for passing seq_len and head_dim sizes safely
    let info = MatrixInfo {
        seq_len: seq_len as u32,
        head_dim: head_dim as u32,
        _padding: [0, 0],
    };

    let info_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Matrix Info Uniform Buffer"),
        contents: bytemuck::cast_slice(&[info]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // 5. Construct Compute Pipeline
    // A Compute Pipeline takes a shader module and an explicit BindGroupLayout.
    // WebGPU can infer layouts implicitly, but being explicit generally provides better type safety and errors.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Flash Attention Pipeline"),
        layout: None, // Let wgpu auto-infer layout from shader bindings for simplicity
        module: &shader,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    // Creating the explicit bind group mapping variables to @binding entries in WGSL
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Flash Attention Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: k_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: v_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: o_buf_storage.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: info_buf.as_entire_binding(),
            },
        ],
    });

    // 6. Execute Compute Shader
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        // Safe scope dropping the compute_pass explicitly so we can call copy_buffer_to_buffer later
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        
        // We dispatch workgroups. Since workgroup size is BLOCK_SIZE x BLOCK_SIZE (16x16)
        // and computes an output block of N rows (seq_len) x BLOCK_SIZE columns (head_dim),
        // we calculate how many workgroups are required.
        // Specifically, one workgroup handles 1 block of size 16x16.
        let workgroups_x = 1; // Since head_dim = 16 = BLOCK_SIZE exactly, we only need 1 block across col axis
        let workgroups_y = (seq_len as u32) / (BLOCK_SIZE as u32);
        
        cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    // Command to copy data from VRAM storage buffer to accessible staging buffer
    encoder.copy_buffer_to_buffer(&o_buf_storage, 0, &o_buf_staging, 0, o_size);

    // Submit work to queue
    queue.submit(Some(encoder.finish()));

    // 7. Map Staging Buffer to Read Data Back
    let buffer_slice = o_buf_staging.slice(..);
    let (sender, receiver) = flume::bounded(1); // MPSC channel from standard library for sync safety
    
    // Explicit map request that queues memory to be available to CPU
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Tell the device to actively poll and execute all mapped work
    device.poll(wgpu::Maintain::Wait);

    // Block until callback confirms map is ready
    if let Ok(Ok(())) = receiver.recv() {
        // Safe lock on mapped view
        let data = buffer_slice.get_mapped_range();
        
        // Safely interpret the raw bytes as slices of floating-point numbers
        let gpu_result: &[f32] = bytemuck::cast_slice(&data);

        println!("GPU Flash Attention Initialized Successfully!");
        println!("  - Example GPU O matrix row 0: {:?}", &gpu_result[0..8]);
        
        // Manual drop to release map handle
        drop(data);
        o_buf_staging.unmap();
    } else {
        panic!("Failed to read data back from GPU.");
    }
}

fn main() {
    // Boilerplate for polling an async context safely
    pollster::block_on(run());
}
