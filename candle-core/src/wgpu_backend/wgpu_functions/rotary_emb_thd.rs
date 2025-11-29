use candle_wgpu_kernels::Constants;

use super::*;

#[allow(clippy::too_many_arguments)]
pub fn queue_rotary_emb_thd(
    dev: &WgpuDevice,
    buffer_src: (BufferReferenceId, u32),
    buffer_cos: (BufferReferenceId, u32),
    buffer_sin: (BufferReferenceId, u32),
    dtype: crate::DType,
    buffer_dest: BufferReferenceId,
    unbatched: bool,
    bthd: (u32, u32, u32, u32),
) -> crate::Result<()> {
    let (b, t, h, d) = bthd;
    let (buffer_src, src_offset) = buffer_src;
    let (buffer_cos, cos_offset) = buffer_cos;
    let (buffer_sin, sin_offset) = buffer_sin;

    // D must be even for contiguous rotary
    debug_assert!(d % 2 == 0, "RotaryEmbThd requires even head_dim (d)");

    // Workgroup layout matches WGSL:
    //   @workgroup_size(8,8,1)
    //   global_id.x -> batch index (0..B-1)
    //   global_id.y -> flat over T * H * (D/2)

    let workgroup_size_x: u32 = 8;
    let workgroup_size_y: u32 = 8;

    let num_invocations_x = b;
    let num_invocations_y = t * h * (d / 2);

    fn ceil_div(a: u32, b: u32) -> u32 {
        if a == 0 {
            0
        } else {
            (a + b - 1) / b
        }
    }

    let workgroup_count_x = ceil_div(num_invocations_x, workgroup_size_x);
    let workgroup_count_y = ceil_div(num_invocations_y, workgroup_size_y);

    let mut queue = dev.get_queue();

    // op_meta[0..7] = B, T, H, D, src_offset, cos_offset, sin_offset
    queue.add(b);
    queue.add(t);
    queue.add(h);
    queue.add(d);
    queue.add(src_offset);
    queue.add(cos_offset);
    queue.add(sin_offset);

    // op_meta[7] = unbatched flag (0/1)
    queue.add_const(Constants::Constv0, unbatched);

    let pipeline = queue.get_pipeline(Pipelines::RotaryEmbThd(
        dev.get_dtype(dtype)?,
        candle_wgpu_kernels::rotary_emb_thd::Functions::RotaryEmbThd,
    ));

    // dest + 3 inputs (src, cos, sin)
    let bind_group = dev.create_bind_group_input3(
        buffer_dest,
        buffer_src,
        buffer_cos,
        buffer_sin,
        dtype.into(),
    );

    queue.enqueue_workgroups(
        pipeline,
        bind_group,
        workgroup_count_x,
        workgroup_count_y,
        1,
        (b * t * h * d) as usize,
    );

    Ok(())
}
