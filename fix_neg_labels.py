"""Fix ALL preprocessed .b2nd seg files:
1. Files corrupted by previous blosc2.save_array() → reconstruct as NDArray
2. Files with -1 values → replace with 0
Uses blosc2.asarray() to preserve NDArray format compatible with nnU-Net."""
import blosc2
import numpy as np
import os

DATA_DIR = '/gscratch/scrubbed/june0604/wisespine_for_abnormal/outputs/nnunet/nnUNet_preprocessed/Dataset500_SpineAbnormal_Original/nnUNetPlans_3d_fullres'

seg_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('_seg.b2nd')])
print(f'Total seg files: {len(seg_files)}')

fixed_count = 0
recovered_count = 0
error_count = 0

for i, f in enumerate(seg_files):
    path = os.path.join(DATA_DIR, f)
    
    # Get the corresponding data file for shape reference
    data_file = f.replace('_seg.b2nd', '.b2nd')
    data_path = os.path.join(DATA_DIR, data_file)
    
    nd = blosc2.open(path)
    is_ndarray = isinstance(nd, blosc2.NDArray)
    
    if is_ndarray:
        # Normal NDArray file — check and fix -1 values
        data = nd[:]
        if np.any(data < 0):
            data[data < 0] = 0
            chunks = nd.chunks
            blocks = nd.blocks
            os.remove(path)
            blosc2.asarray(
                np.ascontiguousarray(data),
                urlpath=path,
                chunks=chunks,
                blocks=blocks,
                cparams={'codec': blosc2.Codec.ZSTD}
            )
            fixed_count += 1
            if fixed_count <= 5 or fixed_count % 100 == 0:
                print(f'  Fixed -1: {f}')
    else:
        # Previously corrupted file (SChunk from save_array) — need to recover
        try:
            # Get shape from corresponding data file  
            data_nd = blosc2.open(data_path)
            # seg shape is (1, z, y, x) where data shape is (C, z, y, x)
            spatial_shape = data_nd.shape[1:]
            seg_shape = (1,) + spatial_shape
            
            # Read raw bytes from SChunk
            raw_bytes = nd[:]
            seg_data = np.frombuffer(raw_bytes, dtype=np.int16).reshape(seg_shape).copy()
            
            # Fix -1 values
            seg_data[seg_data < 0] = 0
            
            # Get chunks/blocks from reference data file
            # For seg files, adjust the first dim chunk from C to 1
            ref_chunks = list(data_nd.chunks)
            ref_chunks[0] = 1
            ref_blocks = list(data_nd.blocks)
            ref_blocks[0] = 1
            
            os.remove(path)
            blosc2.asarray(
                np.ascontiguousarray(seg_data),
                urlpath=path,
                chunks=tuple(ref_chunks),
                blocks=tuple(ref_blocks),
                cparams={'codec': blosc2.Codec.ZSTD}
            )
            recovered_count += 1
            if recovered_count <= 10 or recovered_count % 100 == 0:
                print(f'  Recovered: {f} shape={seg_shape}')
        except Exception as e:
            print(f'  ERROR: {f}: {e}')
            error_count += 1

    if (i + 1) % 200 == 0:
        print(f'  Progress: {i+1}/{len(seg_files)} (fixed={fixed_count}, recovered={recovered_count}, errors={error_count})')

print(f'\nDone!')
print(f'  Fixed -1 values: {fixed_count}')
print(f'  Recovered corrupted: {recovered_count}')
print(f'  Errors: {error_count}')
