"""
5D partition_view -> tile load/store sample.

Matches design doc scenario:
1) make_tensor_view defines 5D global tensor
2) partition_view slices a 5D window
3) tload collapses to 2D tile_buf
4) tstore writes back to the original global tensor
"""

from mlir.ir import (
    Context,
    Location,
    InsertionPoint,
    Module,
    IntegerType,
    IndexType,
    ShapedType,
)
from mlir.dialects import arith, func, pto, builtin


def idx(val: int):
    return arith.ConstantOp(IndexType.get(), val).result


def build_module():
    with Context() as ctx, Location.unknown():
        pto.register_dialect(ctx, load=True)

        f32 = builtin.F32Type.get()
        # Address space helpers
        mat = pto.AddressSpaceAttr.get(2, ctx)  # MAT

        # Types
        tensor_view_ty = pto.TensorViewType.get([1, 1, 16, 1024, 1024], f32)
        part_view_ty = pto.PartitionTensorViewType.get([1, 1, 16, 16, 16], f32)
        tile_buf_ty = pto.TileBufType.get(
            [256, 16], f32, mat, [256, 16], pto.TileBufConfigAttr.get_default(ctx)
        )

        # Func signature: src, dst base pointers (ptr type)
        ptr_f32 = pto.PtrType.get(f32)

        m = Module.create()
        with InsertionPoint(m.body):
            @func.FuncOp.from_py_func(ptr_f32, ptr_f32)
            def run_partition(src, dst):
                c0 = idx(0)
                c1 = idx(1)
                # Shapes/strides for make_tensor_view
                shape = [idx(1), idx(1), idx(16), idx(1024), idx(1024)]
                strides = [idx(1048576), idx(1048576), idx(1048576), idx(1024), idx(1)]

                base_view = pto.MakeTensorViewOp(tensor_view_ty, src, shape, strides).result

                part = pto.PartitionViewOp(
                    part_view_ty,
                    base_view,
                    offsets=[c0, c0, c0, c0, c0],
                    sizes=[idx(1), idx(1), idx(16), idx(16), idx(16)],
                ).result

                # Tile buffer
                tile = pto.AllocTileOp(tile_buf_ty).result

                # Load high-d partition into 2D tile
                pto.TLoadOp(None, part, tile)

                # Store back to destination base pointer (reuse original base view)
                # Reinterpret dst pointer as the same tensor view then partition & store.
                dst_view = pto.MakeTensorViewOp(tensor_view_ty, dst, shape, strides).result
                dst_part = pto.PartitionViewOp(
                    part_view_ty,
                    dst_view,
                    offsets=[c0, c0, c0, c0, c0],
                    sizes=[idx(1), idx(1), idx(16), idx(16), idx(16)],
                ).result
                pto.TStoreOp(None, tile, dst_part)

                return

        return m


if __name__ == "__main__":
    module = build_module()
    print(module)
