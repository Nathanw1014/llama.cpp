#!/usr/bin/env python3
# copy a DSV4-backbone DSpark draft GGUF and inject a selector_hidden.weight
# tensor: a DFlash2-style selector on a backbone whose graph cannot build the lattice
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'gguf-py'))

import numpy as np
import gguf

if len(sys.argv) != 3:
    print(f'usage: {sys.argv[0]} <dspark-draft-in.gguf> <out.gguf>')
    sys.exit(1)

src, dst = sys.argv[1], sys.argv[2]

reader = gguf.GGUFReader(src)
arch = reader.get_field('general.architecture').contents()
assert arch == 'dflash', arch
assert reader.get_field('dflash.hyper_connection.count') is not None, 'input is not a DSV4-backbone draft'

writer = gguf.GGUFWriter(dst, arch, endianess=reader.endianess)
for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
        continue
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
    writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

for tensor in reader.tensors:
    writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)

sel = np.zeros((16, 16), dtype=np.float32)
writer.add_tensor_info('selector_hidden.weight', sel.shape, sel.dtype, sel.nbytes, gguf.GGMLQuantizationType.F32)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()

for tensor in reader.tensors:
    writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
writer.write_tensor_data(sel)

writer.close()
print('done:', dst)
