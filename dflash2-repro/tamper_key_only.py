#!/usr/bin/env python3
# copy a DFlash v1 draft GGUF and inject the dflash.selector_top_k metadata key
# WITHOUT selector tensors: metadata claims DFlash2, the decode graph has no lattice
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'gguf-py'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gguf-py' / 'gguf' / 'scripts'))

import gguf
from gguf_new_metadata import MetadataDetails, copy_with_new_metadata

if len(sys.argv) != 3:
    print(f'usage: {sys.argv[0]} <v1-draft-in.gguf> <out.gguf>')
    sys.exit(1)

src, dst = sys.argv[1], sys.argv[2]

reader = gguf.GGUFReader(src)
arch = reader.get_field('general.architecture').contents()
assert arch == 'dflash', arch
assert reader.get_field('dflash.selector_top_k') is None, 'input is already a DFlash2 draft'

writer = gguf.GGUFWriter(dst, arch, endianess=reader.endianess)
new_metadata = {
    'dflash.selector_top_k': MetadataDetails(gguf.GGUFValueType.UINT32, 16, 'injected'),
}
copy_with_new_metadata(reader, writer, new_metadata, remove_metadata=[])
print('done:', dst)
