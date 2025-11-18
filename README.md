# xxysr-wasm

osu!mania XXY SR calculation for WebAssembly, adapted from [mania-converter-rust](https://github.com/Siflorite/mania-converter-rust).

## Installation

```bash
pnpm add xxysr-wasm
```

## Usage

### Browser

```typescript
import initXxy, { calc_sr } from 'xxysr-wasm';

await initXxy();

const beatmapContent = '...'; // .osu file content
const rate = 1.0; // playback rate (ht=0.75, dt=1.5)
const sr = calc_sr(beatmapContent, rate);
```

### Node.js

```typescript
import { readFile } from 'fs/promises';
import init_xxy, { calc_sr } from 'xxysr-wasm';
import { fileURLToPath } from 'url';

const wasmPath = fileURLToPath(import.meta.resolve('xxysr-wasm/xxysr_wasm_bg.wasm'));
const wasmBuffer = await readFile(wasmPath);
await init_xxy(wasmBuffer);

const beatmapContent = await readFile('beatmap.osu', 'utf-8');
const sr = calc_sr(beatmapContent, 1.0);
```

## License

This project is adapted from [mania-converter-rust](https://github.com/Siflorite/mania-converter-rust), which is also licensed under Apache-2.0.
