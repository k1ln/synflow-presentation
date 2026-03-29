'use strict';
// ── Synflow benchmark v4 ────────────────────────────────────────────
// Buffer-performance comparison: C++ native vs AudioWorklet JS vs WASM
// Each series stops immediately on the first real-time underrun.
// Run with:  node benchmark/run.js

const os   = require('os');
const path = require('path');
const fs   = require('fs');

// ── Config ──────────────────────────────────────────────────────────
const VOICE_COUNTS = [100, 250, 500, 1000, 2500, 5000, 10000, 20000, 50000];
const WARMUP       = 300;
const BUDGET_MS    = (128 / 44100) * 1000;   // 2.9025 ms
const SAMPLE_RATE  = 44100;
const BUFFER_SIZE  = 128;
const TABLE_SIZE   = 512;

// Adaptive iterations: roughly constant benchmark duration per step
function adaptiveIters(voices) {
  return Math.max(100, Math.min(1000, Math.floor(800000 / voices)));
}

// ── WASM binary (real MAcc tight loop) ────────────────────────────
// Built programmatically so f64 constants are correctly encoded.
// WAT equivalent:
//   (func (export "macc") (param $n i32)
//     (local $i i32) (local $acc f64)
//     (block $B (loop $L
//       (br_if $B (i32.ge_u (local.get $i) (local.get $n)))
//       (local.set $acc (f64.add (f64.mul (local.get $acc) (f64.const 0.9998)) (f64.const 0.001)))
//       (local.set $i  (i32.add (local.get $i) (i32.const 1)))
//       (br $L)))
//   )
// Comparability note: JS and C++ run sinf/Math.sin per sample;
// WASM runs a mul-acc DSP loop — same iteration count, lighter per-op cost.
// Label each series clearly in the presentation.
function buildWasmBytes() {
  function f64le(v) {
    const b = new ArrayBuffer(8);
    new DataView(b).setFloat64(0, v, true); // little-endian IEEE 754
    return Array.from(new Uint8Array(b));
  }
  const body = [
    // locals: 1×i32 ($i), 1×f64 ($acc)
    0x02, 0x01, 0x7f, 0x01, 0x7c,
    0x02, 0x40,              // block $B
      0x03, 0x40,            // loop $L
        0x20, 0x01,          // local.get $i
        0x20, 0x00,          // local.get $n
        0x4f,                // i32.ge_u
        0x0d, 0x01,          // br_if $B (exit loop)
        0x20, 0x02,          // local.get $acc
        0x44, ...f64le(0.9998), // f64.const 0.9998
        0xa2,                // f64.mul
        0x44, ...f64le(0.001),  // f64.const 0.001
        0xa0,                // f64.add
        0x21, 0x02,          // local.set $acc
        0x20, 0x01,          // local.get $i
        0x41, 0x01,          // i32.const 1
        0x6a,                // i32.add
        0x21, 0x01,          // local.set $i
        0x0c, 0x00,          // br $L (continue)
      0x0b,                  // end loop
    0x0b,                    // end block
    0x0b,                    // end func
  ];
  const codeVec = [0x01, body.length, ...body];
  return new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // magic + version
    0x01, 0x05, 0x01, 0x60, 0x01, 0x7f, 0x00,        // type section: func(i32)->void
    0x03, 0x02, 0x01, 0x00,                           // function section
    0x07, 0x08, 0x01, 0x04, 0x6d, 0x61, 0x63, 0x63, 0x00, 0x00, // export "macc"
    0x0a, codeVec.length, ...codeVec,                 // code section
  ]);
}

let wasmMacc = null;
(async () => {
  try {
    const mod = await WebAssembly.instantiate(buildWasmBytes());
    wasmMacc = mod.instance.exports.macc;
  } catch(e) {
    console.error('WASM init failed:', e.message);
  }
})();

// ── Benchmarks ─────────────────────────────────────────────────────
function benchJSSin(voices, iters) {
  const buf = new Float32Array(BUFFER_SIZE);
  let phase = 0;
  const freq = 440;
  const inc  = (2 * Math.PI * freq) / SAMPLE_RATE;
  // warmup
  for (let w = 0; w < WARMUP; w++) {
    for (let v = 0; v < voices; v++)
      for (let i = 0; i < BUFFER_SIZE; i++) { buf[i] = Math.sin(phase); phase += inc; }
  }
  const t0 = performance.now();
  for (let it = 0; it < iters; it++)
    for (let v = 0; v < voices; v++)
      for (let i = 0; i < BUFFER_SIZE; i++) { buf[i] = Math.sin(phase); phase += inc; }
  return (performance.now() - t0) / iters;
}

function benchWasmMacc(macc, voices, iters) {
  const n = BUFFER_SIZE * voices;
  for (let w = 0; w < WARMUP; w++) macc(n);
  const t0 = performance.now();
  for (let it = 0; it < iters; it++) macc(n);
  return (performance.now() - t0) / iters;
}

// ── Helpers ─────────────────────────────────────────────────────────
function cpuModel() {
  const cpus = os.cpus();
  if (!cpus || !cpus.length) return { model: 'Unknown', cores: 0, speedMHz: 0 };
  return { model: cpus[0].model.trim(), cores: cpus.length, speedMHz: cpus[0].speed };
}

function pad(s, n) { return String(s).padEnd(n); }
function fmt(ms)   { return ms.toFixed(3) + ' ms'; }
function flag(ms)  { return ms > BUDGET_MS * 2 ? '☠ SEVERE' : ms > BUDGET_MS ? '⚠ OVER' : '✓ OK'; }

// ── Run one benchmark series, stopping on first underrun ─────────────
function runSeries(label, benchFn) {
  console.log(`\n── ${label} ${'─'.repeat(Math.max(0, 53 - label.length))}`);
  const series = [];
  for (const voices of VOICE_COUNTS) {
    const iters = adaptiveIters(voices);
    const ms    = benchFn(voices, iters);
    const over  = ms > BUDGET_MS;
    series.push({ voices, ms, over });
    console.log(`  voices=${pad(voices, 6)}  iters=${pad(iters, 5)}  ${pad(fmt(ms), 11)}  ${flag(ms)}`);
    if (over) {
      console.log(`  ⛔ Underrun — stopping series.\n`);
      break;
    }
  }
  return series;
}

// ── Main ─────────────────────────────────────────────────────────────
async function main() {
  // Wait for WASM init (give async IIFE 200ms)
  await new Promise(r => setTimeout(r, 200));

  const proc = cpuModel();
  console.log(`\n── Synflow Buffer-Performance Benchmark v4 ───────────────`);
  console.log(`CPU:    ${proc.model} (${proc.cores} cores @ ~${proc.speedMHz} MHz)`);
  console.log(`OS:     ${process.platform}  arch: ${process.arch}  Node: ${process.version}`);
  console.log(`Budget: ${BUDGET_MS.toFixed(4)} ms  (${BUFFER_SIZE} samples @ ${SAMPLE_RATE} Hz)`);
  console.log(`WASM:   ${wasmMacc ? 'available' : 'UNAVAILABLE (JS fallback used)'}`);
  console.log(`Each series stops at the first real-time underrun.\n`);
  console.log(`─────────────────────────────────────────────────────────`);

  const results = {};

  // ── 1. AudioWorklet JS (Math.sin — representative AudioWorklet compute)
  results.audioWorkletJS = runSeries(
    'AudioWorklet JS  (Math.sin per voice)',
    (voices, iters) => benchJSSin(voices, iters)
  );

  // ── 2. WebAssembly inside AudioWorklet (tight MAcc mul-acc loop — real measured)
  if (wasmMacc) {
    results.wasm = runSeries(
      'WebAssembly in AudioWorklet (mul-acc DSP loop)',
      (voices, iters) => benchWasmMacc(wasmMacc, voices, iters)
    );
  } else {
    console.log(`\n⚠ WASM series skipped — module failed to initialise.`);
  }

  // ── 3. C++ native (estimated from JS ÷ CPP_FACTOR unless real data available)
  let cppMeta = null;
  const cppPath = path.join(__dirname, 'cpp_results.json');
  if (fs.existsSync(cppPath)) {
    try {
      const cpp = JSON.parse(fs.readFileSync(cppPath, 'utf8'));
      cppMeta = { compiler: cpp.compiler, arch: cpp.arch, generator: cpp.generator };

      // Use real measured sinf data from osc_bench; stop on first underrun
      results.cppNative = runSeries(
        `C++ Native (sinf, ${cpp.compiler})`,
        (voices) => {
          const row = (cpp.results.sinf || []).find(r => r.voices === voices);
          return row ? row.ms : null;
        }
      ).filter(r => r.ms !== null);

      if (cpp.results.wavetable) {
        results.cppWavetable = runSeries(
          `C++ Wavetable (${cpp.compiler})`,
          (voices) => {
            const row = (cpp.results.wavetable || []).find(r => r.voices === voices);
            return row ? row.ms : null;
          }
        ).filter(r => r.ms !== null);
      }
      if (cpp.results.neon) {
        results.cppNeon = runSeries(
          `C++ NEON SIMD (${cpp.compiler})`,
          (voices) => {
            const row = (cpp.results.neon || []).find(r => r.voices === voices);
            return row ? row.ms : null;
          }
        ).filter(r => r.ms !== null);
      }

      console.log(`ℹ  C++ data from: ${cppPath}`);
      console.log(`   Compiler: ${cpp.compiler}  arch: ${cpp.arch}`);
    } catch(e) {
      console.warn(`\n⚠ Could not parse cpp_results.json: ${e.message}`);
    }
  } else {
    console.log(`\n⚠  C++ series skipped — no cpp_results.json found.`);
    console.log(`   Generate real data with:`);
    console.log(`     clang++ -O3 -std=c++17 -march=native -o benchmark/osc_bench benchmark/osc_bench.cpp`);
    console.log(`     ./benchmark/osc_bench > benchmark/cpp_results.json`);
    console.log(`   Then re-run:  node benchmark/run.js`);
  }

  // ── Summary ───────────────────────────────────────────────────────
  console.log(`\n── Underrun summary (first voice count that exceeded budget) `);
  const seriesLabels = {
    audioWorkletJS: 'AudioWorklet JS',
    wasm:           'WebAssembly',
    cppNative:      'C++ Native',
    cppWavetable:   'C++ Wavetable',
    cppNeon:        'C++ NEON SIMD',
  };
  for (const [key, label] of Object.entries(seriesLabels)) {
    if (!results[key]) continue;
    const first = results[key].find(r => r.over);
    const msg   = first
      ? `first underrun at ${first.voices} voices  (${first.ms.toFixed(3)} ms)`
      : `no underrun within tested range`;
    console.log(`  ${pad(label, 18)}: ${msg}`);
  }

  // ── Write results.js ──────────────────────────────────────────────
  const data = {
    timestamp:    new Date().toISOString(),
    processor:    proc,
    platform:     process.platform,
    arch:         process.arch,
    nodeVersion:  process.version,
    sampleRate:   SAMPLE_RATE,
    bufferSize:   BUFFER_SIZE,
    budgetMs:     BUDGET_MS,
    tableSize:    TABLE_SIZE,
    wasmAvailable: !!wasmMacc,
    cppMeta,
    voiceCounts:  VOICE_COUNTS,
    results,
  };

  const out = `// Auto-generated by benchmark/run.js — ${data.timestamp}\n// Do not edit manually.\nwindow.BENCH_DATA = ${JSON.stringify(data, null, 2)};\n`;
  const outPath = path.join(__dirname, 'results.js');
  fs.writeFileSync(outPath, out, 'utf8');
  console.log(`\n✅ results.js written (${fs.statSync(outPath).size} bytes)\n`);
}

main().catch(e => { console.error(e); process.exit(1); });
