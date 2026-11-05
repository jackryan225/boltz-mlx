# Apple GPU generations as they matter for this code (M1 to M5)

Research note, 2026-10-28. CPU-only work: no GPU dispatches. The only device queries were `mx.device_info()` and `otool`.
Every fact carries a source key; the keys are listed under "Sources" at the end. Labels:
**[local]** means measured on the dev machine (Apple M5 Pro, macOS 26.6.2, MLX 0.32.4).
**secondary** means a non-Apple, non-MLX page. **recalled, unverified** means from memory with no source.
**estimate** means our own inference, not a measurement.
MLX source is tag `v0.32.4` (commit `36969c094354086e15d2a327a9c50feb20b9cf11`). The installed 0.32.4 wheel headers match it where compared.

**Our kernel paths** (`boltz_mlx/kernels.py`, the kernel-path detection block; owner decision 2026-10-28 22:30: macOS 26 is a hard requirement, two shipped paths). These apply to `--precision f16`; strict fp32 never builds MPP on any chip.
- **nax**: M5-class GPU (MLX's rule: gen ≥ 17, ≥ 18 for 'p'). The MetalPerformancePrimitives kernels (kernels_nax*, kernels_atom, `kernels_msa.opm_outer_proj`) run on the per-core Neural Accelerators.
- **tensorops**: the same MPP kernels on M1–M4 shader cores — the shipped M1–M4 path (the default there).
- **portable**: internal fallback / debug only — taken with a loud warning if the check fails (or on request); no MPP source is ever built. Not tuned, no validation track.

nax and tensorops both require the on-device tensor-op check: `matmul2d` exact through MLX-steel register fragments, left/destination cooperative-tensor coordinates equal to MLX BaseNAXFrag, and the layout-agnostic API exact. The result is cached per device + OS build + MLX version + probe source.

## Feature × generation

| Feature | M1 | M2 | M3 | M4 | M5 | What our code does | Sources |
|---|---|---|---|---|---|---|---|
| Apple GPU family | Apple7 | Apple8 | Apple9 | Apple9 | Apple10 | n/a (we key on the arch string) | [FST] p.2; [GF7] [GF8] [GF9] ("A17, M3, and M4") |
| `MTLArchitecture.name` gen (`applegpu_g<NN><class>`) | g13 | g14 | g15 | g16 | g17 | `arch_generation()`; `NAX_MIN_GEN = 17` decides nax vs the rest | g13 = M1: [DJ] secondary, and [ARCH] gives `applegpu_g13s` as the example. g15 = M3: [MLX-mm] 1413 ("Pre-M3 generations" = gen < 15). g17 = M5: [local] `applegpu_g17s` on M5 Pro, and [MLX-q] 491-493 (gen 17 's' = "M5 Max"). g14 and g16: recalled, unverified |
| Default kernel path (auto) | tensorops if the check passes, else portable fallback (warning) | same | same | same | nax if the check passes, else fallback | macOS < 26: refused at start-up (`boltz_mlx/__init__.py`) | `kernels.py` `_detect()` |
| Metal 4 (API + MSL 4.0) | yes, macOS 26+ | yes | yes | yes | yes | nax/tensorops need MSL 4.0. Portable needs ≤ MSL 2.3 features | [FST] p.2; [W25-205]; [SDK] `MTLGPUFamilyMetal4 macos(26.0)` |
| `simdgroup_matrix` 8×8 MMA (half/float; bfloat needs MSL 3.1) | yes | yes | yes | yes | yes (shader ALUs, not NAX) | Strict and portable kernels (tri-att `mma`, fused transition, difftx GEMMs, MSA opm/pwa) use `simdgroup_float8x8`; fp16 inputs are converted to float fragments | [FST] p.4 "SIMD-scoped matrix multiply ... Apple7"; [MSL] §2.4 "Metal 2.3 and later"; [TT-M5]: simdgroup-matrix matmul shows 0 % NAX utilisation |
| `simdgroup_matrix` relative speed | FFMA16 ≈ FFMA32 (102.5 vs 101.7 FFMA per core-cycle, secondary) | as M1 (secondary) | not published | not published | NAX matmul is "4 to 8 times faster, depending on precision" | Don't project M5 speedups to M1–M4 | [MB] secondary ("A15+, M1+" column); [TT-M5] |
| SIMD reductions (`simd_sum`, `simd_max`) | yes | yes | yes | yes | yes | LN/add_ln statistics, softmax rows, OPM/PWA (all paths) | [FST] p.4 "SIMD-scoped reduction ... Apple7"; [MSL] §4.4.1 (macOS: Metal 2+) |
| SIMD permutes (`simd_shuffle_xor`) | yes | yes | yes | yes | yes | tri-att online softmax; LN inside the MPP kernels | [FST] p.4 "SIMD-scoped permute ... Apple6" ("shift and fill" is Apple8, unused) |
| Max threads / threadgroup | 1024 theoretical | 1024 | 1024 | 1024 | 1024 | Portable TG sizes 64–256, no thread bound. MPP kernels declare `max_total_threads_per_threadgroup` | [FST] p.7, fn 5 on p.10 (the real limit is per pipeline: `maxTotalThreadsPerThreadgroup`) |
| Max threadgroup memory | 32 KB | 32 KB | 32 KB | 32 KB | 32 KB | Fused transition 25,728 B (estimate from `TRANS_CFG` = (24, 8, 64), D 128, pads 4). Tri-att `mma` 9,344 B (C = 32). MSA asserts ≤ `_TG_BYTES` = 32768 | [FST] p.7 |
| Register / on-chip memory allocation | Static: a SIMD-group holds its *peak* register count for its lifetime, which sets occupancy. 128 GPRs/thread (secondary); ~208 KB register file per core (secondary) | as M1 (Apple8) | Dynamic Caching: registers, threadgroup and tile memory are caches, allocated as used; occupancy adjusted dynamically | as M3 (Apple9) | "second-generation dynamic caching" | R5/R6 lessons came from M5 (see Implications) | [TT-M3]; [NR-M3]; [NR-M5]; [DJ] and [MB] secondary |
| fp16 vs fp32 ALU | F16 FMA = F32 FMA = 256 ops/core-cycle (secondary): no fp16 ALU gain | as M1 | Apple9: FP32/FP16/int in parallel, "up to 2x ALU"; FP16 "at peak throughput", fewer registers | as M3 | NAX fp16 matmul. [local] fp32 GEMM 3.3–3.7 TFLOP/s, TF32 2.4–3× fp32, NAX register peak ~22–26 TFLOP/s | f16 preset: fp16 storage/operands, fp32 accumulate | [MB] secondary; [TT-M3]; [TT-M5]; RESULTS.md:83, :430, :605 [local] |
| `bfloat` type | Apple6+ data type; MSL 3.1 (macOS 14) | same | same | same | same; MPP bfloat tensors from OS 26.1 | Not used in any kernel (bf16 trunk rejected on M5) | [FST] p.5 "MTLDataType.bfloat ... Apple6"; [MSL] §2.1; [MSL] Table 7.3; [TT-M5] |
| Metal 4 tensors + MPP TensorOps (`matmul2d`) | runs on shader cores ("falls back to optimized shader implementations"), macOS 26+ | same | same | same | runs on the Neural Accelerators | M1–M4: **tensorops** path (default, after the check). M5: **nax** path. **portable**: `metal_kernel()` refuses MPP sources | [FST] p.6 "Tensors Metal 4 Apple7"; [MSL] §7; [TT-M5] |
| Neural Accelerators ("NAX") | no | no | no | no | one per GPU core (M6 also has them) | nax path only | [NR-M5]; [TT-M5]; [NR-M6] |
| MLX `is_nax_available()` (MLX's own GEMM/SDPA choice) | false | false | false | false | true if macOS ≥ 26.2 and macOS-26 wheel | Ours (after this note's review): nax = MLX's hardware rule (gen ≥ 17, ≥ 18 for 'p': `kernels.nax_hardware`) AND the check; macOS < 26.2 is noted in the reason (`NAX_MIN_MACOS`) | [MLX-dev] 978-997; [MLX-cm] 89-117 |
| MLX TF32 (fp32 matmul family at reduced precision) | never | never | never | never | default on (`MLX_ENABLE_TF32=1`) when NAX is available | `boltz_mlx/__init__.py:9` sets `MLX_ENABLE_TF32=0` unless `BOLTZ_MLX_TF32=1` | [MLX-mm] 964-966; [MLX-u] 225-227; [MLX-prec]; ISSUES.md:69 |
| HW ray tracing, mesh shading, Dynamic Caching | no | no | yes (first on Mac) | yes | yes | Not used except as Dynamic Caching's effect on occupancy | [NR-M3]; [NR-M5] |

## Metal 4 per chip

| Chip (Mac) | Family | Metal 4 | OS for Metal 4 / MSL 4.0 | `matmul2d` executes on | Our auto path (after the check) |
|---|---|---|---|---|---|
| M1 (MacBook Air/Mac mini/MBP 13 2020, iMac 2021, ...) | Apple7 | yes | macOS 26.0 | shader cores | tensorops |
| M2 | Apple8 | yes | macOS 26.0 | shader cores | tensorops |
| M3 | Apple9 | yes | macOS 26.0 | shader cores | tensorops |
| M4 | Apple9 | yes | macOS 26.0 | shader cores | tensorops |
| A18 Pro (MacBook Neo) | Apple9 | yes | macOS 26.0 | shader cores (A18 = Apple9; NAX is named for "M5 and A19") | depends on its arch string: see Implications ('p' rule) |
| M5 | Apple10 | yes | macOS 26.0 (MLX's own NAX: 26.2) | Neural Accelerators | nax |
| M6 | not verified (online docs list `apple11` with no description) | presumably | – | Neural Accelerators per [NR-M6] | nax if gen ≥ 17 and the check passes |

Sources:
- Metal 4 on "Apple M1 and later, as well as the A14 Bionic and later" [W25-205]; [FST] p.2 lists "Metal 3 & 4" for A14+ and M1–M5.
- `MTLGPUFamilyMetal4` and `MTLLanguageVersion4_0` are `API_AVAILABLE(macos(26.0))` [SDK]; [GF-Metal4] introducedAt 26.0; MSL 4.0 = macOS 26 [MSL] §1.6.10.
- macOS 26 runs on the first M1 Macs and on the MacBook Neo (A18 Pro) [SUP-Tahoe]. That every Apple-silicon Mac is listed is inferred from the oldest M1 models being listed.
- Shader-core fallback: [TT-M5]. A18 = Apple9 [FST] p.2.

## MLX per generation (v0.32.4 source)

- **Arch string.** Read from `MLX_METAL_GPU_ARCH` if set, else `MTLDevice.architecture.name`. `arch_gen` = the 3rd- and 2nd-last characters as digits (`applegpu_g17s` → 17). The device class is the last character: `p` phone, `g` "base, pro", `s` "max", `d` "ultra". The class sets max ops/MB per command buffer (20/40, 40/40, 50/50, 50/50) [MLX-dev] 612-652. [local] The **M5 Pro reports `s`**, which MLX's comment calls "max"; quantized.cpp also calls gen 17 's' "M5 Max" [MLX-q] 491.
- **`mx.device_info()["architecture"]`** returns that same (possibly overridden) string [MLX-di] 32, 46. [local] `MLX_METAL_GPU_ARCH=applegpu_g15s` changes it.
- **`is_nax_available()`** (exact): false if built with `MLX_METAL_NO_NAX`. Otherwise `__builtin_available(macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2)` AND `gen >= (class == 'p' ? 18 : 17)`. It is computed once (static) [MLX-dev] 978-997. `MLX_METAL_NO_NAX` is defined unless `MLX_METAL_VERSION >= 400` AND SDK ≥ 26.2 AND `CMAKE_OSX_DEPLOYMENT_TARGET` ≥ 26.2 [MLX-cm] 89-117, so only the macOS-26.2-target wheel can have NAX.
- **MLX never uses MPP/TensorOps on M1–M4.** Every MPP kernel in MLX is a `*_nax` kernel behind `is_nax_available()`: GEMM regular/thin/split-K [MLX-mm] 964-1090, gather/segmented [MLX-mm] 2964, 3043, SDPA [MLX-sdpa] 412-415, 436, 964, 1436-1452, quantized [MLX-q] 1076-1081, 1298, 1693, and gated-delta. Below gen 17 MLX runs its steel `simdgroup_matrix` kernels. Those convert fp16/bf16 tiles to float (`AccumType`) fragments before the MMA (installed `steel/gemm/mma.h:456,481-483`). MLX has no "TensorOps on shader cores" path. MLX's NAX GEMM is `matmul2d(16, 32, 16, ..., relaxed_precision=true, multiply_accumulate)` (installed `steel/gemm/nax.h:401-408`).
- **What `MLX_METAL_GPU_ARCH` changes.** It is read once [MLX-u] 235-237. It changes:
  1. `arch_gen`, and with it `is_nax_available()`, which moves all the dispatches listed above.
  2. TF32, which only exists on the NAX path.
  3. Non-NAX gen gates: wide GEMV off for gen < 15 [MLX-mm] 1413-1418; qmv batch limits by gen [MLX-q] 90-110; affine `qmv_wide` needs gen ≥ 15 [MLX-q] 545-547; nvfp4 narrow qmv for gen 17 's' [MLX-q] 491-493.
  4. Through the class character: steel tiles (`GEMM_TPARAM_MACRO`: 'g'/'p' small, 'd' large, else 64×64×16, wm = wn = 2) [MLX-mm] 91-160, 387-392; split-K thresholds [MLX-mm] 968; NAX GEMM tiles [MLX-mm] 211-224; SDPA D-256 block on 'd' [MLX-sdpa] 507-512; command-buffer sizes.

  It does **not** change the JIT language version, the physical GPU, or Metal pipeline limits. MLX docs: it "does not change the capabilities of the physical GPU" and a mismatch "can select incompatible kernels" [MLX-env] 134-139. No other gen gating was found in the other fetched Metal backend sources (normalization, softmax, reduce, rope, copy, compiled, conv, sort, scan, fft, indexing, logsumexp).
- **`mx.fast.metal_kernel`** compiles through `Device::build_library_` with `setLanguageVersion(get_metal_version())` [MLX-ck] 50-53, [MLX-dev] 688. The version is **macOS 27 → MSL 4.1, 26 → 4.0, 15 → 3.2, else 3.1** [MLX-dev] 65-80. MLX **throws `invalid_argument`** if the threadgroup size exceeds the pipeline's `maxTotalThreadsPerThreadgroup` [MLX-ck] 84-93. MLX's own SDPA hits this on M1/M2 ("M1/M2 do not have enough threads per threadgroup for default TK=4 kernel" → `_tk1`) [MLX-sdpa] 862-872; see also mlx#4718, ISSUES.md:143.
- **Minimum macOS / wheels.** "MLX requires macOS >= 14.0" [MLX-cmake] 211-212. Release CI builds macOS wheels with targets **14.0, 15.0 and 26.2** [MLX-rel] 155-170. [local] The installed wheel is tagged `macosx_26_0_arm64`; `libmlx.dylib` has `minos 26.2, sdk 26.5` (otool).

## Metal language / macOS minimums

| MSL | macOS | SDK `MTLLanguageVersion` | MLX JIT uses it on | We need it for |
|---|---|---|---|---|
| 2.3 | 11 | `2_3 macos(11.0)` | – | `simdgroup_matrix`, `simdgroup_load(..., ulong2 origin, transpose)` ([MSL] §2.4, Table 6.9) |
| 3.1 | 14 | `3_1 macos(14.0)` | macOS 14 (MLX floor) | nothing extra (`bfloat` would need it; unused) |
| 3.2 | 15 | `3_2 macos(15.0)` | macOS 15 | – |
| 4.0 | 26 | `4_0 macos(26.0)` | macOS 26 | MPP / tensors / cooperative tensors (nax, tensorops) |
| 4.1 | 27 | not in the local Xcode 26.4 SDK (MLX casts `(4<<16)+1`) | macOS 27 | – |

Sources: [SDK] `MTLLibrary.h:207-224`, `MTLDevice.h:231-256` (highest family `Apple10`; online docs also list `apple11` [GF-list]); [MSL] §1.6.10 p.19-20 ("Only Apple silicon supports new features in language standard 3.2 and above"); [MLX-dev] 65-80.
**Net minimum:** macOS 14 (MLX's floor) for the strict and portable paths, which use only ≤ MSL 2.3 features. macOS 26 for the nax/tensorops paths. macOS 26.2 for MLX's own NAX kernels.

## MPP tensor_ops availability

- **Where it runs.** "Metal 4 and later support Metal Performance Primitives", with supported families per the Feature Set Tables [MSL] §7 p.349, which give "Tensors: Metal 4, Apple7" [FST] p.6. Apple on pre-M5 GPUs: "On older GPUs without neural accelerators, TensorOps falls back to optimized shader implementations", and "The same code runs across Apple's entire GPU family from M1 to M5" [TT-M5]. "The OS shader compiler inlines shader code, optimized for the device being used" [W25-205]. MSL §6.8 recommends tensors + MPP over `simdgroup_matrix` as "performant and portable".
- **Performance statements.** Apple gives **no** figures for TensorOps vs `simdgroup_matrix` on M1–M4. The only numbers are for M5: matmul "4 to 8 times faster, depending on precision"; a 4K×4K demo "almost seven times faster" going from `simdgroup_matrix` (0 % NAX use) to TensorOps; app-level M5 vs M1/M4 [TT-M5].
- **Data types.** half×half→float/half and float×float→float since Metal 4. bfloat since OS 26.1. int8/uint8/int4 since OS 26.4. fp8/fp4/int2 need MSL 4.1 [MSL] Table 7.3 p.351-353. `relaxed_precision` "allows the operation to truncate the mantissa before the multiplication" for float data [MSL] Table 7.4 p.354.
- **Layout is implementation-defined and device-dependent.**
  - [MSL] §2.22.3 p.104: "the Layout is device specific ... The tensor operation determines an efficient and performant Layout for a cooperative_tensor based on its usage and the GPU".
  - [SDK-MPP] `MPPTensorOpsMatMul2d.h:216-224`: cooperative_tensor "divides the data ... among threads ... in implementation defined manner"; "The layout of cooperative_tensor depends on operation, data type, number of threads in opscope"; "Though the layout ... is implementation defined, we provide accessor functions".
  - [SDK-MPP] `MPPTensorOpsMatMul2d.h:246-249`: "The layout of data among lanes is implementation defined and not all threads and even all elements within a thread need be valid".
  - Host `MTLTensor`s from a device also have "an opaque layout" [W25-262].
  - Our register-fragment kernels assume MLX's BaseNAXFrag layout for one descriptor (16×32×16, relaxed). RESULTS.md:504 shows that a different descriptor scrambles operands even on M5. Hence the on-device tensor-op check, re-run per device + OS build.
- **Header guard.** The SDK headers compile to nothing unless `defined(__METAL_VERSION__) && defined(__HAVE_TENSOR__)` [SDK-MPP] `MPPTensorOpsMatMul2d.h:328`. `run` lowers to `__tensorops_impl_*` externals (`__impl/MPPTensorOpsMatMul2dImpl.h`), which the OS compiler resolves per device. Under MSL ≤ 3.2 the `mpp::` names do not exist, so the check fails on the compile error and the code falls back to portable.

## TF32

- MLX: `use_nax = is_nax_available() && (enable_tf32() || dtype != float32)` [MLX-mm] 964-966. The same condition is used for SDPA/quantized/gather [MLX-sdpa] 412-415, [MLX-q] 1080. `MLX_ENABLE_TF32` defaults to 1 [MLX-u] 225-227; [MLX-env] 23-27. "TF32" is fp32 fed to the relaxed `matmul2d` on NAX (mantissa truncation [MSL] Table 7.4). [MLX-prec] says results can be off by "several orders of magnitude more than float32 rounding".
- So **TF32 exists only on gen ≥ 17 with macOS ≥ 26.2 and a NAX-enabled MLX build**. On M1–M4 (and on M5 under macOS < 26.2), fp32 matmuls are full fp32 steel GEMMs and `MLX_ENABLE_TF32` has no effect (source reading, not measured).
- Repo notes:
  - ISSUES.md:69: MLX docs PR #3894 says gen-17+ with macOS ≥ 26.2 run at TF32-class precision; "M3 and earlier are unaffected".
  - ISSUES.md:110 [local]: 512³ max rel err 7.7e-4 by default vs 5.1e-7 with TF32 off.
  - RESULTS.md:74-78: TF32 everywhere fails e2e parity (1.44 Å).
  - ISSUES.md:119: #4411 found fp16/bf16 bit-identical with and without `MLX_METAL_GPU_ARCH=applegpu_g14p`.
  - Round 1 measured TF32 and `--turbo` on the M5 only.

## Implications for this code

- **tensorops on M1–M4 may not beat the simdgroup kernels (estimate).** Owner decision (2026-10-28 22:30): tensorops is the shipped M1–M4 path anyway (one kernel family to maintain); the M1 kit records a portable reference timing so a slowdown is visible. The reasons for caution:
  - MLX (Apple's own) never routes pre-M5 work through MPP: its NAX kernels sit behind `is_nax_available()` and M1–M4 get `simdgroup_matrix` steel kernels.
  - Apple publishes no pre-M5 TensorOps speed data.
  - On M1/M2, fp16 and fp32 `simdgroup_matrix` rates are equal (secondary [MB]). A shader fallback has no faster unit to use, and it adds its own registers to kernels designed around NAX register fragments (thread bounds and tiles were tuned on the M5 Pro).
  - With static register allocation on Apple7/8 [TT-M3], that means spills or lower occupancy (estimate). Per-device tile configs (`*_cfg.json`, keyed by chip) let M1–M4 be re-tuned without code changes.
- **The tensor-op check proves layout and integer exactness, not accumulation precision.**
  - Its operands give |ref| ≤ 96 (code comment), which is exact even with fp16 accumulation or truncated operands.
  - A shader fallback under `relaxed_precision=true` could round differently from NAX, and the check would not notice.
  - Add a non-integer probe (e.g. K ≥ 256, random fp16 operands, compare with fp32, expect ~1e-6 rel) before trusting tensorops numerics. Otherwise rely on the e2e gate on the M1.
  - **Partly done**: the check now also runs one 16×32×16 product on random fp16 operands (relative max error ≤ 1e-3 vs fp64, the bound tests/test_nax_layout.py validated on the M5; the value is recorded). A tighter, long-K accumulation probe is still open; the M1 kit's MPP per-kernel tests (vs fp32 references) and its end-to-end accuracy smoke run before tensorops is timed.
- **'p' rule (MacBook Neo).** `nax_hw = gen >= 17` ignores the class letter. MLX requires gen ≥ 18 for 'p' [MLX-dev] 991. That implies a phone-class GPU at gen 17 lacks NAX (inference). The A18 family is Apple9 with no Neural Accelerators [FST] p.2, [TT-M5], and the MacBook Neo has an A18 Pro and runs macOS 26 [SUP-Tahoe]. Its arch string is unknown here.
  - If it reports `applegpu_g17p`, a plain gen ≥ 17 rule would label its shader-core tensor-op path "nax".
  - Fix: mirror MLX (`gen >= (18 if arch.endswith("p") else 17)`). **Done** (`kernels.nax_hardware`, test_portable.py::test_mlx_nax_rules).
- **macOS 26.0/26.1 on an M5.** MLX's `is_nax_available()` is false there (needs 26.2), so MLX uses steel GEMMs and no TF32. Our check can still pass, because MSL 4.0 and half→float `matmul2d` are macOS 26.0 features. The result is an untested mix: our nax kernels with MLX non-NAX GEMMs. Suggest requiring macOS ≥ 26.2 for auto-nax (as MLX does), or at least recording it. **Recorded**: after the owner decision (MPP kernels on every chip), an M5 on macOS 26.0/26.1 still takes the nax path; its reason says MLX's own GEMMs do not use NAX there (`kernels.NAX_MIN_MACOS`), and fingerprints are per chip + OS anyway. Whether the 26.2-minos wheel even loads on 26.0/26.1 is open.
- **Emulating older chips on the M5.**
  - `BOLTZ_MLX_KERNEL_PATH=portable` alone turns off only our MPP kernels; MLX still uses NAX GEMMs/SDPA and TF32.
  - `MLX_METAL_GPU_ARCH=applegpu_g16s` (or g15s/g13s) also flips MLX to its steel kernels, and our auto path to portable. That is the closer emulation of the M1–M4 *software* path.
  - It cannot emulate the hardware. For example, `MLX_METAL_GPU_ARCH=g16s` plus `BOLTZ_MLX_KERNEL_PATH=tensorops` still runs `matmul2d` on the M5's Neural Accelerators, not on shader cores. Occupancy, register limits and speed of M1–M4 are not reproduced (estimate).
- **Pipeline thread limits on M1/M2.** MLX throws at eval if the TG size exceeds the pipeline's `maxTotalThreadsPerThreadgroup` [MLX-ck] 84-93, and MLX's own SDPA hits this on M1/M2 [MLX-sdpa] 862. On M5 every shipped kernel reports 1024 (RESULTS.md:642, compile-only PSOs). On Apple7/8 register-heavy kernels can report less (estimate).
  - Portable candidates: the 256-thread kernels (fused transition, LN/add_ln, 8-simdgroup OPM configs) and `opm_outer` cfg 0 (64 fp32 accumulators per thread).
  - tensorops candidates: every MPP kernel with a large declared thread bound.
  - Suggest a compile-only PSO check on the target chip (the `tools/prof/metal_pipeline.swift` approach), or catch the `ValueError` and fall back.
- **Threadgroup memory** fits the 32 KB limit of every family [FST] p.7 (transition 25.7 KB, tri-att 9.3 KB, MSA ≤ 32 KB asserted), so no guard is needed. On M1/M2 (~60 KB shared memory per core, secondary [MB]) a 25.7 KB TG allows about 2 resident TGs per core (estimate).
- **Register-pressure configs were tuned on the M5.**
  - TAIL3's win came from a *rolled* g k-loop. It stopped the scheduler from hoisting loads into the live p-GEMM region and spilling; on M5 "the compiler spills rather than reducing occupancy". Thread bounds alone moved < 0.3 ms (RESULTS.md:649). TRANS3 applied the same rule (RESULTS.md:557, :652-655).
  - Both are MPP kernels, so they run only on the nax/tensorops paths.
  - On M1/M2, peak register use sets occupancy for the whole kernel [TT-M3]; M3/M4 behave more like M5 (estimate).
  - The portable tri-att uses the small config (RQ 16, BK 16; `_MMA_CFG` comment "bigger tiles spill"). The `opm_outer`/`pwa_avg` and difftx configs were swept on M5 only and should be re-swept on an older chip.
- **fp16 on M1–M4.** fp16 ≈ fp32 MMA rate on M1/M2 (secondary), and MLX steel upconverts fp16 tiles to float. Portable f16 gains come mostly from halved memory traffic and may be small; don't project them (OWNER_QUESTIONS.md R2-6).
- **`BOLTZ_MLX_TF32=1` on M1–M4** has no TF32 to enable: MLX fp32 GEMMs stay full precision. `TRANSITION_IMPL` still switches to "mlx" on that env var (kernels.py `TRANSITION_IMPL`), a choice measured only with TF32 active (RESULTS.md:317). Consider keying it on "MLX TF32 actually active" instead.
- **`simdgroup_matrix` lane layout.** The tri-att MMA, the fused transition and `pack_transition_weights` assume lane → (fm, fn), which MSL calls "unspecified" [MSL] §2.4. MLX steel hard-codes the same mapping on every family without a gen check (installed `steel/gemm/mma.h:49-53`), so the risk is low but not Apple-guaranteed.
- **`mma_half`** (non-default `BOLTZ_MLX_TRIATT_IMPL`) calls `simdgroup_multiply_accumulate(float&, half, half, float)`. [MSL] Table 6.10 documents only a single element type T. It is unverified on Apple7/8; the default `mma` is float-only.
- **SDPA.** Our `mx.fast.scaled_dot_product_attention` head dims are 48 and 32 (estimate from module configs). The M1/M2 TK=4 limit concerns D 192/256 [MLX-sdpa] 862-866, and NAX SDPA needs D ∈ {64, 96, 128, 256, 512} [MLX-sdpa] 412-413. So neither the M1/M2 limit nor TF32 applies to these calls.
- **OS floor.** macOS 26 is now required (refused at start-up and in install_check). Should that refusal be bypassed (`BOLTZ_MLX_ALLOW_OLD_MACOS=1`, debugging only) on macOS 14/15 (MSL 3.1/3.2), the `metal_kernel()` MPP guard plus the check failing closed keep MPP sources from being built.

## Open questions (not verified)

- `applegpu_g14` = M2, `g16` = M4, the class letter per tier ([local] M5 Pro = `s`; MLX comments say `s` = Max), the MacBook Neo / A18 Pro arch string, and the M6 gen. Run `xcrun metal-arch` on each machine.
- TensorOps-on-shader speed vs `simdgroup_matrix` on M1–M4, and whether the fallback's `relaxed_precision` rounding matches NAX. Apple publishes neither; the M1 kit should measure both.
- `maxTotalThreadsPerThreadgroup` of our portable and MPP kernels on Apple7/8/9 (compile-only, needs those machines).
- Native bf16 ALU rates per family; `simdgroup_matrix` throughput on M3/M4/M5 (not published).
- Whether mixed-type `simdgroup_multiply_accumulate` (`mma_half`) compiles on Apple7/8.
- Whether the macOS-26 MLX wheel (`minos 26.2`, tagged `macosx_26_0`) loads on macOS 26.0/26.1.
- The exact mantissa width of relaxed fp32 on NAX. [local] 512³ rel err 7.7e-4 is TF32-like.
- Metal stdlib headers (`__HAVE_TENSOR__` definition, `metal_simdgroup_matrix`) were not readable: the local Xcode 26.4 lacks the Metal Toolchain (`xcrun metal` → "missing Metal Toolchain"), and it was not downloaded.

## Sources (all accessed 2026-10-28)

- [FST] Apple, Metal Feature Set Tables (PDF, May 21 2026), https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf: p.2 GPU → family, p.3-6 features (SIMD p.4, bfloat p.5, Tensors p.6), p.7 limits, footnote 5 on p.10.
- [MSL] Apple, Metal Shading Language Specification v4.1 (2026-06-04), https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf: §1.6.10, §2.1, §2.4, §2.22.3, §4.4.1, §6.8 Tables 6.9/6.10, §7, Tables 7.3/7.4.
- [GF7] [GF8] [GF9] [GF10] [GF11] [GF-Metal4] https://developer.apple.com/tutorials/data/documentation/metal/mtlgpufamily/{apple7,apple8,apple9,apple10,apple11,metal4}.json. [GF-list] https://developer.apple.com/tutorials/data/documentation/metal/mtlgpufamily.md
- [ARCH] https://developer.apple.com/documentation/metal/mtlarchitecture/name.md
- [W25-205] WWDC25 "Discover Metal 4": https://developer.apple.com/videos/play/wwdc2025/205/
- [W25-262] WWDC25 "Combine Metal 4 machine learning and graphics": https://developer.apple.com/videos/play/wwdc2025/262/
- [TT-M3] Apple tech talk 111375, "Explore GPU advancements in M3 and A17 Pro": https://developer.apple.com/videos/play/tech-talks/111375/
- [TT-M5] Apple tech talk 111432, "Accelerate your machine learning workloads with the M5 and A19 GPUs": https://developer.apple.com/videos/play/tech-talks/111432/
- [NR-M3] https://www.apple.com/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/ (2023-10-30)
- [NR-M5] https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/ (2025-10-15)
- [NR-M6] https://www.apple.com/newsroom/2026/08/apple-introduces-m6-and-m5-ultra-for-a-big-leap-in-performance-and-ai-compute/ (2026-08-25)
- [SUP-Tahoe] Apple Support, "macOS Tahoe 26 is compatible with these computers": https://support.apple.com/en-us/122867
- [MLX] ml-explore/mlx tag v0.32.4 via https://raw.githubusercontent.com/ml-explore/mlx/v0.32.4/<path>. Line numbers refer to these files. Tag ref: https://api.github.com/repos/ml-explore/mlx/git/ref/tags/v0.32.4.
  - [MLX-dev] `mlx/backend/metal/device.cpp`
  - [MLX-mm] `matmul.cpp`
  - [MLX-sdpa] `scaled_dot_product_attention.cpp`
  - [MLX-q] `quantized.cpp`
  - [MLX-ck] `custom_kernel.cpp`
  - [MLX-di] `device_info.cpp`
  - [MLX-cm] `mlx/backend/metal/CMakeLists.txt`
  - [MLX-cmake] `CMakeLists.txt`
  - [MLX-u] `mlx/utils.h`
  - [MLX-env] `docs/src/usage/environment_variables.rst`
  - [MLX-prec] `docs/src/usage/precision.rst`
  - [MLX-rel] `.github/workflows/release.yml`
- [SDK] local Xcode 26.4 (17E192) SDK: `Metal.framework/Headers/MTLLibrary.h`, `MTLDevice.h`. [SDK-MPP] `MetalPerformancePrimitives.framework/Headers/MPPTensorOpsMatMul2d.h`, `__impl/`.
- Installed MLX headers: `<venv>/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/steel/gemm/{mma.h,nax.h}`
- [MB] secondary: philipturner/metal-benchmarks README, https://github.com/philipturner/metal-benchmarks (raw README at `main`)
- [DJ] secondary: Dougall Johnson, Apple G13 GPU architecture reference, https://dougallj.github.io/applegpu/docs.html
- Repo: RESULTS.md (development log, not included in this repository), ISSUES.md, `boltz_mlx/kernels*.py` (cited by symbol, because they are edited in parallel).
