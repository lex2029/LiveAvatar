# MaxSpeed Optimization Log

This file is the working journal for performance work on branch `MaxSpeed`.

## Benchmark Policy

- Primary goal: reduce warm generation time in an already-running resident worker.
- Primary metric: `generation_total_s`
- Primary rate metric: `generation_sec_per_audio_sec`
- Diagnostic only:
  - `cold_start`
  - `runner_acquire_s`
  - `pipeline_init_s`
- Always compare warm runs against warm runs.
- Do not treat model load / compile cold start as the optimization target.

## Benchmark Assets

- Image: `examples/boy.jpg`
- Audio:
  - `benchmarks/maxspeed/boy_10s.wav`
  - `benchmarks/maxspeed/boy_30s.wav`
- Note:
  - ElevenLabs was requested, but no `ELEVENLABS_API_KEY` was available in env/config.
  - Local reproducible WAV assets are used until a key is provided.

## Speed Tiers

This section is the practical roadmap ordered by expected speedup and risk.

### Tier A: Low-risk wins inside the current model (`~x1.02` to `~x1.5`)

1. Prompt embedding cache in the resident pipeline.
Status: done
Expected impact:
  Small but effectively free speedup for repeated prompts.
Current result:
  Confirmed win.

2. `torch.compile` on the warm path.
Status: done
Expected impact:
  Large warm-path gain if graph breaks are controlled.
Current result:
  Confirmed strong win.
Latest confirmation on the current best `LightVAE + chunk=384` path:
  - with `compile=true`:
    - `generation_total_s=107.93`
    - `generation_sec_per_audio_sec=10.79`
    - `clip_generation_s=78.72`
  - with `compile=false`:
    - `generation_total_s=223.31`
    - `generation_sec_per_audio_sec=22.33`
    - `clip_generation_s=194.10`
Verdict:
  - `compile=false` is worse by about `115.38s`
  - about `106.9%` slower than the current winner
  - keep testing strong candidates without compile when they are architecturally different, but on the current path `compile=true` remains decisively better
Latest confirmation on the newer winner path (`LightVAE + chunk=512 + rope-cache`):
  - a fresh `compile=false` rerun was started and observed live
  - per-step denoising was clearly much slower than the `compile=true` winner
  - the run was aborted early as a non-promising regression to avoid wasting GPU time
Verdict update:
  - even after the rope-cache win, `compile=true` still remains the only promising mode on
    the current path

3. Remove hot-path compile graph breaks.
Status: in_progress
Expected impact:
  Improve the already-good `compile=true` path by reducing graph fragmentation.
Current work:
  Cleaning `.item()` / Python-size conversions in:
  - `liveavatar/models/wan/causal_model_s2v.py`
  - `liveavatar/models/wan/wan_2_2/modules/s2v/s2v_utils.py`
Latest finding:
  A fresh warm benchmark on the current best `LightVAE` path still reports a compile graph break at:
  - `liveavatar/models/wan/causal_model_s2v.py:1241`
  - `num_frames = original_grid_sizes[0][0].item()`
  This is now the next concrete compile-cleanup target after the current benchmark matrix work.
Update:
  A direct rope-frequency cache in the streaming path produced a real speed win even before
  cleanup is perfect:
  - baseline (`LightVAE + chunk=512`): `100.31s`
  - with rope cache: `99.20s`
  However, the first implementation triggered a `torch.compile` backend exception while
  serializing tensor grid sizes for the cache key, so the next step is to keep the win and
  remove that compile-time noise.
Follow-up:
  A cleanup attempt using `@torch._dynamo.disable` on the cache-key serializer was a bad
  regression:
  - `generation_total_s=109.92`
  - `generation_sec_per_audio_sec=10.99`
  - `clip_generation_s=80.57`
  It also hit `torch._dynamo` recompile limits.
Verdict:
  Revert that cleanup attempt and keep the original rope-cache variant as the winner for now.

4. Reduce unnecessary chunking on 96 GB VRAM.
Status: measured
Expected impact:
  Potential additional win if current chunk sizes are over-conservative.
Notes:
  Must be benchmarked carefully against OOM risk.
Result so far:
  First aggressive test with chunk sizes raised from `256` to `512` was not promising.
  The run reached `complete full-sequence generation`, but hung in the tail and never wrote
  benchmark artifacts to disk.
  A narrower follow-up at `chunk=384` on the current best `LightVAE` path is a real win.
Measured comparison:
  - previous winner: `infer_frames=64 + compile + LightVAE + chunk=256`
    - `generation_total_s=119.39`
    - `generation_sec_per_audio_sec=11.94`
    - `clip_generation_s=90.19`
    - `postprocess_s=21.93`
  - new winner: `infer_frames=64 + compile + LightVAE + chunk=384`
    - `generation_total_s=107.93`
    - `generation_sec_per_audio_sec=10.79`
    - `clip_generation_s=78.72`
    - `postprocess_s=21.93`
Approximate improvement:
  - `11.47s` faster
  - about `9.6%` faster
Latest measured comparison:
  - previous winner: `infer_frames=64 + compile + LightVAE + chunk=448`
    - `generation_total_s=105.33`
    - `generation_sec_per_audio_sec=10.53`
    - `clip_generation_s=76.00`
    - `postprocess_s=21.96`
  - new winner: `infer_frames=64 + compile + LightVAE + chunk=512`
    - `generation_total_s=100.31`
    - `generation_sec_per_audio_sec=10.03`
    - `clip_generation_s=71.06`
    - `postprocess_s=21.90`
Approximate improvement:
  - `5.02s` faster
  - about `4.8%` faster
Verdict:
  - `chunk=512` is the new overall leader on the current path
  - chunking paid off up to `512`
  - `640` is already a slight regression, so treat `512` as the current local optimum
  - move on to the next roadmap item

5. Faster attention backend on the current model.
Status: measured
Expected impact:
  Potential meaningful speedup if Blackwell can run it stably.
Notes:
  Only worth revisiting after compile-break cleanup.
Result so far:
  `flash-attn` is a dead-end on this Blackwell box with the current wheel stack.
  Test config:
  - `size=480*832`
  - `infer_frames=64`
  - `sample_steps=4`
  - `compile=true`
  - `disable_cudnn_attn=true`
  - `disable_flash_attn=false`
  - `warm_runs=1`
  Outcome:
  - pipeline loaded
  - warmup render started
  - then failed inside flash attention with:
    - `CUDA error ... no kernel image is available for execution on the device`
Verdict:
  - no good near-term prospect on the current Blackwell wheel stack
  - move on to the next roadmap item instead of spending more time here
Additional backend check:
  Re-tested the current winner path with `LIVEAVATAR_DISABLE_CUDNN_ATTN=true`.
  Warmup already regressed:
  - current winner warmup (`rope-cache`, `chunk=512`): `166.90s`
  - `disable_cudnn_attn=true` warmup: `183.57s`
  The run was aborted early as a clear regression.
Final verdict:
  Backend toggles are exhausted for the current stack. Keep `cuDNN attention` enabled.

### Tier B: Medium-risk changes inside the current model (`~x1.5` to `~x2.5`)

1. Larger `infer_frames`.
Status: measured
Expected impact:
  Fewer clips, less repeated denoise / postprocess overhead.
Current result:
  `infer_frames=64` is the current winner.
  `infer_frames=72` is a regression even after the `LightVAE` win.
  `infer_frames=80` is unstable / inconclusive.
Latest measured comparison on the current best `LightVAE` path:
  - `infer_frames=64 + compile + LightVAE`
    - `generation_total_s=119.39`
    - `generation_sec_per_audio_sec=11.94`
    - `clip_generation_s=90.19`
    - `postprocess_s=21.93`
  - `infer_frames=72 + compile + LightVAE`
    - `generation_total_s=137.11`
    - `generation_sec_per_audio_sec=13.71`
    - `clip_generation_s=107.16`
    - `postprocess_s=22.34`
Verdict:
  - `infer_frames=72` is worse by about `17.72s`
  - about `14.8%` slower than the current winner
  - do not spend more time pushing `infer_frames` above `64` on the current path before moving to the next roadmap item
Re-check after the newer `chunk=512 + rope-cache` winner:
  `infer_frames=72` is still a regression even on the stronger path.
  Warmup comparison:
  - current winner warmup: `166.90s`
  - `infer_frames=72` warmup: `196.93s`
  The rerun was stopped early after warmup as a clear regression.
Final verdict:
  Keep `infer_frames=64` as the winner on the current stack.

2. Full streaming / online postprocess.
Status: measured
Expected impact:
  Reduce deferred postprocess tail and maybe lower memory spikes.
Notes:
  Current built-in `enable_online_decode` was not enough.
Current result:
  Full online postprocess on top of the current `LightVAE + chunk=384` winner is
  effectively neutral:
  - baseline: `107.93s`
  - full online: `108.20s`
Verdict:
  Move on rather than spending more time here.

3. Fast VAE replacement.
Status: done
Expected impact:
  Reduce postprocess materially.
Notes:
  This is the strongest medium-risk candidate after compile cleanup.
Current result:
  First real `LightVAE` integration benchmark was a major win, and the later `LightTAE` / current TAE path became the stronger confirmed winner.

### Tier C: High-impact changes requiring new weights / alternate inference (`~x3` to `~x10+`)

1. Wan distilled / Lightning / FastVideo path.
Status: blocked_for_current_s2v_path
Expected impact:
  Realistic route to `x3+`.
Notes:
  This is no longer "tune the current path", but change the inference regime.
Current blocker:
  LightX2V provides a practical `Wan2.2` distill path for `i2v`, but not a ready-made `distill + s2v` path
  compatible with the current `LiveAvatar` audio-conditioned pipeline.
  The available `LightX2V` audio path targets a different model family / checkpoint layout (`SekoTalk` /
  `Wan2.1-R2V721-Audio-14B-720P`) and is not a drop-in replacement for the current `LiveAvatar` weights.
Verdict:
  - high upside remains real in principle
  - but not a near-term benchmark candidate for the current product path
  - move on to the next roadmap item unless dedicated new weights are prepared for a separate branch

2. Alternate lightweight VAE / TAE family.
Status: done
Expected impact:
  Potentially very strong postprocess reduction.
Notes:
  Higher quality risk than conservative fast-VAE replacements.
Current result:
  `LightTAE` / local TAE path is already integrated and is the current best confirmed VAE-family winner on this stack.

3. Alternate inference frameworks for Wan2.2.
Status: blocked_for_separate_integration_branch
Expected impact:
  Potential system-level speedup beyond local code tweaks.
Notes:
  Requires integration work and compatibility checks.
Current blocker:
  `cache-dit` is not a near-term drop-in on the current environment, and `LightX2V / FastVideo / Lightning` need separate model/inference branches for this S2V stack.

## Optimization Backlog

1. Unlock larger `infer_frames` on 96 GB VRAM without breaking cache correctness.
Status: done
Expectation:
  Reduce clip count and repeated denoise/prefill overhead.
Result:
  Fixed the blocking `kv_cache` wrap-around bug so `infer_frames=64` and `infer_frames=80` now render successfully.
  This later unlocked a strong warm-performance win together with `compile=true`.
Files:
  - `liveavatar/models/wan/causal_model_s2v.py`

2. Make benchmark comparisons generation-only, not cold-start dominated.
Status: done
Expectation:
  Remove misleading comparisons polluted by model load / compile warmup.
Result:
  Benchmark harness now treats warm generation metrics as primary and stores JSON results to disk.
Files:
  - `benchmarks/maxspeed/run_benchmark.py`

3. Validate FlashAttention fallback on Blackwell-safe path.
Status: done
Expectation:
  Prevent hard failures on unsupported flash kernels while preserving a fast GPU path.
Result:
  Added SDPA fallback when flash attention is unavailable or errors on Blackwell.
Files:
  - `liveavatar/models/wan/wan_2_2/modules/s2v/auxi_blocks.py`

4. Re-evaluate compile path after removing unstable scalar-capture tweak.
Status: done
Expectation:
  Keep `compile=true` only if warm generation becomes faster and stable.
Result so far:
  Removed `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS` / `capture_scalar_outputs` tweak because it caused compile instability.
  Re-ran warm compile benchmark on the resident pipeline.
  Measured run (`infer_frames=48`) improved from prompt-cache warm baseline:
  - before: `generation_total_s=537.3`, `generation_sec_per_audio_sec=53.73`
  - after: `generation_total_s=294.1`, `generation_sec_per_audio_sec=29.41`
  Approximate improvement:
  - `243.1s` faster
  - about `45.3%` faster
Files:
  - `smartblog_worker.py`

5. Measure `enable_online_decode`.
Status: measured
Expectation:
  Reduce deferred postprocess time.
Result so far:
  On the strongest compile profile, `enable_online_decode=true` reduced postprocess time,
  but increased clip-generation time enough to lose overall.
  Warm 10s comparison:
  - compile winner without online decode:
    - `generation_total_s=294.1`
    - `generation_sec_per_audio_sec=29.41`
    - `clip_generation_s=123.2`
    - `postprocess_s=155.3`
  - with online decode:
    - `generation_total_s=304.3`
    - `generation_sec_per_audio_sec=30.43`
    - `clip_generation_s=159.4`
    - `postprocess_s=129.3`
  Verdict:
  - regression in total generation time
  - keep disabled for now

6. Build a warm-run benchmark matrix.
Status: in_progress
Expectation:
  Compare `infer_frames`, `compile`, and decode modes on the metric that matters.
Result so far:
  Current best fully measured warm profile is:
  - `infer_frames=64`
  - `compile=true`
  - `enable_online_decode=false`
  - `LightVAE`
  - chunk sizes `384 / 384 / 384 / 384`
  It improved over the previous warm compile winner (`infer_frames=48`) from:
  - `generation_total_s=294.1`
  - `generation_sec_per_audio_sec=29.41`
  to:
  - `generation_total_s=107.93`
  - `generation_sec_per_audio_sec=10.79`
  Approximate improvement:
  - `186.2s` faster
  - about `63.3%` faster
  `infer_frames=80` reached the end of clip generation on the measured warm run,
  but did not complete cleanly or write `benchmark.json`, so it is currently treated
  as unstable / inconclusive rather than a winner.

7. Investigate fast VAE replacement.
Status: in_progress
Expectation:
  Reduce postprocess time materially without large quality loss.
Result so far:
  `LightVAE` is the current overall winner.
  Compared against the best official-VAE run:
  - official VAE (`infer_frames=64`, compile, warm):
    - `generation_total_s=211.97`
    - `generation_sec_per_audio_sec=21.20`
    - `postprocess_s=106.44`
    - `postprocess_decode_s=69.34`
    - `postprocess_encode_s=35.58`
  - `LightVAE` (`infer_frames=64`, compile, warm):
    - `generation_total_s=119.39`
    - `generation_sec_per_audio_sec=11.94`
    - `postprocess_s=21.93`
    - `postprocess_decode_s=13.94`
    - `postprocess_encode_s=7.08`
  Approximate improvement:
  - `92.57s` faster
  - about `43.7%` faster
  - about `1.78x` speedup
Files:
  - `liveavatar/models/wan/wan_2_2/modules/vae2_1.py`
  - `liveavatar/models/wan/causal_s2v_pipeline.py`

8. Investigate prompt embedding cache in resident worker.
Status: done
Expectation:
  Small but free improvement for repeated prompts.
Result:
  Implemented prompt embedding cache inside the resident pipeline.
  Warm 10s benchmark (`infer_frames=48`, `compile=false`) improved from:
  - `generation_total_s=550.5`
  - `generation_sec_per_audio_sec=55.05`
  to:
  - `generation_total_s=537.3`
  - `generation_sec_per_audio_sec=53.73`
  Approximate improvement:
  - `13.2s` faster
  - about `2.4%` faster
Files:
  - `liveavatar/models/wan/causal_s2v_pipeline.py`

9. Investigate full online decode / streaming postprocess.
Status: done
Expectation:
  Reduce deferred postprocess tail and peak memory.

10. Remove compile graph breaks in hot paths.
Status: in_progress
Expectation:
  Convert the current strong `compile=true` mode into a cleaner, more fully captured graph.
Current findings:
  Confirmed graph breaks in:
  - `causal_model_s2v.py` from `.item()` / tensor-size extraction
  - `wan_2_2/modules/s2v/s2v_utils.py` inside `rope_precompute`
Current state:
  First cleanup patches are already applied locally.
  Fresh warm benchmark after the latest cleanup completed, but it was a slight regression.
  Comparison against the current best `infer_frames=64 + compile=true` warm run:
  - old winner:
    - `generation_total_s=226.3`
    - `generation_sec_per_audio_sec=22.63`
    - `clip_generation_s=104.0`
    - `postprocess_s=106.5`
  - cleanup attempt:
    - `generation_total_s=229.5`
    - `generation_sec_per_audio_sec=22.95`
    - `clip_generation_s=107.4`
    - `postprocess_s=106.3`
  Verdict:
  - slightly slower overall
  - not a good enough direction to prioritize right now
  - move to the next roadmap item

## Measured Results

These are recorded reference points already observed during work. Warm-generation comparison will gradually replace cold-start-heavy numbers below.

### 30s Reference Runs

1. Baseline
- Config:
  - `size=480*832`
  - `infer_frames=48`
  - `sample_steps=4`
  - `compile=true`
  - `disable_flash_attn=true`
- Result:
  - `render_s=854.4`
  - `clip_generation_s=382.4`
  - `postprocess_s=448.8`
  - `upscale_s=1.2`
  - `total_s=955.0`
  - `sec_per_audio_sec=31.83`
- Notes:
  - This was the strongest stable reference before larger `infer_frames` worked.

2. Online decode
- Config:
  - same as baseline
  - `enable_online_decode=true`
- Result:
  - `render_s=848.6`
  - `clip_generation_s=404.7`
  - `postprocess_s=420.5`
  - `upscale_s=1.2`
  - `total_s=949.3`
  - `sec_per_audio_sec=31.64`
- Notes:
  - Tiny total improvement only.

3. Larger clips after KV-cache fix
- Config:
  - `size=480*832`
  - `infer_frames=64`
  - `sample_steps=4`
  - `compile=false`
  - `disable_flash_attn=true`
- Result:
  - `render_s=1259.2`
  - `clip_generation_s=915.5`
  - `postprocess_s=319.6`
  - `upscale_s=1.2`
  - `total_s=1361.1`
  - `sec_per_audio_sec=45.37`
- Notes:
  - Bug fixed successfully, but this mode was slower than baseline in cold-run conditions.

### 10s Control Runs

1. `infer_frames=64`, `compile=true`
- Result:
  - `render_s=452.0`
  - `clip_generation_s=327.0`
  - `postprocess_s=106.4`
  - `upscale_s=1.0`
  - `total_s=553.2`
  - `sec_per_audio_sec=55.32`
- Notes:
  - Stable after removing the scalar-capture tweak.

2. `infer_frames=80`, `compile=false`
- Result:
  - `render_s=513.3`
  - `clip_generation_s=384.2`
  - `postprocess_s=109.2`
  - `upscale_s=1.0`
  - `total_s=614.0`
  - `sec_per_audio_sec=61.40`
- Notes:
  - Stable after KV-cache fix, but not a winner so far.

3. Warm baseline control
- Config:
  - `size=480*832`
  - `infer_frames=48`
  - `sample_steps=4`
  - `compile=false`
  - `disable_flash_attn=true`
  - `warm_runs=1`
- Result:
  - warmup:
    - `generation_total_s=554.3`
    - `generation_sec_per_audio_sec=55.43`
  - measured run:
    - `generation_total_s=550.5`
    - `generation_sec_per_audio_sec=55.05`
    - `clip_generation_s=378.3`
    - `postprocess_s=154.8`
    - `upscale_s=1.0`

4. Warm prompt-cache run
- Config:
  - same as warm baseline control
- Result:
  - warmup:
    - `generation_total_s=552.8`
    - `generation_sec_per_audio_sec=55.28`
  - measured run:
    - `generation_total_s=537.3`
    - `generation_sec_per_audio_sec=53.73`
    - `clip_generation_s=367.2`
    - `postprocess_s=154.5`
    - `upscale_s=0.95`
- Notes:
  - Real but modest win.

5. Warm compile winner
- Config:
  - `size=480*832`
  - `infer_frames=48`
  - `sample_steps=4`
  - `compile=true`
  - `disable_flash_attn=true`
  - `warm_runs=1`
- Result:
  - warmup:
    - `generation_total_s` observed during setup but not used as the comparison target
  - measured run:
    - `generation_total_s=294.1`
    - `generation_sec_per_audio_sec=29.41`
    - `clip_generation_s=123.2`
    - `postprocess_s=155.3`
    - `upscale_s=1.0`
- Notes:
  - Strong compile win over the prompt-cache-only baseline.

6. Warm compile with online decode
- Config:
  - same as warm compile winner
  - `enable_online_decode=true`
- Result:
  - measured run:
    - `generation_total_s=304.3`
    - `generation_sec_per_audio_sec=30.43`
    - `clip_generation_s=159.4`
    - `postprocess_s=129.3`
    - `upscale_s=0.98`
- Notes:
  - Reduced postprocess time
  - Lost overall because clip generation got slower

7. Warm larger clips
- Config:
  - `size=480*832`
  - `infer_frames=64`
  - `sample_steps=4`
  - `compile=true`
  - `disable_flash_attn=true`
  - `warm_runs=1`
- Result:
  - warmup:
    - `generation_total_s=317.4`
  - measured run:
    - `generation_total_s=226.3`
    - `generation_sec_per_audio_sec=22.63`
    - `clip_generation_s=104.0`
    - `postprocess_s=106.5`
    - `upscale_s=0.94`
- Notes:
  - Current best fully measured warm result.

8. Warm very large clips
- Config:
  - `size=480*832`
  - `infer_frames=80`
  - `sample_steps=4`
  - `compile=true`
  - `disable_flash_attn=true`
  - `warm_runs=1`
- Result:
  - warmup completed successfully
  - measured run reached `complete full-sequence generation`
  - but did not finish cleanly or write `benchmark.json`
- Notes:
  - Treat as unstable / inconclusive for now.

9. Warm compile graph-break cleanup attempt
- Config:
  - `size=480*832`
  - `infer_frames=64`
  - `sample_steps=4`
  - `compile=true`
  - `disable_flash_attn=true`
  - `warm_runs=1`
- Result:
  - warmup:
    - `generation_total_s=331.9`
  - measured run:
    - `generation_total_s=229.5`
    - `generation_sec_per_audio_sec=22.95`
    - `clip_generation_s=107.4`
    - `postprocess_s=106.3`
    - `upscale_s=0.95`
- Notes:
  - Completed successfully
  - Slight regression versus the current best warm result (`226.3`)
  - Do not keep prioritizing this path before trying the next roadmap item

5. Warm compile run
- Config:
  - `size=480*832`
  - `infer_frames=48`
  - `sample_steps=4`
  - `compile=true`
  - `disable_flash_attn=true`
  - `warm_runs=1`
- Result:
  - warmup:
    - `generation_total_s=420.0`
    - `generation_sec_per_audio_sec=42.00`
  - measured run:
    - `generation_total_s=294.1`
    - `generation_sec_per_audio_sec=29.41`
    - `clip_generation_s=123.2`
    - `postprocess_s=155.3`
    - `upscale_s=1.0`
- Notes:
  - This is the strongest measured win so far.

6. Warm compile + online decode run
- Config:
  - same as warm compile run
  - `enable_online_decode=true`
- Result:
  - warmup:
    - `generation_total_s=388.1`
    - `generation_sec_per_audio_sec=38.81`
  - measured run:
    - `generation_total_s=304.3`
    - `generation_sec_per_audio_sec=30.43`
    - `clip_generation_s=159.4`
    - `postprocess_s=129.3`
    - `upscale_s=0.98`
- Notes:
  - Postprocess improved, but overall result regressed versus plain warm compile.

7. Warm `infer_frames=64` compile run
- Config:
  - `size=480*832`
  - `infer_frames=64`
  - `sample_steps=4`
  - `compile=true`
  - `disable_flash_attn=true`
  - `warm_runs=1`
- Result:
  - warmup:
    - `generation_total_s=290.0`
    - `generation_sec_per_audio_sec=29.00`
  - measured run:
    - `generation_total_s=212.0`
    - `generation_sec_per_audio_sec=21.20`
    - `clip_generation_s=89.9`
    - `postprocess_s=106.4`
    - `postprocess_decode_s=69.3`
    - `postprocess_encode_s=35.6`
    - `upscale_s=0.96`
- Notes:
  - New best result so far.
  - Improvement versus previous leader (`infer_frames=48`, `compile=true`):
    - from `294.1s` to `212.0s`
    - about `27.9%` faster
  - Compile warmup is expensive, but warm resident-worker generation is much faster.
  - Important postprocess breakdown:
    - `VAE decode` is about `69.3s`
    - `VAE encode` is about `35.6s`
    - together they account for almost the entire `106.4s` postprocess tail
  - Practical implication:
    - `fast VAE` remains a strong next target
    - even a perfect VAE would cap overall warm generation near `106s`, i.e. roughly `2x` from this winner

## Rules For Future Entries

- After each optimization:
  - write what changed
  - record measured result
  - state whether it is a win, neutral change, or regression
  - then move to the next backlog item

## Recent Measurements

8. Warm `infer_frames=64` compile run with `flash-attn`
- Config:
  - `size=480*832`
  - `infer_frames=64`
  - `sample_steps=4`
  - `compile=true`
  - `disable_cudnn_attn=true`
  - `disable_flash_attn=false`
  - `warm_runs=1`
- Result:
  - did not complete
  - pipeline load succeeded
  - warmup render entered clip generation
  - failed with:
    - `CUDA error ... no kernel image is available for execution on the device`
- Notes:
  - Dead-end on the current GPU/software stack.
  - Not a benchmark winner candidate.
  - Move on.

9. Warm `infer_frames=72` compile run
- Config:
  - `size=480*832`
  - `infer_frames=72`
  - `sample_steps=4`
  - `compile=true`
  - `disable_flash_attn=true`
  - `warm_runs=1`
- Result:
  - warmup:
    - `generation_total_s=341.2`
    - `generation_sec_per_audio_sec=34.12`
  - measured run:
    - `generation_total_s=233.3`
    - `generation_sec_per_audio_sec=23.33`
    - `clip_generation_s=109.4`
    - `postprocess_s=107.9`
    - `upscale_s=0.95`
- Notes:
  - Slight regression versus the current winner (`infer_frames=64`):
    - from `212.0s` to `233.3s`
    - about `10.1%` slower
  - Do not keep pushing larger `infer_frames` beyond the current winner before
    moving to a stronger roadmap item.

10. Warm `infer_frames=64` compile run with `LightVAE`
- Config:
  - `size=480*832`
  - `infer_frames=64`
  - `sample_steps=4`
  - `compile=true`
  - `disable_flash_attn=true`
  - `LIVEAVATAR_USE_LIGHTVAE=true`
  - `LIVEAVATAR_VAE_PATH=/root/LiveAvatar/ckpt/Autoencoders/lightvaew2_1.pth`
  - `warm_runs=1`
- Result:
  - warmup:
    - `generation_total_s=197.7`
    - `generation_sec_per_audio_sec=19.77`
    - `postprocess_s=22.15`
    - `postprocess_decode_s=14.14`
    - `postprocess_encode_s=7.10`
  - measured run:
    - `generation_total_s=119.4`
    - `generation_sec_per_audio_sec=11.94`
    - `clip_generation_s=90.19`
    - `postprocess_s=21.93`
    - `postprocess_decode_s=13.94`
    - `postprocess_encode_s=7.08`
    - `upscale_s=0.93`
- Notes:
  - Major win and new best result so far.
  - Improvement versus previous best (`infer_frames=64`, official VAE):
    - from `212.0s` to `119.4s`
    - about `43.7%` faster
    - about `1.78x` speedup
  - Postprocess tail was cut from `106.4s` to `21.9s`.
  - Bottleneck has shifted back toward clip generation, so the next logical test is
    larger `infer_frames` on top of `LightVAE`.

11. Full online postprocess on the current best `LightVAE + chunk=384` path
- Config:
  - `size=480*832`
  - `infer_frames=64`
  - `sample_steps=4`
  - `compile=true`
  - `disable_flash_attn=true`
  - `LIVEAVATAR_USE_LIGHTVAE=true`
  - `LIVEAVATAR_VAE_PATH=/root/LiveAvatar/ckpt/Autoencoders/lightvaew2_1.pth`
  - `LIVEAVATAR_ENABLE_FULL_ONLINE_POSTPROCESS=true`
  - `chunk=384 / 384 / 384 / 384`
  - `warm_runs=1`
- Result:
  - warmup:
    - `generation_total_s=177.11`
    - `generation_sec_per_audio_sec=17.71`
    - `clip_generation_s=166.67`
    - deferred postprocess timers are `null` because decode/encode are now folded into
      the per-clip path
  - measured run:
    - `generation_total_s=108.20`
    - `generation_sec_per_audio_sec=10.82`
    - `clip_generation_s=100.92`
    - deferred postprocess timers are `null` for the same reason
    - `upscale_s=0.94`
- Notes:
  - Works correctly once the benchmark passes the proper LightVAE checkpoint:
    - `LIVEAVATAR_VAE_PATH=/root/LiveAvatar/ckpt/Autoencoders/lightvaew2_1.pth`
  - This is effectively a wash versus the current winner:
    - current winner (`LightVAE + chunk=384`): `107.93s`
    - full online postprocess: `108.20s`
    - about `0.25%` slower
  - Conclusion: no meaningful upside on this stack. Mark as `neutral/slight regression`
    and move on to the next roadmap item instead of spending more time here.

12. Warm `infer_frames=64` compile run with `LightVAE` and `chunk=448`
- Config:
  - `size=480*832`
  - `infer_frames=64`
  - `sample_steps=4`
  - `compile=true`
  - `disable_flash_attn=true`
  - `LIVEAVATAR_USE_LIGHTVAE=true`
  - `LIVEAVATAR_VAE_PATH=/root/LiveAvatar/ckpt/Autoencoders/lightvaew2_1.pth`
  - `chunk=448 / 448 / 448 / 448`
  - `warm_runs=1`
- Result:
  - warmup:
    - `generation_total_s=245.03`
    - `generation_sec_per_audio_sec=24.50`
    - `clip_generation_s=212.41`
    - `postprocess_s=22.09`
  - measured run:
    - `generation_total_s=105.33`
    - `generation_sec_per_audio_sec=10.53`
    - `clip_generation_s=76.00`
    - `postprocess_s=21.96`
    - `postprocess_decode_s=13.97`
    - `postprocess_encode_s=7.08`
    - `upscale_s=0.94`
- Notes:
  - Real win over the previous `chunk=384` leader:
    - from `107.93s` to `105.33s`
    - about `2.4%` faster
  - Almost the entire gain came from clip generation:
    - `78.72s -> 76.00s`
  - This is the new best result so far.
  - Next logical follow-up: re-test `chunk=512` on the same `LightVAE` path instead of
    relying on the earlier non-LightVAE hang.

13. Warm `infer_frames=64` compile run with `LightVAE` and `chunk=512`
- Config:
  - `size=480*832`
  - `infer_frames=64`
  - `sample_steps=4`
  - `compile=true`
  - `disable_flash_attn=true`
  - `LIVEAVATAR_USE_LIGHTVAE=true`
  - `LIVEAVATAR_VAE_PATH=/root/LiveAvatar/ckpt/Autoencoders/lightvaew2_1.pth`
  - `chunk=512 / 512 / 512 / 512`
  - `warm_runs=1`
- Result:
  - warmup:
    - `generation_total_s=166.54`
    - `generation_sec_per_audio_sec=16.65`
    - `clip_generation_s=134.07`
    - `postprocess_s=22.12`
  - measured run:
    - `generation_total_s=100.31`
    - `generation_sec_per_audio_sec=10.03`
    - `clip_generation_s=71.06`
    - `postprocess_s=21.90`
    - `postprocess_decode_s=13.94`
    - `postprocess_encode_s=7.07`
    - `upscale_s=0.93`
- Notes:
  - Another real win over the `chunk=448` leader:
    - from `105.33s` to `100.31s`
    - about `4.8%` faster
  - Again the gain came almost entirely from clip generation:
    - `76.00s -> 71.06s`
  - This is the new best result so far.
  - Next logical follow-up: continue climbing to `chunk=640` until the trend breaks or
    stability becomes worse than the speed gain.

14. Warm `infer_frames=64` compile run with `LightVAE` and `chunk=640`
- Config:
  - `size=480*832`
  - `infer_frames=64`
  - `sample_steps=4`
  - `compile=true`
  - `disable_flash_attn=true`
  - `LIVEAVATAR_USE_LIGHTVAE=true`
  - `LIVEAVATAR_VAE_PATH=/root/LiveAvatar/ckpt/Autoencoders/lightvaew2_1.pth`
  - `chunk=640 / 640 / 640 / 640`
  - `warm_runs=1`
- Result:
  - warmup:
    - `generation_total_s=219.40`
    - `generation_sec_per_audio_sec=21.94`
    - `clip_generation_s=186.98`
    - `postprocess_s=22.10`
  - measured run:
    - `generation_total_s=100.88`
    - `generation_sec_per_audio_sec=10.09`
    - `clip_generation_s=71.60`
    - `postprocess_s=21.94`
    - `postprocess_decode_s=13.95`
    - `postprocess_encode_s=7.05`
    - `upscale_s=0.97`
- Notes:
  - Slight regression versus the `chunk=512` winner:
    - from `100.31s` to `100.88s`
    - about `0.57%` slower
  - The chunking trend has flattened and started to reverse.
  - Treat `chunk=512` as the current local optimum and move on to the next roadmap item
    instead of continuing to squeeze this axis blindly.

15. Rope-frequency cache on the current `LightVAE + chunk=512` winner
- Config:
  - `size=480*832`
  - `infer_frames=64`
  - `sample_steps=4`
  - `compile=true`
  - `disable_flash_attn=true`
  - `LIVEAVATAR_USE_LIGHTVAE=true`
  - `LIVEAVATAR_VAE_PATH=/root/LiveAvatar/ckpt/Autoencoders/lightvaew2_1.pth`
  - `chunk=512 / 512 / 512 / 512`
  - added cache for `rope_precompute(...)` keyed by input shape + rollout grid sizes
  - `warm_runs=1`
- Result:
  - warmup:
    - `generation_total_s=166.90`
    - `generation_sec_per_audio_sec=16.69`
    - `clip_generation_s=134.34`
    - `postprocess_s=22.08`
  - measured run:
    - `generation_total_s=99.20`
    - `generation_sec_per_audio_sec=9.92`
    - `clip_generation_s=69.90`
    - `postprocess_s=21.94`
    - `postprocess_decode_s=13.95`
    - `postprocess_encode_s=7.10`
    - `upscale_s=0.97`
- Notes:
  - Real win over the previous `chunk=512` leader:
    - from `100.31s` to `99.20s`
    - about `1.1%` faster
  - The gain again landed in clip generation:
    - `71.06s -> 69.90s`
  - The first implementation also triggered a `torch.compile` backend exception while
    serializing tensor grid sizes for the cache key.
  - Next step: keep this speed win but clean up the cache-key path so it stops fighting
    `torch.compile`.

16. Rope-cache cleanup attempt with `@torch._dynamo.disable`
- Config:
  - same as item 15
  - added `@torch._dynamo.disable` to the rope-cache key serializer
- Result:
  - warmup:
    - `generation_total_s=237.30`
    - `generation_sec_per_audio_sec=23.73`
    - `clip_generation_s=204.80`
  - measured run:
    - `generation_total_s=109.92`
    - `generation_sec_per_audio_sec=10.99`
    - `clip_generation_s=80.57`
    - `postprocess_s=21.87`
- Notes:
  - Clear regression versus the original rope-cache winner:
    - from `99.20s` to `109.92s`
    - about `10.8%` slower
  - It also triggered `torch._dynamo` recompile-limit warnings.
  - Revert this cleanup attempt. Keep the original rope-cache variant instead of trying to
    “help” compile this way.

17. Backend check: disable `cuDNN attention` on the current winner path
- Config:
  - same as item 15
  - `LIVEAVATAR_DISABLE_CUDNN_ATTN=true`
- Result:
  - warmup only:
    - `generation_total_s=183.57`
    - `generation_sec_per_audio_sec=18.36`
    - `clip_generation_s=150.65`
    - `postprocess_s=22.31`
- Notes:
  - This is already worse than the current winner warmup:
    - `166.90s -> 183.57s`
    - about `10.0%` slower
  - The primary run was stopped early as a non-promising regression.
  - Keep `cuDNN attention` enabled on the current path.

18. Re-check `infer_frames=72` on the newer `chunk=512 + rope-cache` winner
- Config:
  - same as item 15, but `infer_frames=72`
- Result:
  - warmup only:
    - `generation_total_s=196.93`
    - `generation_sec_per_audio_sec=19.69`
    - `clip_generation_s=162.98`
    - `postprocess_s=22.66`
- Notes:
  - This is already worse than the current winner warmup:
    - `166.90s -> 196.93s`
    - about `18.0%` slower
  - The primary run was stopped early as a clear regression.
  - This closes the `infer_frames=72` revisit on the newer stack too.

19. Fixed benchmark harness import order for compile-sensitive env
- Config:
  - benchmark harness only
  - move benchmark-controlled env setup before importing `smartblog_worker.py`
- Result:
  - this changed the measured baseline on the current rope-cache path
  - previous recorded winner artifact (`infer64_compile_lightvae_chunk512_ropecache`) is no longer
    considered authoritative for compile comparisons because the harness was setting compile-related
    env too late
  - corrected baseline on the same general path (`infer64_compile_lightvae_chunk512_ropecache_corrected`):
    - `generation_total_s=101.21`
    - `generation_sec_per_audio_sec=10.12`
    - `clip_generation_s=71.88`
    - `postprocess_s=22.05`
- Notes:
  - treat the corrected harness as the new source of truth for further comparisons
  - do not compare new compile/no-compile work against the older pre-fix harness artifacts

20. Re-check `compile=false` on the corrected winner path
- Config:
  - same path as item 19
  - `compile=false`
- Result:
  - run started with `COMPILE: False` confirmed in logs
  - warmup path was already clearly much slower than corrected `compile=true`
  - the run was stopped early as a non-promising regression to avoid wasting GPU time
- Notes:
  - this closes the concern that the earlier compile comparison might have been purely a harness bug
  - after fixing the harness, `compile=false` still does not look competitive on the current path

21. Capture scalar outputs on the corrected compile winner
- Config:
  - same as item 19
  - add `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1`
- Result:
  - warmup:
    - `generation_total_s=243.81`
    - `generation_sec_per_audio_sec=24.38`
    - `clip_generation_s=211.10`
    - `postprocess_s=22.30`
  - measured run:
    - `generation_total_s=95.09`
    - `generation_sec_per_audio_sec=9.51`
    - `clip_generation_s=65.83`
    - `postprocess_s=22.05`
    - `upscale_s=0.93`
- Notes:
  - despite a much worse warmup, the measured warm generation is a real win over the corrected
    baseline:
    - `101.21s -> 95.09s`
    - about `6.1%` faster
  - almost all of the gain landed in clip generation:
    - `71.88s -> 65.83s`
  - this is now the new corrected leader for warm generation on the current stack

22. Compile `rope_precompute` on top of the capture-scalar winner
- Config:
  - same as item 21
  - enable `@conditional_compile` on `liveavatar/models/wan/wan_2_2/modules/s2v/s2v_utils.py:rope_precompute`
- Result:
  - warmup:
    - `generation_total_s=165.45`
    - `generation_sec_per_audio_sec=16.55`
    - `clip_generation_s=133.08`
    - `postprocess_s=22.19`
  - measured run:
    - `generation_total_s=94.87`
    - `generation_sec_per_audio_sec=9.49`
    - `clip_generation_s=65.54`
    - `postprocess_s=22.02`
    - `upscale_s=0.92`
- Notes:
  - this is a small but real win over item 21:
    - `95.09s -> 94.87s`
    - about `0.23%` faster
  - the gain again lands almost entirely in clip generation:
    - `65.83s -> 65.54s`
  - keep this change; it is not a big win, but it is a clean win on the corrected path

23. Re-check chunk size above `512` on the new corrected winner
- Config:
  - same as item 22
  - `chunk=576 / 576 / 576 / 576`
- Result:
  - warmup was already clearly worse than item 22
  - the run was stopped early before primary completion as a non-promising regression
- Notes:
  - no sign that the local chunk optimum moved upward after the compile fixes
  - keep `512` as the local optimum and move on

24. Validate the new winner on a longer `30s` benchmark
- Config:
  - same as item 22
  - benchmark asset: `benchmarks/maxspeed/boy_30s.wav`
- Result:
  - warmup:
    - `generation_total_s=345.62`
    - `generation_sec_per_audio_sec=11.52`
    - `clip_generation_s=263.43`
    - `postprocess_s=66.37`
  - measured run:
    - `generation_total_s=270.34`
    - `generation_sec_per_audio_sec=9.01`
    - `clip_generation_s=191.50`
    - `postprocess_s=66.25`
    - `upscale_s=1.19`
- Notes:
  - the current winner clearly transfers to a longer clip and is not just overfit to the `10s` benchmark
  - the long-run rate (`9.01 sec/sec`) is even slightly better than the short-run rate on the same path
  - this validates the current leader as a real production-facing improvement, not just a synthetic micro-benchmark win

25. Overall speedup checkpoint before moving attention to Tier C
- Summary:
  - current best `10s` warm result:
    - `generation_total_s=94.87`
    - `generation_sec_per_audio_sec=9.49`
  - current best `30s` warm result:
    - `generation_total_s=270.34`
    - `generation_sec_per_audio_sec=9.01`
- Compared with early local warm baselines:
  - versus `compile_warm_check`:
    - `294.14s -> 94.87s`
    - about `3.10x` speedup
  - versus the earlier official-VAE breakdown baseline:
    - `211.97s -> 94.87s`
    - about `2.23x` speedup
  - versus the previously recorded manual `30s` baseline:
    - `954.97s -> 270.34s`
    - about `3.53x` speedup
- Notes:
  - the current local-path work has already crossed the “strong improvement” threshold
  - further gains inside the same path are now likely to be incremental unless Tier C items become practical

26. Try more aggressive `LightVAE` pruning on the existing `lightvaew2_1` checkpoint
- Config:
  - current winner path
  - `LIVEAVATAR_VAE_PRUNING_RATE=0.875`
- Result:
  - failed immediately at VAE load with state-dict shape mismatches across the encoder and decoder
  - the existing checkpoint only matches the current `LightVAE` architecture at its baked-in pruning level
- Notes:
  - `lightvaew2_1.pth` is not a generic “prune harder” checkpoint
  - it is a trained/distilled checkpoint tied to the existing `75%` pruned architecture
  - this closes the easy idea of pushing `LightVAE` faster just by increasing the pruning rate on the same weights

27. Side benchmark `LightTAE` as the next Tier C candidate
- Config:
  - input video: `benchmarks/maxspeed/infer64_compile_lightvae_chunk512_ropecache_capture_scalar_compilerope/rendered_raw.mp4`
  - benchmarked outside the full S2V path to isolate video autoencoder throughput
  - checkpoint: `ckpt/Autoencoders/lighttaew2_1.pth`
- Result:
  - `parallel=False`:
    - `encode_s=0.80`
    - `decode_s=1.29`
    - `total_s=2.09`
  - `parallel=True`:
    - `encode_s=0.43`
    - `decode_s=1.25`
    - `total_s=1.69`
- Notes:
  - this is an extremely strong standalone result and makes `LightTAE` the most credible remaining candidate for another major speed jump
  - official LightX2V documentation reports the same direction of win for Wan2.1:
    - `lightvaew2_1`: encode `1.5014s`, decode `2.0697s`
    - `lighttaew2_1`: encode `0.3956s`, decode `0.2463s`
    - source: `lightx2v/Autoencoders` model card
  - next step:
    - stop treating alternate VAE/TAE as purely theoretical
    - attempt a real `LightTAE` integration into the current S2V pipeline and benchmark the full generation path

28. Full `LightTAE` integration in the current S2V pipeline
- Config:
  - current best DiT path:
    - `infer_frames=64`
    - `chunk=512`
    - `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1`
    - `compile=true`
  - switch VAE path to:
    - `LIVEAVATAR_USE_TAE=true`
    - `LIVEAVATAR_VAE_PATH=/root/LiveAvatar/ckpt/Autoencoders/lighttaew2_1.pth`
    - `LIVEAVATAR_TAE_PARALLEL=false`
- Integration notes:
  - `parallel=true` in full pipeline was tested first and failed in postprocess decode with a large VRAM spike
  - a minimal in-repo `TAEHV` adapter was added and fixed until the sequential path completed end-to-end
- Result:
  - warmup:
    - `generation_total_s=145.86`
    - `generation_sec_per_audio_sec=14.59`
    - `clip_generation_s=132.70`
    - `postprocess_s=4.54`
  - measured run:
    - `generation_total_s=75.92`
    - `generation_sec_per_audio_sec=7.59`
    - `clip_generation_s=65.74`
    - `postprocess_s=4.34`
    - `postprocess_decode_s=2.04`
    - `postprocess_encode_s=1.54`
    - `upscale_s=0.96`
- Comparison against the previous winner (item 22):
  - previous winner:
    - `generation_total_s=94.87`
    - `postprocess_s=22.02`
  - new `LightTAE` result:
    - `generation_total_s=75.92`
    - `postprocess_s=4.34`
- Improvement:
  - `18.95s` faster
  - about `20.0%` faster end-to-end on the warm benchmark
  - postprocess was reduced by about `17.68s`
- Verdict:
  - `LightTAE` is the new overall leader
  - this is the first Tier C candidate that converted into a real production-path win on the current S2V stack
  - next step:
    - explicitly re-check this new leader with `compile=false`, because this is a materially different architecture and the compile tradeoff may have shifted

29. Re-check `compile=false` on the new `LightTAE` winner
- Config:
  - same as item 28
  - `compile=false`
- Result:
  - warmup:
    - `generation_total_s=185.92`
    - `generation_sec_per_audio_sec=18.59`
    - `clip_generation_s=172.80`
    - `postprocess_s=4.47`
  - measured run:
    - `generation_total_s=167.51`
    - `generation_sec_per_audio_sec=16.75`
    - `clip_generation_s=157.39`
    - `postprocess_s=4.35`
    - `upscale_s=0.93`
- Comparison against the current `LightTAE` compile winner:
  - `compile=true`:
    - `generation_total_s=75.92`
    - `clip_generation_s=65.74`
  - `compile=false`:
    - `generation_total_s=167.51`
    - `clip_generation_s=157.39`
- Verdict:
  - `compile=false` is worse by `91.58s`
  - about `120.6%` slower than the current `LightTAE` winner
  - this closes the compile concern on the new architecture too: `compile=true` still wins decisively here
  - next step:
    - validate the new `LightTAE` winner on the `30s` benchmark before moving to the next roadmap branch

30. Validate the `LightTAE` winner on the longer `30s` benchmark
- Config:
  - same as item 28
  - benchmark asset: `benchmarks/maxspeed/boy_30s.wav`
- Result:
  - warmup:
    - `generation_total_s=285.58`
    - `generation_sec_per_audio_sec=9.52`
    - `clip_generation_s=257.78`
    - `postprocess_s=13.62`
    - `upscale_s=1.18`
  - measured run:
    - `generation_total_s=217.21`
    - `generation_sec_per_audio_sec=7.24`
    - `clip_generation_s=192.24`
    - `postprocess_s=13.84`
    - `postprocess_decode_s=6.50`
    - `postprocess_encode_s=4.97`
    - `upscale_s=1.20`
- Comparison against the previous best long-run result (item 24, `LightVAE` path):
  - previous long winner:
    - `generation_total_s=270.34`
    - `generation_sec_per_audio_sec=9.01`
    - `postprocess_s=66.25`
  - new `LightTAE` long result:
    - `generation_total_s=217.21`
    - `generation_sec_per_audio_sec=7.24`
    - `postprocess_s=13.84`
- Improvement:
  - `53.12s` faster on the `30s` benchmark
  - about `19.6%` faster end-to-end
  - long-run rate improved from `9.01 sec/sec` to `7.24 sec/sec`
  - postprocess dropped by about `52.41s`
- Verdict:
  - `LightTAE + compile=true` is now the confirmed overall production-path winner on both short and long warm benchmarks
  - this is no longer a micro-benchmark anomaly
  - next step:
    - move on from this path and test the next remaining roadmap candidate instead of over-tuning a path that is already clearly dominant

31. Re-check `infer_frames=72` on the new `LightTAE` winner path
- Config:
  - same as item 28
  - `infer_frames=72`
- Result:
  - no full artifact was kept; the run was stopped during warmup as a clear regression
  - live warmup signals were already materially worse than the `infer_frames=64` winner:
    - first denoise block was much slower
    - early clip progression lagged clearly behind the current winner
- Notes:
  - this is consistent with the older `LightVAE` path, where pushing above `64` also regressed
  - there is no good sign that `infer_frames>64` becomes attractive just because the VAE got faster
- Verdict:
  - keep `infer_frames=64` as the winner on the `LightTAE` path too
  - move on to the next candidate instead of burning more time on larger frame blocks

32. Re-check larger chunking (`chunk=640`) on the `LightTAE` winner path
- Config:
  - same as item 28
  - chunk sizes raised from `512 / 512 / 512 / 512` to `640 / 640 / 640 / 640`
- Result:
  - no full artifact was kept; the run was stopped during warmup as a clear regression
  - live warmup signals were already worse than the `chunk=512` winner:
    - first denoise block was materially slower
    - subsequent warmup progression stayed behind the current winner
- Verdict:
  - `chunk=512` remains the local optimum on the `LightTAE` path
  - stop retuning chunk sizes and move on to the next roadmap branch

33. Try a heavier compile mode on the `LightTAE` winner path (`max-autotune-no-cudagraphs`)
- Config:
  - same as item 28
  - `compile_mode=max-autotune-no-cudagraphs`
- Result:
  - no measured artifact was kept
  - the run spent a very long time inside Triton autotune during warmup
  - output directory stayed empty and the process showed no practical path to a timely measured run
  - live logs showed repeated large autotune sweeps and out-of-resource rejections for candidate kernels
- Verdict:
  - not a practical benchmark winner for the current workflow
  - treat this compile mode as a dead-end and move on to lighter compile-mode tuning instead

34. Try `reduce-overhead` compile mode on the `LightTAE` winner path
- Config:
  - same as item 28
  - `compile_mode=reduce-overhead`
- Result:
  - failed during warmup before a measured artifact was produced
  - hit a runtime error in the cudagraph/checkpoint-pool path:
    - `RuntimeError: Expected curr_block->next == nullptr to be true, but got false`
  - logs also showed repeated cudagraph skips from mutated inputs and CPU-device graph fragments
- Verdict:
  - not stable enough to be a production benchmark candidate on the current path
  - keep the default compile mode and move on to the next roadmap item

35. Formalize and re-check `capture_scalar_outputs` on the `LightTAE` winner path
- Config:
  - same as item 28
  - explicitly set `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` through the benchmark harness
- Result:
  - warmup:
    - `generation_total_s=146.49`
    - `generation_sec_per_audio_sec=14.65`
    - `clip_generation_s=133.01`
    - `postprocess_s=4.62`
  - measured run:
    - `generation_total_s=75.69`
    - `generation_sec_per_audio_sec=7.57`
    - `clip_generation_s=65.47`
    - `postprocess_s=4.47`
    - `postprocess_decode_s=2.09`
    - `postprocess_encode_s=1.61`
    - `upscale_s=0.93`
- Comparison against item 28:
  - previous `LightTAE` winner:
    - `generation_total_s=75.92`
    - `clip_generation_s=65.74`
  - explicit capture-scalar winner:
    - `generation_total_s=75.69`
    - `clip_generation_s=65.47`
- Improvement:
  - `0.23s` faster
  - about `0.3%` faster
- Verdict:
  - small but real win
  - keep `capture_scalar_outputs=true` as part of the formal winner config
  - next step:
    - validate this fully formalized winner on the `30s` benchmark

36. Validate the fully formalized `LightTAE` winner on the `30s` benchmark
- Config:
  - same as item 35
  - benchmark asset: `benchmarks/maxspeed/boy_30s.wav`
- Result:
  - warmup:
    - `generation_total_s=284.62`
    - `generation_sec_per_audio_sec=9.49`
    - `clip_generation_s=257.16`
    - `postprocess_s=13.38`
    - `upscale_s=1.18`
  - measured run:
    - `generation_total_s=215.17`
    - `generation_sec_per_audio_sec=7.17`
    - `clip_generation_s=190.84`
    - `postprocess_s=13.25`
    - `postprocess_decode_s=6.14`
    - `postprocess_encode_s=4.83`
    - `upscale_s=1.18`
- Comparison against item 30:
  - previous long `LightTAE` winner:
    - `generation_total_s=217.21`
    - `generation_sec_per_audio_sec=7.24`
    - `clip_generation_s=192.24`
  - formal capture-scalar long winner:
    - `generation_total_s=215.17`
    - `generation_sec_per_audio_sec=7.17`
    - `clip_generation_s=190.84`
- Improvement:
  - `2.05s` faster
  - about `0.9%` faster
- Verdict:
  - small but real long-run win
  - the formal winner configuration is now the best confirmed result on both `10s` and `30s`
  - next step:
    - measure the new `rollout-grid cache` hot-path cleanup on top of this winner

37. Add `rollout-grid cache` on top of the formal `LightTAE` winner
- Config:
  - same as item 35
  - cache `rollout_grid_sizes(...)` results in `causal_model_s2v.py`
- Result:
  - warmup:
    - `generation_total_s=153.02`
    - `generation_sec_per_audio_sec=15.30`
    - `clip_generation_s=139.68`
    - `postprocess_s=4.71`
  - measured run:
    - `generation_total_s=75.99`
    - `generation_sec_per_audio_sec=7.60`
    - `clip_generation_s=65.79`
    - `postprocess_s=4.44`
    - `postprocess_decode_s=2.07`
    - `postprocess_encode_s=1.60`
    - `upscale_s=0.92`
- Comparison against item 35:
  - formal winner:
    - `generation_total_s=75.69`
    - `clip_generation_s=65.47`
  - with rollout-grid cache:
    - `generation_total_s=75.99`
    - `clip_generation_s=65.79`
- Verdict:
  - slight regression
  - about `0.30s` slower
  - do not keep this optimization; move on to the next roadmap item

38. Try `compile_dynamic=false` on the formal `LightTAE` winner path
- Config:
  - same as item 35
  - `compile_dynamic=false`
- Result:
  - no full artifact was kept
  - the run was stopped during warmup as a clear regression
  - live timing signals were already much worse than the current winner:
    - the second and third denoise blocks were materially slower than on the default compile path
- Verdict:
  - `compile_dynamic=false` is not promising on the current winner path
  - keep the default compile dynamic behavior and move on to the next roadmap item

39. Try `LightTAE` with `encode_parallel=true` and `decode_parallel=false`
- Config:
  - same as item 35
  - `LIVEAVATAR_TAE_ENCODE_PARALLEL=true`
  - `LIVEAVATAR_TAE_DECODE_PARALLEL=false`
- Result:
  - warmup:
    - `generation_total_s=157.61`
    - `generation_sec_per_audio_sec=15.76`
    - `clip_generation_s=135.78`
    - `postprocess_s=7.31`
    - `postprocess_encode_s=3.11`
  - measured run:
    - `generation_total_s=75.92`
    - `generation_sec_per_audio_sec=7.59`
    - `clip_generation_s=65.44`
    - `postprocess_s=4.73`
    - `postprocess_decode_s=2.09`
    - `postprocess_encode_s=0.53`
    - `upscale_s=0.93`
- Comparison against item 35:
  - formal winner:
    - `generation_total_s=75.69`
    - `postprocess_encode_s=1.61`
  - encode-parallel-only:
    - `generation_total_s=75.92`
    - `postprocess_encode_s=0.53`
- Verdict:
  - `postprocess_encode` became much faster, but the end-to-end result is still a slight regression
  - about `0.23s` slower overall
  - do not keep this variant; if `LightTAE parallel` is revisited later, it needs a more structural decode-side change rather than this partial flag split

40. Re-check the external framework branch (`LightX2V / TeaCache / SageAttention`) for the current S2V product path
- Sources checked:
  - `LightX2V` upstream repo
  - current Wan2.2 upstream repo
  - recent external notes on Wan2.2 acceleration
- Result:
  - there is still real upside in principle for Wan2.x acceleration frameworks
  - but the practical support remains centered on `T2V/I2V` and adjacent model families, not a drop-in path for the current `LiveAvatar` audio-conditioned S2V stack
  - `LightX2V` does provide useful autoencoder assets and broader acceleration methods, but not a ready-made `S2V + LiveAvatar LoRA + current audio path` runner we can benchmark directly inside this repo
- Verdict:
  - keep this branch as strategically important, but not as an immediate local benchmark candidate
  - for near-term work, only revisit it if we are ready to do a deeper port/integration rather than a quick benchmark


41. Try practical SageAttention install on the current Blackwell stack
- Config:
  - current formal `LightTAE` winner path kept unchanged
  - only testing whether a real `sageattention` package can even be installed/imported on this box before attempting any integration
- Result:
- Update:
  - the first benchmark attempt failed before any model work due to an invalid `LightTAE` checkpoint path in the command
  - this was a harness mistake, not a SageAttention result; rerun with the real local checkpoint path
- Result:
  - `sageattention==1.0.6` installs successfully on this box
  - import works, CUDA micro-test works, and the package exposes both `sageattn` and `sageattn_varlen`
  - however, the first real full-pipeline benchmark on the formal `LightTAE` winner path never reached render
  - observed state during the run:
    - elapsed time already > `1 min`
    - GPU usage stuck around `550 MB`
    - no benchmark artifacts written
    - `/proc/*/stack` and `strace` show the process mostly parked in futex waits rather than active GPU work
- Verdict:
  - this is not a practical near-term speed path for the current `LiveAvatar S2V` stack
  - keep SageAttention out of the active winner path and move on to the next roadmap item


42. Try `LightTAE` with `encode_parallel=false` and `decode_parallel=true`
- Config:
  - same as item 35 formal winner
  - `LIVEAVATAR_TAE_PARALLEL=false`
  - `LIVEAVATAR_TAE_ENCODE_PARALLEL=false`
  - `LIVEAVATAR_TAE_DECODE_PARALLEL=true`
- Result:
  - the run reached `complete full-sequence generation`
  - then failed in `postprocess decode` with:
    - `torch.OutOfMemoryError: Tried to allocate 19.20 GiB`
    - free VRAM at failure was only about `17.11 GiB`
- Verdict:
  - decode-side parallelism is not viable on the current winner path
  - keep `decode_parallel=false` and move back to the remaining compile / hot-path cleanup work


43. Scan real graph breaks on the current formal `LightTAE` winner path
- Config:
  - same as item 35 formal winner
  - `TORCH_LOGS=graph_breaks`
- Result:
  - the scan surfaced concrete current graph-break sites instead of guesses
  - main breaks observed:
    - `causal_model_s2v.py:1319` `start_idx = 30-relative_dist`
    - `causal_model_s2v.py:754` `rollout_grid_cache.get(cache_key)`
    - `s2v_utils.py:36` `if seq_len > 0` inside `rope_precompute`
    - `causal_model_s2v.py:378` slice on `active_cond_cache_size`
    - `causal_model_s2v.py:1368` list-iterator use in the text context stacking path
- Verdict:
  - the graph-break picture is now concrete enough to stop guessing
  - the worst easy target was the `rollout-grid cache`, because it was already a measured regression and also added a graph break

44. Re-check fixed small `relative_dist` after the graph-break scan
- Config:
  - same as item 35 formal winner
  - `relative_dist=8`
- Result:
  - the run reached `complete full-sequence generation`
  - then failed in `postprocess decode` with:
    - `torch.OutOfMemoryError: Tried to allocate 19.20 GiB`
    - free VRAM at failure was about `17.93 GiB`
- Verdict:
  - small fixed `relative_dist` is not a safe near-term winner candidate on the current stack
  - do not pursue this parameter branch further before the hot-path cleanup work is exhausted


45. Try `cond_end_int + pad_sequence context` cleanup on the formal winner path
- Config:
  - same as item 35 formal winner
  - replace tensor-backed `cond_end` reads with a Python-int mirror
  - replace context list-comprehension padding with `pad_sequence`
- Result:
  - the first attempt introduced an initialization bug and was fixed immediately
  - the corrected attempt still did not yield a valid speed measurement
  - instead it exposed a deeper `torch.compile` backend failure on the self-attention path
  - failure mode:
    - `BackendCompilerFailed` / `GuardOnDataDependentSymNode`
    - traced back to data-dependent slicing around `seg_idx` / `roped_key[:, seg_idx[0]:seg_idx[1]]`
- Verdict:
  - this cleanup direction is not a safe near-term win in its current form
  - the attempted changes were rolled back to keep the winner path healthy
  - move on instead of spending more time forcing this branch


46. Try reducing `num_clip` safety margin from `+2` to `+1`
- Config:
  - same as item 35 formal winner
  - `LIVEAVATAR_NUM_CLIP_SAFETY_MARGIN=1`
  - for the `10s` benchmark this reduced `num_clip` from `6` to `5`
- Result:
  - the run advanced further with the smaller clip count
  - but still reached `complete full-sequence generation`
  - then failed in `postprocess decode` with:
    - `torch.OutOfMemoryError: Tried to allocate 19.20 GiB`
    - free VRAM at failure was about `16.97 GiB`
- Verdict:
  - reducing the safety margin is not a safe near-term winner on the current `LightTAE` stack
  - do not continue pushing this branch lower before a structural decode-side memory win exists


47. Try isolated `context` padding cleanup without touching `cond_end`
- Config:
  - same as item 35 formal winner
  - replace the `torch.stack([... for u in context])` padding path with a dedicated helper using `pad_sequence`
  - do not change any `cond_end`, `seg_idx`, or cache slicing logic
- Result:
  - benchmark artifact:
    - `/root/LiveAvatar/benchmarks/maxspeed/infer64_compile_lighttae_chunk512_capture_scalar_contextpad/benchmark.json`
  - measured run:
    - `generation_total_s=76.03`
    - `generation_sec_per_audio_sec=7.60`
    - `clip_generation_s=65.75`
    - `postprocess_s=4.46`
- Comparison against the current formal winner:
  - formal winner:
    - `generation_total_s=75.69`
    - `generation_sec_per_audio_sec=7.57`
  - isolated context-padding cleanup:
    - `generation_total_s=76.03`
    - `generation_sec_per_audio_sec=7.60`
- Verdict:
  - valid and stable, but still a regression
  - about `0.34s` slower overall
  - do not keep this helper as an active speed optimization


48. Try `cond-cache narrow(...)` instead of dynamic `[:, :active_cond_cache_size]` slicing
- Config:
  - same as item 35 formal winner
  - replace `cond_k[:, :active_cond_cache_size]` / `cond_v[:, :active_cond_cache_size]` with `narrow(...)`
  - replace prefill writes with `narrow(...).copy_(...)`
- Result:
  - warmup artifact:
    - `/root/LiveAvatar/benchmarks/maxspeed/infer64_compile_lighttae_chunk512_capture_scalar_condnarrow/warmup_1_metrics.json`
  - warmup metrics:
    - `generation_total_s=203.41`
    - `generation_sec_per_audio_sec=20.34`
  - the primary measured run never produced `benchmark.json`
  - after a long wall-clock wait it was still running and had already exceeded the practical wall time of the formal winner path
- Comparison against the current formal winner:
  - formal warmup:
    - `generation_total_s=146.49`
  - `condnarrow` warmup:
    - `generation_total_s=203.41`
- Verdict:
  - this branch is not a practical near-term winner
  - even before the main measured run, warmup is already drastically worse
  - do not keep this slicing variant in the active winner path


49. Try early Python-int conversion inside `rope_precompute`
- Config:
  - same as item 35 formal winner
  - convert the `grid_sizes` triplets to Python ints at the top of each loop iteration
  - remove the tensor-backed `int(...) / .item()` pattern around `seq_len > 0`
- Result:
  - the run started normally and used the GPU heavily
  - but it did not produce even the warmup artifacts within a practical wall-clock budget
  - after waiting, the output directory still had no files at all
- Verdict:
  - this is not a practical near-term winner for the current stack
  - treat it as another compile-cleanup dead-end and do not keep it in the active winner path


50. Try caching initial non-streaming `rope_precompute` across warm runs in the resident process
- Config:
  - same as item 35 formal winner
  - reuse `_get_cached_rope_freqs(...)` for the initial prefill/full-sequence `rope_precompute` calls too
- Result:
  - the run started and used the GPU heavily
  - but again failed to produce even the warmup artifacts within a practical wall-clock budget
  - the output directory remained empty while elapsed time was already well into the same range as the whole formal winner path
- Verdict:
  - this reuse path is not a practical near-term winner on the current stack
  - do not keep this cache variant in the active path


51. Evaluate the external `LightX2V seko_talk` acceleration branch for talking-avatar S2V
- Sources inspected:
  - `/root/bench_external/LightX2V/test_cases/run_seko_talk_01_base.sh`
  - `/root/bench_external/LightX2V/scripts/seko_talk/run_seko_talk_15_base_compile.sh`
  - `/root/bench_external/LightX2V/scripts/seko_talk/run_seko_talk_16_fp8_dist_compile.sh`
  - `/root/bench_external/LightX2V/configs/seko_talk/`
  - `/root/bench_external/LightX2V/lightx2v/models/runners/wan/wan_audio_runner.py`
  - `/root/bench_external/LightX2V/tools/convert/seko_talk_converter.py`
- Result:
  - `LightX2V` does have a real talking-avatar `S2V` path:
    - `model_cls=seko_talk`
    - `task=s2v`
    - `image_path + audio_path`
  - it also has dedicated configs for:
    - `compile`
    - `fp8`
    - `dist`
    - `5090`
    - `nbhd_attn`
  - however, this is not a drop-in path for the current `LiveAvatar` stack:
    - the runner is built around its own `WanAudioModel` + separate audio adapter
    - the code path is explicitly tied to a `SekoTalk-Distill` model family
    - the referenced model repos appear gated/private from this machine (`401 Unauthorized` via HF API checks)
    - the LoRA/adapter conversion tooling expects separate `r2v_model`, `distill_model`, and `audio_adapter` assets
- Verdict:
  - this is a strong strategic Tier C branch for talking-avatar acceleration
  - but it is not an immediately benchmarkable or drop-in replacement for the current `LiveAvatar Wan2.2-S2V` stack without separate model access and a deeper migration effort
  - keep it as the strongest external acceleration lead, but move back to profiling the current winner path for near-term wins


52. Try full `torch.profiler` on the current formal winner path
- Config:
  - same as the current formal winner
  - enable `--torch-profiler` for the primary run only
- Result:
  - the benchmark rendered all expected media artifacts:
    - `/root/LiveAvatar/benchmarks/maxspeed/infer64_compile_lighttae_chunk512_capture_scalar_profiled/rendered_raw.mp4`
    - `/root/LiveAvatar/benchmarks/maxspeed/infer64_compile_lighttae_chunk512_capture_scalar_profiled/rendered.mp4`
  - but it never produced:
    - `benchmark.json`
    - `primary_metrics.json`
    - profiler summary tables
  - `status.json` remained stuck at `primary_run_started`
  - the process had to be killed after render completion because profiler teardown/output never finished within a practical wall-clock budget
- Verdict:
  - full `torch.profiler` is too heavy / impractical for this optimization loop on the current stack
  - do not use it as the main diagnostic tool here
  - switch to lightweight event-level timing instead


53. Add lightweight event-level timing on the current formal winner path
- Config:
  - same as the current formal winner
  - record clip-level and postprocess-clip-level stage timestamps in the benchmark harness
  - artifact:
    - `/root/LiveAvatar/benchmarks/maxspeed/infer64_compile_lighttae_chunk512_capture_scalar_eventtrace_30s/benchmark.json`
- Result:
  - end-to-end time:
    - `generation_total_s=216.74`
    - `generation_sec_per_audio_sec=7.22`
  - this is slightly worse than the current formal `30s` winner (`215.17s`), so it is not itself a speed win
  - but it exposed two useful facts:
    - the first clip is materially slower than the steady-state clips:
      - first clip `21.32s`
      - clips `2..12` average `15.55s`
    - `LightTAE` postprocess is already very flat and cheap:
      - postprocess clip average `1.085s`
    - there is still a notable raw-save tail after `postprocess_complete` and before final upscale starts
- Verdict:
  - keep lightweight event timing in the benchmark harness as a diagnostic tool
  - use it to target the remaining non-denoise tail instead of returning to full `torch.profiler`


54. Defer intermediate audio merge into the final GPU normalize pass
- Config:
  - skip `merge_video_audio(...)` during raw render
  - pass `audio_path` directly into final `normalize_video(...)`
  - benchmark artifact:
    - `/root/LiveAvatar/benchmarks/maxspeed/infer64_compile_lighttae_chunk512_capture_scalar_deferaudiomerge_10s/benchmark.json`
- Result:
  - `generation_total_s=75.71`
  - previous `10s` formal winner was `75.69`
  - effectively neutral / tiny regression
- Verdict:
  - this change by itself is not a meaningful speed win
  - keep only if it helps a larger raw-save improvement, otherwise do not count it as an optimization on its own


55. Replace slow intermediate `imageio/libx264` raw save with a direct NVENC raw-video path
- Config:
  - keep the deferred final audio merge from item 54
  - replace intermediate raw save in `ResidentLiveAvatarRunner.render(...)` with direct `ffmpeg` rawvideo -> `h264_nvenc`
  - benchmark artifact:
    - `/root/LiveAvatar/benchmarks/maxspeed/infer64_compile_lighttae_chunk512_capture_scalar_fastrawsave_fix1_10s/benchmark.json`
- Result:
  - `generation_total_s=74.55`
  - `generation_sec_per_audio_sec=7.46`
  - previous `10s` formal winner was `75.69`
  - improvement: about `1.14s`, roughly `1.5%`
  - render tail after `postprocess_complete` also became visibly shorter on the event trace
- Verdict:
  - this is a real short-run win
  - the mandatory next step is the long `30s` confirmation


56. Confirm the fast intermediate NVENC raw-save path on `30s`
- Config:
  - same as item 55
  - long benchmark artifact:
    - `/root/LiveAvatar/benchmarks/maxspeed/infer64_compile_lighttae_chunk512_capture_scalar_fastrawsave_fix1_30s/benchmark.json`
- Result:
  - `generation_total_s=213.91`
  - `generation_sec_per_audio_sec=7.13`
  - previous `30s` formal winner was `215.17`
  - improvement: about `1.25s`, roughly `0.6%`
  - warmup also improved:
    - was `284.62s`
    - now `282.63s`
- Verdict:
  - fast intermediate NVENC raw save is now the confirmed overall winner
  - keep this path
  - next useful branch is to remove the intermediate raw file entirely and test direct final encode


57. Try direct final NVENC encode with audio and scaling in one pass
- Config:
  - bypass the intermediate raw file entirely in benchmark mode
  - use `runner.render(..., return_video_tensor=True)` and stream frames directly into final `ffmpeg` with `h264_nvenc + audio + scale_cuda`
  - short benchmark artifact:
    - `/root/LiveAvatar/benchmarks/maxspeed/infer64_compile_lighttae_chunk512_capture_scalar_directfinal_fix1_10s/benchmark.json`
- Result:
  - `generation_total_s=73.94`
  - `generation_sec_per_audio_sec=7.39`
  - previous winner on `10s` was the fast raw-save path at `74.55s`
  - improvement vs fast raw-save: about `0.61s`, roughly `0.8%`
  - improvement vs the older formal winner `75.69s`: about `1.75s`, roughly `2.3%`
- Verdict:
  - this is the new short-run winner
  - the next mandatory step is long `30s` confirmation before keeping it as the overall winner


58. Re-check direct final NVENC encode on `30s`
- Config:
  - same as item 57
  - long benchmark output dir:
    - `/root/LiveAvatar/benchmarks/maxspeed/infer64_compile_lighttae_chunk512_capture_scalar_directfinal_fix1_30s`
- Result:
  - the run stayed alive well past the practical point where the current long winner already produces warmup artifacts
  - even after an extended wait, the output directory still had no files at all
  - the process had to be abandoned as impractical for the current optimization loop
- Verdict:
  - direct final encode is a valid short-run win
  - but on long runs it is currently not practical / not yet production-viable
  - keep the fast intermediate NVENC raw-save path as the overall confirmed winner


59. Re-test `compile=false` on the new fast raw-save overall winner
- Config:
  - keep the current overall winner path:
    - `infer_frames=64`
    - `LightTAE`
    - `chunk=512`
    - fast intermediate NVENC raw save
  - disable compile explicitly:
    - `--compile false`
  - output dir:
    - `/root/LiveAvatar/benchmarks/maxspeed/infer64_nocompile_lighttae_fastrawsave_10s`
- Result:
  - after about `237s` of wall-clock, the run still had no `events.json` and no `benchmark.json` at all
  - GPU was still heavily occupied (`~66 GiB`), but the run had not even reached the point where the compiled winner-path already produces artifacts
  - for comparison, the compiled fast raw-save winner on `10s` finishes completely in `74.55s`
- Verdict:
  - `compile=false` remains an obvious regression even on the newer overall winner path
  - do not keep this branch in the active search space


60. Try a minimal `simple TeaCache`-style residual skip on the current S2V path
- Config:
  - add an opt-in experimental cache-reuse path inside `CausalWanModel_S2V._forward_inference(...)`
  - cache the transformer-body residual per streaming block
  - decide skip by relative L1 distance on the timestep embedding path
  - expose knobs through benchmark env/CLI:
    - `LIVEAVATAR_ENABLE_SIMPLE_TEACACHE`
    - `LIVEAVATAR_SIMPLE_TEACACHE_THRESH`
    - `LIVEAVATAR_SIMPLE_TEACACHE_FORCE_CALC_STEPS`
- Result A (`compile=true`, threshold `0.15`):
  - output dir:
    - `/root/LiveAvatar/benchmarks/maxspeed/infer64_compile_lighttae_chunk512_capture_scalar_simpleteacache015`
  - the run failed during warmup with a `torch.compile` / inductor C++ compile failure
  - this is exactly the kind of case where `compile=true` can break an otherwise testable idea
- Result B (`compile=false`, threshold `0.15`):
  - benchmark artifact:
    - `/root/LiveAvatar/benchmarks/maxspeed/infer64_nocompile_lighttae_chunk512_simpleteacache015/benchmark.json`
  - `generation_total_s=169.29`
  - `generation_sec_per_audio_sec=16.93`
  - this is much faster than the older no-compile non-TeaCache path (`316.21s`)
  - but it is still far slower than the compiled production winner (`74.55s`)
- Verdict:
  - the idea itself is not dead: it materially improves the no-compile path
  - but the first threshold is nowhere near enough to beat the compiled winner
  - the only reasonable next move is a more aggressive threshold sweep; if that still cannot approach the winner, close the branch


61. Sweep a more aggressive `simple TeaCache` threshold (`0.26`)
- Config:
  - same as item 60 result B
  - raise `simple_teacache_thresh` from `0.15` to `0.26`
  - benchmark artifact:
    - `/root/LiveAvatar/benchmarks/maxspeed/infer64_nocompile_lighttae_chunk512_simpleteacache026/benchmark.json`
- Result:
  - `generation_total_s=168.99`
  - `generation_sec_per_audio_sec=16.90`
  - this is only trivially different from the `0.15` run (`169.29s`)
  - the branch is still far slower than the compiled production winner (`74.55s`)
- Verdict:
  - the threshold sweep did not unlock a meaningful new regime
  - close the `simple TeaCache` branch for the current product path
  - move on to the next remaining roadmap candidate instead of tuning this further


62. Re-test the cache-skip path with LightX2V-style polynomial Tea coefficients for `480p`
- Config:
  - same as item 61, but add the official `wan_i2v_tea_480p` polynomial coefficients from LightX2V
  - output dir:
    - `/root/LiveAvatar/benchmarks/maxspeed/infer64_nocompile_lighttae_chunk512_simpleteacache_poly480`
- Result:
  - early warmup timing stayed effectively the same as the plain `0.26` threshold run
  - first denoise block remained around `3.4s`, then settled into the same `~2.0s` per step pattern
  - there was no sign of a new faster regime before the run was stopped
- Verdict:
  - adding the polynomial rescale did not materially change behavior on the current product path
  - close the entire `simple TeaCache` family for now and move on


63. Re-check long `direct final NVENC encode` on the current best stack
- Config:
  - `infer_frames=64`
  - `LightTAE`
  - `chunk=512`
  - `compile=true`
  - `capture_scalar_outputs=true`
  - bypass the intermediate raw file and stream frames directly into final NVENC encode
  - benchmark artifact:
    - `/root/LiveAvatar/benchmarks/maxspeed/infer64_compile_lighttae_chunk512_capture_scalar_directfinal_fix2_30s/benchmark.json`
- Result:
  - `generation_total_s=213.45`
  - `generation_sec_per_audio_sec=7.115`
  - `clip_generation_s=193.60`
  - `postprocess_s=13.53`
  - `upscale_s=4.85`
- Comparison:
  - previous confirmed long winner (fast intermediate NVENC raw save):
    - `/root/LiveAvatar/benchmarks/maxspeed/infer64_compile_lighttae_chunk512_capture_scalar_fastrawsave_fix1_30s/benchmark.json`
    - `generation_total_s=213.91`
    - `generation_sec_per_audio_sec=7.130`
  - improvement: about `0.46s`, roughly `0.2%`
- Verdict:
  - direct-final encode is now a confirmed overall long-run winner too
  - keep it as the new production-path leader
  - remaining work should focus on other major roadmap items rather than more encode-tail micro-tuning


59. `compile=false` retest on the fast-raw-save winner path.
Artifact:
  - `benchmarks/maxspeed/infer64_nocompile_lighttae_fastrawsave_10s`
Result:
  - after about `237s` wall-clock there was still no `events.json` / `benchmark.json`
  - obvious regression vs the compiled winner, so the run was stopped early
Verdict:
  - keep `compile=true` on the current production winner path

60. Minimal `simple TeaCache`-style residual skip.
Artifacts:
  - `benchmarks/maxspeed/infer64_nocompile_lighttae_chunk512_simpleteacache015/benchmark.json`
Result:
  - `generation_total_s=169.29`
  - `generation_sec_per_audio_sec=16.93`
Notes:
  - `compile=true` failed on this branch with an inductor C++ compile error
  - `compile=false` was dramatically better than the old no-compile baseline, but still far from the production winner
Verdict:
  - idea is not dead, but this specific TeaCache-style branch is not competitive enough

61. More aggressive `simple TeaCache` threshold `0.26`.
Artifact:
  - `benchmarks/maxspeed/infer64_nocompile_lighttae_chunk512_simpleteacache026/benchmark.json`
Result:
  - `generation_total_s=168.99`
  - `generation_sec_per_audio_sec=16.90`
Verdict:
  - effectively identical to `0.15`; close the simple TeaCache threshold tuning branch

62. Polynomial `simple TeaCache` variant using LightX2V-style coefficients.
Artifact:
  - `benchmarks/maxspeed/infer64_nocompile_lighttae_chunk512_simpleteacache_poly480`
Result:
  - warmup denoising looked essentially identical to the simple threshold branch
  - no new speed regime appeared early enough to justify a full long run
Verdict:
  - close the simple TeaCache family and move on

63. Direct final encode long-run recheck.
Artifact:
  - `benchmarks/maxspeed/infer64_compile_lighttae_chunk512_capture_scalar_directfinal_fix2_30s/benchmark.json`
Result:
  - `generation_total_s=213.45`
  - `generation_sec_per_audio_sec=7.12`
Comparison vs previous long winner:
  - previous `fast raw-save` long winner: `213.91s`
  - new `direct-final` long result: `213.45s`
Verdict:
  - direct final encode is now the confirmed overall long-run winner too

64. Minimal `simple AdaCache`-style residual reuse.
Artifact:
  - `benchmarks/maxspeed/infer64_nocompile_lighttae_chunk512_simpleadacache/benchmark.json`
Result:
  - `generation_total_s=92.95`
  - `generation_sec_per_audio_sec=9.29`
  - `clip_generation_s=84.01`
  - `postprocess_s=4.43`
Comparison:
  - much better than the simple TeaCache no-compile branch (`~169s`)
  - still slower than the current compiled production winner (`73.94s` on `10s`)
Verdict:
  - first no-compile cache-skip branch that is competitive enough to deserve one more serious pass
  - next step: test whether `compile=true` helps or breaks it; then, if needed, try one codebook tuning pass


65. `simple AdaCache` with `compile=true`.
Artifact:
  - `benchmarks/maxspeed/infer64_compile_lighttae_chunk512_simpleadacache`
Result:
  - no warmup artifacts were written in reasonable wall-clock
  - process stayed CPU-bound with only about `550 MiB` on GPU and no normal render progression
Verdict:
  - treat `compile=true + simple AdaCache` as impractical on the current stack
  - keep exploring this branch, if at all, in `compile=false` mode only


66. More aggressive `simple AdaCache` codebook.
Artifact:
  - `benchmarks/maxspeed/infer64_nocompile_lighttae_chunk512_simpleadacache_aggr/warmup_1_metrics.json`
Config:
  - codebook `0.03:16,0.05:14,0.07:12,0.09:10,0.11:8,1.0:5`
Result:
  - `warmup generation_total_s=118.48`
  - baseline `simple AdaCache` warmup was `111.83s`
Comparison:
  - about `6.65s` slower than the baseline AdaCache branch already during warmup
Verdict:
  - aggressive AdaCache tuning is a regression
  - close the simple AdaCache family as non-competitive with the compiled production winner


67. Cross-check external `TAEHV` branch.
Source:
  - `madebyollin/taehv`
Result:
  - local `liveavatar/models/wan/wan_2_2/modules/tae2_1.py` is already effectively a TAEHV-style implementation for Wan `taew2_1`
  - there is no obvious drop-in external TAEHV win left beyond the current `LightTAE`/`TAE` path already benchmarked
Verdict:
  - close this as already-covered by the current alternate TAE work rather than a fresh acceleration branch

68. External `cache-dit` feasibility check.
Result:
  - `cache-dit` installs, but pulls `transformers 5.5.0` / `huggingface-hub 1.9.0`
  - on this stack, importing `cache_dit` fails through `diffusers/peft` with `cannot import name 'HybridCache' from transformers`
  - docs show a potentially relevant transformer/block adapter path for Wan-style transformers, but this is not a near-term drop-in on the current environment
Verdict:
  - treat `cache-dit` as a separate dependency/integration branch, not a quick local optimization
  - restore the working env and move on instead of destabilizing the main benchmark stack


70. Post-restore sanity check of the current production winner.
Artifact:
  - `benchmarks/maxspeed/infer64_compile_lighttae_chunk512_capture_scalar_directfinal_postrestore_10s/benchmark.json`
Result:
  - `generation_total_s=74.50`
  - `generation_sec_per_audio_sec=7.45`
  - `clip_generation_s=66.79`
  - `postprocess_s=4.53`
  - `upscale_s=1.95`
Comparison:
  - original short direct-final winner: `73.94s`
  - post-restore sanity check: `74.50s`
  - delta is only about `+0.57s` (`~0.8%`)
Verdict:
  - after reverting failed branches and restoring the env, the production-path winner remains intact
  - use this as the clean baseline for any further work


71. Final `compile=false` retest on the clean direct-final production winner path.
Artifact:
  - `benchmarks/maxspeed/infer64_nocompile_lighttae_chunk512_capture_scalar_directfinal_postrestore_10s`
Result:
  - even after substantial wall-clock there were still no warmup metrics or `benchmark.json`
  - live progression was clearly slower than the compiled winner path on the same config
Verdict:
  - close `compile=false` as non-competitive on the current best production path too
  - the remaining promising work is no longer in disabling compile, but in separate integration/model branches


72. External `TurboDiffusion` feasibility check.
Source:
  - official `thu-ml/TurboDiffusion` README / model list
Result:
  - official available checkpoints currently target `TurboWan2.2-I2V-A14B-720P` and `Wan2.1 T2V` variants
  - there is no ready-made `audio-conditioned S2V` drop-in path matching the current `LiveAvatar + LoRA` stack
Verdict:
  - high upside remains real in principle
  - but this is a separate model/inference branch, not a near-term optimization for the current product path


73. External `xDiT / xFuser` feasibility check.
Source:
  - official xDiT / xFuser docs and package notes
Result:
  - the strongest published gains are primarily from multi-GPU parallelism and Diffusers-based Wan `I2V/T2V` pipelines
  - single-card support exists, but the expected gain is much smaller and still assumes a different serving/runtime stack than the current `LiveAvatar audio-conditioned S2V`
Verdict:
  - not a strong near-term branch for the current one-GPU product path
  - treat it as a separate serving/runtime migration rather than a direct optimization of the current stack
