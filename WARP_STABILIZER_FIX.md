# FilmCraft — Warp Stabilizer Responsiveness Fix

**Branch:** gpu-effects-optimization  
**Implementation commit:** 6b203c4b

## What was wrong?

FilmCraft experienced severe GUI freezes during Warp Stabilizer analysis, particularly when seeking through long, high-resolution recordings.

Testing included a 30-second clip and a 31-minute, 44-second dashcam recording (3.66 GiB, 3200×1800, 30 FPS).

Three problems were identified:

1. **Blocking audio shutdown:** CPAL/ALSA stream destruction could block the GUI thread during seeking.
2. **Synchronous stabilization analysis:** The rendering path could initiate expensive frame decoding and motion tracking when no cached analysis existed.
3. **Insufficient progress feedback:** Users could not clearly distinguish ongoing analysis from an application freeze.

## What changed?

### 1. Asynchronous audio shutdown

Modified `apps/filmcraft/src/audio.rs`.

Audio streams are now destroyed on background threads instead of the GUI thread, preventing slow CPAL/ALSA teardown from blocking interface interaction.

### 2. Background audio recovery

Automatic audio stream recovery was moved into a background worker.

Added cancellation handling, non-blocking completion checks, failure reporting, and playback-frame accounting while recovery is pending.

### 3. Background Warp Stabilizer analysis

Modified:

- `crates/render/src/vfx/mod.rs`
- `crates/render/src/vfx/temporal.rs`

Stabilization analysis now supports background execution, progress reporting, and cancellation.

The normal rendering path only retrieves completed analysis from the cache. It no longer initiates expensive analysis while rendering a frame.

### 4. Dedicated stabilization job manager

Added `crates/ui-egui/src/stabilizer_jobs.rs`.

This module coordinates background analysis, progress, cancellation, retry, and completion.

### 5. Preview protection

Modified `crates/ui-egui/src/panels/monitor.rs`.

During relevant stabilization analysis, FilmCraft suspends preview requests and frame prefetching while preserving the last available frame.

The monitor displays an analysis banner and refreshes after completed analysis becomes available.

### 6. Improved effect controls

Modified `crates/ui-egui/src/panels/effect_controls.rs`.

Added:

- Amber **Preview disabled** indicator during analysis.
- Green **Stabilised** indicator after successful analysis.
- Analysis progress information.
- Cancel and Retry controls.
- Larger, 12-point status labels.

### 7. Cache and cancellation safety

Added explicit cache lookup helpers and a shared cache-key calculation.

A cache-generation counter prevents obsolete background analysis from publishing results after cache invalidation.

Cancelled analysis does not publish incomplete stabilization paths.

### 8. Test and compiler cleanup

Updated the Warp Stabilizer regression test to explicitly prepare analysis before rendering.

Removed an unused Rayon import from the export module.

## Testing

The application compiled successfully in release mode.

Interactive testing covered both short and long video clips, including the 31-minute high-resolution dashcam recording.

The following were tested successfully:

- Starting stabilization analysis.
- Timeline seeking during analysis.
- Cancelling analysis.
- Changing effect settings.
- Normal playback and audio.
- Preview recovery after analysis.
- Successful stabilization status display.

The Warp Stabilizer regression test passed:

    test vfx::tests::warp_stabilizer_removes_camera_jitter ... ok

    test result: ok. 1 passed; 0 failed

Automatic audio recovery under a deliberately simulated device failure has not yet been separately tested.

## Result

FilmCraft no longer exhibits the original GUI freezes in the tested Warp Stabilizer workflows.

Expensive stabilization analysis and potentially blocking audio operations have been separated from GUI-critical execution paths.

The changes preserve the existing GPU rendering architecture and do not introduce a CPU-rendering fallback.

**Implementation commit:** 6b203c4b
