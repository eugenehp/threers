// The threers volumetric fluid, for iOS and iPadOS.
//
// Hand-written rather than generated: there are nine functions and one struct,
// and a header you can read is worth more than a build step. It must match
// `src/lib.rs` — `the_header_matches_the_library` checks that it does.
#ifndef THREERS_FLUID_H
#define THREERS_FLUID_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle. One per view.
typedef struct ThreersFluid ThreersFluid;

/// What the fluid came out as, for a debug overlay.
typedef struct {
  /// Velocity and pressure grid, in voxels a side.
  uint32_t solver;
  /// Dye grid, in voxels a side.
  uint32_t dye;
  /// What the fields occupy.
  uint32_t megabytes;
  /// Frames a second as the display delivers them. Present is `Fifo`, so
  /// this is the refresh rate until the fluid cannot keep up — not the
  /// number to tune against.
  float fps;
  /// Milliseconds the GPU spends in the solver, from timestamp queries; 0
  /// where the device has none. This is the one to tune against — a
  /// wall-clock timer around the frame measures the refresh rate, because
  /// acquiring a drawable blocks on vsync. Excludes the ray march.
  float solver_ms;
} ThreersFluidInfo;

/// What `threers_fluid_set` is setting, and what `a`, `b`, `c` mean.
///
/// One entry point rather than five: five were five calling conventions to
/// get wrong across a language boundary, and this is one. An unknown key
/// does nothing, which is the right answer for an ABI that will outlive one
/// version of the app linking it.
typedef enum {
  /// The view resized. a = width, b = height, in pixels.
  THREERS_FLUID_RESIZE = 0,
  /// A finger moved. a = dx, b = dy in points, c non-zero while touching.
  THREERS_FLUID_TOUCH = 1,
  /// A pinch. a = scale since the last call (1.0 is no change).
  THREERS_FLUID_PINCH = 2,
  /// Add a puffs of dye at random places, up to 32.
  THREERS_FLUID_SPLAT = 3,
  /// Speed. a = a multiple of real time; 0 holds it still.
  THREERS_FLUID_SPEED = 4,
  /// Average the ray march over a frames, 1 to 32. Resets when the camera
  /// moves, so it costs nothing while a finger is down.
  THREERS_FLUID_ACCUMULATE = 5,
  /// Draw the frame rate over the picture; a non-zero for on.
  THREERS_FLUID_SHOW_FPS = 6,
} ThreersFluidKey;

/// Create a fluid drawing into `layer`, a `CAMetalLayer`. Null on failure.
///
/// `width` and `height` are in pixels: multiply the view's bounds by the
/// screen's `nativeScale`. `budget_mb` of 0 takes a conservative default.
/// The layer must outlive the handle.
ThreersFluid *threers_fluid_create(void *layer, uint32_t width, uint32_t height,
                                   uint32_t budget_mb);

/// Set one thing; see `ThreersFluidKey`.
void threers_fluid_set(ThreersFluid *handle, uint32_t key, float a, float b, float c);

/// Step by `dt` seconds and draw. Returns false if the drawable could not be
/// acquired, which means the view is off screen — stop the display link.
bool threers_fluid_frame(ThreersFluid *handle, float dt);

/// Fill `out` with what was allocated and how fast it is going.
void threers_fluid_info(ThreersFluid *handle, ThreersFluidInfo *out);

/// Free it. Passing null is fine; the handle is invalid afterwards.
void threers_fluid_destroy(ThreersFluid *handle);

#ifdef __cplusplus
}
#endif

#endif // THREERS_FLUID_H
