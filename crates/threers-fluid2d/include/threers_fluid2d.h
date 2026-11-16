// The threers 2D fluid on Metal, for an app that owns its MTLDevice.
//
// Hand-written, and held to src/lib.rs by `the_header_declares_every_export`.
// The plain calls commit on the fluid's own queue; the `_on` calls encode into
// a command buffer the app passes (uncommitted, on the same device) and commit
// nothing. Every call from one thread at a time.
#ifndef THREERS_FLUID2D_H
#define THREERS_FLUID2D_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ThreersFluid2d ThreersFluid2d;

// Fill with threers_fluid2d_config_default, then change what you need.
typedef struct {
    uint32_t sim_resolution;       // solver grid, short side, texels
    uint32_t dye_resolution;       // dye grid, short side, texels
    uint32_t pressure_iterations;  // Jacobi sweeps, 1-64
    float    density_dissipation;  // how fast the dye fades, per second
    float    velocity_dissipation;
    float    pressure;             // fraction of last frame's pressure kept
    float    curl;                 // vorticity confinement
    float    splat_radius;         // for threers_fluid2d_splat_random, percent
    uint32_t shading;              // non-zero: threers' shading in render_on
    uint32_t maccormack;           // non-zero: second-order advection, sharper dye
    uint32_t multigrid;            // non-zero: a V-cycle instead of Jacobi
    uint32_t narrow_formats;       // non-zero: rg16f velocity, r16f scalars; fixed at create
} ThreersFluid2dConfig;

// x, y: uv, origin top left. dx, dy: the push. radius: Gaussian variance in
// canvas fractions, already widened for the aspect. erase != 0 removes rgb.
typedef struct {
    float    x, y, dx, dy;
    float    r, g, b;
    float    radius;
    uint32_t erase;
} ThreersFluid2dSplat;

void threers_fluid2d_config_default(ThreersFluid2dConfig *out);

// device: id<MTLDevice>. cfg may be NULL for the defaults. NULL on failure.
ThreersFluid2d *threers_fluid2d_create(void *device, uint32_t width, uint32_t height,
                                       const ThreersFluid2dConfig *cfg);
void threers_fluid2d_destroy(ThreersFluid2d *h);
// The last failure's message, or "". Valid until the next call on h.
const char *threers_fluid2d_last_error(ThreersFluid2d *h);

// id<MTLCommandQueue> the plain calls commit on. Borrowed.
void *threers_fluid2d_queue(ThreersFluid2d *h);
// id<MTLTexture>, rgba16Float. Borrowed; changes after every splat, step or clear.
void *threers_fluid2d_dye_texture(ThreersFluid2d *h);
void  threers_fluid2d_grid_sizes(ThreersFluid2d *h, uint32_t *sim_w, uint32_t *sim_h,
                                 uint32_t *dye_w, uint32_t *dye_h);

// These return 0 on success, -1 on failure (see threers_fluid2d_last_error).
int threers_fluid2d_configure(ThreersFluid2d *h, const ThreersFluid2dConfig *cfg);
int threers_fluid2d_resize(ThreersFluid2d *h, uint32_t width, uint32_t height);

// Committing on the fluid's queue.
int  threers_fluid2d_splats(ThreersFluid2d *h, const ThreersFluid2dSplat *list, uint32_t count);
void threers_fluid2d_splat_random(ThreersFluid2d *h, uint32_t count);
int  threers_fluid2d_step(ThreersFluid2d *h, float dt);
int  threers_fluid2d_clear(ThreersFluid2d *h);

// Into the app's command buffer (id<MTLCommandBuffer>, uncommitted).
int threers_fluid2d_splats_on(ThreersFluid2d *h, void *cmd, const ThreersFluid2dSplat *list,
                              uint32_t count);
int threers_fluid2d_step_on(ThreersFluid2d *h, void *cmd, float dt);
int threers_fluid2d_clear_on(ThreersFluid2d *h, void *cmd);
// threers' own composite into target (id<MTLTexture>, MTLPixelFormat pixel_format).
int threers_fluid2d_render_on(ThreersFluid2d *h, void *cmd, void *target, uint64_t pixel_format,
                              uint32_t width, uint32_t height);

#ifdef __cplusplus
}
#endif

#endif // THREERS_FLUID2D_H
