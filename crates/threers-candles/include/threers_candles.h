// threers candles on a table, drawn on Metal into a CAMetalLayer.
//
// Hand-written, and held to src/lib.rs by `the_header_declares_every_export`.
// Positions on the table are metres from its centre (x right, z towards the
// viewer); screen points are uv, 0-1, origin top left. Every call from one
// thread at a time.
#ifndef THREERS_CANDLES_H
#define THREERS_CANDLES_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ThreersCandles ThreersCandles;

// ---- the table ---------------------------------------------------------------

// layer: a CAMetalLayer that outlives the handle. realistic != 0: volumetric
// flames; 0: the cheap ones, for many candles. NULL on failure.
ThreersCandles *threers_candles_create(void *layer, uint32_t width, uint32_t height,
                                       uint32_t realistic);
void threers_candles_destroy(ThreersCandles *h);
void threers_candles_resize(ThreersCandles *h, uint32_t width, uint32_t height);

// ---- candles -------------------------------------------------------------------

// A candle at x, z on the table; its id, or 0 when the point is off the table.
uint32_t threers_candles_add(ThreersCandles *h, float x, float z);
void     threers_candles_move(ThreersCandles *h, uint32_t id, float x, float z);
void     threers_candles_remove(ThreersCandles *h, uint32_t id);
uint32_t threers_candles_count(ThreersCandles *h);
// Burn in r, g, b (linear), amount 0-1 of the way from the flame's own colour.
void     threers_candles_set_color(ThreersCandles *h, uint32_t id, float r, float g, float b,
                                   float amount);
// bodies != 0: flames on candles (the default); 0: the flames alone. Rebuilds
// those placed, keeping ids and colours.
void     threers_candles_set_bodies(ThreersCandles *h, uint32_t bodies);
// fire != 0: the flames burn as one volume of gas (threers' 3D fluid) in place
// of their meshes; flames close together feed the same gas, so they run
// together and their colours mix. 1 when the fire burns, 0 off or unavailable.
uint32_t threers_candles_set_fire(ThreersCandles *h, uint32_t fire);
// How much fire, 0-1: how bright the gas burns and how high it rises.
void     threers_candles_set_fire_amount(ThreersCandles *h, float amount);
// The sweep on the table at `turns` around (0 the far side, clockwise from
// above), strength 0-1; 0 hides it.
void     threers_candles_set_sweep(ThreersCandles *h, float turns, float strength);
// Candle id flares (0-1): thrown up brighter for a moment, as it is played.
void     threers_candles_flare(ThreersCandles *h, uint32_t id, float amount);

// ---- the screen and the table --------------------------------------------------

// Where the screen point u, v meets the table: 1 and x, z, or 0 off the table.
uint32_t threers_candles_table_point(ThreersCandles *h, float u, float v, float *x, float *z);
// The candle under the screen point, or 0.
uint32_t threers_candles_pick(ThreersCandles *h, float u, float v);
// Where candle id's flame is on screen: 1 and u, v, or 0 if there is none.
uint32_t threers_candles_screen_point(ThreersCandles *h, uint32_t id, float *u, float *v);
// Where the point x, y, z (metres) is on screen: 1 and u, v, or 0 behind the camera.
uint32_t threers_candles_project(ThreersCandles *h, float x, float y, float z, float *u, float *v);
// Turn the view by a drag of dx, dy points; zoom by a pinch's scale since last call.
void     threers_candles_orbit(ThreersCandles *h, float dx, float dy);
void     threers_candles_zoom(ThreersCandles *h, float scale);

// ---- the frame -----------------------------------------------------------------

// Flicker by dt seconds and draw. False when the view is off screen.
bool     threers_candles_frame(ThreersCandles *h, float dt);
// As threers_candles_frame, and a copy of the frame into texture (an id<MTLTexture> the
// view's size, bgra8Unorm_srgb, system default device; the layer not framebufferOnly).
// Returns the copy's id<MTLCommandBuffer>, +1 and NOT committed: add a completion handler,
// commit it (the caller must), release it. NULL when nothing was drawn.
void    *threers_candles_frame_record(ThreersCandles *h, float dt, void *texture);

#ifdef __cplusplus
}
#endif

#endif // THREERS_CANDLES_H
