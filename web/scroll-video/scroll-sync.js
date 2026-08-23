/**
 * Map page scroll to video playback time.
 *
 * Modes:
 * - progress: scroll position scrubs through the clip
 * - velocity: scroll speed integrates into playback time
 */

export class ScrollVideoSync {
    constructor(options) {
        options = options || {};
        this.mode = options.mode != null ? options.mode : 'progress';
        this.scrollStart = options.scrollStart != null ? options.scrollStart : 0;
        this.scrollEnd = options.scrollEnd != null ? options.scrollEnd : 4000;
        this.videoDuration = options.videoDuration != null ? options.videoDuration : 10;
        this.clipIn = options.clipIn != null ? options.clipIn : 0;
        this.clipOut = options.clipOut != null ? options.clipOut : null;
        this.pxPerSecond = options.pxPerSecond != null ? options.pxPerSecond : 900;
        this.velocityGain = options.velocityGain != null ? options.velocityGain : 1;
        this.smoothing = options.smoothing != null ? options.smoothing : 0.18;
        this.easing = options.easing != null ? options.easing : false;
        this.markers = options.markers != null ? options.markers : [];

        this._time = this.clipIn;
        this._scrollY = 0;
        this._scrollSpeed = 0;
        this._lastScrollY = null;
        this._lastTs = null;
    }

    effectiveDuration() {
        var out = this.clipOut != null ? this.clipOut : this.videoDuration;
        return Math.max(0, out - this.clipIn);
    }

    get time() {
        return this._time;
    }

    set time(t) {
        this._time = this.clampTime(t);
    }

    get progress() {
        var d = this.effectiveDuration();
        return d > 0 ? (this._time - this.clipIn) / d : 0;
    }

    get scrollSpeed() {
        return this._scrollSpeed;
    }

    get playbackRate() {
        if (this.pxPerSecond <= 0) return 0;
        return (this._scrollSpeed * this.velocityGain) / this.pxPerSecond;
    }

    clampTime(t) {
        var lo = this.clipIn;
        var hi = this.clipOut != null ? this.clipOut : this.videoDuration;
        return Math.min(Math.max(t, lo), hi);
    }

    /** Smoothstep for progress mapping when easing is enabled. */
    ease(u) {
        u = Math.min(Math.max(u, 0), 1);
        if (!this.easing) return u;
        return u * u * (3 - 2 * u);
    }

    update(state) {
        var scrollY = state.scrollY;
        var deltaY = state.deltaY != null ? state.deltaY : 0;
        var now = state.now != null ? state.now : performance.now();
        this._scrollY = scrollY;
        var dt = this._lastTs != null
            ? Math.max((now - this._lastTs) / 1000, 1 / 120)
            : 0;

        if (this._lastScrollY != null && dt > 0) {
            var dy = scrollY - this._lastScrollY;
            var instant = dy / dt;
            var alpha = 1 - Math.min(Math.max(this.smoothing, 0), 0.99);
            this._scrollSpeed += (instant - this._scrollSpeed) * alpha;
        } else if (deltaY !== 0) {
            this._scrollSpeed = deltaY * 60;
        }

        if (this.mode === 'progress') {
            var span = this.scrollEnd - this.scrollStart;
            var u = span > 0 ? (scrollY - this.scrollStart) / span : 0;
            u = this.ease(u);
            this._time = this.clampTime(this.clipIn + u * this.effectiveDuration());
        } else if (dt > 0) {
            this._time = this.clampTime(this._time + this.playbackRate * dt);
        }

        this._lastScrollY = scrollY;
        this._lastTs = now;
        return this._time;
    }

    applyToVideo(video, opts) {
        opts = opts || {};
        var threshold = opts.threshold != null ? opts.threshold : (
            this.mode === 'progress' ? 0.001 : 0.04
        );
        var t = this._time;
        if (this.mode === 'progress') {
            try {
                video.currentTime = t;
            } catch (_err) {
                /* seek while metadata loading */
            }
        } else if (Math.abs(video.currentTime - t) > threshold) {
            try {
                video.currentTime = t;
            } catch (_err) {
                /* seek while metadata loading */
            }
        }
        if (this.mode === 'velocity') {
            var moving = Math.abs(this._scrollSpeed) > 8;
            if (opts.playWhenMoving !== false && moving) {
                if (video.paused) {
                    var playPromise = video.play();
                    if (playPromise && playPromise.catch) {
                        playPromise.catch(function () {});
                    }
                }
                video.playbackRate = Math.min(Math.max(Math.abs(this.playbackRate), 0.25), 4);
            } else {
                video.pause();
            }
        } else {
            video.pause();
        }
    }

    markerErrors() {
        var span = this.scrollEnd - this.scrollStart;
        var d = this.effectiveDuration();
        var self = this;
        return this.markers.map(function (m) {
            var u = span > 0 ? (m.scrollY - self.scrollStart) / span : 0;
            var mapped = self.clipIn + u * d;
            return {
                label: m.label != null ? m.label : '',
                scrollY: m.scrollY,
                time: m.time,
                error: mapped - m.time,
            };
        });
    }

    /**
     * Re-read DOM markers into this sync instance.
     * @param {Document|Element} root
     * @param {object} [opts]
     */
    refreshMarkers(root, opts) {
        this.markers = readScrollMarkers(root, opts);
        return this.markers;
    }

    toJSON() {
        return {
            mode: this.mode,
            scrollStart: this.scrollStart,
            scrollEnd: this.scrollEnd,
            videoDuration: this.videoDuration,
            clipIn: this.clipIn,
            clipOut: this.clipOut,
            pxPerSecond: this.pxPerSecond,
            velocityGain: this.velocityGain,
            smoothing: this.smoothing,
            easing: this.easing,
            markers: this.markers,
        };
    }

    static fromJSON(json) {
        return new ScrollVideoSync(json);
    }
}

export function readScrollMarkers(root, opts) {
    opts = opts || {};
    var attr = opts.attr != null ? opts.attr : 'data-scroll-marker';
    var timeAttr = opts.timeAttr != null ? opts.timeAttr : 'data-time';
    var labelAttr = opts.labelAttr != null ? opts.labelAttr : 'data-label';
    var scrollRoot = opts.scrollRoot != null ? opts.scrollRoot : null;
    var out = [];
    root.querySelectorAll('[' + attr + ']').forEach(function (el) {
        var scrollY;
        if (scrollRoot) {
            scrollY = el.getBoundingClientRect().top
                - scrollRoot.getBoundingClientRect().top
                + scrollRoot.scrollTop;
        } else {
            scrollY = el.getBoundingClientRect().top + window.scrollY;
        }
        var rawTime = el.getAttribute(timeAttr);
        var time = Number(rawTime != null ? rawTime : '0');
        var rawLabel = el.getAttribute(labelAttr);
        var text = el.textContent ? el.textContent.trim() : '';
        var label = rawLabel != null ? rawLabel : text;
        out.push({ scrollY: scrollY, time: time, label: label });
    });
    return out.sort(function (a, b) { return a.scrollY - b.scrollY; });
}

export function autoFitScrollRange(sync, markers) {
    if (markers.length < 2) return;
    sync.scrollStart = markers[0].scrollY;
    sync.scrollEnd = markers[markers.length - 1].scrollY;
    sync.markers = markers;
    var last = markers[markers.length - 1];
    sync.videoDuration = Math.max(sync.videoDuration, last.time + 0.5);
    if (sync.clipOut == null) sync.clipOut = sync.videoDuration;
}

/**
 * Keep marker scroll positions aligned when copy layout changes.
 * @param {ScrollVideoSync} sync
 * @param {Element} scrollRoot
 * @param {Document|Element} markerRoot
 * @param {object} [opts]
 * @returns {ResizeObserver|null}
 */
export function attachMarkerResizeObserver(sync, scrollRoot, markerRoot, opts) {
    if (!scrollRoot || typeof ResizeObserver === 'undefined') return null;
    var ro = new ResizeObserver(function () {
        sync.refreshMarkers(markerRoot, Object.assign({ scrollRoot: scrollRoot }, opts || {}));
    });
    ro.observe(scrollRoot);
    return ro;
}
