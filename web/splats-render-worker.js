// The splat viewer's renderer in a Worker, drawing to the OffscreenCanvas the
// page handed over: see web/splats-view.js.
import { run } from "./splats-view.js";

run(self, true);
