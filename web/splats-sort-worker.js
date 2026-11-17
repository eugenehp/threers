// The splat viewer's sort, on its own thread: the renderer sends the splats'
// centres once, then a view axis whenever it turns, and gets back the order
// to draw them in (splat indices as floats, back to front), transferred
// rather than copied. The radix sort is threers-splat-sort's, through
// `WebSplatSort`; with a SIMD build its keys are made in simd128.
//
// The port to the renderer arrives as the first message.
import init, { WebSplatSort } from "./pkg/threers.js";

const ready = init();
let sorter = null, gen = -1;

self.onmessage = ({ data }) => {
  const port = data.port;
  port.onmessage = async ({ data: m }) => {
    await ready;
    if (m.type === "centres") {
      sorter?.free();
      sorter = new WebSplatSort(m.centres);
      gen = m.gen;
    } else if (m.type === "sort" && sorter && m.gen === gen) {
      // Into the buffer the renderer handed back (two of them shuttle
      // between the threads), so sorting every frame makes no garbage.
      const t = performance.now();
      const order = m.buffer && m.buffer.length === sorter.length ? m.buffer : new Float32Array(sorter.length);
      sorter.sortInto(m.axis[0], m.axis[1], m.axis[2], order);
      port.postMessage({ type: "order", order, gen, axis: m.axis, ms: performance.now() - t }, [order.buffer]);
    }
  };
};
