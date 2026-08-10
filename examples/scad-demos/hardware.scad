// A tiny reusable 2D-profile library (modules with defaults → `use`-friendly).
module rounded_bar(l, h, r) offset(r = r) square([l, h]);
module hole(r = 1.5) circle(r, $fn = 24);
