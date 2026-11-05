//! What a query costs per row, on the paths games use most.
#![recursion_limit = "512"]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use threers::prelude::*;
use threers_ecs::prelude::*;

struct A(f32);
impl Component for A {}
struct Rare;
impl Component for Rare {}

fn world(n: usize) -> World {
    let mut w = World::new();
    for i in 0..n {
        let e = w.spawn(Object3D::group());
        w.insert(e, A(i as f32));
        if i % 100 == 0 {
            w.insert(e, Rare);
        }
    }
    w
}

fn queries(c: &mut Criterion) {
    let mut w = world(100_000);
    c.bench_function("each &mut A", |b| {
        b.iter(|| w.each::<&mut A>(|a| a.0 += 1.0))
    });
    c.bench_function("each (Node, &A)", |b| {
        b.iter(|| w.each::<(Node, &A)>(|(node, a)| node.position.x = a.0))
    });
    c.bench_function("each (&A, With<Rare>)", |b| {
        b.iter(|| {
            w.each::<(&A, With<Rare>)>(|(a, _)| {
                black_box(a.0);
            })
        })
    });
    c.bench_function("each Node", |b| {
        b.iter(|| w.each::<Node>(|node| node.position.y += 1.0))
    });
    c.bench_function("par_query &mut A", |b| {
        b.iter(|| w.par_query::<&mut A>(|a| a.0 += 1.0))
    });
}

criterion_group!(benches, queries);
criterion_main!(benches);
