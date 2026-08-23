//! The RLX bridge from outside the crate.
//!
//! The unit tests inside `src/rlx/` check the conversions in isolation; this
//! file checks the promise the feature makes — that data can leave threers as
//! a tensor, be transformed by a graph that rlx actually compiled and ran, and
//! come back as the geometry or the pixels it started as.

#![cfg(feature = "rlx")]

use threers::core::{BufferAttribute, BufferGeometry};
use threers::rlx::prelude::*;
use threers::rlx::{
    geometry_to_tensor, preferred_device, tensor_to_geometry, tensor_to_texture, texture_to_tensor,
    ColorSpace, GraphRunner, Layout, Tensor,
};
use threers::textures::{Texture, TextureFormat};

/// `y = x * factor`, over a statically shaped input.
fn scale_graph(dims: &[usize], factor: f64) -> Graph {
    let mut g = Graph::new("scale");
    let x = g.input("x", Shape::new(dims, DType::F32));
    let k = g.constant(factor, DType::F32);
    let y = g.mul(x, k);
    g.set_outputs(vec![y]);
    g
}

#[test]
fn a_graph_moves_a_geometry_and_the_geometry_notices() {
    let mut geometry = BufferGeometry::new();
    geometry.set_attribute(
        "position",
        BufferAttribute::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 3),
    );
    let version = geometry.geometry_version;

    let positions = geometry_to_tensor(&geometry, "position").expect("position");
    assert_eq!(positions.dims(), &[2, 3]);

    let mut runner = GraphRunner::new(scale_graph(positions.dims(), 2.0), preferred_device());
    let scaled = runner.run(&[("x", &positions)]).remove(0);
    assert_eq!(scaled.dims(), &[2, 3]);
    assert_eq!(scaled.data(), &[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);

    tensor_to_geometry(&mut geometry, "position", &scaled).expect("write back");
    let back = geometry.get_attribute("position").expect("position");
    assert_eq!(back.item_size, 3);
    assert_eq!(back.array, vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
    // Writing through the tensor bridge is still writing: the version stamp
    // moves, so the renderer re-uploads.
    assert_ne!(geometry.geometry_version, version);
}

#[test]
fn a_texture_survives_a_round_trip_through_a_graph() {
    let width = 4;
    let height = 3;
    let data: Vec<u8> = (0..(width * height * 4))
        .map(|i| (i * 5 % 256) as u8)
        .collect();
    let texture = Texture::new(width, height, TextureFormat::Rgba8UnormSrgb, data.clone());

    let tensor = texture_to_tensor(&texture, Layout::Nhwc, ColorSpace::Linear).expect("to tensor");
    assert_eq!(tensor.dims(), &[1, height as usize, width as usize, 4]);

    // Multiplying by one is the identity that still proves the trip: the
    // values went to rlx, were compiled into a graph's arena, and came back.
    let mut runner = GraphRunner::new(scale_graph(tensor.dims(), 1.0), preferred_device());
    let out = runner.run(&[("x", &tensor)]).remove(0);

    let back = tensor_to_texture(
        &out,
        TextureFormat::Rgba8UnormSrgb,
        Layout::Nhwc,
        ColorSpace::Linear,
    )
    .expect("to texture");
    assert_eq!((back.width, back.height), (width, height));
    // 8 bits in, linear f32 through, 8 bits out: exact, not merely close.
    assert_eq!(*back.data, data);
}

#[test]
fn output_shapes_come_back_from_the_graph_not_from_the_caller() {
    let input = Tensor::new((0..24).map(|i| i as f32).collect(), &[2, 3, 4]).unwrap();
    let mut runner = GraphRunner::new(scale_graph(input.dims(), 0.5), preferred_device());
    let out = runner.run(&[("x", &input)]).remove(0);
    assert_eq!(out.dims(), &[2, 3, 4]);
    assert_eq!(out.data()[23], 11.5);
}

#[test]
fn the_device_is_one_rlx_says_it_has() {
    let device = preferred_device();
    assert!(
        available_devices().contains(&device),
        "{device:?} is not among {:?}",
        available_devices()
    );
}
