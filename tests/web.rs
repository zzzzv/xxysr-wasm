//! Test suite for the Web and headless browsers.

#![cfg(target_arch = "wasm32")]

extern crate wasm_bindgen_test;
use wasm_bindgen_test::*;
use xxysr_wasm::calc_sr;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_calc_sr() {
    let text = include_str!("../tests/1.osu");
    let speed = 1.0;
    let sr = calc_sr(text, speed).unwrap();
    let expected = 6.617031;
    let epsilon = 1e-6;
    assert!((sr - expected).abs() < epsilon, "sr={}, expected={}", sr, expected);
}