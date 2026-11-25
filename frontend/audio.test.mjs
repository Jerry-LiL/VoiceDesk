import assert from "node:assert/strict";
import test from "node:test";

import { downsample, floatToPcm16, rmsOf } from "./src/audio.js";

test("downsamples a steady tone from 48 kHz to 16 kHz", () => {
  const input = new Float32Array(4800).fill(0.5);
  const output = downsample(input, 48000, 16000);
  assert.equal(output.length, 1600);
  for (const sample of output) {
    assert.ok(Math.abs(sample - 0.5) < 0.001);
  }
});

test("leaves 16 kHz audio alone", () => {
  const input = new Float32Array([0.1, -0.2, 0.3]);
  const output = downsample(input, 16000, 16000);
  assert.equal(output, input);
});

test("converts floats to 16-bit little-endian samples", () => {
  const buffer = floatToPcm16(new Float32Array([1, -1, 0]));
  const view = new DataView(buffer);
  assert.equal(view.getInt16(0, true), 32767);
  assert.equal(view.getInt16(2, true), -32768);
  assert.equal(view.getInt16(4, true), 0);
});

test("rms of silence is zero", () => {
  assert.equal(rmsOf(new Float32Array(8)), 0);
  assert.equal(rmsOf(new Float32Array([1, -1])), 1);
});
