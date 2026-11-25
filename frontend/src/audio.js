export function downsample(input, fromRate, toRate) {
  if (!input || input.length === 0) return new Float32Array(0);
  if (fromRate === toRate) {
    return input instanceof Float32Array ? input : Float32Array.from(input);
  }
  const length = Math.max(1, Math.round((input.length * toRate) / fromRate));
  const output = new Float32Array(length);
  const last = input.length - 1;
  for (let i = 0; i < length; i += 1) {
    const position = (i * fromRate) / toRate;
    const index = Math.min(Math.floor(position), last);
    const next = Math.min(index + 1, last);
    const fraction = position - index;
    output[i] = input[index] * (1 - fraction) + input[next] * fraction;
  }
  return output;
}

export function floatToPcm16(samples) {
  const pcm = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    pcm[i] = clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767);
  }
  return pcm.buffer;
}

export function rmsOf(samples) {
  if (!samples.length) return 0;
  let sum = 0;
  for (let i = 0; i < samples.length; i += 1) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

export async function startCapture(onChunk) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
    },
    video: false,
  });
  const context = new AudioContext();
  await context.resume();
  await context.audioWorklet.addModule("/pcm-worklet.js");
  const source = context.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(context, "pcm-processor");
  // A silent gain keeps the worklet in the audio graph without playing the mic back.
  const mute = context.createGain();
  mute.gain.value = 0;
  const inputRate = context.sampleRate;

  node.port.onmessage = (event) => {
    const down = downsample(event.data, inputRate, 16000);
    if (!down.length) return;
    onChunk(floatToPcm16(down), rmsOf(down));
  };

  source.connect(node);
  node.connect(mute);
  mute.connect(context.destination);

  return () => {
    node.port.onmessage = null;
    source.disconnect();
    node.disconnect();
    mute.disconnect();
    stream.getTracks().forEach((track) => track.stop());
    return context.close();
  };
}
