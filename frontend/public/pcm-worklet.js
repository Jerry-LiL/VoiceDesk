class PCMProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel && channel.length) {
      // Copy: the browser reuses this buffer on the next callback.
      this.port.postMessage(channel.slice(0));
    }
    return true;
  }
}

registerProcessor("pcm-processor", PCMProcessor);
