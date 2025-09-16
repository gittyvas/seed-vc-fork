class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.micGain = 1.0;   // Default gain
    this.outputGain = 1.0;

    // Listen for config messages from main thread
    this.port.onmessage = (event) => {
      if (event.data.type === "config") {
        if (event.data.micGain !== undefined) {
          this.micGain = event.data.micGain;
        }
        if (event.data.outputGain !== undefined) {
          this.outputGain = event.data.outputGain;
        }
      }
    };
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input && input.length > 0) {
      const inputChannel = input[0];

      // Apply mic gain
      const processed = new Float32Array(inputChannel.length);
      for (let i = 0; i < inputChannel.length; i++) {
        processed[i] = inputChannel[i] * this.micGain;
      }

      // Send processed audio to main thread
      this.port.postMessage(processed.buffer, [processed.buffer]);
    }

    // Optional: you could also write back to outputs[] if you want live monitoring
    // (e.g., apply outputGain here before passing audio to speakers).

    return true;
  }
}

registerProcessor('audio-processor', AudioProcessor);
