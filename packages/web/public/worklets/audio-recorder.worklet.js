"use strict";
class AudioRecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // Fixed settings that cannot be changed
    this.defaultTargetSampleRate = 16000; // Output at 16kHz for speech
    this.defaultChunkDuration = 100; // ms per chunk
    this.nativeSampleRate = 0; // Will be set during initialization
    this.bufferIndex = 0;
    // Stats and state
    this.processCounter = 0;
    this.chunksSent = 0;
    this.lastLevelReportFrame = 0;
    // Initialize with default values
    this.configuredTargetSampleRate = this.defaultTargetSampleRate;
    this.configuredChunkDuration = this.defaultChunkDuration;
    // Default to worklet's sampleRate until explicitly configured
    this.nativeSampleRate = sampleRate;
    // Calculate buffer size for the target duration
    const bufferSize = Math.ceil(
      (this.configuredTargetSampleRate * this.configuredChunkDuration) / 1000,
    );
    this.buffer = new Float32Array(bufferSize);
    // Add message handler to receive initialization data
    this.port.onmessage = (event) => {
      if (event.data && event.data.command === "init") {
        const config = event.data.config || {};
        // Update sample rates if provided
        if (config.nativeSampleRate) {
          this.nativeSampleRate = config.nativeSampleRate;
        }
        // Update target sample rate if provided
        if (config.targetSampleRate) {
          this.configuredTargetSampleRate = config.targetSampleRate;
        }
        // Update buffer duration if provided
        if (config.bufferDuration) {
          this.configuredChunkDuration = config.bufferDuration;
          // Recalculate buffer size with new settings
          const newBufferSize = Math.ceil(
            (this.configuredTargetSampleRate * this.configuredChunkDuration) /
              1000,
          );
          this.buffer = new Float32Array(newBufferSize);
          this.bufferIndex = 0;
        }
        this.port.postMessage({
          type: "debug",
          message: `Recorder initialized: native=${this.nativeSampleRate}Hz, target=${this.configuredTargetSampleRate}Hz, buffer=${this.buffer.length} samples`,
        });
      }
    };
  }
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  process(inputs, outputs) {
    this.processCounter++;
    // Check for valid input
    if (!inputs || !inputs[0] || !inputs[0][0] || inputs[0][0].length === 0) {
      return true;
    }
    // Get mono input (or mix down to mono if stereo)
    let inputSamples;
    if (inputs[0].length > 1) {
      // Mix stereo channels to mono
      inputSamples = new Float32Array(inputs[0][0].length);
      for (let i = 0; i < inputSamples.length; i++) {
        inputSamples[i] = 0.5 * (inputs[0][0][i] + inputs[0][1][i]);
      }
    } else {
      inputSamples = inputs[0][0];
    }
    // Calculate audio level for this frame
    let maxLevel = 0;
    for (let i = 0; i < inputSamples.length; i++) {
      maxLevel = Math.max(maxLevel, Math.abs(inputSamples[i]));
    }
    // Report audio levels periodically
    if (this.processCounter - this.lastLevelReportFrame >= 100) {
      this.port.postMessage({
        type: "level",
        value: maxLevel,
      });
      this.lastLevelReportFrame = this.processCounter;
    }
    // Downsample to target sample rate if needed
    let processedSamples;
    if (this.nativeSampleRate === this.configuredTargetSampleRate) {
      processedSamples = inputSamples;
    } else {
      // Use the correct sample rates for accurate resampling
      processedSamples = this.downsample(
        inputSamples,
        this.nativeSampleRate,
        this.configuredTargetSampleRate,
      );
    }
    // Add processed samples to our buffer
    for (let i = 0; i < processedSamples.length; i++) {
      // If buffer is full, send it and start a new one
      if (this.bufferIndex >= this.buffer.length) {
        this.sendBuffer();
      }
      // Add sample to buffer
      this.buffer[this.bufferIndex++] = processedSamples[i];
    }
    // Send buffer if it's 3/4 full, to avoid holding audio too long
    if (this.bufferIndex >= this.buffer.length * 0.75) {
      this.sendBuffer();
    }
    return true;
  }
  // Improved downsampling function
  downsample(float32Arr, fromSampleRate, toSampleRate) {
    if (fromSampleRate === toSampleRate) {
      return float32Arr;
    }
    const sampleRateRatio = fromSampleRate / toSampleRate;
    const newLength = Math.round(float32Arr.length / sampleRateRatio);
    const result = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;
    while (offsetResult < result.length) {
      const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
      // Use average value of skipped samples for better quality
      let accum = 0,
        count = 0;
      for (
        let i = offsetBuffer;
        i < nextOffsetBuffer && i < float32Arr.length;
        i++
      ) {
        accum += float32Arr[i];
        count++;
      }
      result[offsetResult] = count > 0 ? accum / count : 0;
      offsetResult++;
      offsetBuffer = nextOffsetBuffer;
    }
    return result;
  }
  // Helper to send current buffer
  sendBuffer() {
    if (this.bufferIndex === 0) return;
    // Create buffer with just the filled part
    const filledBuffer = this.buffer.slice(0, this.bufferIndex);
    // Convert to Int16 for transmission (common PCM format)
    const int16Data = new Int16Array(filledBuffer.length);
    for (let i = 0; i < filledBuffer.length; i++) {
      // Clamp values to [-1,1] and scale to Int16 range
      const sample = Math.max(-1, Math.min(1, filledBuffer[i]));
      int16Data[i] = Math.round(sample < 0 ? sample * 32768 : sample * 32767);
    }
    // Convert to Uint8Array for transmission
    const uint8Data = new Uint8Array(int16Data.buffer);
    // Send using transferable objects for efficiency
    this.port.postMessage(uint8Data, [uint8Data.buffer]);
    this.chunksSent++;
    // Reset buffer
    this.buffer = new Float32Array(this.buffer.length);
    this.bufferIndex = 0;
  }
}
registerProcessor("audio-recorder-processor", AudioRecorderProcessor);
