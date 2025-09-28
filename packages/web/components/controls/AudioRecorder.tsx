import RecordingContext from "@/context/RecordingContext";
import { MicNone, StopCircle } from "@mui/icons-material";
import { Button, Typography } from "@mui/joy";
import { FC, useCallback, useContext, useEffect, useRef } from "react";

const AudioRecorder: FC = () => {
  const recordingContext = useContext(RecordingContext);

  const audioCtxRef = useRef<AudioContext | null>(null);
  const recorderNodeRef = useRef<AudioWorkletNode | null>(null);
  const audioChunkCountRef = useRef(0);

  if (!recordingContext) {
    throw new Error("AudioRecorder must be used within a RecordingContext");
  }

  const {
    isRecording,
    setIsRecording,
    onStartRecording,
    onAudioData,
    onStopRecording,
  } = recordingContext;

  const handleAudioData = useCallback(
    async (audioData: Uint8Array) => {
      console.debug("AudioRecorder: handleAudioData", isRecording);
      audioChunkCountRef.current++;

      if (audioChunkCountRef.current % 10 === 0) {
        console.log(
          `AudioRecorder: Received ${audioChunkCountRef.current} audio chunks`,
        );
      }

      if (isRecording) {
        await onAudioData?.(audioData);
      }
    },
    [isRecording, onAudioData],
  );

  useEffect(() => {
    if (recorderNodeRef.current) {
      recorderNodeRef.current.port.onmessage = (event: MessageEvent) => {
        if (event.data?.type) {
          console.debug(
            `AudioRecorder: Received message type: ${event.data.type}`,
          );
          return; // Skip debug/level messages
        }
        const pcmBytes = event.data;
        if (pcmBytes instanceof Uint8Array && pcmBytes.byteLength > 0) {
          handleAudioData(new Uint8Array(pcmBytes));
        } else {
          console.warn(
            "AudioRecorder: Received invalid audio data",
            typeof pcmBytes,
            pcmBytes,
          );
        }
      };
    }
  }, [handleAudioData, recorderNodeRef]);

  // Start recording
  const startAudioRecording = useCallback(async (): Promise<boolean> => {
    try {
      // Reset audio chunk counter
      audioChunkCountRef.current = 0;
      console.log("AudioRecorder: Starting recording...");

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true },
      });

      // Create and configure AudioContext
      const audioCtx = new AudioContext({ latencyHint: "interactive" });
      audioCtxRef.current = audioCtx;

      if (audioCtx.state === "suspended") await audioCtx.resume();
      console.log(
        `AudioRecorder: Audio context created, state: ${audioCtx.state}, sample rate: ${audioCtx.sampleRate}`,
      );

      // Create audio source from microphone stream
      const source = audioCtx.createMediaStreamSource(stream);

      // Add audio worklet for processing
      await audioCtx.audioWorklet.addModule(
        "/worklets/audio-recorder.worklet.js",
      );
      console.debug("AudioRecorder: Audio worklet module loaded");

      // Create recorder node with specified options
      const recorderNode = new AudioWorkletNode(
        audioCtx,
        "audio-recorder-processor",
        {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          channelCount: 1,
          processorOptions: {
            nativeContextSampleRate: audioCtx.sampleRate,
            targetSampleRate: 16000,
          },
        },
      );

      recorderNodeRef.current = recorderNode;

      // Initialize audio worklet
      recorderNode.port.postMessage({
        command: "init",
        config: {
          bufferDuration: 100, // 100ms buffer - should send audio chunks every 100ms
          nativeSampleRate: audioCtx.sampleRate,
          targetSampleRate: 16000,
        },
      });
      console.log("AudioRecorder: Worklet initialized with 100ms buffer");

      // Connect audio nodes
      source.connect(recorderNode);
      const silentGain = audioCtx.createGain();
      silentGain.gain.value = 0;
      recorderNode.connect(silentGain).connect(audioCtx.destination);
      console.log("AudioRecorder: Audio nodes connected");

      return true;
    } catch (error) {
      console.error("Error starting audio recording:", error);
      return false;
    }
  }, []);

  const startRecording = useCallback(async () => {
    // Start audio recording
    const success = await startAudioRecording();
    if (success) {
      try {
        await onStartRecording?.();
        setIsRecording(true);
      } catch (error) {
        console.error("Error starting recording:", error);
        // toast.error("Failed to start recording session.");
      }
    } else {
      console.error(
        "Failed to start recording. Please check microphone permissions.",
      );
      // toast.error(
      //   "Failed to start recording. Please check microphone permissions.",
      // );
    }
  }, [onStartRecording, setIsRecording, startAudioRecording]);

  const stopRecording = useCallback(async () => {
    console.debug("AudioRecorder#stopRecording", isRecording);
    if (!isRecording) return;
    console.log("AudioRecorder: Stopping recording...");

    // Clean up audio resources
    if (recorderNodeRef.current) {
      try {
        recorderNodeRef.current.disconnect();
        recorderNodeRef.current.port.postMessage({ command: "stop" });
        recorderNodeRef.current = null;
      } catch (error) {
        console.error("Error stopping recorder node:", error);
      }
    }

    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(console.error);
      audioCtxRef.current = null;
    }

    await onStopRecording?.();
    setIsRecording(false);
  }, [isRecording, setIsRecording, onStopRecording]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (isRecording) {
        stopRecording().catch(console.error);
      }
    };
  }, [isRecording, stopRecording]);

  useEffect(() => {
    // Listening for external isRecording changes from context => stopRecording
    if (!isRecording) {
      stopRecording();
    }
  }, [isRecording, stopRecording]);

  return (
    <Button
      variant="plain"
      color="neutral"
      size="sm"
      startDecorator={
        isRecording ? (
          <Typography color="danger">
            <StopCircle />
          </Typography>
        ) : (
          <MicNone />
        )
      }
      onClick={isRecording ? stopRecording : startRecording}
    >
      {isRecording ? "Stop" : "Dictate"}
    </Button>
  );
};

export default AudioRecorder;
