import { FC, useCallback, useState } from "react";
import RecordingContextProvider from "@/context/RecordingContext";
import AudioRecorder from "./AudioRecorder";
import { ChunkType, MemoryChunk } from "protos/generated/ts/stt";
import { useWebSocket } from "@/hooks/useWebSocket";

const TranscribedRecorder: FC<{
  onTranscription: (transcript: MemoryChunk) => void;
}> = ({ onTranscription }) => {
  const [isRecording, setIsRecording] = useState(false);
  const { addEventListener, connect, disconnect, send } = useWebSocket(
    "ws://localhost:8080/ws/transcribe",
  );

  const stopTranscription = useCallback(() => {
    disconnect();
  }, [disconnect]);

  const startTranscription = useCallback(() => {
    connect();

    const success = addEventListener("message", async (event: MessageEvent) => {
      const data =
        event.data instanceof Blob ? await event.data.bytes() : event.data;
      if (data) {
        const message = MemoryChunk.decode(new Uint8Array(data));
        console.debug("TranscribedRecorder: Message received", message);
        if (message.metadata?.type === ChunkType.TRANSCRIPT) {
          onTranscription(message);
        } else {
          console.log("Ignored message:", message);
        }
      } else {
        console.warn("TranscribedRecorder: Received empty message", data);
      }
    });
    if (!success) {
      console.error("Failed to add message event listener");
      stopTranscription();
      return;
    }

    addEventListener("error", (error) => {
      console.error("WebSocket error: ", error);
      stopTranscription();
    });
  }, [addEventListener, connect, onTranscription, stopTranscription]);

  const onStartRecording = useCallback(async () => {
    console.debug("TranscribedRecorder#onStartRecording");
    startTranscription();
  }, [startTranscription]);

  const onAudioData = useCallback(
    async (audioData: Uint8Array) => {
      console.debug("TranscribedRecorder#onAudioData", audioData);
      const data: MemoryChunk = {
        audioData: audioData,
        metadata: {
          sessionId: "",
          memoryId: "",
          type: ChunkType.TRANSCRIPT,
          isFinal: false,
          score: 0,
        },
      };
      // send as binary encoded protobuf message
      const message = MemoryChunk.encode(data).finish();
      send(message);
    },
    [send],
  );

  const onStopRecording = useCallback(async () => {
    console.debug("TranscribedRecorder#onStopRecording");
    stopTranscription();
  }, [stopTranscription]);

  return (
    <RecordingContextProvider
      value={{
        isRecording,
        setIsRecording,
        onStartRecording,
        onAudioData,
        onStopRecording,
      }}
    >
      <AudioRecorder />
    </RecordingContextProvider>
  );
};

export default TranscribedRecorder;
