import { FC, useCallback, useEffect, useRef, useState } from "react";
import RecordingContextProvider from "@/context/RecordingContext";
import AudioRecorder from "./AudioRecorder";
import { ChunkType, MemoryChunk } from "protos/generated/ts/stt";

const TranscribedRecorder: FC<{
  onTranscription: (transcript: MemoryChunk) => void;
}> = ({ onTranscription }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [, setIsConnected] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  const socketClosingTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const socket = socketRef.current;

    return () => {
      if (socket) {
        socket.close();
      }
    };
  }, [socketRef]);

  const stopTranscription = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.send(
        MemoryChunk.encode({
          metadata: {
            sessionId: "",
            memoryId: "",
            type: ChunkType.TRANSCRIPT,
            isFinal: true,
            score: 0,
          },
        }).finish(),
      );

      if (socketClosingTimeoutRef.current) {
        console.debug("Clearing existing socket close timeout");
        clearTimeout(socketClosingTimeoutRef.current);
      }
      socketClosingTimeoutRef.current = setTimeout(() => {
        if (socketRef.current) {
          socketRef.current.close();
          socketRef.current = null;
        }
      }, 5000);
    }
  }, []);

  const startTranscription = useCallback(() => {
    socketRef.current = new WebSocket("ws://localhost:8080/ws/transcribe");

    socketRef.current.addEventListener("open", () => {
      console.log("WebSocket connection established");
      setIsConnected(true);
    });

    socketRef.current.addEventListener("message", async (event) => {
      const data =
        event.data instanceof Blob ? await event.data.bytes() : event.data;
      if (data) {
        const message = MemoryChunk.decode(new Uint8Array(data));
        console.debug("TranscribedRecorder: Message received", message);
        if (message.metadata?.type === ChunkType.TRANSCRIPT) {
          if (message.metadata.isFinal) {
            console.debug("Final transcript: ", message.textData);
            if (socketClosingTimeoutRef.current) {
              console.debug(
                "TranscribedRecorder: isFinal received, clearing existing socket close timeout",
              );
              clearTimeout(socketClosingTimeoutRef.current);
            }
            socketRef.current?.close();
            socketRef.current = null;
          }
          onTranscription(message);
        } else {
          console.log("Ignored message:", message);
        }
      } else {
        console.warn("TranscribedRecorder: Received empty message", data);
      }
    });

    socketRef.current.addEventListener("close", () => {
      console.log("WebSocket connection closed");
      setIsConnected(false);
    });

    socketRef.current.addEventListener("error", (error) => {
      console.error("WebSocket error: ", error);
      stopTranscription();
    });
  }, [onTranscription, stopTranscription]);

  const onStartRecording = useCallback(async () => {
    console.debug("TranscribedRecorder#onStartRecording");
    startTranscription();
    setIsConnected(true);
  }, [startTranscription]);

  const onAudioData = useCallback(async (audioData: Uint8Array) => {
    console.debug("TranscribedRecorder#onAudioData", audioData);
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
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
      socketRef.current.send(message);
    } else {
      console.warn("Unable to send audio data.");
    }
  }, []);

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
