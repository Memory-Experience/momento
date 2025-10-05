import "@testing-library/jest-dom";
import { render, screen, waitFor, act } from "@testing-library/react";
import { useContext } from "react";
import TranscribedRecorder from "@/components/controls/TranscribedRecorder";
import AudioRecorder from "@/components/controls/AudioRecorder";
import RecordingContext from "@/context/RecordingContext";
import { ChunkType, MemoryChunk } from "protos/generated/ts/stt";

const states = {
  OPEN: WebSocket.OPEN.valueOf(),
  CLOSED: WebSocket.CLOSED.valueOf(),
};

// Simple WebSocket mock focusing on the API interaction
const mockWebSocket = {
  send: jest.fn(),
  close: jest.fn(),
  addEventListener: jest.fn(),
  readyState: states.OPEN,
};

const mockWebSocketConstructor = jest.fn(() => mockWebSocket);

// Helper to trigger WebSocket events
const triggerWebSocketEvent = (eventType: string, data?: Uint8Array) => {
  const calls = mockWebSocket.addEventListener.mock.calls;
  const handler: EventListener[] = calls
    .filter((call) => call[0] === eventType)
    .map((call) => call[1]);
  if (handler) {
    if (eventType === "message") {
      handler.forEach((curr) => {
        curr({ data } as MessageEvent);
      });
    } else if (eventType === "open") {
      mockWebSocket.readyState = states.OPEN;
      handler.forEach((curr) => curr({} as Event));
    } else if (eventType === "close") {
      mockWebSocket.readyState = states.CLOSED;
      handler.forEach((curr) => curr({} as Event));
    } else {
      handler.forEach((curr) => curr({} as Event));
    }
  }
};

let originalWebSocket: typeof WebSocket;
beforeAll(() => {
  originalWebSocket = global.WebSocket;
  // Mocking WebSocket contructor returning the mockWebSocket instance
  (global as { WebSocket: Partial<typeof WebSocket> }).WebSocket =
    mockWebSocketConstructor;
  for (const [key, value] of Object.entries(states)) {
    (global.WebSocket as unknown as Record<string, unknown>)[key] = value;
  }
  jest.useFakeTimers();
});

afterAll(() => {
  global.WebSocket = originalWebSocket;
});

jest.mock("./AudioRecorder", () => {
  return jest.fn(() => {
    const recordingContext = useContext(RecordingContext);

    if (!recordingContext) {
      throw new Error("recordingContext is null");
    }

    const { onAudioData, onStartRecording, onStopRecording } = recordingContext;

    const sendAudioData = () => {
      const value =
        document.querySelector<HTMLInputElement>("#audio-recorder")?.value;
      if (!value) {
        return onAudioData?.(new Uint8Array());
      }

      onAudioData?.(
        new Uint8Array(value.split(",").map((v) => parseInt(v, 10))),
      );
    };

    return (
      <div>
        <button onClick={onStartRecording}>Start Recording</button>
        <button onClick={onStopRecording}>Stop Recording</button>
        <input
          id="audio-recorder"
          type="text"
          readOnly
          value="AudioRecorder Component"
        />
        <button onClick={sendAudioData}>Send Transcription</button>
      </div>
    );
  });
});

beforeEach(() => {
  jest.clearAllMocks();
});

describe("TranscribedRecorder", () => {
  it("renders TranscribedRecorder with AudioRecorder child", () => {
    const onTranscription = jest.fn();
    expect(AudioRecorder).toHaveBeenCalledTimes(0);

    render(<TranscribedRecorder onTranscription={onTranscription} />);

    expect(AudioRecorder).toHaveBeenCalledTimes(1);
    expect(onTranscription).toHaveBeenCalledTimes(0);
    expect(
      screen.getByRole("button", { name: "Start Recording" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Stop Recording" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Send Transcription" }),
    ).toBeInTheDocument();
  });

  it("starts and stops WebSocket connection", async () => {
    render(<TranscribedRecorder onTranscription={jest.fn()} />);

    const startButton = screen.getByRole("button", { name: "Start Recording" });

    act(() => {
      startButton.click();
    });

    expect(mockWebSocketConstructor).toHaveBeenCalledWith(
      "ws://localhost:8080/ws/transcribe",
    );

    const stopButton = screen.getByRole("button", { name: "Stop Recording" });
    act(() => {
      stopButton.click();
      // Trigger timeout (no final message sent)
      jest.advanceTimersByTime(5000);
    });

    await waitFor(() => {
      expect(mockWebSocket.close).toHaveBeenCalled();
    });
  });

  it("sends audio data through WebSocket when recording", async () => {
    render(<TranscribedRecorder onTranscription={jest.fn()} />);

    const startButton = screen.getByRole("button", { name: "Start Recording" });
    act(() => {
      startButton.click();
      triggerWebSocketEvent("open");
    });

    // Set up audio data and send
    const audioRecorderInput = screen.getByRole<HTMLInputElement>("textbox", {
      name: "",
    });
    audioRecorderInput.value = "1,2,3,4,5";

    const sendButton = screen.getByRole("button", {
      name: "Send Transcription",
    });
    sendButton.click();

    await waitFor(() => {
      expect(mockWebSocket.send).toHaveBeenCalledTimes(1);
      expect(mockWebSocket.send).toHaveBeenCalledWith(
        MemoryChunk.encode({
          audioData: new Uint8Array([1, 2, 3, 4, 5]),
          metadata: {
            type: ChunkType.TRANSCRIPT,
            isFinal: false,
            sessionId: "",
            memoryId: "",
            score: 0,
          },
        }).finish(),
      );
    });
  });

  it("handles incoming WebSocket messages and calls onTranscription", async () => {
    const onTranscription = jest.fn();
    render(<TranscribedRecorder onTranscription={onTranscription} />);

    const startButton = screen.getByRole("button", { name: "Start Recording" });
    act(() => {
      startButton.click();
      triggerWebSocketEvent("open");
    });

    // Create a mock transcript message
    const mockTranscriptMessage: MemoryChunk = {
      textData: "Hello world",
      metadata: {
        sessionId: "test-session",
        memoryId: "test-memory",
        type: ChunkType.TRANSCRIPT,
        isFinal: false,
        score: 0.95,
      },
    };
    const encodedMessage = MemoryChunk.encode(mockTranscriptMessage).finish();

    // Simulate receiving the message
    triggerWebSocketEvent("message", encodedMessage);

    expect(onTranscription).toHaveBeenCalledTimes(1);
    expect(onTranscription).toHaveBeenCalledWith(mockTranscriptMessage);
  });

  it("sends final message when stopping recording", async () => {
    render(<TranscribedRecorder onTranscription={jest.fn()} />);

    // Start recording
    const startButton = screen.getByRole("button", { name: "Start Recording" });
    act(() => {
      startButton.click();
      triggerWebSocketEvent("open");
    });

    // Stop recording
    const stopButton = screen.getByRole("button", { name: "Stop Recording" });
    act(() => {
      stopButton.click();
    });

    // Verify final message was sent
    await waitFor(() => {
      expect(mockWebSocket.send).toHaveBeenCalledTimes(1);
      expect(mockWebSocket.send).toHaveBeenCalledWith(
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
    });
  });

  it("handles final transcript messages and closes connection", async () => {
    const onTranscription = jest.fn();
    render(<TranscribedRecorder onTranscription={onTranscription} />);

    // Start recording
    const startButton = screen.getByRole("button", { name: "Start Recording" });
    act(() => {
      startButton.click();
      triggerWebSocketEvent("open");
    });

    // Create a final transcript message
    const finalMessage: MemoryChunk = {
      textData: "Final transcript",
      metadata: {
        sessionId: "test-session",
        memoryId: "test-memory",
        type: ChunkType.TRANSCRIPT,
        isFinal: true,
        score: 0.98,
      },
    };

    const encodedMessage = MemoryChunk.encode(finalMessage).finish();

    act(() => {
      triggerWebSocketEvent("message", encodedMessage);
      jest.runAllTimers();
    });

    expect(onTranscription).toHaveBeenCalledTimes(1);
    expect(onTranscription).toHaveBeenCalledWith(finalMessage);

    expect(mockWebSocket.close).toHaveBeenCalledTimes(1);
  });
});
