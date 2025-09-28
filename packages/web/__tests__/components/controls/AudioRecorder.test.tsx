import "@testing-library/jest-dom";
import { render, screen, waitFor, act } from "@testing-library/react";
import { FC, ReactElement, useState, PropsWithChildren } from "react";
import AudioRecorder from "@/components/controls/AudioRecorder";
import RecordingContext from "@/context/RecordingContext";
import { RecordingContextType } from "@/types/RecordingContextType";

const renderWithContext = (
  ui: ReactElement,
  contextValue: RecordingContextType,
) => {
  return render(
    <RecordingContext.Provider value={contextValue}>
      {ui}
    </RecordingContext.Provider>,
  );
};

type NonReactiveRecordingContextType = Exclude<
  RecordingContextType,
  "isRecording" | "setIsRecording"
>;

const ReactiveRecordingStateContextProvider: FC<
  PropsWithChildren<NonReactiveRecordingContextType>
> = ({ children, ...restProps }) => {
  const [isRecording, setIsRecording] = useState(false);
  return (
    <RecordingContext.Provider
      value={{
        ...restProps,
        isRecording,
        setIsRecording,
      }}
    >
      {children}
    </RecordingContext.Provider>
  );
};

// Mock Web APIs
const mockMediaStream = {
  getTracks: jest.fn(() => []),
  getAudioTracks: jest.fn(() => []),
  getVideoTracks: jest.fn(() => []),
};

const mockMediaStreamSource = {
  connect: jest.fn(),
  disconnect: jest.fn(),
};

const mockGainNode = {
  connect: jest.fn(() => ({ connect: jest.fn() })),
  gain: { value: 0 },
};

const mockAudioWorkletNode = {
  connect: jest.fn(() => mockGainNode),
  disconnect: jest.fn(),
  port: {
    postMessage: jest.fn(),
    onmessage: null as ((event: MessageEvent) => void) | null,
  },
};

const mockAudioContext = {
  state: "running",
  sampleRate: 48000,
  destination: {},
  resume: jest.fn(() => Promise.resolve()),
  close: jest.fn(() => Promise.resolve()),
  createMediaStreamSource: jest.fn(() => mockMediaStreamSource),
  createGain: jest.fn(() => mockGainNode),
  audioWorklet: {
    addModule: jest.fn(() => Promise.resolve()),
  },
};

// Mock getUserMedia
const mockGetUserMedia = jest.fn(() => Promise.resolve(mockMediaStream));

// Mock AudioContext constructor
const mockAudioContextConstructor = jest.fn(() => mockAudioContext);

// Mock AudioWorkletNode constructor
const mockAudioWorkletNodeConstructor = jest.fn(() => mockAudioWorkletNode);

let originalAudioContext: typeof AudioContext;
let originalAudioWorkletNode: typeof AudioWorkletNode;
beforeAll(() => {
  // Mock navigator.mediaDevices.getUserMedia
  Object.defineProperty(navigator, "mediaDevices", {
    writable: true,
    value: {
      getUserMedia: mockGetUserMedia,
    },
  });

  originalAudioContext = global.AudioContext;
  (global as { AudioContext: Partial<typeof AudioContext> }).AudioContext =
    mockAudioContextConstructor;

  originalAudioWorkletNode = global.AudioWorkletNode;
  (
    global as { AudioWorkletNode: Partial<typeof AudioWorkletNode> }
  ).AudioWorkletNode = mockAudioWorkletNodeConstructor;
});

afterAll(() => {
  global.AudioContext = originalAudioContext;
  global.AudioWorkletNode = originalAudioWorkletNode;
});

beforeEach(() => {
  jest.clearAllMocks();
  // Reset mock state
  mockAudioContext.state = "running";
  mockAudioWorkletNode.port.onmessage = null;
});

describe("AudioRecorder", () => {
  it("must be rendered in a RecordingContext", () => {
    const errorySpy = jest.fn();
    jest.spyOn(console, "error").mockImplementationOnce(errorySpy);

    expect(() => render(<AudioRecorder />)).toThrow(
      "AudioRecorder must be used within a RecordingContext",
    );
    // expect(errorySpy).toHaveBeenCalledTimes(1);
  });

  it("renders an AudioRecorder with dictation controls", () => {
    renderWithContext(<AudioRecorder />, {
      isRecording: false,
      setIsRecording: () => {},
    });

    expect(screen.getByRole("button", { name: "Dictate" })).toBeInTheDocument();
  });

  it("renders an AudioRecorder with stop dictation controls if isRecording is true", () => {
    renderWithContext(<AudioRecorder />, {
      isRecording: true,
      setIsRecording: () => {},
    });

    expect(screen.getByRole("button", { name: "Stop" })).toBeInTheDocument();
  });

  it("request user media and initializes audio context when starting recording", async () => {
    const setIsRecording = jest.fn();
    const onStartRecording = jest.fn(() => Promise.resolve());

    renderWithContext(<AudioRecorder />, {
      isRecording: false,
      setIsRecording,
      onStartRecording,
    });

    const startButton = screen.getByRole("button", { name: "Dictate" });
    startButton.click();

    // Verify microphone permission was requested
    await waitFor(() => {
      expect(mockGetUserMedia).toHaveBeenCalledTimes(1);
      expect(mockGetUserMedia).toHaveBeenCalledWith({
        audio: { channelCount: 1, echoCancellation: true },
      });
    });

    // Verify AudioContext was created
    expect(mockAudioContextConstructor).toHaveBeenCalledTimes(1);
    expect(mockAudioContextConstructor).toHaveBeenCalledWith({
      latencyHint: "interactive",
    });

    // Verify audio worklet was requested
    expect(mockAudioContext.audioWorklet.addModule).toHaveBeenCalledTimes(1);
    expect(mockAudioContext.audioWorklet.addModule).toHaveBeenCalledWith(
      "/worklets/audio-recorder.worklet.js",
    );

    // Verify AudioWorkletNode was created
    expect(mockAudioWorkletNodeConstructor).toHaveBeenCalledTimes(1);
    expect(mockAudioWorkletNodeConstructor).toHaveBeenCalledWith(
      mockAudioContext,
      "audio-recorder-processor",
      expect.objectContaining({
        numberOfInputs: 1,
        numberOfOutputs: 1,
        channelCount: 1,
      }),
    );

    // Verify media stream source connects to worklet node
    expect(mockMediaStreamSource.connect).toHaveBeenCalledTimes(1);
    expect(mockMediaStreamSource.connect).toHaveBeenCalledWith(
      mockAudioWorkletNode,
    );

    // Verify worklet node connects to gain node
    expect(mockAudioWorkletNode.connect).toHaveBeenCalledTimes(1);
    expect(mockAudioWorkletNode.connect).toHaveBeenCalledWith(mockGainNode);
    expect(mockAudioWorkletNode.port.postMessage).toHaveBeenCalledTimes(1);
    expect(mockAudioWorkletNode.port.postMessage).toHaveBeenCalledWith({
      command: "init",
      config: {
        bufferDuration: 100,
        nativeSampleRate: 48000,
        targetSampleRate: 16000,
      },
    });

    // Verify gain node connects to destination
    expect(mockGainNode.connect).toHaveBeenCalledTimes(1);
    expect(mockGainNode.connect).toHaveBeenCalledWith(
      mockAudioContext.destination,
    );

    // Verify context callbacks were called
    expect(onStartRecording).toHaveBeenCalledTimes(1);
    expect(onStartRecording).toHaveBeenCalledWith();
    expect(setIsRecording).toHaveBeenCalledTimes(1);
    expect(setIsRecording).toHaveBeenCalledWith(true);
  });

  it("handles audio data from worklet", async () => {
    const onAudioData = jest.fn();
    render(
      <ReactiveRecordingStateContextProvider onAudioData={onAudioData}>
        <AudioRecorder />
      </ReactiveRecordingStateContextProvider>,
    );

    const startButton = screen.getByRole("button", { name: "Dictate" });
    startButton.click();

    // Wait for worklet setup
    await waitFor(() => {
      expect(mockAudioWorkletNode.port.postMessage).toHaveBeenCalled();
    });

    // Simulate receiving audio data from worklet
    const mockAudioData = new Uint8Array([1, 2, 3, 4, 5]);
    act(() => {
      mockAudioWorkletNode.port.onmessage?.({
        data: mockAudioData,
      } as MessageEvent);
    });

    // Verify audio data was passed to context
    await waitFor(() => {
      expect(onAudioData).toHaveBeenCalledTimes(1);
      expect(onAudioData).toHaveBeenCalledWith(mockAudioData);
    });
  });

  it("stops recording when Stop button is clicked", async () => {
    const onStartRecording = jest.fn(() => Promise.resolve());
    const onStopRecording = jest.fn(() => Promise.resolve());

    render(
      <ReactiveRecordingStateContextProvider
        onStartRecording={onStartRecording}
        onStopRecording={onStopRecording}
      >
        <AudioRecorder />
      </ReactiveRecordingStateContextProvider>,
    );

    const startButton = screen.getByRole("button", { name: "Dictate" });
    startButton.click();

    await waitFor(() => {
      expect(onStartRecording).toHaveBeenCalledTimes(1);
    });

    const stopButton = screen.getByRole("button", { name: "Stop" });
    stopButton.click();

    // Verify cleanup was performed
    await waitFor(() => {
      expect(onStopRecording).toHaveBeenCalledTimes(1);
    });
    expect(mockAudioWorkletNode.disconnect).toHaveBeenCalledTimes(1);
    expect(mockAudioWorkletNode.disconnect).toHaveBeenCalledWith();
    expect(mockAudioWorkletNode.port.postMessage).toHaveBeenCalledTimes(2);
    expect(mockAudioWorkletNode.port.postMessage).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({ command: "init" }),
    );
    expect(mockAudioWorkletNode.port.postMessage).toHaveBeenCalledWith({
      command: "stop",
    });
    expect(mockAudioContext.close).toHaveBeenCalledTimes(1);
    expect(mockAudioContext.close).toHaveBeenCalledWith();
  });

  it("handles microphone permission errors gracefully", async () => {
    const onStartRecording = jest.fn(() => Promise.resolve());
    const setIsRecording = jest.fn();
    const consoleSpy = jest.spyOn(console, "error").mockImplementation();
    mockGetUserMedia.mockRejectedValueOnce(new Error("Permission denied"));

    renderWithContext(<AudioRecorder />, {
      isRecording: false,
      setIsRecording,
      onStartRecording,
    });

    const startButton = screen.getByRole("button", { name: "Dictate" });
    startButton.click();

    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith(
        "Error starting audio recording:",
        expect.any(Error),
      );
    });

    // Verify recording was not started
    expect(onStartRecording).not.toHaveBeenCalled();
    expect(setIsRecording).not.toHaveBeenCalledWith(true);

    consoleSpy.mockRestore();
  });

  it("handles invalid audio data gracefully", async () => {
    const consoleSpy = jest
      .spyOn(console, "warn")
      .mockImplementationOnce(() => {});

    render(
      <ReactiveRecordingStateContextProvider>
        <AudioRecorder />
      </ReactiveRecordingStateContextProvider>,
    );

    const startButton = screen.getByRole("button", { name: "Dictate" });

    await act(async () => {
      startButton.click();
    });

    await waitFor(() => {
      expect(mockAudioWorkletNode.port.postMessage).toHaveBeenCalled();
    });

    // Simulate receiving invalid audio data
    act(() => {
      mockAudioWorkletNode.port.onmessage?.({
        data: "invalid data",
      } as MessageEvent);
    });

    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith(
        "AudioRecorder: Received invalid audio data",
        "string",
        "invalid data",
      );
    });
  });
});
