import { Dispatch, SetStateAction } from "react";

export type RecordingContextType = {
  isRecording: boolean;
  setIsRecording: Dispatch<SetStateAction<boolean>>;
  onStartRecording?: () => Promise<void>;
  onAudioData?: (audioData: Uint8Array) => Promise<void>;
  onStopRecording?: () => Promise<void>;
};
