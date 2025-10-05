import { createContext } from "react";
import { RecordingContextType } from "@/types/RecordingContextType";

export const RecordingContext = createContext<RecordingContextType | null>(
  null,
);

export default RecordingContext;
