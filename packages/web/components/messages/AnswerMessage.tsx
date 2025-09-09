import {
  useImperativeHandle,
  forwardRef,
  useEffect,
  useRef,
  useReducer,
} from "react";
import { MemoryChunk, ChunkType } from "protos/generated/ts/stt";
import { Card, CardContent } from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ChevronDown, ChevronRight } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface Memory {
  id: string;
  text: string;
  score: number;
}

interface AnswerState {
  memories: Memory[];
  thinkingOpen: boolean;
  thinkingText: string;
  isThinking: boolean;
  thinkingComplete: boolean;
  thinkingTime: number;
  answerText: string;
  isComplete: boolean;
}

enum ActionType {
  PROCESS_CHUNK = "PROCESS_CHUNK",
  MARK_COMPLETE = "MARK_COMPLETE",
  TOGGLE_THINKING = "TOGGLE_THINKING",
}

type Action =
  | {
      type: ActionType.PROCESS_CHUNK;
      payload: { chunk: MemoryChunk; startTime: number };
    }
  | { type: ActionType.MARK_COMPLETE; payload: { startTime: number } }
  | { type: ActionType.TOGGLE_THINKING };

const initialState: AnswerState = {
  memories: [],
  thinkingOpen: false,
  thinkingText: "",
  isThinking: false,
  thinkingComplete: false,
  thinkingTime: 0,
  answerText: "",
  isComplete: false,
};

function reducer(state: AnswerState, action: Action): AnswerState {
  switch (action.type) {
    case ActionType.PROCESS_CHUNK: {
      const { chunk } = action.payload;
      const newState = { ...state };

      if (chunk.metadata?.type === ChunkType.MEMORY && chunk.textData) {
        newState.memories = [
          ...newState.memories,
          {
            id: chunk.metadata?.memoryId || "",
            text: chunk.textData!,
            score: /*chunk.metadata?.score || */ 0.99, // Using placeholder score
          },
        ];
      }

      if (chunk.metadata?.type === ChunkType.ANSWER) {
        let textToProcess = chunk.textData || "";

        const appendToAnswer = (text: string) => {
          if (!text) return;
          // Only add a space if there's existing text.
          const prefix = newState.answerText ? " " : "";
          newState.answerText += prefix + text;
        };

        const appendToThinking = (text: string) => {
          if (!text) return;
          // Only add a space if there's existing text.
          const prefix = newState.thinkingText ? " " : "";
          newState.thinkingText += prefix + text;
        };

        while (textToProcess.length > 0) {
          if (newState.isThinking && !newState.thinkingComplete) {
            const thinkEndIndex = textToProcess.indexOf("</think>");
            if (thinkEndIndex !== -1) {
              appendToThinking(textToProcess.substring(0, thinkEndIndex));
              newState.thinkingComplete = true;
              newState.isThinking = false;
              newState.thinkingTime =
                (Date.now() - action.payload.startTime) / 1000;
              textToProcess = textToProcess.substring(
                thinkEndIndex + "</think>".length,
              );
            } else {
              appendToThinking(textToProcess);
              textToProcess = "";
            }
          } else {
            const thinkStartIndex = textToProcess.indexOf("<think>");
            if (thinkStartIndex !== -1) {
              const beforeText = textToProcess.substring(0, thinkStartIndex);
              appendToAnswer(beforeText);
              newState.isThinking = true;
              action.payload.startTime = Date.now(); // Reset start time for thinking
              textToProcess = textToProcess.substring(
                thinkStartIndex + "<think>".length,
              );
            } else {
              appendToAnswer(textToProcess);
              textToProcess = "";
            }
          }
        }
      }

      if (chunk.metadata?.isFinal) {
        newState.isComplete = true;
        if (newState.isThinking && !newState.thinkingComplete) {
          newState.thinkingComplete = true;
          newState.isThinking = false;
          newState.thinkingTime =
            (Date.now() - action.payload.startTime) / 1000;
        }
      }
      return newState;
    }
    case ActionType.MARK_COMPLETE: {
      const newState = { ...state, isComplete: true };
      if (newState.isThinking && !newState.thinkingComplete) {
        newState.thinkingComplete = true;
        newState.isThinking = false;
        newState.thinkingTime = (Date.now() - action.payload.startTime) / 1000;
      }
      return newState;
    }
    case ActionType.TOGGLE_THINKING:
      return { ...state, thinkingOpen: !state.thinkingOpen };
    default:
      return state;
  }
}

export interface AnswerMessageHandle {
  processChunk: (chunk: MemoryChunk) => void;
  markComplete: () => void;
}

interface AnswerMessageProps {
  onComplete?: () => void;
}

const AnswerMessage = forwardRef<AnswerMessageHandle, AnswerMessageProps>(
  function AnswerMessage({ onComplete }, ref) {
    const [state, dispatch] = useReducer(reducer, initialState);
    const thinkingStartTime = useRef<number | null>(null);

    useEffect(() => {
      if (state.isComplete && onComplete) {
        onComplete();
      }
    }, [state.isComplete, onComplete]);

    useEffect(() => {
      if (state.isThinking && thinkingStartTime.current === null) {
        thinkingStartTime.current = Date.now();
      }
      if (!state.isThinking && thinkingStartTime.current !== null) {
        thinkingStartTime.current = null;
      }
    }, [state.isThinking]);

    useImperativeHandle(ref, () => ({
      processChunk: (chunk: MemoryChunk) => {
        dispatch({
          type: ActionType.PROCESS_CHUNK,
          payload: {
            chunk,
            startTime: thinkingStartTime.current || Date.now(),
          },
        });
      },
      markComplete: () => {
        dispatch({
          type: ActionType.MARK_COMPLETE,
          payload: { startTime: thinkingStartTime.current || Date.now() },
        });
      },
    }));

    const PulsatingThinking = () => (
      <div className="flex items-center text-gray-400">
        <span className="animate-pulse">Thinking...</span>
      </div>
    );

    const ThoughtHeader = () => (
      <div className="flex items-center text-gray-400">
        <span>Thought</span>
        <span className="ml-2 text-xs text-gray-500">
          ({state.thinkingTime.toFixed(1)}s)
        </span>
      </div>
    );

    return (
      <Card className="w-full">
        <CardContent className="p-4">
          {/* Main content container with relative positioning */}
          <div className="relative">
            {/* Memories shown on the right side regardless of thinking state */}
            {state.memories.length > 0 && (
              <div className="absolute right-0 top-0 z-10 flex flex-row-reverse flex-wrap gap-2 max-w-[60%] justify-end">
                <TooltipProvider>
                  {state.memories.map((mem) => (
                    <Tooltip key={mem.id}>
                      <TooltipTrigger asChild>
                        <div className="rounded-full bg-primary/10 px-3 py-1 text-xs text-primary/80 transition-colors hover:bg-primary/20">
                          <p className="truncate max-w-[150px]">{mem.text}</p>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>
                          Retrieved memory deemed relevant with a matching score
                          of: {mem.score.toFixed(2)}
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  ))}
                </TooltipProvider>
              </div>
            )}

            {/* Thinking section - use full width with dynamic spacing for memories */}
            {(state.isThinking || state.thinkingText) && (
              <div
                className={`mb-4 ${state.memories.length > 0 ? "pr-4" : "w-full"}`}
              >
                <Collapsible
                  open={state.thinkingOpen}
                  onOpenChange={() =>
                    dispatch({ type: ActionType.TOGGLE_THINKING })
                  }
                  className="w-full"
                >
                  <CollapsibleTrigger className="flex w-full items-center justify-between text-sm">
                    <div className="flex items-center">
                      {state.thinkingOpen ? (
                        <ChevronDown className="mr-2 h-4 w-4" />
                      ) : (
                        <ChevronRight className="mr-2 h-4 w-4" />
                      )}
                      {state.thinkingComplete ? (
                        <ThoughtHeader />
                      ) : (
                        <PulsatingThinking />
                      )}
                    </div>
                  </CollapsibleTrigger>

                  <CollapsibleContent className="prose prose-sm mt-2 max-w-none w-full rounded-md border border-transparent bg-transparent p-2 text-gray-500">
                    {state.thinkingText}
                  </CollapsibleContent>
                </Collapsible>
              </div>
            )}

            {state.answerText && (
              <div className="prose prose-invert max-w-none mt-4">
                {state.answerText}
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    );
  },
);

export default AnswerMessage;
