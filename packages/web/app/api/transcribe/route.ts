import { NextRequest } from "next/server";
import * as grpc from "@grpc/grpc-js";
import {
  MemoryChunk,
  ChunkMetadata,
  ChunkType,
  TranscriptionServiceClient,
} from "protos/generated/ts/clients/stt";

// Store active sessions with their gRPC streams
const sessions = new Map<
  string,
  {
    controller: ReadableStreamDefaultController;
    grpcStream: grpc.ClientDuplexStream<MemoryChunk, MemoryChunk>;
    audioBuffer: Uint8Array[];
    sessionType: ChunkType;
    memoryId: string;
    isFirstChunk: boolean;
  }
>();

// Create gRPC client
const grpcClient = new TranscriptionServiceClient(
  "localhost:50051",
  grpc.credentials.createInsecure(),
);

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const sessionId = searchParams.get("sessionId");
  const sessionTypeParam = searchParams.get("type");
  const memoryId = searchParams.get("memoryId") || crypto.randomUUID();

  if (!sessionId) {
    return new Response("Missing sessionId", { status: 400 });
  }

  const sessionType =
    sessionTypeParam === "question" ? ChunkType.QUESTION : ChunkType.MEMORY;

  // Create Server-Sent Events stream
  const stream = new ReadableStream({
    start(controller) {
      console.log(`Starting transcription session: ${sessionId}`);

      try {
        // Create bidirectional gRPC stream
        const grpcStream = grpcClient.transcribe();

        // Set a flag to track if we've confirmed the connection works
        let connectionConfirmed = false;

        // Set a timeout to detect if backend never responds
        const connectionTimeout = setTimeout(() => {
          if (!connectionConfirmed) {
            console.log("Backend connection timeout - no response received");
            const timeoutMessage = JSON.stringify({
              type: "error",
              message: "Backend service timeout. Please check if it's running.",
              timestamp: Date.now(),
            });
            try {
              controller.enqueue(`data: ${timeoutMessage}\n\n`);
            } catch (error) {
              console.log("Failed to enqueue timeout message:", error);
            }
            // Clean up
            grpcStream.destroy();
            sessions.delete(sessionId);
          }
        }, 10000); // 10 second timeout to match frontend

        // Handle responses from gRPC server
        grpcStream.on("data", (response: MemoryChunk) => {
          console.log("Received gRPC response:", response);

          // Check if controller is still open before trying to enqueue
          if (controller.desiredSize === null) {
            console.log("Controller already closed, ignoring response");
            return;
          }

          // Send connection confirmation on first successful response
          if (!connectionConfirmed) {
            connectionConfirmed = true;
            clearTimeout(connectionTimeout); // Clear the timeout since we got a response
            const connectMessage = JSON.stringify({
              type: "connected",
              sessionId,
              memoryId,
              timestamp: Date.now(),
            });
            try {
              controller.enqueue(`data: ${connectMessage}\n\n`);
            } catch (error) {
              console.log("Failed to enqueue connection message:", error);
            }
          }

          const responseType = response.metadata?.type;

          if (responseType === ChunkType.TRANSCRIPT && response.textData) {
            const message = JSON.stringify({
              type: "transcript",
              text: response.textData,
              timestamp: Date.now(),
            });

            try {
              controller.enqueue(`data: ${message}\n\n`);
            } catch (error) {
              console.log("Failed to enqueue transcript message:", error);
            }
          } else if (responseType === ChunkType.MEMORY && response.textData) {
            const message = JSON.stringify({
              type: "memory",
              text: response.textData,
              timestamp: Date.now(),
            });

            try {
              controller.enqueue(`data: ${message}\n\n`);
              console.log("Memory context chunk sent");
            } catch (error) {
              console.log("Failed to enqueue memory message:", error);
            }
          } else if (responseType === ChunkType.ANSWER && response.textData) {
            const message = JSON.stringify({
              type: "answer",
              text: response.textData,
              timestamp: Date.now(),
            });

            try {
              controller.enqueue(`data: ${message}\n\n`);
              console.log("Answer chunk sent");
              // Don't close the controller here - let the gRPC stream end event handle cleanup
            } catch (error) {
              console.log("Failed to enqueue answer message:", error);
            }
          }
        });

        grpcStream.on("error", (error: Error) => {
          console.error("gRPC stream error:", error);
          clearTimeout(connectionTimeout); // Clear timeout since we got an error response

          // If connection was never confirmed, this means backend is down
          const message = JSON.stringify({
            type: "error",
            message: connectionConfirmed
              ? "Backend connection lost during operation."
              : "Backend service unavailable. Please check if it's running.",
            timestamp: Date.now(),
          });
          try {
            controller.enqueue(`data: ${message}\n\n`);
          } catch (enqueueError) {
            console.log("Failed to enqueue error message:", enqueueError);
          }
        });

        grpcStream.on("end", () => {
          console.log("gRPC stream ended");
          clearTimeout(connectionTimeout); // Clear timeout since stream ended
          // Clean up the session
          sessions.delete(sessionId);
          try {
            if (controller.desiredSize !== null) {
              controller.close();
            }
          } catch (error: unknown) {
            if (
              !(
                error instanceof Error &&
                error.message.includes("already closed")
              )
            ) {
              console.error("Error closing controller:", error);
            }
          }
        });

        // Store the session
        sessions.set(sessionId, {
          controller,
          grpcStream,
          audioBuffer: [],
          sessionType,
          memoryId,
          isFirstChunk: true,
        });

        // Don't send connection message here - wait for first gRPC response
      } catch (error) {
        console.error("Error setting up gRPC stream:", error);
        const errorMessage = JSON.stringify({
          type: "error",
          message: "Failed to connect to transcription service",
          timestamp: Date.now(),
        });
        controller.enqueue(`data: ${errorMessage}\n\n`);
        controller.close();
      }
    },
    cancel() {
      // Cleanup when client disconnects
      console.log(`Cleaning up session: ${sessionId}`);
      const session = sessions.get(sessionId);
      if (session?.grpcStream) {
        session.grpcStream.end();
      }
      sessions.delete(sessionId);
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "Cache-Control",
    },
  });
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { sessionId } = body;

    if (!sessionId) {
      return new Response("Missing sessionId", { status: 400 });
    }

    const session = sessions.get(sessionId);
    if (!session) {
      return new Response("Session not found", { status: 404 });
    }

    // Check if this is a text or audio input
    if (body.textData !== undefined) {
      // Handle text input
      const { textData } = body;

      if (!textData) {
        return new Response("Empty text data", { status: 400 });
      }

      console.log(
        `Processing text input for session ${sessionId}: ${textData}`,
      );

      // Send text data to gRPC server
      if (session.grpcStream && !session.grpcStream.destroyed) {
        const metadata: ChunkMetadata = {
          sessionId,
          memoryId: session.memoryId,
          type: session.sessionType,
        };

        const textChunk: MemoryChunk = {
          textData,
          metadata: session.isFirstChunk ? metadata : undefined,
        };

        // Mark that we've sent the first chunk
        if (session.isFirstChunk) {
          session.isFirstChunk = false;
        }

        try {
          session.grpcStream.write(textChunk);
        } catch (writeError) {
          console.error("Error writing text to gRPC stream:", writeError);
          return new Response(
            JSON.stringify({
              success: false,
              error: "Stream already closed",
            }),
            {
              status: 500,
              headers: { "Content-Type": "application/json" },
            },
          );
        }

        return new Response(
          JSON.stringify({
            success: true,
            type: "text",
            textLength: textData.length,
          }),
          {
            headers: { "Content-Type": "application/json" },
          },
        );
      }
    } else if (body.audioData !== undefined) {
      // Handle audio input (existing functionality)
      const { audioData } = body;

      if (!audioData) {
        return new Response("Empty audio data", { status: 400 });
      }

      // Convert audio data back to Uint8Array
      const audioBytes = new Uint8Array(audioData);

      // Send audio chunk to gRPC server
      if (session.grpcStream && !session.grpcStream.destroyed) {
        const metadata: ChunkMetadata = {
          sessionId,
          memoryId: session.memoryId,
          type: session.sessionType,
        };

        const audioChunk: MemoryChunk = {
          audioData: audioBytes,
          metadata: session.isFirstChunk ? metadata : undefined,
        };

        // Mark that we've sent the first chunk
        if (session.isFirstChunk) {
          session.isFirstChunk = false;
        }

        try {
          session.grpcStream.write(audioChunk);
        } catch (writeError) {
          console.error("Error writing to gRPC stream:", writeError);
          return new Response(
            JSON.stringify({
              success: false,
              error: "Stream already closed",
            }),
            {
              status: 500,
              headers: { "Content-Type": "application/json" },
            },
          );
        }

        return new Response(
          JSON.stringify({
            success: true,
            type: "audio",
            bytesReceived: audioBytes.length,
          }),
          {
            headers: { "Content-Type": "application/json" },
          },
        );
      }
    } else {
      return new Response("Missing audioData or textData", { status: 400 });
    }

    if (!session.grpcStream || session.grpcStream.destroyed) {
      console.error("gRPC stream not available for session:", sessionId);
      return new Response(
        JSON.stringify({
          success: false,
          error: "Session ended or gRPC stream not available",
        }),
        {
          status: 410, // Gone - indicates the resource is no longer available
          headers: { "Content-Type": "application/json" },
        },
      );
    }

    console.error("Unexpected error - should not reach here");
    return new Response("Internal server error", { status: 500 });
  } catch (error) {
    console.error("Error processing input data:", error);
    return new Response(
      JSON.stringify({
        success: false,
        error: "Failed to process input data",
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
}

export async function DELETE(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const sessionId = searchParams.get("sessionId");

  if (!sessionId) {
    return new Response("Missing sessionId", { status: 400 });
  }

  const session = sessions.get(sessionId);
  if (session) {
    console.log(
      `Ending session ${sessionId} (${session.sessionType === ChunkType.QUESTION ? "question" : "memory"})`,
    );

    // For both session types, just end the gRPC stream to signal no more audio is coming
    // The cleanup will be handled by the gRPC stream 'end' event
    if (session.grpcStream && !session.grpcStream.destroyed) {
      session.grpcStream.end();
    }
    // Don't delete the session or close controller here - let the gRPC end event handle it
  }

  return new Response(JSON.stringify({ success: true }), {
    headers: { "Content-Type": "application/json" },
  });
}
