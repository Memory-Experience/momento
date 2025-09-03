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
    sessionType: ChunkType;
    memoryId: string;
    isFirstChunk: boolean;
    isClosing: boolean;
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

  // Check if this is an initialization session
  const isInitSession = sessionId.startsWith("init_");

  // Create Server-Sent Events stream
  const stream = new ReadableStream({
    start(controller) {
      console.log(
        `Starting ${isInitSession ? "initialization" : "recording"} session: ${sessionId}`,
      );

      try {
        // Send immediate connection confirmation
        const connectMessage = JSON.stringify({
          type: "connected",
          sessionId,
          timestamp: Date.now(),
        });

        controller.enqueue(`data: ${connectMessage}\n\n`);

        // For initialization sessions, we don't need a gRPC stream
        if (isInitSession) {
          // Just close the stream after confirmation
          setTimeout(() => controller.close(), 200);
          return;
        }

        // For actual recording sessions, create gRPC stream
        const grpcStream = grpcClient.transcribe();

        // Handle responses from gRPC server
        grpcStream.on("data", (response: MemoryChunk) => {
          if (controller.desiredSize === null) return; // Stream closed

          // Send response to client
          const jsonData = MemoryChunk.toJSON(response);
          controller.enqueue(`data: ${JSON.stringify(jsonData)}\n\n`);
        });

        grpcStream.on("error", (error) => {
          console.error("gRPC error:", error);
          if (controller.desiredSize === null) return; // Stream closed

          controller.enqueue(
            `data: ${JSON.stringify({
              type: "error",
              message: "Backend error occurred",
            })}\n\n`,
          );
        });

        grpcStream.on("end", () => {
          console.log("gRPC stream ended");
          const session = sessions.get(sessionId);
          if (session && !session.isClosing) {
            session.isClosing = true;
            sessions.delete(sessionId);
            if (controller.desiredSize !== null) {
              try {
                controller.close();
              } catch (err) {
                console.warn("Failed to close controller on stream end:", err);
              }
            }
          }
        });

        // Store session
        sessions.set(sessionId, {
          controller,
          grpcStream,
          sessionType,
          memoryId,
          isFirstChunk: true,
          isClosing: false,
        });
      } catch (error) {
        console.error("Error creating session:", error);
        controller.enqueue(
          `data: ${JSON.stringify({
            type: "error",
            message: "Failed to create session",
          })}\n\n`,
        );
        controller.close();
      }
    },

    cancel() {
      console.log(`Canceling session: ${sessionId}`);
      const session = sessions.get(sessionId);
      if (!session) return;

      // Mark as closing to prevent double-close attempts
      session.isClosing = true;

      if (session.grpcStream && !session.grpcStream.destroyed) {
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
    const { sessionId, audioData, textData } = body;

    if (!sessionId) {
      return new Response("Missing sessionId", { status: 400 });
    }

    const session = sessions.get(sessionId);
    if (!session) {
      return new Response("Session not found", { status: 404 });
    }

    if (!session.grpcStream || session.grpcStream.destroyed) {
      return new Response(
        JSON.stringify({
          success: false,
          error: "Session closed",
        }),
        {
          status: 410,
          headers: { "Content-Type": "application/json" },
        },
      );
    }

    const metadata: ChunkMetadata = {
      sessionId,
      memoryId: session.memoryId,
      type: session.sessionType,
    };

    try {
      if (textData !== undefined) {
        // Text input
        const textChunk = {
          textData,
          metadata: session.isFirstChunk ? metadata : undefined,
        };

        session.grpcStream.write(textChunk);
      } else if (audioData !== undefined) {
        // Audio input
        const audioChunk = {
          audioData: new Uint8Array(audioData),
          metadata: session.isFirstChunk ? metadata : undefined,
        };

        session.grpcStream.write(audioChunk);
      } else {
        return new Response("Missing audioData or textData", { status: 400 });
      }

      // Mark that first chunk was sent
      if (session.isFirstChunk) {
        session.isFirstChunk = false;
      }

      return new Response(JSON.stringify({ success: true }), {
        headers: { "Content-Type": "application/json" },
      });
    } catch (writeError) {
      console.warn(writeError);
      return new Response(
        JSON.stringify({
          success: false,
          error: "Error writing to stream",
        }),
        {
          status: 500,
          headers: { "Content-Type": "application/json" },
        },
      );
    }
  } catch (error) {
    console.warn(error);
    return new Response(
      JSON.stringify({
        success: false,
        error: "Server error",
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

  // For initialization sessions, just return success
  if (sessionId.startsWith("init_")) {
    return new Response(JSON.stringify({ success: true }), {
      headers: { "Content-Type": "application/json" },
    });
  }

  const session = sessions.get(sessionId);
  if (session) {
    // End the gRPC stream
    if (!session.grpcStream.destroyed) {
      session.grpcStream.end();
    }

    // Remove the session
    sessions.delete(sessionId);
  }

  return new Response(JSON.stringify({ success: true }), {
    headers: { "Content-Type": "application/json" },
  });
}
