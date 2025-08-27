import grpc
import wave

from pathlib import Path
from typing import Iterator

# Use the class-based Indexer from the pipelines module
from pipelines.index_corpus import Indexer

# Import the generated protobuf code
from protos.generated.py import stt_pb2, stt_pb2_grpc

CHUNK_SIZE = 4096  # number of frames to read per chunk (not bytes)

def create_audio_chunks(audio_file: str) -> Iterator[stt_pb2.AudioChunk]:
    """
    Generator that yields audio chunks from a WAV file.

    Args:
        audio_file: Path to the WAV file.

    Yields:
        stt_pb2.AudioChunk: Audio data chunks to stream over gRPC.
    """
    with wave.open(audio_file, 'rb') as wav_file:
        while True:
            data = wav_file.readframes(CHUNK_SIZE)
            if not data:
                break
            yield stt_pb2.AudioChunk(data=data)

def transcribe_audio(audio_file: str, server_address: str = 'localhost:50051') -> None:
    """
    Transcribe audio using the gRPC streaming service and index the transcript.

    Args:
        audio_file: Path to the WAV file to transcribe.
        server_address: Address of the gRPC server.
    """
    text = ""
    with grpc.insecure_channel(server_address) as channel:
        # Mirror the original client stub and RPC method names
        stub = stt_pb2_grpc.TranscriptionServiceStub(channel)

        try:
            audio_stream = create_audio_chunks(audio_file)
            responses = stub.Transcribe(audio_stream)

            print(f"Transcribing {audio_file}...")
            # Initialize an Indexer to send transcript(s) to your index
            indexer = Indexer(collection="transcripts", corpus="", grpc_stub=None)
            
            for response in responses:
                # Adjust this if your proto field names differ
                if hasattr(response, "transcript") and hasattr(response.transcript, "text"):
                    text = response.transcript.text
                    print(f"Transcript: {text}")
                    # Keep the 'index_document' concept via the Indexer class
        except grpc.RpcError as e:
            print(f"RPC Error: {e.code()}: {e.details()}")
    indexer.index_document(doc_id=audio_file, content=text)


def main():
    # Example usage (update the path as needed)
    audio_file = "/Users/jamosharif/Developer/uzh-masterproject/packages/api/recordings/recording_20250820_141700.wav"
    if Path(audio_file).exists():
        transcribe_audio(audio_file)
    else:
        print(f"Audio file not found: {audio_file}")

if __name__ == "__main__":
    main()
