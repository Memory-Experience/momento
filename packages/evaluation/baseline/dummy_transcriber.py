from api.models.transcription.transcriber_interface import TranscriberInterface


class DummyTranscriber(TranscriberInterface):
    """A dummy transcriber that returns a fixed transcription."""

    def __init__(self, transcription: str = "This is a dummy transcription."):
        self.transcription = transcription

    def initialize(self):
        """No initialization needed for the dummy transcriber."""
        pass

    def transcribe(self, audio: bytes) -> str:
        """Return the fixed transcription."""
        return self.transcription

    def cleanup(self):
        """No cleanup needed for the dummy transcriber."""
        pass
