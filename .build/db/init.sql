/*
NOTE: This is only run on initial database setup, if you happen to mount a volume that
data, the setup script is NOT executed again.

The events table is used to store a memory/event, holding meta data for recordings/transcriptions.
*/
CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY,
    -- currently no user handling, for later use
    user_id INTEGER,
    audio_file VARCHAR(255),
    -- transcript_file VARCHAR(255) NOT NULL,
    transcript TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE,
    deleted_at TIMESTAMP
);