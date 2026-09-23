CREATE TABLE IF NOT EXISTS accounts (
    account_id          INTEGER PRIMARY KEY,
    signup_ts           INTEGER NOT NULL,
    ip                  TEXT    NOT NULL,
    asn                 INTEGER NOT NULL,
    device_fingerprint  TEXT    NOT NULL,
    email_domain        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    account_id    INTEGER NOT NULL REFERENCES accounts(account_id),
    ts            INTEGER NOT NULL,
    action        TEXT    NOT NULL,   -- login | api_call | message_send
    payload_hash  TEXT,               -- content hash for message_send
    target_id     INTEGER,            -- recipient for message_send
    ip            TEXT    NOT NULL
);

-- Ground-truth labels for the injected patterns (normal | farm | quota | spam_fanout | ato).
CREATE TABLE IF NOT EXISTS labels (
    account_id  INTEGER PRIMARY KEY REFERENCES accounts(account_id),
    label       TEXT NOT NULL
);

-- Synthetic ground truth for evaluation slices, never detector input.
CREATE TABLE IF NOT EXISTS account_metadata (
    account_id INTEGER PRIMARY KEY REFERENCES accounts(account_id),
    behavior TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_account_ts ON events(account_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_action     ON events(action);
