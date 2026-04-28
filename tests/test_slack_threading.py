from cosmere_rag.slack.threading import thread_key_for_event


def test_dm_uses_channel_id():
    event = {"channel": "D123", "ts": "1700000000.000100", "channel_type": "im"}
    assert thread_key_for_event(event) == "dm:D123"


def test_top_level_channel_mention_uses_event_ts():
    event = {"channel": "C123", "ts": "1700000000.000100", "channel_type": "channel"}
    assert thread_key_for_event(event) == "thread:1700000000.000100"


def test_threaded_channel_mention_uses_thread_ts():
    event = {
        "channel": "C123",
        "ts": "1700000000.000200",
        "thread_ts": "1700000000.000100",
        "channel_type": "channel",
    }
    assert thread_key_for_event(event) == "thread:1700000000.000100"


def test_dm_with_thread_ts_prefers_thread():
    # Slack rarely produces this, but if it does, treat the thread as primary.
    event = {
        "channel": "D123",
        "ts": "1700000000.000200",
        "thread_ts": "1700000000.000100",
        "channel_type": "im",
    }
    assert thread_key_for_event(event) == "thread:1700000000.000100"
