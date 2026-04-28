from cosmere_rag.slack.text import strip_bot_mention


def test_strip_leading_mention():
    assert strip_bot_mention("<@UBOT> who is Kelsier?", "UBOT") == "who is Kelsier?"


def test_strip_mention_with_label():
    assert (
        strip_bot_mention("<@UBOT|cosmerebot> who is Kelsier?", "UBOT")
        == "who is Kelsier?"
    )


def test_strip_mid_mention():
    assert strip_bot_mention("hey <@UBOT> what now?", "UBOT") == "hey what now?"


def test_only_mention_returns_empty():
    assert strip_bot_mention("<@UBOT>", "UBOT") == ""


def test_other_user_mention_preserved():
    assert (
        strip_bot_mention("<@UBOT> say hi to <@UALICE>", "UBOT")
        == "say hi to <@UALICE>"
    )
