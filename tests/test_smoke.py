"""The smallest possible test: it prints, and it checks what it printed."""


def test_hello(capsys):
    print("Hello, I am a test")
    assert capsys.readouterr().out == "Hello, I am a test\n"
