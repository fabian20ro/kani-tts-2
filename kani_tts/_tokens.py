"""Shared token ID definitions for KaniTTS-2 backends."""


class TokenLayout:
    """
    Audio codec token layout.

    Computes special token IDs from a base tokenizer length.
    Shared between PyTorch (NemoAudioPlayer) and MLX (MLXAudioPlayer).
    """

    def __init__(self, tokeniser_length: int = 64400, codebook_size: int = 4032):
        self.tokeniser_length = tokeniser_length
        self.codebook_size = codebook_size
        self.start_of_speech = tokeniser_length + 1
        self.end_of_speech = tokeniser_length + 2
        self.start_of_human = tokeniser_length + 3
        self.end_of_human = tokeniser_length + 4
        self.start_of_ai = tokeniser_length + 5
        self.end_of_ai = tokeniser_length + 6
        self.pad_token = tokeniser_length + 7
        self.audio_tokens_start = tokeniser_length + 10
