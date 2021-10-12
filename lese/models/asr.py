import torch
from speechbrain.pretrained import EncoderDecoderASR
from lese.models._base_lexical import BaseLexical


class AsrFeatExtractor(BaseLexical):
    def __init__(self, device='cuda'):
        super(AsrFeatExtractor, self).__init__()
        self.model = EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-crdnn-rnnlm-librispeech",
            savedir="pretrained_models/asr-crdnn-rnnlm-librispeech",
            run_opts = {'device': device}
        )

    def extract_feats(self, x, sr=16000):
        """ x - audio of shape [B, T] """
        lengths = torch.FloatTensor([xi.shape[-1] for xi in x])
        x = self.model.audio_normalizer(x, sr)
        x = self.model.encode_batch(x, lengths)
        return x


if __name__ == "__main__":
    x = torch.zeros(1, 16000)
    model = AsrFeatExtractor()
    yhat = model(x)
    print(yhat.shape)
