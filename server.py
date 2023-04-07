import asyncio
import logging
import re
from logging.config import fileConfig

import torch
import websockets
from torch import no_grad
from torch.cuda import LongTensor

import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence

fileConfig('logging_config.ini')
logging.getLogger('numba').setLevel(logging.WARNING)
device = torch.device('cuda')


def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def get_label_value(text, label, default, warning_name='value'):
    value = re.search(rf'\[{label}=(.+?)\]', text)
    if value:
        try:
            text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
            value = float(value.group(1))
        except:
            print(f'Invalid {warning_name}!')
    else:
        value = default
    return value, text


def get_label(text, label):
    if f'[{label}]' in text:
        return True, text.replace(f'[{label}]', '')
    else:
        return False, text


model = 'model/1026_epochs.pth'
config = 'model/1026_config.json'

hps_ms = utils.get_hparams_from_file(config)
n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
speakers = hps_ms.speakers if 'speakers' in hps_ms.keys() else ['0']
use_f0 = hps_ms.data.use_f0 if 'use_f0' in hps_ms.data.keys() else False

net_g_ms = SynthesizerTrn(
    n_symbols,
    hps_ms.data.filter_length // 2 + 1,
    hps_ms.train.segment_size // hps_ms.data.hop_length,
    n_speakers=n_speakers,
    emotion_embedding=False,
    device=device,
    **hps_ms.model)
_ = net_g_ms.eval()
utils.load_checkpoint(model, net_g_ms)


async def handler(websocket):
    while True:
        text: str = await websocket.recv()
        text = text.strip()
        if len(text) <= 0:
            continue

        speaker_id = 2

        suffix = '[ZH]'
        for i in range(0, 2):
            if '\u3040' <= text[i] <= '\u309f' or '\u30a0' <= text[i] <= '\u30ff':
                suffix = '[JA]'
                break

        text = suffix + text + suffix

        print(text)
        length_scale, text = get_label_value(text, 'LENGTH', 1.35, 'length scale')
        noise_scale, text = get_label_value(text, 'NOISE', 0.3, 'noise scale')
        noise_scale_w, text = get_label_value(text, 'NOISEW', 0.8, 'deviation of noise')
        cleaned, text = get_label(text, 'CLEANED')

        stn_tst = get_text(text, hps_ms, cleaned=cleaned)

        with no_grad():
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = LongTensor([stn_tst.size(0)])
            sid = LongTensor([speaker_id])
            audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                   noise_scale_w=noise_scale_w, length_scale=length_scale)[0][
                0, 0].data.cpu().float().numpy()

        print(hps_ms.data.sampling_rate)

        await websocket.send((0).to_bytes(1, "big", signed=False))
        await websocket.send(hps_ms.data.sampling_rate.to_bytes(4, "big", signed=False))
        await websocket.send(audio.tobytes())
        await websocket.send((1).to_bytes(1, "big", signed=False))

        print('Successfully send!')


async def main():
    async with websockets.serve(handler, "", 2235):
        await asyncio.Future()  # run forever


if __name__ == '__main__':
    asyncio.run(main())
