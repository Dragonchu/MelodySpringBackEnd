import numpy as np


def convert_to_16_bit_wav(data):
    # Based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
    if data.dtype in [np.float64, np.float32, np.float16]:
        data = data / np.abs(data).max()
        data = data * 32767
        data = data.astype(np.int16)
    elif data.dtype == np.int32:
        data = data / 65536
        data = data.astype(np.int16)
    elif data.dtype == np.int16:
        pass
    elif data.dtype == np.uint16:
        data = data - 32768
        data = data.astype(np.int16)
    elif data.dtype == np.uint8:
        data = data * 257 - 32768
        data = data.astype(np.int16)
    elif data.dtype == np.int8:
        data = data * 256
        data = data.astype(np.int16)
    else:
        raise ValueError(
            "Audio data cannot be converted automatically from "
            f"{data.dtype} to 16-bit int format."
        )
    return data
