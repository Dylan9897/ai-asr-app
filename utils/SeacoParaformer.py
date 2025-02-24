# encoding : utf-8 -*-
# @author  : 冬瓜
# @mail    : dylan_han@126.com
# @Time    : 2025/2/17 14:29

import json
import numpy as np
import copy
import os


from typing import List,Union,Tuple
from pathlib import Path
from funasr_onnx import Paraformer
from funasr_onnx.utils.postprocess_utils import sentence_postprocess, sentence_postprocess_sentencepiece
from funasr_onnx.utils.timestamp_utils import time_stamp_lfr6_onnx
from funasr_onnx.utils.utils import pad_list,ONNXRuntimeError,read_yaml,TokenIDConverter,CharTokenizer,OrtInferSession,Hypothesis
from funasr_onnx.utils.frontend import WavFrontend

class SeacoParaformer(Paraformer):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
            self,
            model_dir: Union[str, Path] = None,
            batch_size: int = 16,
            device_id: Union[str, int] = "-1",
            plot_timestamp_to: str = "",
            quantize: bool = False,
            intra_op_num_threads: int = 4,
            cache_dir: str = None,
            **kwargs,
    ):

        if not Path(model_dir).exists():
            try:
                from modelscope.hub.snapshot_download import snapshot_download
            except:
                raise "You are exporting model from modelscope, please install modelscope and try it again. To install modelscope, you could:\n" "\npip3 install -U modelscope\n" "For the users in China, you could install with the command:\n" "\npip3 install -U modelscope -i https://mirror.sjtu.edu.cn/pypi/web/simple"
            try:
                model_dir = snapshot_download(model_dir, cache_dir=cache_dir)
            except:
                raise "model_dir must be model_name in modelscope or local path downloaded from modelscope, but is {}".format(
                    model_dir
                )

        if quantize:
            model_bb_file = os.path.join(model_dir, "model_quant.onnx")
            model_eb_file = os.path.join(model_dir, "model_eb_quant.onnx")
        else:
            model_bb_file = os.path.join(model_dir, "model.onnx")
            model_eb_file = os.path.join(model_dir, "model_eb.onnx")

        if not (os.path.exists(model_eb_file) and os.path.exists(model_bb_file)):
            print(".onnx does not exist, begin to export onnx")
            try:
                from funasr import AutoModel
            except:
                raise "You are exporting onnx, please install funasr and try it again. To install funasr, you could:\n" "\npip3 install -U funasr\n" "For the users in China, you could install with the command:\n" "\npip3 install -U funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple"

            model = AutoModel(model=model_dir)
            model_dir = model.export(type="onnx", quantize=quantize, **kwargs)

        config_file = os.path.join(model_dir, "config.yaml")
        cmvn_file = os.path.join(model_dir, "am.mvn")
        config = read_yaml(config_file)
        token_list = os.path.join(model_dir, "tokens.json")
        with open(token_list, "r", encoding="utf-8") as f:
            token_list = json.load(f)

        # revert token_list into vocab dict
        self.vocab = {}
        for i, token in enumerate(token_list):
            self.vocab[token] = i

        self.converter = TokenIDConverter(token_list)
        self.tokenizer = CharTokenizer()
        self.frontend = WavFrontend(cmvn_file=cmvn_file, **config["frontend_conf"])
        self.ort_infer_bb = OrtInferSession(
            model_bb_file, device_id, intra_op_num_threads=intra_op_num_threads
        )
        self.ort_infer_eb = OrtInferSession(
            model_eb_file, device_id, intra_op_num_threads=intra_op_num_threads
        )

        self.batch_size = batch_size
        self.plot_timestamp_to = plot_timestamp_to
        if "predictor_bias" in config["model_conf"].keys():
            self.pred_bias = config["model_conf"]["predictor_bias"]
        else:
            self.pred_bias = 0


    def __call__(
            self, wav_content: Union[str, np.ndarray, List[str], List[np.ndarray]], hotwords: str, **kwargs
    ) -> List:
        # make hotword list
        hotwords, hotwords_length = self.proc_hotword(hotwords)
        [bias_embed] = self.eb_infer(hotwords, hotwords_length)


        # index from bias_embed
        bias_embed = bias_embed.transpose(1, 0, 2)


        _ind = np.arange(0, len(hotwords)).tolist()
        bias_embed = bias_embed[_ind, hotwords_length.tolist()]


        # 在循环外部进行 expand_dims 和 repeat 操作
        bias_embed = np.expand_dims(bias_embed, axis=0)


        waveform_nums = len(wav_content)
        asr_res = []
        for beg_idx in range(0, waveform_nums, self.batch_size):
            end_idx = min(waveform_nums, beg_idx + self.batch_size)
            feats, feats_len = self.extract_feat(wav_content[beg_idx:end_idx])


            # 在循环内部重复 bias_embed
            repeated_bias_embed = np.repeat(bias_embed, feats.shape[0], axis=0)


            try:
                outputs = self.bb_infer(feats, feats_len, repeated_bias_embed)
                am_scores, valid_token_lens = outputs[0], outputs[1]

                if len(outputs) == 4:
                    # for BiCifParaformer Inference
                    us_alphas, us_peaks = outputs[2], outputs[3]
                else:
                    us_alphas, us_peaks = None, None

            except ONNXRuntimeError:
                preds = [""]
            else:
                preds = self.decode(am_scores, valid_token_lens)
                if us_peaks is None:
                    for pred in preds:
                        if self.language == "en-bpe":
                            pred = sentence_postprocess_sentencepiece(pred)
                        else:
                            pred = sentence_postprocess(pred)
                        asr_res.append({"preds": pred})
                else:
                    for pred, us_peaks_ in zip(preds, us_peaks):
                        raw_tokens = pred
                        timestamp, timestamp_raw = time_stamp_lfr6_onnx(
                            us_peaks_, copy.copy(raw_tokens)
                        )
                        text_proc, timestamp_proc, _ = sentence_postprocess(
                            raw_tokens, timestamp_raw
                        )
                        # logging.warning(timestamp)
                        if len(self.plot_timestamp_to):
                            self.plot_wave_timestamp(
                                wav_content[0], timestamp, self.plot_timestamp_to
                            )
                        asr_res.append(
                            {
                                "preds": text_proc,
                                "timestamp": timestamp_proc,
                                "raw_tokens": raw_tokens,
                            }
                        )
        return asr_res

    def proc_hotword(self, hotwords):
        hotwords = hotwords.split(" ")
        hotwords_length = [len(i) - 1 for i in hotwords]
        hotwords_length.append(0)
        hotwords_length = np.array(hotwords_length)

        # hotwords.append('<s>')
        def word_map(word):
            hotwords = []
            for c in word:
                if c not in self.vocab.keys():
                    hotwords.append(8403)
                else:
                    hotwords.append(self.vocab[c])
            return np.array(hotwords)

        hotword_int = [word_map(i) for i in hotwords]

        hotword_int.append(np.array([1]))
        hotwords = pad_list(hotword_int, pad_value=0, max_len=10)

        return hotwords, hotwords_length

    def bb_infer(
            self, feats: np.ndarray, feats_len: np.ndarray, bias_embed
    ) -> Tuple[np.ndarray, np.ndarray]:
        outputs = self.ort_infer_bb([feats, feats_len, bias_embed])
        return outputs

    def eb_infer(self, hotwords, hotwords_length):
        outputs = self.ort_infer_eb([hotwords.astype(np.int32), hotwords_length.astype(np.int32)])
        return outputs

    def decode(self, am_scores: np.ndarray, token_nums: int) -> List[str]:
        return [
            self.decode_one(am_score, token_num)
            for am_score, token_num in zip(am_scores, token_nums)
        ]

    def decode_one(self, am_score: np.ndarray, valid_token_num: int) -> List[str]:
        yseq = am_score.argmax(axis=-1)
        score = am_score.max(axis=-1)
        score = np.sum(score, axis=-1)


        yseq = np.array([1] + yseq.tolist() + [2])
        hyp = Hypothesis(yseq=yseq, score=score)


        last_pos = -1
        token_int = hyp.yseq[1:last_pos].tolist()


        token_int = list(filter(lambda x: x not in (0, 2), token_int))


        token = self.converter.ids2tokens(token_int)
        token = token[: valid_token_num - self.pred_bias]
        return token

