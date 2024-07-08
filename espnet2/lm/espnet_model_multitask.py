from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.lm.abs_model import AbsLM
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask, pad_list, th_accuracy
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)

import logging
import time
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter


class ESPnetMultitaskLanguageModel(AbsESPnetModel):
    @typechecked
    def __init__(
        self,
        lm: AbsLM,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        ignore_id: int = 0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        sos_syms: List[str] = ["<generatetext>", "<generatespeech>"],
        eos_sym: str = "<sos/eos>",
        bpemodel: str = "bpemodel",
    ):
        super().__init__()
        self.lm = lm
        self.sos_ids = [token_list.index(t) for t in sos_syms]
        self.eos_id = token_list.index(eos_sym)

        self.start_cond_ids = [
            token_list.index(t) for t in ["<startoftext>", "<startofspeech>"]
        ]

        # ignore_id may be assumed as 0, shared with CTC-blank symbol for ASR.
        self.ignore_id = ignore_id

        self.token_list = token_list.copy()

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        self.bpemodel = bpemodel
        self.tokenizer = build_tokenizer(
            token_type="bpe",
            bpemodel=bpemodel,
        )
        self.converter = TokenIDConverter(
            token_list=token_list,
            unk_symbol="<unk>",
        )

        logging.info(
            "Using bpemodel {} tokenizer: {} token_id_converter {}".format(
                bpemodel, self.tokenizer, self.converter
            )
        )

    def nll(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        prefix: Optional[torch.Tensor] = None,
        prefix_lengths: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute negative log likelihood (nll)

        NOTE(yifan): We only use nll to calculate perplexity,
            so there is no condition in each sentence.

        Normally, this function is called in batchify_nll.
        Args:
            text: (Batch, Length)
            text_lengths: (Batch,)
            max_lengths: int
        """
        assert max_length is None

        batch_size = text.size(0)
        # For data parallel
        if max_length is None:
            text = text[:, : text_lengths.max()]
        else:
            text = text[:, :max_length]

        # NOTE(yifan): The first token is space when using bpe
        text = text[:, 1:]
        text_lengths = text_lengths - 1

        # 1. Create a sentence pair like '<sos> w1 w2 w3' and 'w1 w2 w3 <eos>'
        # text: (Batch, Length) -> x, y: (Batch, Length + 1)
        x, x_lengths = text, text_lengths  # text already has <sos>
        t = F.pad(text, [0, 1], "constant", self.ignore_id)
        for i, l in enumerate(text_lengths):
            t[i, l] = self.eos_id
        t = t[:, 1:]  # remove <sos>

        # 2. Forward Language model
        # x: (Batch, Length) -> y: (Batch, Length, NVocab)
        y, _ = self.lm(x, None)

        if prefix is not None:
            acc = th_accuracy(
                y[:, prefix.shape[0] :].view(-1, y.shape[-1]),
                t[:, prefix.shape[0] :],
                ignore_label=self.ignore_id,
            )
        else:
            acc = th_accuracy(
                y.view(-1, y.shape[-1]),
                t,
                ignore_label=self.ignore_id,
            )

        # 3. Calc negative log likelihood
        # nll: (BxL,)
        nll = F.cross_entropy(y.view(-1, y.shape[-1]), t.view(-1), reduction="none")
        # nll: (BxL,) -> (BxL,)
        if max_length is None:
            nll.masked_fill_(make_pad_mask(x_lengths).to(nll.device).view(-1), 0.0)
        else:
            raise NotImplementedError
        # nll: (BxL,) -> (B, L)
        nll = nll.view(batch_size, -1)
        return nll, x_lengths, acc

    def batchify_nll(
        self, text: torch.Tensor, text_lengths: torch.Tensor, batch_size: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute negative log likelihood(nll) from transformer language model

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            text: (Batch, Length)
            text_lengths: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase

        """
        total_num = text.size(0)
        if total_num <= batch_size:
            nll, x_lengths, _ = self.nll(text, text_lengths)
        else:
            nlls = []
            x_lengths = []
            max_length = text_lengths.max()

            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_text = text[start_idx:end_idx, :]
                batch_text_lengths = text_lengths[start_idx:end_idx]
                # batch_nll: [B * T]
                batch_nll, batch_x_lengths, _ = self.nll(
                    batch_text, batch_text_lengths, max_length=max_length
                )
                nlls.append(batch_nll)
                x_lengths.append(batch_x_lengths)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nlls)
            x_lengths = torch.cat(x_lengths)
        assert nll.size(0) == total_num
        assert x_lengths.size(0) == total_num
        return nll, x_lengths

    def _calc_att_loss(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        # NOTE(yifan): The first token is space when using bpe
        # text = text[:, 1:]
        # text_lengths = text_lengths - 1

        # 1. Prepare input and target
        input = pad_list(
            [t[:t_len] for t, t_len in zip(text, text_lengths)], self.eos_id
        )

        target = []
        for cur_text, cur_text_len in zip(text, text_lengths):
            cur_text = cur_text[:cur_text_len]
            # mask out the condition text
            for sos in self.sos_ids:
                if sos in cur_text:
                    cur_text[: (cur_text == sos).nonzero()[0][0] + 1] = self.ignore_id
                    break
            cur_text = cur_text[1:]  # left shift
            cur_text = F.pad(cur_text, (0, 1), value=self.eos_id)  # add eos
            target.append(cur_text)
        target = pad_list(target, self.ignore_id)

        # 2. Compute attention loss
        pred, _ = self.lm(input, None)

        loss = self.criterion_att(pred, target)
        acc = th_accuracy(
            pred.view(-1, pred.shape[-1]),
            target,
            ignore_label=self.ignore_id,
        )

        return loss, acc

    def gen_sequence(
        self,
        prefix: torch.Tensor,
        prefix_lengths: torch.Tensor,
        max_gen_length: int = 100,
        use_grad=False,
    ):

        states = [None]

        # Generate
        # prefix: Shape B x seq_len+max_gen_length

        for i in range(max_gen_length):
            if use_grad:
                scores, states = self.lm.batch_score(prefix, states, None)
            else:
                with torch.no_grad():
                    scores, states = self.lm.batch_score(prefix, states, None)
            prefix = torch.cat(
                (prefix, torch.argmax(scores, dim=-1, keepdim=True)), dim=1
            )

        prefix_lengths = prefix_lengths + max_gen_length
        return prefix, prefix_lengths

    # text_gen has input text and generates speech
    def text_gen(
        self,
        text_gen: torch.Tensor,
        text_gen_lengths: torch.Tensor,
        max_gen_length: int = 250,
    ):

        batch_size = text_gen.shape[0]
        seq_length = text_gen.shape[1]

        # Generate tokens
        text_gen, text_gen_lengths = self.gen_sequence(
            text_gen, text_gen_lengths, max_gen_length
        )

        # Cycle - consistency reformat
        # Input format:  text, token
        # Output format: token, text
        text_prefix = text_gen[:, :seq_length].clone()  # Shape B x seq_len
        token_pseudo = text_gen[
            :, seq_length:
        ].clone()  # token_pseudo: Shape B x max_gen_length

        # Reformat tokens to reverse the task as speech-to-text
        # 1. add new token <startofspeech>, start_cond_ids[1] --> index of "<startofspeech>"
        # 2. change <startoftext> --> <generatetext>
        # 3. Remove <generatespeech> with <eos>, sos_ids[1] is index of "<generatespeech>"
        start_token_tag = (
            torch.ones((batch_size, 1)).to(device=text_gen.device)
        ) * self.start_cond_ids[1]
        text_gen[:, 0:1] = start_token_tag
        text_prefix = torch.where(
            text_prefix == self.start_cond_ids[0], self.sos_ids[0], text_prefix
        )
        text_prefix = torch.where(
            text_prefix == self.sos_ids[1], self.eos_id, text_prefix
        )

        # Swap for output format
        text_gen[:, 1 : (max_gen_length + 1)] = token_pseudo[:, :]
        text_gen[:, (max_gen_length + 1) :] = text_prefix[:, :-1]  # Remove last token

        # Generated sequence can be of different length, to address that
        # First find first <eos> in every row as eos_index
        # Store prefix from eos_index+1
        eos_lengths = []
        # Move prefix to after first eos
        for i in range(batch_size):
            idx = torch.arange(max_gen_length, 0, -1).to(device=text_gen.device)
            token_pseudo2 = (token_pseudo[i] == self.eos_id) * idx
            eos_index = torch.argmax(token_pseudo2)
            eos_index = torch.where(
                token_pseudo2.sum() == 0, (max_gen_length - 1), eos_index
            )

            text_gen[i, (eos_index + 1) : (eos_index + seq_length + 1)] = text_prefix[
                i, :
            ]

            # clean all indices
            eos_lengths.append((eos_index + seq_length + 1))

        # Pad end of sequence with eos_id
        text_gen.masked_fill_(make_pad_mask(eos_lengths, text_gen), self.eos_id)

        # compute loss for the second iteration
        return self._calc_att_loss(text_gen, text_gen_lengths)

    # speech_gen has input speech and generates text
    def speech_gen(
        self,
        speech_gen: torch.Tensor,
        speech_gen_lengths: torch.Tensor,
        max_gen_length: int = 60,  # 300, # 6 sec of speech
    ):

        batch_size = speech_gen.shape[0]
        seq_length = speech_gen.shape[1]

        # Generate tokens
        speech_gen, speech_gen_lengths = self.gen_sequence(
            speech_gen, speech_gen_lengths, max_gen_length
        )

        # Cycle - consistency reformat
        # Input format:  speech, text
        # Output format: text, token,
        speech_prefix = speech_gen[:, :seq_length].clone()  # Shape B x seq_len
        text_pseudo = speech_gen[
            :, seq_length:
        ].clone()  # token_pseudo: Shape B x max_gen_length

        # Reformat tokens to reverse the task as text-to-speech
        # 1. add new token <startoftext>, start_cond_ids[0] --> index of "<startoftext>"
        # 2. change <startofspeech> --> <generatespeech>
        # 3. Remove <generatetext> with <eos>, sos_ids[0] is index of "<generatetext>"
        start_token_tag = (
            torch.ones((batch_size, 1)).to(device=speech_gen.device)
        ) * self.start_cond_ids[0]
        speech_gen[:, 0:1] = start_token_tag
        speech_prefix = torch.where(
            speech_prefix == self.start_cond_ids[1], self.sos_ids[1], speech_prefix
        )
        speech_prefix = torch.where(
            speech_prefix == self.sos_ids[0], self.eos_id, speech_prefix
        )

        # Generated sequence can be of different length, to address that
        # First find first <eos> in every row as eos_index
        # Store prefix from eos_index+1
        eos_lengths = []
        # Move prefix to after first eos
        for i in range(batch_size):
            idx = torch.arange(max_gen_length, 0, -1).to(device=speech_gen.device)
            text_pseudo2 = (text_pseudo[i] == self.eos_id) * idx
            eos_index = torch.argmax(text_pseudo2)
            # Set to last index, if eos was not generated
            eos_index = torch.where(
                text_pseudo2.sum() == 0, (max_gen_length - 1), eos_index
            )

            speech_gen[i, (eos_index + 1) : (eos_index + seq_length + 1)] = (
                speech_prefix[i, :]
            )

            # clean all indices
            eos_lengths.append((eos_index + seq_length + 1))

        # Pad end of sequence with eos_id
        speech_gen.masked_fill_(make_pad_mask(eos_lengths, speech_gen), self.eos_id)

        return self._calc_att_loss(speech_gen, speech_gen_lengths)

    def forward_no_cycle(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        batch_size = text.shape[0]
        loss, acc = self._calc_att_loss(text, text_lengths)
        stats = dict(
            loss=loss.detach(),
            acc=acc,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        text_taskid: Optional[torch.Tensor] = None,
        text_taskid_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        if text_taskid == None:
            # task id missing - no cycle generation is needed
            return self.forward_no_cycle(text, text_lengths)

        batch_size = text.shape[0]

        # If taskid is present
        # select three types of data
        # text --> text_og, text_gen, speech_gen
        # text_og - text lm data - no generation is needed (taskid=0)
        # text_gen - text data only, run two iteration: text-> speech and speech-> text (taskid=1)
        # speech_gen - speech data only, run two iteration: speech-> text and text-> speech (taskid=2)
        text_og = text[torch.where(text_taskid == 0)[0]]
        text_og_lengths = text_lengths[torch.where(text_taskid == 0)[0]]
        text_gen = text[torch.where(text_taskid == 1)[0]]
        text_gen_lengths = text_lengths[torch.where(text_taskid == 1)[0]]
        speech_gen = text[torch.where(text_taskid == 2)[0]]
        speech_gen_lengths = text_lengths[torch.where(text_taskid == 2)[0]]

        if text_og.shape[0] > 0:
            loss, acc = self._calc_att_loss(text_og, text_og_lengths)
        else:
            loss = 0
            acc = 0

        if text_gen.shape[0] > 0:
            loss_text_gen, acc_text_gen = self.text_gen(text_gen, text_gen_lengths)
            loss = loss + loss_text_gen
            acc = acc + acc_text_gen

        if speech_gen.shape[0] > 0:
            loss_speech_gen, acc_speech_gen = self.speech_gen(
                speech_gen, speech_gen_lengths
            )
            loss = loss + loss_speech_gen
            acc = acc + acc_speech_gen

        stats = dict(
            loss=loss.detach(),
            acc=acc,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        return {}
