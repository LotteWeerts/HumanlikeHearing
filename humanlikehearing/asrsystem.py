from .sound import Sound

import warnings
import numpy as np
import os
import time

try:
    from deepspeech import Model as DPModel
except ModuleNotFoundError:
    warnings.warn("Missing some of the required libraries for running DeepSpeech.", UserWarning)

try:
    import json
    import vosk
    from library.audio_format import float_to_byte
except ModuleNotFoundError:
    warnings.warn("Missing some of the required libraries for running Vosk.", UserWarning)

try:
    import argparse
    import torch
    from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder
    from fairseq import checkpoint_utils
    from fairseq.data import Dictionary
    from omegaconf.errors import MissingMandatoryValue
except ModuleNotFoundError:
    warnings.warn("Missing some of the required libraries for running Wav2vec 2.0.", UserWarning)

class ASRSystem(object):
    """
    Parent class for all automatic speech recognition systems.

    Dependants of ASRSystem must implement self.transcribe(sound_or_path) 
    and set self.samplerate_hz (the expected samplerate of input sounds for the ASR system)
    on initialisation.
    """

    def __init__(self, model_path, identifier=None):
        self.__samplerate_hz = None
        self.__samplerate_hz_is_set = False
        if identifier is None:
            self.id = int(time.time())
        else:
            self.id = identifier
        self.model_path = model_path

    def transcribe(self, sound_or_path):
        """
        Transcribe the given audiofile (path to file or array type).
        """
        raise NotImplementedError()

    def _load_sound(self, sound_or_path):
        """
        Loads a sound from file path or directly array into a Sound object.
        
        Sounds are always resampled to self.samplerate_Hz (usually 16kHz, but this may differ
        per ASR system) to ensure the input sound is at the samplerate that is expected by the 
        ASR system.
        """
        if isinstance(sound_or_path, str):
            sound = Sound(sound_or_path, samplerate_Hz=self.samplerate_hz)
        else:
            sound = sound_or_path
            if self.samplerate_hz:
                sound = Sound.resample(sound, self.samplerate_hz)
        return sound

    @property
    def samplerate_hz(self):
        """The sample rate in Hz the ASRsystem expects the input sound to be sampled at."""
        if not self.__samplerate_hz_is_set:
            raise NotImplementedError('Descendants from ASRSystem must set samplerate_hz.')
        else:
            return self.__samplerate_hz

    @samplerate_hz.setter
    def samplerate_hz(self, value):
        self.__samplerate_hz = value
        self.__samplerate_hz_is_set = True

    @classmethod
    def create(cls, asrsystem_name, model_path):
        """Creates an instance of asrsystem_name using the model path."""
        if asrsystem_name == 'MozillaDeepSpeech':
            return MozillaDeepSpeech(model_path)
        elif asrsystem_name == 'Vosk':
            return Vosk(model_path)
        elif asrsystem_name == 'Wav2Vec2':
            return Wav2Vec2(model_path)
        elif asrsystem_name == 'DummyASR':
            return DummyASR(model_path)
        else:
            raise ValueError('ASR system `%s` has not been implemented.')

class DummyASR(ASRSystem):
    """
    Dummy ASR system that always transcribes 'hello world'.
    """

    def __init__(self, model_path, use_language_model=False, identifier=None):
        super(DummyASR, self).__init__(model_path, identifier)
        self.samplerate_hz = 16000

    def transcribe(self, sound_or_path, fs=None):
        return 'hello world'

class Vosk(ASRSystem):
    """
    Implements a Vosk Kaldi model based on the model file at model_path. 

    See https://alphacephei.com/vosk/install for installation instructions. 
    """

    def __init__(self, model_path, samplerate=16000, identifier=None):
        super(Vosk, self).__init__(model_path, identifier)
        vosk.SetLogLevel(0)
        self.model = vosk.Model(self.model_path)
        self.rec = vosk.KaldiRecognizer(self.model, samplerate)
        self.samplerate_hz = samplerate

    def transcribe(self, sound_or_path, fs=None):
        def frame_generator(sound):
            c = 0
            nr_frames = 4000
            while c + nr_frames <= len(sound):
                yield float_to_byte(sound[c:c+nr_frames])
                c += nr_frames
            if c < len(sound):
                yield float_to_byte(sound[c:])

        sound = self._load_sound(sound_or_path)
        for data in frame_generator(sound):
            if self.rec.AcceptWaveform(data):
                self.rec.Result()
            else:
                self.rec.PartialResult()
        final_result = json.loads(self.rec.FinalResult())
        return final_result['text']


class MozillaDeepSpeech(ASRSystem):
    """
    Implements a Mozilla DeepSpeech model based on the model file at model_path. 

    This code assumes the model follows Mozilla DeepSpeech version 6.1 and may not
    work for later models. See https://deepspeech.readthedocs.io/en/v0.6.1/USING.html
    for installation instructions.
    """

    def __init__(self, model_path, use_language_model=False, identifier=None):
        super(MozillaDeepSpeech, self).__init__(model_path, identifier)
        model_path = os.path.join(self.model_path, 'output_graph.pbmm')
        alphabet_path = os.path.join(self.model_path, 'alphabet.txt')
        language_model_path = os.path.join(self.model_path, 'lm.binary')
        trie_path = os.path.join(self.model_path, 'trie')
        self._model = DPModel(model_path, 500)
        self.samplerate_hz = 16000

        if use_language_model:
            self._model.enableDecoderWithLM(language_model_path, trie_path, 0.75, 1.85)

    def transcribe(self, sound_or_path, fs=None):
        sound = self._load_sound(sound_or_path)
        sound = (np.iinfo(np.int16).max * sound).astype(np.int16)
        res = self._model.stt(sound)
        return res


class Wav2Vec2(ASRSystem):

    """
    Implements Wav2Vec 2.0 model based on the model file at model_path. 

    This code assumes the model follows Wav2Vec 2.0 as per 
    https://github.com/pytorch/fairseq/commit/1bba712622b8ae4efb3eb793a8a40da386fe11d0 
    and may not work for later models. See link above for installation instructions.
    Note that this code requires a dictionary file (dict_file) to be present in the same 
    folder as where the main model (model_path) is located. 
    """

    def __init__(self, model_path, identifier=None, dict_file='dict.ltr.txt'):
        super(Wav2Vec2, self).__init__(model_path, identifier)
        self._init_model(dict_file)
        self.samplerate_hz = 16000

    def _init_model(self, dict_file):
        parser = self._create_parser(dict_file)
        model_dir = os.path.dirname(self.model_path)
        target_dict_path = os.path.join(model_dir, dict_file)
        args = parser.parse_args(
            ['--target_dict_path', target_dict_path,
            '--w2v_path', self.model_path]
        )
        target_dict = Dictionary.load(args.target_dict_path)
        self.model = self._load_model(args.w2v_path, target_dict)[0]
        self.model.eval()
        self.generator = W2lViterbiDecoder(args, target_dict)
        self.args = args
        self.target_dict = target_dict

    def _create_parser(self, dict_file):
        parser = argparse.ArgumentParser(description='Wav2vec-2.0 Recognize')
        parser.add_argument('--w2v_path', type=str,
                            default='~/wav2vec2_vox_960h.pt',
                            help='path of pre-trained wav2vec-2.0 model')
        parser.add_argument('--target_dict_path', type=str,
                            default=dict_file,
                            help='path of target dict (dict.ltr.txt)')
        parser.add_argument('--nbest', type=int,
                            default=1,
                            help='nbest')
        parser.add_argument('--criterion', type=str,
                            default='ctc',
                            help='type of criterion')
        return parser

    def _load_model(self, model_path, target_dict):

        class BasicTask(object):
            # Class to get around creating a full task object.
            def __init__(self, target_dict):
                self.target_dictionary = target_dict

        data_folder = self.model_path.replace(self.model_path.split('/')[-1], '')
        args_overrides = {
            'data': data_folder,
        }

        try:
            model, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
                [model_path],
                arg_overrides=args_overrides
            )
        except MissingMandatoryValue as e:
            args_overrides['w2v_path'] = model_path
            model, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
                [model_path], 
                arg_overrides=args_overrides
            )
        return model

    def transcribe(self, sound_or_path):
        sound = self._load_sound(sound_or_path)

        # Reformat input sound so it is shaped as expected by Wav2Vec 2.0.
        sample = dict()
        net_input = dict()
        feature = self._extract_feature(sound)
        net_input["source"] = feature.unsqueeze(0)
        padding_mask = torch.BoolTensor(net_input["source"].size(1)).fill_(False).unsqueeze(0)
        net_input["padding_mask"] = padding_mask
        sample["net_input"] = net_input

        # Move to GPU if required.
        if next(self.model.parameters()).is_cuda:
            sample['net_input']['source'] = sample['net_input']['source'].to(torch.device("cuda:0"))
            sample['net_input']['padding_mask'] = sample['net_input']['padding_mask'].to(torch.device("cuda:0"))
        
        # Run network input through network to obtain output.
        with torch.no_grad():
            hypo = self.generator.generate([self.model], sample, prefix_tokens=None)

        # Clean up output into letter output. 
        hyp_pieces = self.target_dict.string(hypo[0][0]["tokens"].int().cpu())
        transcription = hyp_pieces.replace(" ", "").replace("|", " ").strip()
        return transcription

    def _extract_feature(self, sound):
        def postprocess(feats):
            if feats.dim() == 2:
                feats = feats.mean(-1)
            assert feats.dim() == 1, feats.dim()
            with torch.no_grad():
                feats = torch.nn.functional.layer_norm(feats, feats.shape)
            return feats
        feats = torch.from_numpy(sound).float()
        feats = postprocess(feats)
        return feats