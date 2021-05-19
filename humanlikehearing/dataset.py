from .sound import Sound

import PyPDF2
import librosa
import numpy as np
import os
import pandas as pd
import re
import scipy
import soundfile
import warnings
from collections import defaultdict


class DataSet(object):
    """Class to access data in a unified way.

    A data set is stored as a list of tuples (wav_path, transcription)
    where wav_path refers to the path at which the wav file is stored and
    transcription contains a string with the transcription of the given wavfile.
    Note that all files in a data set are assumed to have the same sample rate. 
    """

    def __init__(self, data_path):
        self.data = list()
        self.data_path = data_path
        self._samplerate = None 
        self._longterm_fft = None
        

    def get(self, index):
        """Return data point at given index."""

        return self.data[index]

    @property
    def samplerate(self):
        """The samplerate of all sound files in the data set."""

        if self._samplerate is None:
            _, samplerate = soundfile.read(self.data[0][0])
            self._samplerate = samplerate
        return self._samplerate

    def iterator(self, ):
        """ Returns an iterator for all data points."""

        return iter(self.data)

    def create_subset(self, sex=None, country=None, excluded_speakers=None):
        """Return copy of data set with only a subset of speakers (e.g. only male speakers).

        Note that this function only works if the underlying dataset has the following 
        data_info associated with it:

        * speaker_id (per sentence)
        * sex of each speaker (stored in self.SEXES)
        * country of each speaker (stored in self.COUNTRY)
        * speaker_id

        See ARUDataSet for an example.
        
        """
        include = np.ones(len(self.SEXES))
        if sex is not None:
            include *= np.array(self.SEXES) == sex
        if country is not None:
            include *= np.array(self.COUNTRY) == country
        if excluded_speakers is not None:
            include[np.atleast_1d(excluded_speakers) - 1] = False
        speakers = np.where(include)[0] + 1
        subset = list(self.data_info['speaker'].isin(speakers))        
        new_dataset = type(self)(self.data_path)
        new_dataset.data = [x for i, x in zip(subset, self.data) if i]
        new_dataset.data_info = self.data_info[subset]
        return new_dataset

    @classmethod
    def create(cls, dataset_name, data_path):
        """Creates an instance of dataset_name using the data in data_path."""
        if dataset_name == 'LibriSpeechDataSet':
            return LibriSpeechDataSet(data_path)
        elif dataset_name == 'ARUDataSet':
            return ARUDataSet(data_path)
        elif dataset_name == 'CRMDataSet':
            return CRMDataSet(data_path)
        elif dataset_name == 'DummyDataSet':
            return DummyDataSet(data_path)
        else:
            raise ValueError('Dataset {} not known.'.format(dataset_name))

    def longterm_fft(self, freq):
        """
        Returns the longterm fft for the given frequencies freq.
        
        Note that this function may take a long time on first call, as the longterm fft will
        be estimated from the whole dataset. After the first call, a file 'longterm_fft.npz'
        will be stored in the data_path for speeding up subsequent calls.
        """
        if self._longterm_fft is None:
            # Initialise longterm fft function.
            longterm_fft_path = os.path.join(self.data_path, 'longterm_fft.npz')
            if not os.path.isfile(longterm_fft_path):
                self._compute_and_store_longterm_fft()
            longterm_fft_dict = np.load(longterm_fft_path)
            fft = longterm_fft_dict['fft']
            freqs = longterm_fft_dict['freqs']
            self._longterm_fft = scipy.interpolate.interp1d(freqs, fft, kind='cubic')

        max_freq = np.max(self._longterm_fft.x)
    
        return self._longterm_fft(np.clip(freq, None, max_freq))

    def _compute_and_store_longterm_fft(self):
        """
        Computes the longterm FFT from the dataset. This may take a while, but the results
        are stored s.t. it should only happen once. 
        """
        warnings.warn(
            'Computing and storing longterm fft for data set located in \n{}\n'
            'This may take a while but only happens once.'.format(self.data_path),
            UserWarning
        )
        sound_pointer = 0
        array_pointer = 0
        array = np.zeros(self.samplerate)
        fft = 0
        for wavfile, transcription in self.data:
            sound = Sound(wavfile)
            while sound_pointer < len(sound):
                left_in_array = len(array) - array_pointer
                left_in_sound = len(sound) - sound_pointer
                to_add = min(left_in_array, left_in_sound)
                array[array_pointer:array_pointer + to_add] = sound[sound_pointer:sound_pointer + to_add]
                sound_pointer += to_add
                array_pointer += to_add
                if array_pointer == len(array):
                    fft += np.abs(scipy.fft.fft(array))
                    array_pointer = 0
            sound_pointer = 0
        freqs = scipy.fft.fftfreq(len(array), 1.0/sound.samplerate_Hz)
        np.savez_compressed(os.path.join(self.data_path, '/longterm_fft.npz'), fft=fft, freqs=freqs)

class NoiseDataSet(DataSet):
    """
    A NoiseDataSet stores data that is meant to be added as noise.

    A NoiseDataSet does not need to have transcriptions, but does need to 
    implement random_sample(), a function that randomly selects a noise
    of a given length from the noise data set.
    """
    def __init__(self, data_path):
        super(NoiseDataSet, self).__init__(data_path)

    def random_sample(self):
        raise NotImplementedError()

class VBNoiseDataSet(NoiseDataSet):
    """
    Noise dataset consisting of one large six speaker babble file from here:
    https://homepages.inf.ed.ac.uk/cvbotinh/se/noises/
    """
    def __init__(self, data_path):
        super(VBNoiseDataSet, self).__init__(data_path)

        wav_path = os.path.join(data_path, 'babble.wav')
        self.data = [(wav_path, '')]
        self._data_raw, self._samplerate = soundfile.read(wav_path)

    def random_sample(self, target_duration_frames, target_samplerate_hz):
        target_duration_sec = target_duration_frames / target_samplerate_hz
        source_duration_frames = int(np.ceil(target_duration_sec * self.samplerate))
        start_id = np.random.randint(0, len(self._data_raw) - source_duration_frames)
        sample = self._data_raw[start_id:start_id + source_duration_frames]
        resampled_sample = librosa.core.resample(sample, self.samplerate, target_samplerate_hz)
        return resampled_sample[:target_duration_frames]


class DummyDataSet(DataSet):
    """
    Dummy data set consisting of a single audio file.
    """

    def __init__(self, data_path):
        super(DummyDataSet, self).__init__(data_path)
        self.data.append((os.path.join(data_path, 'sample.wav'), 'hello world.'))
        self.data_info = []
        self.data_info = pd.DataFrame(
            [[1, 0, 0]],
            columns=['speaker', 'sentence_number', 'list_number']
        )
        self.SEXES = ['M']


class ARUDataSet(DataSet):
    """
    The ARUDataSet contains 1200 IEEE sentences spoken by 12 British speakers.

    The corpus can be downloaded on http://datacat.liverpool.ac.uk/681/.

    In the code below it is assumed the data folder, besides all .wav files, als 
    contains 'IEEE_wordlists.pdf', which contains the IEEE transcriptions in a PDF file.
    """

    def __init__(self, data_path, to_sort=True):
        """Initialises ARUDataSet.

        When to_sort is True the data will be sorted by 'speaker', 'list_number' 
        and 'sentence_number'.
        """

        super(ARUDataSet, self).__init__(data_path)
        self.data_info = []
        self.SEXES = ['M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'F', 'M', 'F', 'M']
        self.COUNTRY = [
            'England', 'Wales', 'England', 'England', 'Wales', 'England', 
            'England', 'England', 'England', 'England', 'England', 'England'
        ]
        transcription_dict = self._extractTranscriptionsFromPDF(os.path.join(data_path, 'IEEE_wordlists.pdf'))
        for wav_path in os.listdir(data_path):
            if '.wav' in wav_path:
                list_number = [x for x in wav_path.split(' - ') if 'List' in x][0]
                sentence_number = [int(x.split(' ')[1]) for x in wav_path.split(' - ') if 'Sentence' in x][0]
                speaker_id = int(wav_path.split('_')[0].replace('ID', ''))
                self.data.append((os.path.join(data_path, wav_path), transcription_dict[list_number][sentence_number - 1]))
                self.data_info.append(
                    [speaker_id, sentence_number, int(list_number.split(' ')[1])]
                )
        self.data_info = pd.DataFrame(
            self.data_info, 
            columns=['speaker', 'sentence_number', 'list_number']
        )

        if to_sort:
            sorted_data_info = self.data_info.sort_values(
                ['speaker', 'list_number', 'sentence_number']
            )
            sorted_idx = list(sorted_data_info.index)
            self.data = [self.data[i] for i in sorted_idx]
            self.data_info = sorted_data_info.set_index(
                pd.Series(range(len(self.data_info)))
            )

    def _extractTranscriptionsFromPDF(self, filename):
        read_pdf = PyPDF2.PdfFileReader(filename)
        text = ''
        for page_number in range(read_pdf.getNumPages()):
            page = read_pdf.getPage(page_number)
            page_content = page.extractText()
            text += page_content

        transcriptions_per_list = defaultdict(list)
        for line in text.split('.'):
            has_list = re.match(r'List \d+', line.strip())
            if has_list:
                current_list = has_list.group(0)
                line = line.replace(current_list, '')
            line = line.replace('\n', '').strip()
            transcriptions_per_list[current_list].append(line)
        return transcriptions_per_list



class LibriSpeechDataSet(DataSet):
    """
    The LibriSpeech dataset contains audio files extracted from audiobooks.

    The corpus can be downloaded from https://www.openslr.org/12

    We recommend to only use the "test-clean.tar.gz" subset of this data set, as
    many freely available ASR systems are trained using the LibriSpeech data set, 
    and performance may be overestimated when testing on data the ASR system was
    trained on. 

    When the LibriSpeech dataset is loaded for the first time, all .flac files
    are converted to .wav files and stored in the same folder. 
    """

    def __init__(self, data_path):
        super(LibriSpeechDataSet, self).__init__(data_path)

        for speaker in os.listdir(data_path):
            speaker_path = os.path.join(data_path, speaker)
            for chapter in os.listdir(speaker_path):
                chapter_path = os.path.join(speaker_path, chapter)
                transcription_path = os.path.join(
                    chapter_path,
                    '{}-{}.trans.txt'.format(speaker, chapter)
                )
                
                # Load transcriptions of chapter
                with open(transcription_path) as transcription_file:
                    transcripts = dict(x.split(' ', maxsplit=1) for x in transcription_file.readlines())

                for utterance_flac_file in os.listdir(chapter_path):
                    if utterance_flac_file.endswith('.flac'):
                        utterance = utterance_flac_file.replace('.flac', '')
                        flac_path = os.path.join(chapter_path, utterance_flac_file)
                        wav_file = '{}.wav'.format(utterance)
                        wav_path = os.path.join(chapter_path, wav_file)

                        # Store .wav files
                        if not os.path.exists(wav_path):
                            pcm, sample_rate = soundfile.read(flac_path)
                            soundfile.write(wav_path, pcm, sample_rate)

                        self.data.append((wav_path, transcripts[utterance]))

class CRMDataSet(DataSet):
    """
    The Coordinate Response Measure corpus contains short structered sentences.

    The CRM corpus can be downloaded at https://github.com/NSNC-Lab/CRM
    """

    def __init__(self, data_path, to_sort=True):
        """Initialises CRMDataSet.

        When to_sort is True the data will be sorted by 'speaker' and 'call_id' 
        """
        super(CRMDataSet, self).__init__(data_path)

        self.SEXES = ['M', 'M', 'M', 'M', 'F', 'F', 'F', 'F']
        self.callsigns = ['charlie', 'ringo', 'laker', 'hopper', 'arrow', 'tiger', 'eagle', 'baron']
        self.colors = ['blue', 'red', 'white', 'green']
        self.numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight']

        self.data_info = []

        for talker in os.listdir(data_path):
            print(talker)
            if 'Talker' in talker:
                for file in os.listdir(os.path.join(data_path, talker)):
                    if '.BIN' not in file:
                        continue

                    full_path = os.path.join(data_path, talker, file)
                    wav_path = full_path.replace('.BIN', '.wav')                    
                    if not os.path.exists(wav_path):
                        self._bin_to_wav(full_path)
                    speaker_id = int(talker.split(' ')[-1])
                    transcription = self._name_to_transcription(file)
                    self.data.append((wav_path, transcription))
                    self.data_info.append([speaker_id, file.replace('.BIN', '')])

        self.data_info = pd.DataFrame(
            self.data_info, 
            columns=['speaker', 'call_id']
        )

        if to_sort:
            sorted_data_info = self.data_info.sort_values(
                ['speaker', 'call_id']
            )
            sorted_idx = list(sorted_data_info.index)
            self.data = [self.data[i] for i in sorted_idx]
            self.data_info = sorted_data_info.set_index(
                pd.Series(range(len(self.data_info)))
            )

    def _bin_to_wav(self, file_name):
        with open(file_name, 'rb') as fid:
            data_array = np.fromfile(fid, np.int16)
        sound = Sound(data_array/16384, samplerate_Hz=40000)
        sound.write(file_name.replace('.BIN', '.wav'))

    def _name_to_transcription(self, filename):
        callsign_id = int(filename[:2])
        color_id = int(filename[2:4])
        number_id = int(filename[4:6])

        return "ready {} go to {} {} now.".format(
            self.callsigns[callsign_id],
            self.colors[color_id],
            self.numbers[number_id]
        )

class PregeneratedTargetMaskerDataSet(DataSet):
    """DataSet subclass for data sets that have pregenerated target and maskers. 

    For some speech in noise experiments, the target stimuli and maskers may have been 
    previously generated. The PregeneratedTargetMaskerDataSet assumes targets/maskers are 
    stored in subfolder named 'Targets'/'Maskers'.  

    A PregeneratedTargetMaskerDataSet needs to implement extract_data_info_from_filename(),
    a function that generates the correct data info if required (e.g. speaker_id) as well 
    as transcription based on the name of the file. 
    """

    def extract_details_from_filename(self, wav_path):
        raise NotImplementedError()

    def __init__(self, data_path, to_sort=True):
        super(PregeneratedTargetMaskerDataSet, self).__init__(data_path)
        self.data_info = []

        targets_path = os.path.join(data_path, 'Targets')
        transcription_dict = self._extractTranscriptionsFromPDF(os.path.join(targets_path, 'IEEE_wordlists.pdf'))
        for target_type in os.listdir(targets_path):
            target_path = os.path.join(targets_path, target_type)
            if os.path.isdir(target_path):
                for wav_path in os.listdir(target_path):
                    full_wav_path = os.path.join(target_path, wav_path)
                    if '.wav' in wav_path:
                        list_number, sentence_number, speaker_id = self.extract_details_from_filename(full_wav_path)
                        self.data.append((
                            full_wav_path, 
                            transcription_dict['List {}'.format(list_number)][sentence_number - 1]
                        ))
                        self.data_info.append([
                            speaker_id, 
                            sentence_number, 
                            list_number
                        ])

        self.maskers, self.masker_info = [], []
        masker_path = os.path.join(data_path, 'Maskers')
        for masker_type in os.listdir(masker_path):
            full_masker_type_path = os.path.join(masker_path, masker_type)
            if os.path.isdir(full_masker_type_path):
                for i, wav_path in enumerate(os.listdir(full_masker_type_path)):
                    if '.wav' in wav_path:
                        self.maskers.append(
                            os.path.join(full_masker_type_path, wav_path)
                        )
                        self.masker_info.append([i, masker_type])

        self.masker_info = pd.DataFrame(
            self.masker_info, 
            columns=['masker_id', 'masker_type']
        )

        self.data_info = pd.DataFrame(
            self.data_info, 
            columns=['speaker', 'sentence_number', 'list_number']
        )

        if to_sort:
            sorted_data_info = self.data_info.sort_values(
                ['speaker', 'list_number', 'sentence_number']
            )
            sorted_idx = list(sorted_data_info.index)
            self.data = [self.data[i] for i in sorted_idx]
            self.data_info = sorted_data_info.set_index(
                pd.Series(range(len(self.data_info)))
            )

    def _extractTranscriptionsFromPDF(self, filename):
        doc = fitz.open(filename)
        text = ''
        for page in doc.pages():
            text += page.getText()
        transcriptions_per_list = defaultdict(list)
        for line in text.split('.'):
            has_list = re.match(r'List \d+', line.strip())
            if has_list:
                current_list = has_list.group(0)
                line = line.replace(current_list, '')
            line = line.replace('\n', '').strip()
            transcriptions_per_list[current_list].append(line)
        return transcriptions_per_list

class SteinmetzgerDataSet(PregeneratedTargetMaskerDataSet):
    def extract_details_from_filename(self, full_wav_path):
        wav_path = full_wav_path.split('/')[-1]
        list_number = int(wav_path[4:6])
        sentence_number = int(wav_path[6:8])
        speaker_id = wav_path.replace('.wav', '').split('_')[-1]
        return list_number, sentence_number, speaker_id

class RosenDataSet(PregeneratedTargetMaskerDataSet):
    def extract_details_from_filename(self, full_wav_path):
        wav_path = full_wav_path.split('/')[-1]
        speaker_id = full_wav_path.split('/')[-2]
        list_number = int(wav_path[4:6])
        sentence_number = int(wav_path[7:9])
        return list_number, sentence_number, speaker_id
