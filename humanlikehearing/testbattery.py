from . import filtertools
from .sound import Sound
from .library.voice_activity_detection import extract_voiced_segments
from .dataset import NoiseDataSet

import bisect
import librosa.core
import nltk
import numpy as np
import os 
import pandas as pd
import scipy.fft
import scipy.signal
import time

class Test(object):

    def __init__(self, folder='', store_wavs=False, sentences_per_condition=100):
        self.folder = folder
        self.id = int(time.time())
        self._download_nltk_dependencies()
        self.results = dict()
        self.results_counter = dict()
        self.store_wavs = store_wavs
        self.to_store = []
        self.other_data = []
        self.sentences_per_condition = sentences_per_condition
        self.parameters = dict()

    def run(self, asr_system, dataset):
        raise NotImplementedError()

    def _extract_keywords(self, sentence):

        def is_keyword(word, pos):
            is_key_word = pos[:2] in ['NN', 'VB', 'JJ', 'CD', 
                                      'MD', 'RP', 'PR', 'RB',
                                      'WR', 'IN', 'DT', 'EX',
                                      'WP']
            #is_posessive = '$' in pos
            banned = word in ['be', 'am', 'are', 'is', 
                             'was', 'been', 'for',
                             'a', 'the', 'so', 'will',
                             'from', 'can', 'any', "n't"]

            include = word in ['us', 'we', 'he']
            long_word = len(word) > 2
            return (is_key_word and not banned and long_word) or include

        tokenized = nltk.word_tokenize(sentence)
        keywords = [word for (word, pos) in nltk.pos_tag(tokenized) if is_keyword(word, pos)] 
        return keywords

    def _keyword_accuracy(self, prediction, transcription):
        """
        TODO(lweerts): 
        * Ensure every keyword can only count towards the accuracy as often as it occurs in the sentence.
        * Find the maximum number of keywords that occur in the correct order
        """

        keywords_prediction = self._extract_keywords(prediction.lower())
        keywords_transcription = self._extract_keywords(transcription.lower())

        # Measure how many of the keywords in the transcription were in the prediction
        correct_keywords = [1 if k in keywords_prediction else 0 for k in keywords_transcription]

        return np.sum(correct_keywords)/len(correct_keywords), correct_keywords        

    def _add_noise(self, speech_signal, noise_signal, snr, a_weighting=False, speech_volt_meter=False, speech_reference_level=None, noise_reference_level=None):

        if noise_signal is None:
            if speech_reference_level:
                speech_signal.set_advanced_level(speech_reference_level, a_weighting, speech_volt_meter)
            return speech_signal

        samplerate = min(speech_signal.samplerate_Hz, noise_signal.samplerate_Hz)
        if speech_signal.samplerate_Hz != noise_signal.samplerate_Hz:
            if speech_signal.samplerate_Hz > samplerate:
                speech_signal = Sound.resample(speech_signal, samplerate)
            else:
                noise_signal = Sound.resample(noise_signal, samplerate)

        # Resize or lengthen noise signal to match with speech signal.
        noise_aligned, noise_start, noise_end = self._align(speech_signal, noise_signal)
        noise_signal = Sound(noise_aligned, noise_signal.samplerate_Hz)

        if speech_reference_level is not None:
            speech_signal.set_advanced_level(speech_reference_level, a_weighting, speech_volt_meter)
            noise_signal.set_advanced_level(speech_reference_level - snr, a_weighting, False)
        elif noise_reference_level is not None:
            speech_signal.set_advanced_level(noise_reference_level + snr, a_weighting, speech_volt_meter)
            noise_signal.set_advanced_level(noise_reference_level, a_weighting, False)
        else:
            speech_level = speech_signal.get_advanced_level(a_weighting, speech_volt_meter)
            noise_level = noise_signal.get_advanced_level(a_weighting, False)
            noise_signal *= 10**(((speech_level - snr) - noise_level)/20.)
        return speech_signal + noise_signal

    def _align(self, speech_signal, noise):

        if speech_signal.samplerate_Hz != noise.samplerate_Hz:
            noise = Sound.resample(noise, speech_signal.samplerate_Hz)
        goal_len = len(speech_signal)

        if(goal_len == len(noise)):
            return noise, 0, goal_len
        elif goal_len <= len(noise):
            start = np.random.randint(0, len(noise) - goal_len)
            aligned_noise = Sound(noise[start:start + goal_len], noise.samplerate_Hz)
            return aligned_noise, start, start + goal_len
        else: 
            raise ValueError('Noise needs to be longer than or equal to speech signal in duration.')

    def _generate_longterm_speech_shaped_noise(self, sound, dataset):
        samples = len(sound)
        freqs = np.abs(scipy.fft.fftfreq(samples, 1/sound.samplerate_Hz))
        power = dataset.longterm_fft(freqs)
        power = np.array(power, dtype='complex')
        Np = (len(power) - 1) // 2
        phases = np.random.rand(Np) * 2 * np.pi
        phases = np.cos(phases) + 1j * np.sin(phases)
        power[1:Np+1] *= phases
        power[-1:-1-Np:-1] = np.conj(power[1:Np+1])
        return Sound(scipy.fft.ifft(power).real, sound.samplerate_Hz)

    def _generate_speech_shaped_noise(self, sound, modulated, modulation_rate_Hz=8):
        spectrogram = np.abs(scipy.fft.fft(sound))*np.exp(2j*np.pi*np.random.rand(len(sound)))
        noise = np.real(scipy.fft.ifft(spectrogram))

        if modulated:
            t = np.arange(len(sound))/sound.samplerate_Hz
            modulation = 10**(30*(np.sin(2*np.pi*modulation_rate_Hz*t)-1)/40)
            noise = noise*modulation
        return Sound(noise, sound.samplerate_Hz)

    def _generate_gaussian_white_noise(self, duration_frames, samplerate):
        return Sound(np.random.randn(duration_frames), samplerate)

    def _generate_noise(self, sound, noise_type, longterm_dataset=None, kwargs={}):

        if noise_type is None:
            return None

        if isinstance(noise_type, NoiseDataSet):
            return Sound(noise_type.random_sample(len(sound), sound.samplerate_Hz), sound.samplerate_Hz)
        elif 'speech_shaped_longterm' in noise_type:
            return self._generate_longterm_speech_shaped_noise(sound, longterm_dataset, **kwargs)
        elif 'speech_shaped' in noise_type:
            if 'modulated' in noise_type:
                return self._generate_speech_shaped_noise(sound, True, **kwargs)
            else:
                return self._generate_speech_shaped_noise(sound, False, **kwargs)
        elif 'white_noise' in noise_type:
            return self._generate_gaussian_white_noise(len(sound), sound.samplerate_Hz, **kwargs)

    def _download_nltk_dependencies(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')

        return

    def _allocate_result_dataframe(self, index_values, index_names, column_names, result_name='standard'):
        midx = pd.MultiIndex.from_product(
            index_values,
            names=index_names
        )
        empty_results = np.full_like(np.zeros((len(midx), len(column_names))), np.nan)
        self.results[result_name] = pd.DataFrame(empty_results, columns=column_names, index=midx)
        self.results_counter[result_name] = 0

    def _allocate_result_dataframe_from_dataframe(self, old_results, result_name):
        self.results[result_name] = pd.DataFrame(columns=old_results.columns, index=old_results.index)
        self.results_counter[result_name] = 0

    def _add_result(self, trial_outcome, sound=None, result_name='standard', display=False, other_data=None):

        c = self.results_counter[result_name]
        self.results[result_name].iloc[c] = [
            trial_outcome[column] for column in self.results[result_name].columns
        ]
        self.results_counter[result_name] += 1

        if self.store_wavs:
            self.to_store.append((c, sound))

        if other_data is not None:
            self.other_data.append((c, other_data))

        if display:
            added_values = ' '.join([
                '{}: {}'.format(x, y) for x, y in zip(trial_outcome.keys(), trial_outcome.values())
            ])
            print("{} ({}): trial {} / {} ({})".format(
                type(self).__name__,
                result_name, 
                self.results_counter[result_name], 
                len(self.results[result_name]), 
                added_values, 
            ), end="\r")

    def _get_result_folder(self, result_name):
        test_class_name = type(self).__name__
        file_folder_name = '{}_{}_{}'.format(
            test_class_name,
            result_name,
            self.id
        )
        file_folder_path = os.path.join(self.folder, file_folder_name)
        return file_folder_path

    def _store_results(self, result_name='standard'):
        file_folder_path = self._get_result_folder(result_name)
        if not os.path.exists(file_folder_path):
            os.mkdir(file_folder_path)

        if not os.path.isfile(os.path.join(file_folder_path, 'params.pk1')):
            self.results[result_name].index.to_frame().to_pickle(os.path.join(file_folder_path, 'params.pk1'))

        result_path = os.path.join(file_folder_path, 'results.pk1')
        self.results[result_name].to_pickle(result_path)
        
        for c, sound in self.to_store:
            sound.write(os.path.join(file_folder_path, "{}.wav".format(c)))
        self.to_store = []

        for c, other in self.other_data:
            np.savez_compressed(os.path.join(file_folder_path, '{}.npz'.format(c)), **other)
        self.other_data = []

    def random_soundfile_for_speaker(self, dataset, speaker_id, remove_pauses=True):
        all_but_speaker = list(range(1, speaker_id)) + list(range(speaker_id + 1, 13))
        subset = dataset.create_subset(excluded_speakers=all_but_speaker)
        sounds = []
        for i in np.random.choice(range(len(subset.data)), 100):
            sounds.append(Sound(subset.data[i][0]))
        all_sounds = Sound(np.concatenate(sounds), sounds[0].samplerate_Hz)
        if remove_pauses:
            all_sounds = self.remove_pauses(all_sounds)
        return all_sounds

    def remove_pauses(self, sound, frame_rate_ms=10, pause_length_ms=100):

        def find_closest_samplerate(samplerate):
            samplerate_options = np.array([0, 8000, 16000, 32000, 48000])
            downsampled_rate = samplerate_options[samplerate >= samplerate_options][-1]
            if downsampled_rate == 0:
                return 8000
            return downsampled_rate

        def time_to_sample_number(time_s, samplerate_hz):
            return int(samplerate_hz*time_s)

        def selected_voiced_parts_from_segments(sound, segments):
            chunks = []
            for segment in segments:
                start, end = segment[0].timestamp, segment[-1].timestamp
                start_idx = time_to_sample_number(start, sound.samplerate_Hz)
                end_idx = time_to_sample_number(end, sound.samplerate_Hz)
                chunks.append(sound[start_idx:end_idx])
            return Sound(np.concatenate(chunks), sound.samplerate_Hz)

        downsample_rate = find_closest_samplerate(sound.samplerate_Hz)
        if downsample_rate != sound.samplerate_Hz:
            downsampled_sound = Sound.resample(sound, downsample_rate)
        else:
            downsampled_sound = sound
        segments = extract_voiced_segments(
            downsampled_sound, 
            downsampled_sound.samplerate_Hz, 
            frame_rate_ms, 
            pause_length_ms
        )
        sound_without_pauses = selected_voiced_parts_from_segments(sound, segments)
        return sound_without_pauses


class SpeechReceptionThresholdTest(Test):    
    """Tests whether the ASR model is suspectable to audio loudness."""

    def __init__(self, folder='', store_wavs=False, sentences_per_condition=100, test_parameters=None):
        super(SpeechReceptionThresholdTest, self).__init__(folder, store_wavs, sentences_per_condition)
        self.results_adaptive = dict()

        default_parameters = {
            'snr_step_db': 3,
            'min_snr_db': -6,
            'max_snr_db': 36,
            'n_per_srt': 20,
            'srt_threshold': 0.5,
            'speech_target_level_db': 65
        }

        self.parameters.update(default_parameters)
        if test_parameters:
            self.parameters.update(test_parameters)

    def get_adaptive_snr_stepsize(self, results_in_group):
        reversals = np.array(list(results_in_group['reversal']))
        if np.sum(np.isnan(reversals)) == len(reversals):
            nr_reversals = None
        else:
            nr_reversals = np.nansum(reversals)

        if nr_reversals is None:
            return 6
        elif nr_reversals < 2:
            return 4
        else:
            return 2

    def run_static_snr_trial(self, snr, asr_system, transcription, sound, noise):
        """Runs static trial of the experiment with the given snr.

        Returns outcome, a dictionary with both the sound used to make the prediction and the
        trial outcome, which incorporates all trial outputs such as 'accuracy' and 'predicted_transcription'. 
        """

        if sound is None or (noise is None and snr > self.parameters['min_snr_db']): # Do not compute multiple baselines.
            mixed_sound = None
            accuracy = np.nan
            predicted_transcription = None
        else:
            mixed_sound = self._add_noise(
                sound, noise, snr, a_weighting=True, speech_volt_meter=True, 
                speech_reference_level=self.parameters['speech_target_level_db']
            )
            predicted_transcription = asr_system.transcribe(mixed_sound)
            accuracy, _ = self._keyword_accuracy(predicted_transcription, transcription)
        outcome = {
            'trial_outcome': {
                'keyword_accuracy': accuracy,
                'transcription': transcription,
                'predicted_transcription': predicted_transcription
            },
            'sound': mixed_sound
        }
        return accuracy, outcome

    def _run_and_store_srt_trial(self, trial_input, snrs, adaptive):
        if adaptive:
            outcome = self._run_adaptive_snr_trial(trial_input)
            self._add_result(**outcome, display=True)
            self._store_results()
        else:
            for i, snr in enumerate(snrs):
                _, outcome = self.run_static_snr_trial(snr, *trial_input)
                self._add_result(**outcome, display=True)
                self._store_results()
 
    def _run_adaptive_snr_trial(self, static_trial_input, results_name='standard'):

        if not self.results_adaptive[results_name]:
            return ValueError(
                'Result dataframe for name \'{}\' were not initialised as adaptive. Set adaptive_srt_estimation to '
                'True when calling _allocate_result_dataframe().'.format(results_name)
            )

        def compute_next_snr(previous_state, results_in_group):
            reversal = False
            step_size = self.get_adaptive_snr_stepsize(results_in_group)
            if previous_state['snr_accuracy'] < self.parameters['srt_threshold']:
                snr = previous_state['snr'] + step_size
                if previous_state['snr_change_direction'] == -1:
                    reversal = True
                change = 1
            else:
                snr = previous_state['snr'] - step_size
                if previous_state['snr_change_direction'] == 1:
                    reversal = True
                change = -1

            return snr, reversal, change

        def is_first_in_adaptive_snr_block(results_name):
            counter = self.results_counter[results_name]
            distance = self.distance_to_previous_in_condition
            counter_in_srt_group = (counter // distance) % self.parameters['n_per_srt']
            return counter_in_srt_group == 0

        def get_previous_adaptive_snr_state(results_name):
            results = self.results[results_name]
            counter = self.results_counter[results_name]
            return results.iloc[counter- self.distance_to_previous_in_condition]

        def get_results_in_srt_group(result_name):
            counter = self.results_counter[results_name]
            distance = self.distance_to_previous_in_condition
            srt_group_number = (counter // distance) // self.parameters['n_per_srt']
            group_offset = self.parameters['n_per_srt']*distance*srt_group_number
            first_sentence_in_srt_group = group_offset + (counter % distance)
            final_sentence_in_srt_group = group_offset + self.parameters['n_per_srt']*distance
            all_sentences_in_srt_group = np.arange(
                first_sentence_in_srt_group,
                final_sentence_in_srt_group,
                distance,
            )
            return self.results[result_name].iloc[all_sentences_in_srt_group]

        results_in_group = get_results_in_srt_group(results_name)
        if is_first_in_adaptive_snr_block(results_name):
            step_size = self.get_adaptive_snr_stepsize(results_in_group)
            max_accuracy, _ = self.run_static_snr_trial(self.parameters['max_snr_db'], *static_trial_input)
            snr = self.parameters['max_snr_db']
            accuracy, outcome = self.run_static_snr_trial(snr, *static_trial_input)
            while accuracy < max_accuracy and snr < self.parameters['max_snr_db']:
                snr += step_size
                accuracy, outcome = self.run_static_snr_trial(snr, *static_trial_input)
            reversal, change = 0, 0
        else:
            previous_state = get_previous_adaptive_snr_state(results_name)
            snr, reversal, change = compute_next_snr(previous_state, results_in_group)
            accuracy, outcome = self.run_static_snr_trial(snr, *static_trial_input)

        adaptive_snr_outcome = {
            'snr_accuracy': accuracy,
            'snr': snr,
            'snr_change_direction': change,
            'reversal': reversal
        }
        outcome['trial_outcome'].update(adaptive_snr_outcome)
        return outcome

    def _allocate_result_dataframe(self, index_values, index_names, column_names, result_name='standard', adaptive_srt_estimation=False):
        if adaptive_srt_estimation:
            assert index_names[-1] == 'SNR'
            assert index_names[0] == 'sentence_id'

            # Replace last column 'SNR' with extra columns required for an adaptive SNR procedure.
            index_values = index_values[:-1]
            index_names = index_names[:-1]
            column_names = np.concatenate([column_names, ['snr_accuracy', 'snr', 'snr_change_direction', 'reversal']])

        super(SpeechReceptionThresholdTest, self)._allocate_result_dataframe(index_values, index_names, column_names, result_name)

        if adaptive_srt_estimation:
            total_nr_sentences = len(index_values[0])
            self.distance_to_previous_in_condition = len(self.results[result_name])//total_nr_sentences
            self.results_adaptive[result_name] = True
        else:
            self.results_adaptive[result_name] = False

class NormalisationTest(Test):
    """Tests whether the ASR model is suspectable to audio loudness."""

    def __init__(self, folder='', store_wavs=False, sentences_per_condition=100, test_parameters=None):
        super(NormalisationTest, self).__init__(folder, store_wavs, sentences_per_condition)

        default_parameters = {
            'levels_db': range(10, 150, 5)
        }

        self.parameters.update(default_parameters)
        if test_parameters:
            self.parameters.update(test_parameters)

    def run(self, asr_system, dataset):

        self._allocate_result_dataframe(
            [range(self.sentences_per_condition), self.parameters['levels_db']],
            ['sentence_id', 'level'],
            ['keyword_accuracy'],
        )

        for wav_file, transcription in dataset.data[:self.sentences_per_condition]:
            sound = Sound(wav_file)
            for i, level in enumerate(self.parameters['levels_db']):
                sound.level_dB = level
                prediction = asr_system.transcribe(sound)
                accuracy, _ = self._keyword_accuracy(prediction.lower(), transcription.lower())
                trial_outcome = {
                    'keyword_accuracy': accuracy
                }
                self._add_result(trial_outcome, sound=sound, display=True)
                self._store_results()

class PeakClippingTest(Test):
    """
    https://asa.scitation.org/doi/abs/10.1121/1.1862575

    TODO(lweerts): remove silence from beginning and end of sentences.

    In the original paper, the total test set consisted of 250 sentences. These
    were presented to 13 listeners in blocks of 80 sentences per condition. With approx
    8 thresholds per condition, this gives 10 sentences per threshold per listener, 
    and 130 sentences per threshold per listener overall. Each sentence thus has a chance
    of approximately 50% to have been presented in a certain condition. Here, we have
    only one listener and there are no learning effects, so we simplify the procedure
    to present 100 sentences for each threshold, giving a total testet of 700 and 800 
    sentences for the peak clipping and center clipping experiments, respectively.
    """

    def __init__(self, folder='', store_wavs=False, sentences_per_condition=100, test_parameters=None):
        super(PeakClippingTest, self).__init__(folder, store_wavs, sentences_per_condition)

        default_parameters = {
            'peak_clipping_thresholds': (None, 0, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99),
            'center_clipping_thresholds': (None, 0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98)
        }

        self.parameters.update(default_parameters)
        if test_parameters:
            self.parameters.update(test_parameters)

    def _cumulative_histogram(self, array):
        hist, bin_edges = np.histogram(array, bins=len(array), density=True)
        cumulative_sum = np.cumsum(hist)
        return cumulative_sum/cumulative_sum[-1], bin_edges

    def _get_clipping_value(self, sound, threshold):
        n, bins = self._cumulative_histogram(np.abs(sound))
        clipping_value = bins[bisect.bisect(n, threshold) + 1]
        return clipping_value

    def _peak_clipping(self, sound, threshold):
        clipping_value = self._get_clipping_value(sound, threshold)
        new_sound = np.clip(sound, -clipping_value, clipping_value)
        return Sound(new_sound, sound.samplerate_Hz)

    def _center_clipping(self, sound, threshold):
        clipping_value = self._get_clipping_value(sound, threshold)
        new_sound = sound[:]
        new_sound[(sound <= clipping_value)*(sound >= -clipping_value)] = 0
        return Sound(new_sound, sound.samplerate_Hz)

    def run_clipping_method(self, clip_type, thresholds, asr_system, dataset):
        N = self.sentences_per_condition

        self._allocate_result_dataframe(
            [range(N), thresholds], 
            ['sentence_id', 'threshold'],
            ['keyword_accuracy', 'predicted_transcription', 'transcription'],
            result_name=clip_type
        )

        for wav_file, transcription in dataset.data[:N]:
            for threshold in thresholds:
                sound = Sound(wav_file)
                if threshold is None: # Baseline prediction
                    prediction = asr_system.transcribe(sound)
                    distorted_sound = sound
                else:
                    if clip_type == 'peak_clipping':
                        distortion = self._peak_clipping
                        args = [threshold]
                    elif clip_type == 'center_clipping': 
                        distortion = self._center_clipping
                        args = [threshold]
                    else:
                        distortion = lambda x: x
                        args = []

                    distorted_sound = Sound(distortion(sound, *args), sound.samplerate_Hz)
                    distorted_sound.level_dB = 65
                    prediction = asr_system.transcribe(distorted_sound)

                accuracy, _ = self._keyword_accuracy(prediction.lower(), transcription.lower())
                trial_outcome = {
                    'keyword_accuracy': accuracy,
                    'predicted_transcription': prediction,
                    'transcription': transcription
                }

                self._add_result(
                    trial_outcome,
                    sound=distorted_sound,
                    result_name=clip_type,
                    display=True
                )

            # Store intermediate results
            self._store_results(clip_type)

    def run(self, asr_system, dataset):

        self.run_clipping_method(
            'peak_clipping', 
            self.parameters['peak_clipping_thresholds'], 
            asr_system, 
            dataset
        )

        print('Peak clipping test completed.')

        self.run_clipping_method(
            'center_clipping', 
            self.parameters['center_clipping_thresholds'],
            asr_system, 
            dataset
        )

        print('Center clipping test completed.')

class BandpassTest(Test):

    def __init__(self, folder='', store_wavs=False, sentences_per_condition=100, test_parameters=None):
        super(BandpassTest, self).__init__(folder, store_wavs, sentences_per_condition)

        default_parameters = {
            'bandpass_widths_semitone': [None] + list(range(42, 3, -1)),
            'center_frequency_bandpass': 1500,
            'speech_target_level_db': 65
        }

        self.parameters.update(default_parameters)
        if test_parameters:
            self.parameters.update(test_parameters)

    def _construct_bandpass_filter(self, width_in_semitones, order=2000, samplerate_hz=16000):

        centre_hz = self.parameters['center_frequency_bandpass']
        half_width = width_in_semitones / 2
        cutoff = [
            np.exp(-half_width*np.log(2)/12)*centre_hz,
            np.exp(half_width*np.log(2)/12)*centre_hz
        ]

        b = scipy.signal.firwin(
            order + 1,
            cutoff,
            pass_zero=False,
            fs=samplerate_hz
        )
        return b

    def run(self, asr_system, dataset):

        filters = []
        for width_in_semitones in self.parameters['bandpass_widths_semitone']:
            if width_in_semitones is None:
                filters.append(None)
            else:
                fir_filter = self._construct_bandpass_filter(
                    width_in_semitones, 
                    samplerate_hz=dataset.samplerate
                )
                filters.append(fir_filter)

        self._allocate_result_dataframe(
            [range(self.sentences_per_condition), self.parameters['bandpass_widths_semitone']],
            ['sentence_id', 'bandpass_width_semitone'],
            ['keyword_accuracy', 'predicted_transcription', 'transcription'],
        )

        for wav_file, transcription in dataset.data[:self.sentences_per_condition]:
            sound = Sound(wav_file)
            for f in filters:
                if f is None:
                    filtered_sound = sound
                else:
                    filtered_sound = Sound(scipy.signal.lfilter(f, 1, sound), sound.samplerate_Hz)
                filtered_sound.level_dB = self.parameters['speech_target_level_db']
                prediction = asr_system.transcribe(filtered_sound)
                accuracy, _  = self._keyword_accuracy(prediction.lower(), transcription.lower())

                trial_outcome = {
                    'keyword_accuracy': accuracy,
                    'predicted_transcription': prediction,
                    'transcription': transcription
                }

                self._add_result(trial_outcome, sound=filtered_sound, display=True)
                self._store_results()


class PeriodicityTest(SpeechReceptionThresholdTest):

    def __init__(self, folder='', store_wavs=False, sentences_per_condition=100, test_parameters=None):
        super(PeriodicityTest, self).__init__(folder, store_wavs, sentences_per_condition)

        default_parameters = {
            'min_snr_db': -6,
            'max_snr_db': 21, 
            'snr_step_db': 3,
            'nr_channels': [6, 7, 8, 10, 12, 16, 24, 32],
            'maskers': [
                None, 
                'harmonic_complex_modulated', 
                'harmonic_complex', 
                'speech_shaped_modulated', 
                'speech_shaped'
            ],
            'target_filters': [
                None, 
                'noise_vocoded', 
                'dudley_vocoded', 
                'periodic_vocoded'
            ],
            'speech_target_level': 65,
            'vocode_low': 100,
            'vocode_high': 11000
        }

        self.parameters.update(default_parameters)
        if test_parameters:
            self.parameters.update(test_parameters)

    def run_static_snr_trial(self, snr, asr_system, transcription, sound, noise):
        if sound is None or (noise is None and snr > self.parameters['min_snr_db']): # Do not compute multiple baselines.
            mixed_sound = None
            accuracy = np.nan
            predicted_transcription = np.nan
        else:
            keywords = self._extract_keywords(transcription.lower())
            mixed_sound = self._add_noise(
                sound, noise, snr, a_weighting=True, 
                speech_volt_meter=True, 
                speech_reference_level=self.parameters['speech_target_level']
            )
            predicted_transcription = asr_system.transcribe(mixed_sound)
            accuracy, _ = self._keyword_accuracy(predicted_transcription, transcription)

        outcome = {
            'trial_outcome': {
                'keyword_accuracy': accuracy,
                'predicted_transcription': predicted_transcription,
                'transcription': transcription
            },
            'sound': mixed_sound
        }
        return accuracy, outcome

    def create_harmonic_complex(self, speech_signal, long_noise_files):
        speaker_id = np.random.choice(range(len(long_noise_files)))
        speaker_sound, _, _ = self._align(speech_signal, long_noise_files[speaker_id])
        speaker_sound = Sound(speaker_sound, speech_signal.samplerate_Hz)
        harmonic_complex, _, _ = filtertools.generate_f0_pulses(speaker_sound, interpolate=True)
        harmonic_complex.level_dB = 65
        return harmonic_complex

    def select_and_align_masker(self, sound, masker_type, dataset):

        if masker_type is None:
            return None

        possible_ids = np.where(dataset.masker_info['masker_type'] == masker_type)[0]
        pick_id = np.random.choice(possible_ids)
        noise = Sound(dataset.maskers[pick_id])
        aligned_noise, _, _ = self._align(sound, noise)
        return Sound(aligned_noise, noise.samplerate_Hz)


    def run(self, asr_system, dataset, noise_dataset=None, adaptive_srt_estimation_flag=False):

        SNRs = list(reversed(np.arange(
            self.parameters['min_snr_db'],
            self.parameters['max_snr_db'], 
            self.parameters['snr_step_db']
        )))

        self._allocate_result_dataframe(
            [
                range(self.sentences_per_condition), 
                self.parameters['nr_channels'],
                self.parameters['maskers'],
                self.parameters['target_filters'],
                SNRs
            ],
            ['sentence_id', 'nr_channels', 'noise_filter', 'target_filter', 'SNR'], 
            ['keyword_accuracy', 'predicted_transcription', 'transcription'],
            adaptive_srt_estimation=adaptive_srt_estimation_flag,
        )

        vocoder_settings = {
            'low_Hz': self.parameters['vocode_low'], 
            'high_Hz': self.parameters['vocode_high'],
            'spacing': 'greenwood', 
            'filterbank_type': ('butter', 6),
            'envelope_method': ('butter', 4, 30),
            'equalise': True,
        }

        # Select all male speakers and create long noise files
        print('Generating concatenated files for male speakers.')
        long_noise_files = []

        if not noise_dataset:
            for speaker in np.where(np.array(dataset.SEXES) == 'M')[0]:
                long_noise_files.append(self.random_soundfile_for_speaker(dataset, int(speaker + 1)))

        for j, (wav_file, transcription) in enumerate(dataset.data[:self.sentences_per_condition]):
            sound = Sound(wav_file)
            for n, nr_channels in enumerate(self.parameters['nr_channels']):
                for f, masker in enumerate(self.parameters['maskers']):
                    if noise_dataset:
                        noise = self.select_and_align_masker(sound, masker, noise_dataset)
                    else:
                        if masker is None:
                            noise = None
                        elif 'harmonic_complex' in masker:
                            noise = self.create_harmonic_complex(sound, long_noise_files)
                        elif 'speech_shaped' in masker:
                            noise = self._generate_speech_shaped_noise(sound, False)
                        else:
                            ValueError('Masker {} not implemented.'.format(masker))

                        if masker is not None and 'modulated' in masker:
                            t = np.arange(len(noise))/noise.samplerate_Hz
                            modulation = np.sin(2*np.pi*10*t) + 1
                            noise = Sound(noise*modulation, noise.samplerate_Hz)
 
                    for t, target_filter in enumerate(self.parameters['target_filters']):
                        if target_filter is None:
                            filtered_sound = sound
                        elif target_filter == 'noise_vocoded':
                            filtered_sound = filtertools.vocode(
                                sound,
                                vocoder_function='noise',  
                                nr_channels=nr_channels, 
                                **vocoder_settings
                            )
                        elif target_filter == 'dudley_vocoded':
                            filtered_sound = filtertools.vocode(
                                sound,
                                vocoder_function='dudley',  
                                nr_channels=nr_channels, 
                                **vocoder_settings
                            )
                        elif target_filter == 'periodic_vocoded':
                            filtered_sound = filtertools.vocode(
                                sound,
                                vocoder_function='periodic',  
                                nr_channels=nr_channels, 
                                **vocoder_settings
                            )
                        else:
                            raise ValueError('Target filter {} not known.'.format(target_filter))

                        filtered_sound = filtertools.elliptic_filter(filtered_sound, 6, 5, 40, low=10000, high=None)
                        trial_input = [asr_system, transcription, filtered_sound, noise]
                        self._run_and_store_srt_trial(trial_input, SNRs, adaptive_srt_estimation_flag)


class CompetingTalkerTest(SpeechReceptionThresholdTest):

    def __init__(self, folder='', store_wavs=False, sentences_per_condition=100, test_parameters=None):
        super(CompetingTalkerTest, self). __init__(folder, store_wavs, sentences_per_condition)

        default_parameters = {
            'snr_step_db': 3,
            'min_snr_db': -6, 
            'max_snr_db': 36,
            'noise_filters': [
                None, 'noise_vocode', 'modulate',
            ],
            'target_filters': [None],
            'nr_talkers': [0, 1, 2, 4, 8, 16],
            'speech_target_level': 65,
            'vocode_low': 100,
            'vocode_high': 11000,
            'vocode_channels': 12
        }
        self.parameters.update(default_parameters)
        if test_parameters:
            self.parameters.update(test_parameters)

    def create_babble_noises(self, speech_signal, long_noise_files):
        noises = [0]*len(self.parameters['nr_talkers'])
        speaker_subsets = np.random.choice(range(len(long_noise_files)), np.max(self.parameters['nr_talkers']))
        for speaker_i, x in enumerate(speaker_subsets):
            speaker_sound, _, _ = self._align(speech_signal, long_noise_files[x])
            speaker_sound.level_dB = 65
            for i in range(len(noises)):
                if speaker_i < self.parameters['nr_talkers'][i]:
                    noises[i] += speaker_sound
        for i in range(len(noises)):
            noises[i] = Sound(noises[i], long_noise_files[0].samplerate_Hz)
        return noises

    def modulated_babble(self, noise, reference_babble):
        babble_shaped_noise = self._generate_speech_shaped_noise(reference_babble, False)
        b, a = scipy.signal.butter(4, 30, fs=noise.samplerate_Hz)
        envelope = scipy.signal.filtfilt(b, a, np.abs(noise))
        return babble_shaped_noise*envelope

    def noise_vocode(self, noise):

        vocoder_settings = {
            'low_Hz': self.parameters['vocode_low'], 
            'high_Hz': self.parameters['vocode_high'],
            'spacing': 'greenwood', 
            'filterbank_type': ('butter', 6),
            'envelope_method': ('butter', 4, 30),
            'equalise': True,
            'nr_channels': self.parameters['vocode_channels']
        }

        vocoded_noise = filtertools.vocode(
            noise,
            vocoder_function='noise',
            **vocoder_settings
        )

        return vocoded_noise

    def run(self, asr_system, dataset, noise_dataset=None, adaptive_srt_estimation_flag=False):

        SNRs = list(reversed(np.arange(
            self.parameters['min_snr_db'],
            self.parameters['max_snr_db'],
            self.parameters['snr_step_db'] 
        )))

        self._allocate_result_dataframe(
            [
                range(self.sentences_per_condition), 
                self.parameters['nr_talkers'], 
                self.parameters['noise_filters'], 
                self.parameters['target_filters'], 
                SNRs
            ], 
            ['sentence_id', 'nr_talkers', 'noise_filter', 'target_filter', 'SNR'], 
            ['keyword_accuracy', 'transcription', 'predicted_transcription'],
            adaptive_srt_estimation=adaptive_srt_estimation_flag
        )

        # Select all male speakers and create long noise files
        print('Generating concatenated files for male speakers.')
        long_noise_files = []
        for speaker in np.where(np.array(dataset.SEXES) == 'M')[0]:
            long_noise_files.append(self.random_soundfile_for_speaker(dataset, int(speaker + 1)))

        for wav_file, transcription in dataset.data[:self.sentences_per_condition]:
            sound = Sound(wav_file)
            noises = self.create_babble_noises(sound, long_noise_files)

            for n, nr_talkers in enumerate(self.parameters['nr_talkers']):
                if nr_talkers == 0:
                    noise = None
                else:
                    noise = noises[n]
                for noise_filter in self.parameters['noise_filters']:
                    if noise is not None:
                        if noise_filter is None:
                            noise = noise
                        elif noise_filter == 'noise_vocode':
                            noise = self.noise_vocode(noise)
                        elif noise_filter == 'modulate':
                            noise = self.modulated_babble(noise, noises[-1])
                        else:
                            ValueError('Noise filter {} not implemented.'.format(noise_filter))

                    for target_filter in self.parameters['target_filters']:
                        if target_filter == 'noise_vocode':
                            sound = self.noise_vocode(sound)

                        trial_input = [asr_system, transcription, sound, noise]
                        self._run_and_store_srt_trial(trial_input, SNRs, adaptive_srt_estimation_flag)

 
class TemporalFineStructureTest(SpeechReceptionThresholdTest):


    def __init__(self, folder='', store_wavs=False, sentences_per_condition=100, test_parameters=None):
        super(TemporalFineStructureTest, self).__init__(folder, store_wavs, sentences_per_condition)

        default_parameters = {
            'cutoff_values': [0, 6, 12, 18, 24, 30],
            'region_width': 6,
            'erb_min': 100,
            'erb_max': 7999,
            'erb_channels': 30,
            'vocoder_types': [
                'retain_tfs_below_cutoff',
                'retain_tfs_above_cutoff', 
                'retain_tfs_region',
                'remove_envelope_region'
            ],
            'snr_min': -4,
            'snr_max': 24,
            'snr_step_db': 2,
            'speech_target_level': 65,
            'noise_types': ['white_noise']
        }

        self.parameters.update(default_parameters)
        if test_parameters:
            self.parameters.update(test_parameters)

    def run_static_snr_trial(self, snr, asr_system, transcription, sound, noise):
        if sound is None or (noise is None and snr > self.parameters['min_snr_db']): # Do not compute multiple baselines.
            mixed_sound = None
            accuracy = np.nan
        else:
            keywords = self._extract_keywords(transcription.lower())
            mixed_sound = self._add_noise(
                sound, noise, snr, a_weighting=True, speech_volt_meter=True, 
                speech_reference_level=self.parameters['speech_target_level']
            )
            predicted_transcription = asr_system.transcribe(mixed_sound)
            accuracy, _ = self._keyword_accuracy(predicted_transcription, transcription)

        trial_outcome = {
            'keyword_accuracy': accuracy,
            'predicted_transcription': predicted_transcription,
            'transcription': transcription
        }
        outcome = {
            'trial_outcome': trial_outcome,
            'sound': mixed_sound
        }
        return accuracy, outcome

    def _vocode(self, sound, cutoff_value, vocoder_type):

        def remove_tfs_vocoder_function(
            sound, channel_index, freqs, 
            channel, channel_envelope, tone_vocoded_channels):
            (_, center) = freqs
            if channel_index not in tone_vocoded_channels:
                return channel
            else:
                carrier = filtertools.tone(center, channel.samplerate_Hz, frames=len(channel))
                return carrier*channel_envelope 

        def remove_envelope_vocoder_function(
            sound, channel_index, freqs, 
            channel, channel_envelope, tone_vocoded_channels):
            (_, center)  = freqs
            if channel_index not in tone_vocoded_channels:
                return Sound(np.zeros(len(channel)), channel.samplerate_Hz)
            else:
                carrier = filtertools.tone(center, channel.samplerate_Hz, frames=len(channel))
                return carrier*channel_envelope           

        if vocoder_type == 'retain_tfs_above_cutoff':
            tone_vocoded_channels = range(0, cutoff_value + 1)
            tfs_vocoder_function = remove_tfs_vocoder_function
        elif vocoder_type == 'retain_tfs_below_cutoff':
            tone_vocoded_channels = range(cutoff_value + 1, self.parameters['erb_channels'])
            tfs_vocoder_function = remove_tfs_vocoder_function
        elif 'region' in vocoder_type:
            tone_vocoded_channels = list(range(0, cutoff_value)) 
            tone_vocoded_channels += list(range(
                cutoff_value + self.parameters['region_width'], self.parameters['erb_channels']
            ))
            if vocoder_type == 'retain_tfs_region':
                tfs_vocoder_function = remove_tfs_vocoder_function
            elif vocoder_type == 'remove_envelope_region':
                tfs_vocoder_function = remove_envelope_vocoder_function
            else:
                raise ValueError('vocoder type {} not known'.format(vocoder_type))
        else:
            raise ValueError('vocoder_type {} not known'.format(vocoder_type))

        vocoded = filtertools.vocode(sound, self.parameters['erb_min'], self.parameters['erb_max'], self.parameters['erb_channels'],
            vocoder_function=(tfs_vocoder_function, [tone_vocoded_channels]), 
            spacing='erb',
            filterbank_type=('fir'),
            envelope_method=('hilbert'),
            equalise=True
        )

        return vocoded

    def run(self, asr_system, dataset, adaptive_srt_estimation_flag=False):
        SNRs = np.arange(
            self.parameters['snr_min'], 
            self.parameters['snr_max'], 
            self.parameters['snr_step_db']
        ) 
        unzipped_cutoff_values = [
            ' - '.join([str(i) for i in np.atleast_1d(x)]) for x in self.parameters['cutoff_values']
        ]

        self._allocate_result_dataframe(
            [
                range(self.sentences_per_condition), 
                [str(x) for x in self.parameters['noise_types']],
                self.parameters['vocoder_types'], 
                unzipped_cutoff_values, 
                SNRs
            ],
            ['sentence_id', 'noise_type', 'vocoder_type', 'cutoff_values', 'SNR'],
            ['keyword_accuracy', 'predicted_transcription', 'transcription'],
            adaptive_srt_estimation=adaptive_srt_estimation_flag
        )

        for wav_file, transcription in dataset.data[:self.sentences_per_condition]:
            for noise_type in self.parameters['noise_types']:
                for vocoder_type in self.parameters['vocoder_types']:
                    for co in self.parameters['cutoff_values']:
                        sound = Sound(wav_file)
                        vocoded_sound = self._vocode(sound, co, vocoder_type=vocoder_type)
                        noise_data = self._generate_noise(vocoded_sound, noise_type)
                        trial_input = [asr_system, transcription, vocoded_sound, noise_data]
                        self._run_and_store_srt_trial(trial_input, SNRs, adaptive_srt_estimation_flag)


class ModulationSensitivityTest(Test):
    """
    As per The Modulation Transfer Function for Speech Intelligibility by Elliot and Theunissen (2009).
    """

    def __init__(self, folder='', store_wavs=False, sentences_per_condition=100, test_parameters=None):
        super(ModulationSensitivityTest, self).__init__(folder, store_wavs, sentences_per_condition)

        default_parameters = {
            'fband': 32,                        # Width of the frequency band.
            'n_std': 6,                         # Width of Gaussian window is N_STD times standard deviation.
            'db_noise': 80,                     # db in noise for log compression - values below are set to zero.
            'dfi': 0,                           # Ramp added to filtering in spectral domain in cycle/kHz.
            'dti': 0,                           # Ramp added to filtering in temporal domain in Hz.
            'sil_duration': 0.5,                # Duration of silence pre- and appended to sound in seconds.
            'spectrogram_sampling_rate': 1000,  # Sampling rate used to generate spectrogram.
            'speech_target_level_db': 65,
            'spectral_filters': [],
            'temporal_filters': [], 
            'combined_filters': [],
            'snr_levels_db': [], 
            'noise_types': [None], 
            'prefilter': None
        }

        self.parameters.update(default_parameters)
        if test_parameters:
            self.parameters.update(test_parameters)

    def _prepare_sound(self, sound, samplerate):
        # Prepend and append silence of SIL_DURATION to the sound.
        silence_in_frames = int(self.parameters['sil_duration']*samplerate)
        sound_with_silence = np.zeros(silence_in_frames*2 + len(sound))
        sound_with_silence[silence_in_frames:silence_in_frames + len(sound)] = sound
        # Normalise to have zero mean.
        sound_with_silence = sound_with_silence - np.mean(sound_with_silence)
        return Sound(sound_with_silence, samplerate)

    def _remove_silence(self, sound, samplerate):
        silence_in_frames = int(self.parameters['sil_duration']*samplerate)
        return sound[silence_in_frames:-silence_in_frames]
    
    def _put_at(self, inds, axis=-1, slc=(slice(None),)): 
        return (axis<0)*(Ellipsis,) + axis*slc + (inds,) + (-1-axis)*slc 
    
    def _apply_filter(self, mps_amplitude, mps_phase, 
                    thresholds, resolution, boundary, 
                    axis, ramp):

        low_bound, high_bound = thresholds
        if low_bound is None and high_bound is None:
            return

        if low_bound is None: # Highpass filter.
            low_bound = 0
            low_ramp = False
        else:
            low_bound += ramp
            low_ramp = True
        if high_bound is None: # Lowpass filter.
            high_bound = boundary
            high_ramp = False
        else:
            high_bound -= ramp
            high_ramp = True

        start_idx = int(np.ceil(low_bound/resolution))
        end_idx = int(np.floor(high_bound/resolution))

        # Remove amplitude of positive and negative modulations.
        positive_half_indices = self._put_at(range(start_idx, end_idx + 1), axis)
        mps_amplitude[positive_half_indices] = 0.0
        negative_half_indices = self._put_at(range(-(end_idx + 1), -start_idx ), axis)
        mps_amplitude[negative_half_indices] = 0.0

        # Randomise phase of positive and negative modulations.
        indices_shape = mps_phase[positive_half_indices].shape
        mps_phase[positive_half_indices] = (np.random.rand(*indices_shape) - 0.5)*2*np.pi
        mps_phase[negative_half_indices] = (np.random.rand(*indices_shape) - 0.5)*2*np.pi
        mps_phase[0, 0] = 0.0

        # Add ramp to amplitude for smoother transition. 
        ramp_width = int(ramp//resolution)
        ramp_up = 0.5*np.cos(np.linspace(0, np.pi, ramp_width)) + 0.5
        ramp_down = 0.5*np.cos(np.linspace(-np.pi, 0, ramp_width)) + 0.5

        if axis == 0: # Needed for row-wise multiplication.
            ramp_up, ramp_down = ramp_up[:, np.newaxis], ramp_down[:, np.newaxis]
        
        # Ramp is only added if the bound was given by the user; if the 
        # thresholds indicate a lowpass or highpass filter (i.e. (x, None)
        # or (None, x) for lowpass/highpass filter) no ramp is added for
        # the boundary threshold values.
        if low_ramp:
            mps_amplitude[
                self._put_at(range(start_idx - ramp_width, start_idx), axis)
            ] *= ramp_up
            mps_amplitude[
                self._put_at(range(-start_idx, -(start_idx - ramp_width)), axis)
            ] *= ramp_down
        if high_ramp:
            mps_amplitude[
                self._put_at(range(end_idx, end_idx + ramp_width), axis)
            ] *= ramp_down
            mps_amplitude[
                self._put_at(range(-(end_idx + ramp_width), -end_idx), axis)
            ] *= ramp_up

        return

    def _filter_modulation_power_spectrum(self, mps, temporal_thresholds,
                                          spectral_thresholds,
                                          temporal_boundary,
                                          spectral_boundary):

        spectral_resolution = spectral_boundary/(mps.shape[0]//2)
        temporal_resolution = temporal_boundary/(mps.shape[1]//2)
        mps_amplitude = np.abs(mps)
        mps_phase = np.angle(mps)
        self._apply_filter(
            mps_amplitude, mps_phase, spectral_thresholds, 
            spectral_resolution, spectral_boundary, 0, self.parameters['dfi'],
        )
        self._apply_filter(
            mps_amplitude, mps_phase, temporal_thresholds, 
            temporal_resolution, temporal_boundary, 1, self.parameters['dti'],
        )
        return mps_amplitude*np.exp(1j*mps_phase) 

    def filter_modulations(self, sound, temporal_thresholds, spectral_thresholds, plot=False):

        """
        spectral_thresholds are given in cycles/Hz, temporal thresholds in Hz
        """

        samplerate = sound.samplerate_Hz

        # Compute spectrogram and modulation power spectrum parameters.
        window_width_ms = 1000*self.parameters['n_std']/(self.parameters['fband']*2.0*np.pi)
        window_length = int(samplerate*(window_width_ms/1000))
        gauss_window = ('gaussian', window_length/6.0)
        n_overlap = int(window_length - samplerate/self.parameters['spectrogram_sampling_rate'])
        f_step = samplerate/window_length
        amplitude_modulation_rate = samplerate/int((1/self.parameters['spectrogram_sampling_rate'])*samplerate)
        temporal_boundary = 0.5*amplitude_modulation_rate 
        spectral_boundary = 0.5*(1/f_step)*1000
        sound_frames = len(sound)
        sound = self._prepare_sound(sound, samplerate)

        # Create log spectrogram.
        _, _, spectrogram = scipy.signal.stft(sound,
                                    fs=samplerate,
                                    window=gauss_window,
                                    nperseg=window_length,
                                    noverlap=n_overlap)
        max_spectrogram = np.max(np.abs(spectrogram))
        # Remove any part of the spectrogram under self.parameters['db_noise']. 
        log_spec = 20*np.log10(np.abs(spectrogram)/max_spectrogram) + self.parameters['db_noise']
        log_spec[log_spec < 0] = 0.0
        mps = scipy.fft.fft2(log_spec) # Modulation Power Spectrum is fft of the log spectrogram.
        mps_filtered = self._filter_modulation_power_spectrum(mps,
                                                              temporal_thresholds,
                                                              spectral_thresholds,
                                                              temporal_boundary,
                                                              spectral_boundary)
        # Invert filtered MPS back to sound.
        inverse_log_spec = np.real(scipy.fft.ifft2(mps_filtered))
        inverse_log_spec = (10**((inverse_log_spec - 80)/20.0))*np.abs(max_spectrogram)
        inverse_sound = librosa.core.griffinlim(
            inverse_log_spec,
            n_iter=20,
            hop_length=window_length - n_overlap,
            win_length=window_length - 1,
            window=gauss_window,
            init=None
        )
        inverse_sound = self._remove_silence(inverse_sound, samplerate)

        return Sound(inverse_sound[:sound_frames], samplerate)
    
    def run(self, asr_system, dataset):

        filters_all = [
            self.parameters['spectral_filters'], 
            self.parameters['temporal_filters'],
            self.parameters['combined_filters']
        ]
        filter_names = ['spectral', 'temporal', 'combined']

        for filter_name, filters in zip(filter_names, filters_all):
            unzipped_filters = [' - '.join([str(i) for i in np.atleast_1d(x)]) for x in filters]
            self._allocate_result_dataframe(
                [
                    range(self.sentences_per_condition), 
                    self.parameters['noise_types'], 
                    self.parameters['snr_levels_db'], 
                    unzipped_filters
                ],
                ['sentence_id', 'noise_type', 'SNR', '{}_filter'.format(filter_name)],
                ['keyword_accuracy', 'predicted_transcription', 'transcription'],
                result_name = filter_name
            )

        for k in range(self.sentences_per_condition):
            wav_file, transcription = dataset.data[k]
            sound = Sound(wav_file)

            # Reconstructing the modulation filtered sound is computationally expensive, 
            # so downsample beforehand if the ASR system allows us to. Don't downsample
            # when the test also stores the input files, as future users of those datasets
            # might want to test at different samplerate. 
            if not self.store_wavs and asr_system.samplerate_hz is not None:
                if sound.samplerate_Hz > asr_system.samplerate_hz:
                    sound = Sound.resample(sound, asr_system.samplerate_hz)

            if self.parameters['prefilter'] is not None:
                sound = self.filter_modulations(
                    sound, 
                    self.parameters['prefilter'][0:2], 
                    self.parameters['prefilter'][2:4], 
                )

            sound.level_dB = 65

            for n, noise_type in enumerate(self.parameters['noise_types']):
                noise = self._generate_noise(sound, noise_type)
                for i, SNR_level in enumerate(self.parameters['snr_levels_db']):
                    for filter_name, filters in zip(filter_names, filters_all):
                        for filter_i in filters:
                            if noise_type is None and i > 0: # Skip SNRs if no noise is added. 
                                word_accuracy = np.nan
                            else:
                                if filter_name == 'spectral':
                                    filtered_sound = self.filter_modulations(
                                        sound, (None, None), filter_i,
                                    )
                                elif filter_name == 'temporal':
                                    filtered_sound = self.filter_modulations(
                                        sound, filter_i, (None, None),
                                    )
                                elif filter_name == 'combined':
                                    filtered_sound = self.filter_modulations(
                                        sound, filter_i[0:2], filter_i[2:4]
                                    )

                                filtered_sound.level_dB = 65
                                mixed_sound = self._add_noise(
                                    filtered_sound, noise, SNR_level, 
                                    a_weighting=True, speech_volt_meter=True,
                                    speech_reference_level=self.parameters['speech_target_level_db']
                                )
                                prediction = asr_system.transcribe(mixed_sound)
                                accuracy, _ = self._keyword_accuracy(prediction.lower(), transcription.lower())

                            trial_outcome = {
                                'keyword_accuracy': accuracy,
                                'predicted_transcription': prediction,
                                'transcription': transcription
                            }
                            self._add_result(
                                trial_outcome,
                                sound=mixed_sound,
                                result_name=filter_name,
                                display=True)
                            self._store_results(filter_name)